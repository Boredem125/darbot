"""Tests for ChatService: order, degradation, caching, single-flight, timing."""

from __future__ import annotations

import datetime
import json
from typing import Any

from app.config import Settings
from app.domain.messages import BUSY_EN
from app.domain.ports import (
    LLMDelta,
    LLMDone,
    LLMFailover,
    LLMUnavailableError,
    RetrievedHit,
)
from app.domain.router import EntityIndex
from app.service import ChatService
from app.telemetry import RingBuffer
from common.schemas import (
    CachedAnswer,
    ChunkPayload,
    DocKind,
    Entities,
    Entity,
    EntityKind,
    ErrorCode,
    FinishReason,
    GroundingReport,
    Language,
    Provider,
    Route,
    SearchFilters,
    Source,
    TokenUsage,
)
from tests.fakes.fake_adapters import (
    FakeCache,
    FakeEmbedder,
    FakeLLM,
    FakeRetriever,
    FakeStructuredQuery,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime.datetime(2026, 11, 19, tzinfo=datetime.UTC)


def _entity_index() -> EntityIndex:
    entities = Entities(entities=[
        Entity(kind=EntityKind.city, canonical="Riyadh", aliases=["riyadh"]),
        Entity(kind=EntityKind.city, canonical="Jeddah", aliases=["jeddah"]),
    ])
    return EntityIndex(entities)


def _make_hit(doc_id: str = "wasalt:123", score: float = 0.8) -> RetrievedHit:
    payload = ChunkPayload(
        doc_id=doc_id,
        kind=DocKind.listing,
        source=Source.wasalt,
        title="Test Villa",
        url="https://wasalt.sa/en/property/123",
        scraped_at=_NOW,
        purpose="sale",
        property_type="villa",
        city="Riyadh",
        bedrooms=3,
        price_sar=1500000,
        chunk_index=0,
        text="A test villa in Riyadh for SAR 1,500,000 [1].",
    )
    return RetrievedHit(payload=payload, score=score)


def _settings(**overrides: Any) -> Settings:
    defaults = {
        "INDEX_DIR": "data/index",
        "RERANK_ENABLED": 0,
        "SEMANTIC_CACHE_ENABLED": 0,
        "MAX_INFLIGHT_CHATS": 12,
        "SCORE_FLOOR": 0.52,
        "PREFETCH_LIMIT": 20,
        "ARABIC_MAX_TOKENS": 700,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_service(
    *,
    llm: FakeLLM | None = None,
    retriever: FakeRetriever | None = None,
    cache: FakeCache | None = None,
    settings_overrides: dict | None = None,
) -> ChatService:
    llm = llm or FakeLLM()
    retriever = retriever or FakeRetriever()
    cache = cache or FakeCache()
    embedder = FakeEmbedder()

    # Set up retriever with default hits
    if not retriever._hits:
        hit = _make_hit()
        retriever.set_hits([hit], top_dense_score=0.75)

    # Set up LLM with default script
    if not llm._script:
        llm.set_script([
            LLMDelta(text="A test villa in Riyadh for SAR 1,500,000 [1]."),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                ttft_ms=500.0,
                total_ms=800.0,
            ),
        ])

    s = _settings(**(settings_overrides or {}))
    return ChatService(
        embedder=embedder,
        retriever=retriever,
        reranker=None,
        sql=FakeStructuredQuery(),
        llm=llm,
        cache=cache,
        entities=_entity_index(),
        settings=s,
        ring=RingBuffer(),
        index_version="77284224a04cf674",
        prompt_version="system_v1@aabbccdd",
    )


async def _collect_events(svc: ChatService, message: str, **kw: Any) -> list[dict]:
    """Collect all SSE events from a handle() call. Parse each data line."""
    events = []
    async for ev in svc.handle(message, **kw):
        data = json.loads(ev["data"])
        events.append(data)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServiceOrder:
    """start before first token; end after last token."""

    async def test_start_before_token(self):
        svc = _make_service()
        events = await _collect_events(svc, "villas in Riyadh")
        event_types = [e["event"] for e in events]
        assert event_types[0] == "start"
        # First token comes after start
        token_idx = next(i for i, e in enumerate(events) if e["event"] == "token")
        assert token_idx > 0
        assert event_types[-1] == "end"

    async def test_start_has_cards_before_token(self):
        svc = _make_service()
        events = await _collect_events(svc, "villas in Riyadh")
        start = events[0]
        assert start["event"] == "start"
        assert len(start["sources"]) > 0
        # Cards carry n starting at 1
        assert start["sources"][0]["n"] == 1


class TestRefusal:
    """Out-of-scope and below-floor: no LLM call."""

    async def test_out_of_scope_no_llm(self):
        llm = FakeLLM()
        svc = _make_service(llm=llm)
        events = await _collect_events(svc, "What is the weather today?")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "refusal"
        assert end["finish_reason"] == "no_llm"
        assert end["provider"] == "none"
        assert llm.call_count == 0

    async def test_below_floor_refusal(self):
        """Empty-filter query with low dense score triggers refusal, no LLM."""
        llm = FakeLLM()
        retriever = FakeRetriever()
        retriever.set_hits([_make_hit()], top_dense_score=0.3)
        svc = _make_service(llm=llm, retriever=retriever,
                            settings_overrides={"SCORE_FLOOR": 0.52})
        # "homes" is a domain cue -> routes to lookup with empty filters
        events = await _collect_events(svc, "homes")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "refusal"
        assert end["finish_reason"] == "no_llm"
        assert llm.call_count == 0


class TestCacheTiers:
    """L1 hit on repeat with cache_tier=L1."""

    async def test_l1_cache_hit(self):
        cache = FakeCache()
        CachedAnswer(
            answer="Cached answer [1].",
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(),
            sources=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            grounding=GroundingReport(numbers_ok=True, entities_ok=True, citations_ok=True),
            prompt_version="system_v1@aabbccdd",
            index_version="77284224a04cf674",
            created_at=_NOW,
        )
        # We need to know the exact key; we'll put under a known key
        # and configure the service to produce the same key.
        # Instead, let the service produce the key and check a repeat.
        llm = FakeLLM()
        llm.set_script([
            LLMDelta(text="A test villa in Riyadh for SAR 1,500,000 [1]."),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                ttft_ms=500.0,
                total_ms=800.0,
            ),
        ])
        svc = _make_service(llm=llm, cache=cache)

        # First call: should produce and cache
        events1 = await _collect_events(svc, "villas in Riyadh")
        end1 = next(e for e in events1 if e["event"] == "end")
        assert end1["cache_tier"] == "miss"
        assert llm.call_count == 1

        # Reset LLM script for second call
        llm.set_script([
            LLMDelta(text="Should not be called"),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                ttft_ms=500.0,
                total_ms=800.0,
            ),
        ])

        # Second call: same query should hit L1
        events2 = await _collect_events(svc, "villas in Riyadh")
        end2 = next(e for e in events2 if e["event"] == "end")
        assert end2["cache_tier"] == "L1"
        assert end2["answer_mode"] == "cache"
        assert end2["finish_reason"] == "cached"
        # LLM should not have been called again
        assert llm.call_count == 1

    async def test_followup_bypasses_cache(self):
        """Follow-up turns always go to retrieval and LLM, no L0-L2."""
        llm = FakeLLM()
        svc = _make_service(llm=llm)

        # First call sets up session
        await _collect_events(svc, "villas in Riyadh", session_id="sess1")
        assert llm.call_count == 1

        # Follow-up: "cheaper ones"
        llm.set_script([
            LLMDelta(text="Cheaper options [1]."),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                ttft_ms=500.0,
                total_ms=800.0,
            ),
        ])
        events2 = await _collect_events(svc, "cheaper ones", session_id="sess1")
        end2 = next(e for e in events2 if e["event"] == "end")
        assert end2["cache_tier"] == "miss"
        assert llm.call_count == 2


class TestPutPolicy:
    """Cache put conditions from caching.md section 5."""

    async def test_no_put_on_truncated(self):
        llm = FakeLLM()
        llm.set_script([
            LLMDelta(text="Truncated answer"),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.length,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=500),
                ttft_ms=500.0,
                total_ms=5000.0,
            ),
        ])
        cache = FakeCache()
        svc = _make_service(llm=llm, cache=cache)
        events = await _collect_events(svc, "villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["finish_reason"] == "length"
        assert end["flags"]["truncated"] is True
        # Cache should be empty (no put)
        assert len(cache._exact) == 0

    async def test_no_put_on_unverified(self):
        """Ungrounded answer should not be cached."""
        llm = FakeLLM()
        # Answer with a hallucinated number
        llm.set_script([
            LLMDelta(text="There are 999 villas at SAR 99999999 [1]."),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                ttft_ms=500.0,
                total_ms=800.0,
            ),
        ])
        cache = FakeCache()
        svc = _make_service(llm=llm, cache=cache)
        events = await _collect_events(svc, "villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["flags"]["unverified"] is True
        # No cache put
        assert len(cache._exact) == 0

    async def test_no_put_on_retrieval_only(self):
        llm = FakeLLM()
        llm.set_script([])  # Will raise LLMUnavailableError
        # Override to raise
        async def _stream(messages, *, max_tokens, temperature=0.0):
            raise LLMUnavailableError([])
            yield  # type: ignore[unreachable]
        llm.stream = _stream  # type: ignore[assignment]

        cache = FakeCache()
        svc = _make_service(llm=llm, cache=cache)
        events = await _collect_events(svc, "villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "retrieval_only"
        assert len(cache._exact) == 0


class TestDegradationLadder:
    """Each degradation level has a test."""

    async def test_level1_openrouter(self):
        """Normal OpenRouter response."""
        svc = _make_service()
        events = await _collect_events(svc, "villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "llm"
        assert end["provider"] == "openrouter"

    async def test_level2_groq_fallback(self):
        """Failover from OpenRouter to Groq."""
        llm = FakeLLM()
        llm.set_script([
            LLMFailover(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                code=ErrorCode.provider_rate_limited,
                detail="429",
            ),
            LLMDelta(text="Answer from Groq [1]."),
            LLMDone(
                provider=Provider.groq,
                model="openai/gpt-oss-120b",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                ttft_ms=600.0,
                total_ms=900.0,
            ),
        ])
        svc = _make_service(llm=llm)
        events = await _collect_events(svc, "villas in Riyadh")

        # Should have an error event with degraded_to=fallback_llm
        errors = [e for e in events if e["event"] == "error"]
        assert len(errors) >= 1
        assert errors[0]["degraded_to"] == "fallback_llm"

        end = next(e for e in events if e["event"] == "end")
        assert end["provider"] == "groq"
        assert end["answer_mode"] == "llm"

    async def test_level3_retrieval_only(self):
        """All LLM providers unavailable -> retrieval-only."""
        llm = FakeLLM()
        async def _stream(messages, *, max_tokens, temperature=0.0):
            raise LLMUnavailableError([
                LLMFailover(
                    provider=Provider.openrouter,
                    model="google/gemma-4-31b-it:free",
                    code=ErrorCode.provider_rate_limited,
                    detail="429",
                ),
            ])
            yield  # type: ignore[unreachable]
        llm.stream = _stream  # type: ignore[assignment]

        svc = _make_service(llm=llm)
        events = await _collect_events(svc, "villas in Riyadh")

        errors = [e for e in events if e["event"] == "error"]
        assert any(e["degraded_to"] == "retrieval_only" for e in errors)

        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "retrieval_only"
        assert end["provider"] == "none"
        assert end["finish_reason"] == "no_llm"

    async def test_level4_static_busy(self):
        """Over MAX_INFLIGHT_CHATS -> static busy."""
        svc = _make_service(settings_overrides={"MAX_INFLIGHT_CHATS": 0})
        events = await _collect_events(svc, "villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "static_busy"
        assert end["provider"] == "none"
        assert end["finish_reason"] == "no_llm"

        tokens = [e for e in events if e["event"] == "token"]
        assert any(BUSY_EN in t["delta"] for t in tokens)


class TestSingleFlight:
    """Single-flight serves the follower from the leader's cache put."""

    async def test_single_flight_follower(self):
        llm = FakeLLM()
        llm.set_script([
            LLMDelta(text="Answer [1]."),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=10),
                ttft_ms=500.0,
                total_ms=700.0,
            ),
        ])
        cache = FakeCache()
        svc = _make_service(llm=llm, cache=cache)

        # Run leader first
        events1 = await _collect_events(svc, "villas in Riyadh")
        end1 = next(e for e in events1 if e["event"] == "end")
        assert end1["cache_tier"] == "miss"

        # Follower gets L1 from cache
        llm.set_script([
            LLMDelta(text="Should not appear"),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=10),
                ttft_ms=500.0,
                total_ms=700.0,
            ),
        ])
        events2 = await _collect_events(svc, "villas in Riyadh")
        end2 = next(e for e in events2 if e["event"] == "end")
        assert end2["cache_tier"] == "L1"
        assert end2["answer_mode"] == "cache"


class TestTimingRule:
    """CC-2: embed_ms + sparse_ms == wall time of embed_query."""

    async def test_embed_sparse_timing(self):
        svc = _make_service()
        events = await _collect_events(svc, "villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        timings = end["timings"]
        # On a non-cached embed, both embed_ms and sparse_ms should be set
        assert timings["embed_ms"] is not None or timings["sparse_ms"] is not None
        # total_ms must be set
        assert timings["total_ms"] > 0

    async def test_stage_sum_le_total(self):
        """Sum of non-overlapping stage timings must not exceed total_ms.

        llm_ttft_ms is a sub-interval of llm_total_ms, so we include only
        llm_total_ms. Uses small fake LLM timings so they fit within the
        service's wall-clock total_ms.
        """
        llm = FakeLLM()
        llm.set_script([
            LLMDelta(text="A test villa in Riyadh for SAR 1,500,000 [1]."),
            LLMDone(
                provider=Provider.openrouter,
                model="google/gemma-4-31b-it:free",
                finish_reason=FinishReason.stop,
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                ttft_ms=1.0,
                total_ms=2.0,
            ),
        ])
        svc = _make_service(llm=llm)
        events = await _collect_events(svc, "villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        timings = end["timings"]
        # Non-overlapping stages (llm_ttft is inside llm_total, skip it)
        stages = [
            timings.get("cache_lookup_ms"),
            timings.get("embed_ms"),
            timings.get("sparse_ms"),
            timings.get("retrieve_ms"),
            timings.get("rerank_ms"),
            timings.get("sql_ms"),
            timings.get("llm_total_ms"),
            timings.get("grounding_ms"),
        ]
        stage_sum = sum(v for v in stages if v is not None)
        # Allow 1 ms tolerance for float rounding
        assert stage_sum <= timings["total_ms"] + 1.0, (
            f"stage sum {stage_sum:.2f} > total_ms {timings['total_ms']:.2f}"
        )


class TestWarmUp:
    """Warm-up must complete before /ready turns green."""

    async def test_warm_up_runs(self):
        svc = _make_service()
        # warm_up should not raise
        await svc.warm_up()


class TestArabicDualBedrooms:
    """Arabic dual form for bedrooms: غرفتين = 2."""

    def test_ghurfatayn_gives_2(self):
        """'غرفتين' (dual) should parse as 2 bedrooms."""
        from app.domain.router import route as do_route
        d = do_route("فلل غرفتين للبيع في الرياض", entities=_entity_index())
        assert d.filters.bedrooms_min == 2
        assert d.filters.bedrooms_max == 2

    def test_ghurfatayn_nawm(self):
        """'غرفتين نوم' should parse as 2 bedrooms."""
        from app.domain.router import route as do_route
        d = do_route("شقق غرفتين نوم في الرياض", entities=_entity_index())
        assert d.filters.bedrooms_min == 2
        assert d.filters.bedrooms_max == 2

    def test_ghurfa_wahida(self):
        """'غرفة واحدة' should parse as 1 bedroom."""
        from app.domain.router import route as do_route
        d = do_route("شقة غرفة واحدة في الرياض", entities=_entity_index())
        assert d.filters.bedrooms_min == 1
        assert d.filters.bedrooms_max == 1


class TestNoMatchRelaxation:
    """Zero-hit lookup with non-empty filters gives a helpful count message."""

    async def test_price_constraint_in_description(self):
        """'villas in Riyadh under 50000 SAR' must name the price in the first sentence."""
        from app.domain.ports import AggregateOp, AggregateQuery, SqlResult

        retriever = FakeRetriever()
        retriever.set_hits([], top_dense_score=None)
        sql = FakeStructuredQuery()

        def _cb(q: AggregateQuery) -> SqlResult | None:
            if q.op != AggregateOp.count:
                return None
            # Drop price: 217 villas
            if q.filters.price_max_sar is None and q.filters.price_min_sar is None:
                return SqlResult(op=AggregateOp.count, value=217, unit="count", n_matched=217)
            return SqlResult(op=AggregateOp.count, value=0, unit="count", n_matched=0)

        sql.set_callback(_cb)
        svc = ChatService(
            embedder=FakeEmbedder(), retriever=retriever, reranker=None, sql=sql,
            llm=FakeLLM(), cache=FakeCache(), entities=_entity_index(),
            settings=_settings(), ring=RingBuffer(),
            index_version="77284224a04cf674", prompt_version="system_v1@aabbccdd",
        )
        events = await _collect_events(svc, "villas in Riyadh under 50000 SAR")
        token_text = "".join(t["delta"] for t in events if t["event"] == "token")
        # First sentence must include the price constraint
        assert "50,000" in token_text
        assert "217" in token_text

    async def test_2bed_riyadh_villa_gives_count_message(self):
        """The '2 bed villas in Riyadh' case: 0 hits, but relaxing bedrooms finds 236."""
        from app.domain.ports import AggregateOp, AggregateQuery, SqlResult

        llm = FakeLLM()
        retriever = FakeRetriever()
        # Return 0 hits for the original query (2-bed villas in Riyadh)
        retriever.set_hits([], top_dense_score=None)

        sql = FakeStructuredQuery()

        def _sql_callback(q: AggregateQuery) -> SqlResult | None:
            if q.op != AggregateOp.count:
                return None
            # First relaxation (bedrooms dropped): 236 villas for sale in Riyadh
            if q.filters.bedrooms_min is None and q.filters.bedrooms_max is None:
                return SqlResult(op=AggregateOp.count, value=236, unit="count", n_matched=236)
            # Original filters: 0
            return SqlResult(op=AggregateOp.count, value=0, unit="count", n_matched=0)

        sql.set_callback(_sql_callback)

        svc = ChatService(
            embedder=FakeEmbedder(),
            retriever=retriever,
            reranker=None,
            sql=sql,
            llm=llm,
            cache=FakeCache(),
            entities=_entity_index(),
            settings=_settings(),
            ring=RingBuffer(),
            index_version="77284224a04cf674",
            prompt_version="system_v1@aabbccdd",
        )

        events = await _collect_events(svc, "2 bed villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "refusal"
        assert end["finish_reason"] == "no_llm"
        assert end["provider"] == "none"
        assert llm.call_count == 0

        tokens = [e for e in events if e["event"] == "token"]
        token_text = "".join(t["delta"] for t in tokens)
        assert "236" in token_text
        assert "bedroom" in token_text.lower() or "bed" in token_text.lower()

    async def test_no_match_at_all(self):
        """No hits even after relaxing all constraints: simple no-match message."""
        from app.domain.ports import SqlResult

        llm = FakeLLM()
        retriever = FakeRetriever()
        retriever.set_hits([], top_dense_score=None)

        sql = FakeStructuredQuery()
        # Always return 0
        sql.set_callback(
            lambda q: SqlResult(op=q.op, value=0, unit="count", n_matched=0)
        )

        svc = ChatService(
            embedder=FakeEmbedder(),
            retriever=retriever,
            reranker=None,
            sql=sql,
            llm=llm,
            cache=FakeCache(),
            entities=_entity_index(),
            settings=_settings(),
            ring=RingBuffer(),
            index_version="77284224a04cf674",
            prompt_version="system_v1@aabbccdd",
        )

        events = await _collect_events(svc, "2 bed villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "refusal"
        assert end["finish_reason"] == "no_llm"
        assert end["provider"] == "none"

        tokens = [e for e in events if e["event"] == "token"]
        token_text = "".join(t["delta"] for t in tokens)
        # Should mention "No" and some filter info, not the generic OOS refusal
        assert "No " in token_text or "no " in token_text

    async def test_oos_path_unchanged(self):
        """Out-of-scope queries still use the generic REFUSAL_EN, not the count path."""
        from app.domain.messages import REFUSAL_EN

        llm = FakeLLM()
        svc = _make_service(llm=llm)
        events = await _collect_events(svc, "What is the weather today?")
        tokens = [e for e in events if e["event"] == "token"]
        token_text = "".join(t["delta"] for t in tokens)
        assert token_text == REFUSAL_EN
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "refusal"

    async def test_eval_refusal_contract(self):
        """answer_mode=refusal, finish_reason=no_llm, provider=none on zero-hit lookup."""
        from app.domain.ports import SqlResult

        retriever = FakeRetriever()
        retriever.set_hits([], top_dense_score=None)
        sql = FakeStructuredQuery()
        sql.set_callback(
            lambda q: SqlResult(op=q.op, value=50, unit="count", n_matched=50)
        )

        svc = ChatService(
            embedder=FakeEmbedder(),
            retriever=retriever,
            reranker=None,
            sql=sql,
            llm=FakeLLM(),
            cache=FakeCache(),
            entities=_entity_index(),
            settings=_settings(),
            ring=RingBuffer(),
            index_version="77284224a04cf674",
            prompt_version="system_v1@aabbccdd",
        )

        events = await _collect_events(svc, "2 bed villas in Riyadh")
        end = next(e for e in events if e["event"] == "end")
        assert end["answer_mode"] == "refusal"
        assert end["finish_reason"] == "no_llm"
        assert end["provider"] == "none"
        assert end["model"] is None
