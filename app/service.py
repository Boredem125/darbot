"""ChatService: the single entry point for a chat turn.

Imports: common, domain, domain.ports, telemetry, config.  Never adapters.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from cachetools import TTLCache

from app.config import Settings
from app.domain.cache_keys import build_key, build_scope, parts_from
from app.domain.grounding import check as grounding_check
from app.domain.grounding import normalise_citations
from app.domain.messages import (
    BUSY_AR,
    BUSY_EN,
    REFUSAL_AR,
    REFUSAL_EN,
    describe_filters_ar,
    describe_filters_en,
    nomatch_none_ar,
    nomatch_none_en,
    nomatch_relaxed_ar,
    nomatch_relaxed_en,
)
from app.domain.policy import degrade_text_retrieval_only, is_cacheable
from app.domain.ports import (
    LLM,
    AggregateOp,
    AggregateQuery,
    Cache,
    ChatMessage,
    Embedder,
    EmbeddingError,
    LLMDelta,
    LLMDone,
    LLMFailover,
    LLMUnavailableError,
    QueryEmbedding,
    Reranker,
    RetrievalError,
    RetrievalResult,
    RetrievedHit,
    Retriever,
    SqlResult,
    StructuredQuery,
    StructuredQueryError,
)
from app.domain.prompting import build_messages, max_tokens_for
from app.domain.retrieval_post import (
    TOP_K_DOCS,
    DocHit,
    below_floor,
    collapse_to_docs,
    dedup_docs,
    to_cards,
)
from app.domain.router import EntityIndex, RouteDecision, SessionSlots, route
from app.telemetry import RequestRecord, RingBuffer, log_request
from common.cards import render_card
from common.schemas import (
    AnswerCacheTier,
    AnswerFlags,
    AnswerMode,
    CachedAnswer,
    Card,
    ChatEventEnd,
    ChatEventError,
    ChatEventStart,
    ChatEventToken,
    DegradedTo,
    ErrorCode,
    FinishReason,
    GroundingReport,
    InnerCacheTier,
    Language,
    Provider,
    Route,
    SearchFilters,
    SortKey,
    StageTimings,
    TokenUsage,
)
from common.text import arabic_ratio, normalise_for_embedding

logger = logging.getLogger("app.service")


def _listings_to_cards(examples: tuple) -> list[Card]:
    """Convert Listing examples from DuckDB to Cards for sorted lookup."""
    from common.schemas import ChunkPayload, DocKind, Source

    doc_hits: list[DocHit] = []
    for ex in examples:
        payload = ChunkPayload(
            doc_id=ex.id,
            kind=DocKind.listing,
            source=Source.wasalt,
            title=ex.title,
            url=ex.url,
            scraped_at=ex.scraped_at,
            purpose=ex.purpose,
            property_type=ex.property_type,
            city=ex.city,
            district=ex.district,
            price_sar=ex.price_sar,
            rent_period=ex.rent_period,
            annual_rent_sar=ex.annual_rent_sar,
            bedrooms=ex.bedrooms,
            bathrooms=ex.bathrooms,
            area_sqm=ex.area_sqm,
            chunk_index=0,
            text=ex.title,
        )
        doc_hits.append(DocHit(ex.id, 1.0, payload))
    return to_cards(doc_hits)


# ---------------------------------------------------------------------------
# ChatService
# ---------------------------------------------------------------------------


class ChatService:
    """Stateful service: holds sessions, caches, and the in-flight map."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        retriever: Retriever,
        reranker: Reranker | None,
        sql: StructuredQuery,
        llm: LLM,
        cache: Cache,
        entities: Any,  # EntityIndex or Entities (builds EntityIndex if needed)
        settings: Settings,
        ring: RingBuffer,
        index_version: str,
        prompt_version: str,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._sql = sql
        self._llm = llm
        self._cache = cache
        # Accept either EntityIndex or Entities (build from it)
        if isinstance(entities, EntityIndex):
            self._entities = entities
        else:
            self._entities = EntityIndex(entities)
        self._settings = settings
        self._ring = ring
        self._index_version = index_version
        self._prompt_version = prompt_version

        self._sessions: TTLCache[str, SessionSlots] = TTLCache(maxsize=1000, ttl=1800)
        self._inflight: int = 0
        # Single-flight map: exact cache key -> asyncio.Future[CachedAnswer | None]
        self._single_flight: dict[str, asyncio.Future[CachedAnswer | None]] = {}

    async def warm_up(self) -> None:
        """Run one embed_query and one trivial DuckDB query to warm ONNX sessions
        and DuckDB connection before /ready turns green."""
        logger.info("Warming up embedder...")
        await self._embedder.embed_query("warm up")
        logger.info("Warming up DuckDB...")
        try:
            await self._sql.aggregate(
                AggregateQuery(
                    op=AggregateOp.count,
                    filters=SearchFilters(),
                    example_limit=0,
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("DuckDB warm-up query failed (non-fatal)")
        logger.info("Warm-up complete")

    @property
    def sessions(self) -> TTLCache:
        return self._sessions

    @property
    def inflight(self) -> int:
        return self._inflight

    # ------------------------------------------------------------------
    # handle: async generator of SSE events
    # ------------------------------------------------------------------

    async def handle(
        self,
        message: str,
        *,
        session_id: str | None = None,
        lang_hint: Language | None = None,
    ) -> AsyncIterator[dict[str, str]]:
        """Process one chat turn. Yields SSE event dicts."""
        request_id = uuid.uuid4().hex
        t0 = time.perf_counter()

        # Admission control: static busy
        if self._inflight >= self._settings.MAX_INFLIGHT_CHATS:
            async for ev in self._static_busy(request_id, t0, lang_hint):
                yield ev
            return

        self._inflight += 1
        try:
            async for ev in self._handle_inner(
                message,
                request_id=request_id,
                t0=t0,
                session_id=session_id,
                lang_hint=lang_hint,
            ):
                yield ev
        finally:
            self._inflight -= 1

    async def _handle_inner(
        self,
        message: str,
        *,
        request_id: str,
        t0: float,
        session_id: str | None,
        lang_hint: Language | None,
    ) -> AsyncIterator[dict[str, str]]:
        # 1. Normalise
        norm_query = normalise_for_embedding(message)

        # 2. Route
        session = self._sessions.get(session_id) if session_id else None
        decision = route(
            message,
            session=session,
            entities=self._entities,
            lang_hint=lang_hint,
        )

        standalone = not decision.carried_slots
        cache_tier = AnswerCacheTier.miss
        cache_hits: list[InnerCacheTier] = []
        t_cache_lookup: float | None = None
        cached_answer: CachedAnswer | None = None
        dense_vec: tuple[float, ...] | None = None
        qe: QueryEmbedding | None = None
        t_embed: float | None = None
        t_sparse: float | None = None

        # Build cache key parts
        ck_parts = parts_from(decision, message, self._index_version, self._prompt_version)
        exact_key = build_key(ck_parts)
        scope = build_scope(ck_parts)

        # 3. Cache lookup (standalone turns only)
        # cache_lookup_ms times only cache work (get_exact + get_semantic);
        # the embed call is timed separately as embed_ms/sparse_ms per CC-2.
        if standalone:
            t_cache_ms = 0.0

            # Check single-flight: if same key already in-flight, await it
            if exact_key in self._single_flight:
                future = self._single_flight[exact_key]
                try:
                    result = await asyncio.wait_for(asyncio.shield(future), timeout=30.0)
                    if result is not None:
                        cached_answer = result
                        cache_tier = AnswerCacheTier.L1
                except (TimeoutError, asyncio.CancelledError, OSError):
                    logger.debug("Single-flight wait failed, proceeding on own")

            if cached_answer is None:
                # L0/L1 exact
                t_ex0 = time.perf_counter()
                hit = await self._cache.get_exact(exact_key)
                t_cache_ms += (time.perf_counter() - t_ex0) * 1000
                if hit is not None:
                    cached_answer = hit.entry
                    cache_tier = AnswerCacheTier(hit.tier)

            if cached_answer is None and self._settings.SEMANTIC_CACHE_ENABLED:
                # Need embedding for L2 (timed separately)
                try:
                    t_e0 = time.perf_counter()
                    qe = await self._embedder.embed_query(norm_query)
                    t_e1 = time.perf_counter()
                    w_ms = (t_e1 - t_e0) * 1000
                    if qe.cached:
                        cache_hits.append(InnerCacheTier.L3)
                        t_embed = None
                        t_sparse = None
                    else:
                        t_sparse = qe.sparse_ms
                        t_embed = max(w_ms - (qe.sparse_ms or 0), 0)
                    dense_vec = qe.dense

                    t_sem0 = time.perf_counter()
                    hit = await self._cache.get_semantic(dense_vec, scope=scope)
                    t_cache_ms += (time.perf_counter() - t_sem0) * 1000
                    if hit is not None:
                        cached_answer = hit.entry
                        cache_tier = AnswerCacheTier.L2
                except EmbeddingError:
                    pass  # fall through to retrieval

            t_cache_lookup = t_cache_ms

        # Register single-flight
        sf_future: asyncio.Future[CachedAnswer | None] | None = None
        if standalone and cached_answer is None and exact_key not in self._single_flight:
            sf_future = asyncio.get_event_loop().create_future()
            self._single_flight[exact_key] = sf_future

        try:
            # Serve from cache
            if cached_answer is not None:
                async for ev in self._serve_cached(
                    request_id, decision, cached_answer, cache_tier,
                    cache_hits, t0, t_cache_lookup, t_embed, t_sparse,
                ):
                    yield ev
                return

            # 4. Route-specific handling
            cards: list[Card] = []
            sql_result: SqlResult | None = None
            context_texts: list[str] = []
            t_retrieve: float | None = None
            t_rerank: float | None = None
            t_sql: float | None = None
            retrieval_result: RetrievalResult | None = None

            if decision.route == Route.out_of_scope:
                async for ev in self._serve_refusal(
                    request_id, decision, cache_tier, cache_hits, t0, t_cache_lookup,
                ):
                    yield ev
                return

            if decision.route == Route.aggregate:
                # DuckDB aggregate
                try:
                    t_s0 = time.perf_counter()
                    sql_result = await self._sql.aggregate(
                        AggregateQuery(
                            op=decision.op,
                            filters=decision.filters,
                            example_limit=decision.example_limit,
                        )
                    )
                    t_sql = (time.perf_counter() - t_s0) * 1000
                    # Build cards from examples
                    if sql_result.examples:
                        from common.schemas import ChunkPayload, DocKind, Source
                        doc_hits = []
                        for ex in sql_result.examples:
                            payload = ChunkPayload(
                                doc_id=ex.id,
                                kind=DocKind.listing,
                                source=Source.wasalt,
                                title=ex.title,
                                url=ex.url,
                                scraped_at=ex.scraped_at,
                                purpose=ex.purpose,
                                property_type=ex.property_type,
                                city=ex.city,
                                district=ex.district,
                                price_sar=ex.price_sar,
                                rent_period=ex.rent_period,
                                annual_rent_sar=ex.annual_rent_sar,
                                bedrooms=ex.bedrooms,
                                bathrooms=ex.bathrooms,
                                area_sqm=ex.area_sqm,
                                chunk_index=0,
                                text=ex.title,
                            )
                            doc_hits.append(DocHit(ex.id, 1.0, payload))
                        cards = to_cards(doc_hits)
                        context_texts = [
                            render_card(card, mode="context", n=card.n, snippet=card.snippet)
                            for card in cards
                        ]
                except StructuredQueryError:
                    # Degrade: emit error then static busy
                    yield ChatEventError(
                        request_id=request_id,
                        code=ErrorCode.sql_failed,
                        message="Query processing failed. Please try again.",
                        degraded_to=DegradedTo.static_busy,
                    ).to_sse()
                    async for ev in self._serve_static_busy_after_error(
                        request_id, decision, cache_hits, t0, t_cache_lookup,
                    ):
                        yield ev
                    return
            elif (
                decision.route == Route.lookup
                and decision.filters.sort != SortKey.relevance
            ):
                # Sorted lookup: use DuckDB for sorted results instead of relevance
                try:
                    t_s0 = time.perf_counter()
                    sorted_result = await self._sql.aggregate(
                        AggregateQuery(
                            op=AggregateOp.count,
                            filters=decision.filters,
                            example_limit=TOP_K_DOCS,
                        )
                    )
                    t_sql = (time.perf_counter() - t_s0) * 1000
                    if sorted_result.examples:
                        cards = _listings_to_cards(sorted_result.examples)
                        context_texts = [
                            render_card(card, mode="context", n=card.n, snippet=card.snippet)
                            for card in cards
                        ]
                except StructuredQueryError:
                    pass  # fall through to retrieval below

                # If DuckDB returned cards, skip retrieval
                if cards:
                    # Still need embedding for the LLM prompt context (and cache key)
                    if qe is None:
                        try:
                            t_e0 = time.perf_counter()
                            qe = await self._embedder.embed_query(norm_query)
                            t_e1 = time.perf_counter()
                            w_ms = (t_e1 - t_e0) * 1000
                            if qe.cached:
                                cache_hits.append(InnerCacheTier.L3)
                            else:
                                t_sparse = qe.sparse_ms
                                t_embed = max(w_ms - (qe.sparse_ms or 0), 0)
                            dense_vec = qe.dense
                        except EmbeddingError:
                            pass  # proceed without embedding
                else:
                    # Fallback to normal retrieval if DuckDB returned nothing
                    cards = []  # reset

            if not cards and decision.route in (Route.lookup, Route.project_info):
                # Lookup / project_info: embed + retrieve
                # Get embedding if not already done
                if qe is None:
                    try:
                        t_e0 = time.perf_counter()
                        qe = await self._embedder.embed_query(norm_query)
                        t_e1 = time.perf_counter()
                        w_ms = (t_e1 - t_e0) * 1000
                        if qe.cached:
                            cache_hits.append(InnerCacheTier.L3)
                            t_embed = None
                            t_sparse = None
                        else:
                            t_sparse = qe.sparse_ms
                            t_embed = max(w_ms - (qe.sparse_ms or 0), 0)
                        dense_vec = qe.dense
                    except EmbeddingError:
                        # Fatal: cannot retrieve without embedding
                        yield ChatEventError(
                            request_id=request_id,
                            code=ErrorCode.retrieval_failed,
                            message="Search is temporarily unavailable.",
                            degraded_to=DegradedTo.static_busy,
                        ).to_sse()
                        async for ev in self._serve_static_busy_after_error(
                            request_id, decision, cache_hits, t0, t_cache_lookup,
                        ):
                            yield ev
                        return

                # Retrieve
                try:
                    t_r0 = time.perf_counter()
                    retrieval_result = await self._retriever.search(
                        qe, decision.filters, limit=self._settings.PREFETCH_LIMIT,
                    )
                    t_retrieve = (time.perf_counter() - t_r0) * 1000
                    if retrieval_result.cached:
                        cache_hits.append(InnerCacheTier.L4)
                except RetrievalError:
                    yield ChatEventError(
                        request_id=request_id,
                        code=ErrorCode.retrieval_failed,
                        message="Search is temporarily unavailable.",
                        degraded_to=DegradedTo.static_busy,
                    ).to_sse()
                    async for ev in self._serve_static_busy_after_error(
                        request_id, decision, cache_hits, t0, t_cache_lookup,
                    ):
                        yield ev
                    return

                # Score floor / zero-hit check
                if below_floor(
                    n_hits=len(retrieval_result.hits),
                    filters_is_empty=decision.filters.is_empty(),
                    top_dense_score=retrieval_result.top_dense_score,
                    floor=self._settings.SCORE_FLOOR,
                ):
                    # Zero hits with non-empty filters: try relaxation
                    if (
                        len(retrieval_result.hits) == 0
                        and not decision.filters.is_empty()
                    ):
                        async for ev in self._serve_nomatch_relaxed(
                            request_id, decision, cache_tier, cache_hits,
                            t0, t_cache_lookup, t_embed, t_sparse, t_retrieve,
                            qe=qe,
                        ):
                            yield ev
                        return

                    async for ev in self._serve_refusal(
                        request_id, decision, cache_tier, cache_hits, t0,
                        t_cache_lookup, t_embed=t_embed, t_sparse=t_sparse,
                        t_retrieve=t_retrieve,
                    ):
                        yield ev
                    return

                # Collapse to docs + optional rerank
                hits_for_collapse = [
                    (hit.payload, hit.score) for hit in retrieval_result.hits
                ]
                project_id = (
                    decision.filters.project_id
                    if decision.route == Route.project_info
                    else None
                )
                rerank_enabled = bool(
                    self._settings.RERANK_ENABLED
                    and self._reranker is not None
                    and decision.language == Language.en
                )
                collapse_k = 10 if rerank_enabled else TOP_K_DOCS
                doc_hits = collapse_to_docs(
                    hits_for_collapse, top_k=collapse_k, project_id=project_id,
                )

                if rerank_enabled and self._reranker is not None and doc_hits:
                    t_rr0 = time.perf_counter()
                    rh_list = [
                        RetrievedHit(payload=dh.best_chunk, score=dh.best_score)
                        for dh in doc_hits
                    ]
                    reranked = await self._reranker.rerank(
                        norm_query, rh_list, top_k=TOP_K_DOCS,
                    )
                    t_rerank = (time.perf_counter() - t_rr0) * 1000
                    doc_hits = [
                        DocHit(rh.payload.doc_id, rh.score, rh.payload)
                        for rh in reranked
                    ]

                doc_hits = dedup_docs(doc_hits, top_k=TOP_K_DOCS)
                cards = to_cards(doc_hits)
                context_texts = [
                    render_card(card, mode="context", n=card.n, snippet=card.snippet)
                    for card in cards
                ]

            # 5. start event (before first token)
            start_event = ChatEventStart(
                request_id=request_id,
                route=decision.route,
                language=decision.language,
                filters=decision.filters,
                cache_tier=cache_tier,
                sources=cards,
            )
            yield start_event.to_sse()

            # 6. Prompt
            system_prompt = self._load_system_prompt()
            cards_context = context_texts
            include_history = bool(decision.carried_slots) and session is not None
            history = session.history if include_history and session else ()
            prompt_plan = build_messages(
                system_prompt=system_prompt,
                cards_context=cards_context,
                sql_block=sql_result,
                history=history,
                question=message,
                include_history=include_history,
                language=decision.language,
            )

            max_tok = max_tokens_for(decision.language)
            if decision.language == Language.ar and self._settings.ARABIC_MAX_TOKENS:
                max_tok = self._settings.ARABIC_MAX_TOKENS

            # 7. LLM call (service-side timing for disjoint stage spans)
            answer_text = ""
            done_info: LLMDone | None = None
            failovers: list[LLMFailover] = []
            llm_error_after_first_token = False
            t_llm0 = time.perf_counter()
            t_llm_ttft: float | None = None

            try:
                async for item in self._llm.stream(
                    prompt_plan.messages,
                    max_tokens=max_tok,
                    temperature=0.0,
                ):
                    if isinstance(item, LLMFailover):
                        failovers.append(item)
                        yield ChatEventError(
                            request_id=request_id,
                            code=item.code,
                            message="Primary model is busy; switching to a fallback model.",
                            degraded_to=DegradedTo.fallback_llm,
                        ).to_sse()
                    elif isinstance(item, LLMDelta):
                        if t_llm_ttft is None:
                            t_llm_ttft = (time.perf_counter() - t_llm0) * 1000
                        answer_text += item.text
                        yield ChatEventToken(
                            request_id=request_id,
                            delta=item.text,
                        ).to_sse()
                    elif isinstance(item, LLMDone):
                        done_info = item
                        if item.finish_reason == FinishReason.error and answer_text:
                            llm_error_after_first_token = True
                            yield ChatEventError(
                                request_id=request_id,
                                code=ErrorCode.llm_interrupted,
                                message="The response was interrupted.",
                                degraded_to=None,
                            ).to_sse()
            except LLMUnavailableError as e:
                failovers = list(e.attempts)
                yield ChatEventError(
                    request_id=request_id,
                    code=ErrorCode.all_providers_unavailable,
                    message="All language models are currently busy.",
                    degraded_to=DegradedTo.retrieval_only,
                ).to_sse()
                # Retrieval-only fallback
                async for ev in self._serve_retrieval_only(
                    request_id, decision, cards, cache_tier, cache_hits,
                    t0, t_cache_lookup, t_embed, t_sparse, t_retrieve,
                    t_rerank, t_sql,
                ):
                    yield ev
                return

            t_llm_total = (time.perf_counter() - t_llm0) * 1000

            if done_info is None:
                # Shouldn't happen, but safety
                return

            # Normalise fullwidth citation brackets (Groq gpt-oss writes 【n】)
            answer_text = normalise_citations(answer_text)

            # 8. Grounding
            t_g0 = time.perf_counter()
            grounding = grounding_check(
                answer_text,
                context_texts=context_texts,
                sql=sql_result,
                query=message,
                entities=self._entities,
                k=len(cards),
                sql_n_matched=sql_result.n_matched if sql_result is not None else None,
            )
            t_grounding = (time.perf_counter() - t_g0) * 1000

            # Determine finish_reason and flags
            finish = done_info.finish_reason
            truncated = finish in (FinishReason.length, FinishReason.error)
            if llm_error_after_first_token:
                finish = FinishReason.error
                truncated = True
            unverified = not grounding.grounded

            # 9. end event
            total_ms = (time.perf_counter() - t0) * 1000
            timings = StageTimings(
                cache_lookup_ms=t_cache_lookup,
                embed_ms=t_embed,
                sparse_ms=t_sparse,
                retrieve_ms=t_retrieve,
                rerank_ms=t_rerank,
                sql_ms=t_sql,
                llm_ttft_ms=t_llm_ttft,
                llm_total_ms=t_llm_total,
                grounding_ms=t_grounding,
                total_ms=total_ms,
            )
            end_event = ChatEventEnd(
                request_id=request_id,
                answer_mode=AnswerMode.llm,
                cache_tier=cache_tier,
                cache_hits=list(cache_hits),
                provider=done_info.provider,
                model=done_info.model,
                tokens=done_info.usage,
                finish_reason=finish,
                timings=timings,
                grounding=grounding,
                flags=AnswerFlags(truncated=truncated, unverified=unverified),
                prompt_version=self._prompt_version,
                index_version=self._index_version,
            )
            yield end_event.to_sse()

            # 10. Cache put (caching.md section 5)
            # Also reject answers whose script contradicts the decision language
            ar = arabic_ratio(answer_text)
            script_ok = not (
                (decision.language == Language.en and ar > 0.5)
                or (decision.language == Language.ar and ar < 0.2)
            )
            if script_ok and is_cacheable(
                standalone=standalone,
                route=decision.route,
                answer_mode=AnswerMode.llm,
                finish_reason=finish,
                grounding=grounding,
            ):
                import datetime
                cached = CachedAnswer(
                    answer=answer_text,
                    route=decision.route,
                    language=decision.language,
                    filters=decision.filters,
                    sources=cards,
                    provider=done_info.provider,
                    model=done_info.model,
                    grounding=grounding,
                    prompt_version=self._prompt_version,
                    index_version=self._index_version,
                    created_at=datetime.datetime.now(datetime.UTC),
                )
                await self._cache.put(
                    exact_key,
                    cached,
                    dense=dense_vec,
                    scope=scope if dense_vec is not None else None,
                )
                # Resolve single-flight
                if sf_future is not None and not sf_future.done():
                    sf_future.set_result(cached)
            else:
                if sf_future is not None and not sf_future.done():
                    sf_future.set_result(None)

            # 11. Telemetry
            record = RequestRecord(
                request_id=request_id,
                route=decision.route,
                cache_tier=cache_tier,
                cache_hits=tuple(cache_hits),
                provider=done_info.provider,
                model=done_info.model,
                prompt_tokens=done_info.usage.prompt_tokens,
                completion_tokens=done_info.usage.completion_tokens,
                timings=timings,
                ts=time.time(),
            )
            self._ring.push(record)
            log_request(record)

            # Update session
            if session_id:
                new_history = ()
                if session is not None and session.history:
                    new_history = session.history
                new_history = new_history + (
                    ChatMessage(role="user", content=message),
                    ChatMessage(role="assistant", content=answer_text),
                )
                # Keep last 2 turns (4 messages)
                if len(new_history) > 4:
                    new_history = new_history[-4:]
                prices = tuple(c.price_sar for c in cards if c.price_sar is not None)
                areas = tuple(c.area_sqm for c in cards if c.area_sqm is not None)
                self._sessions[session_id] = SessionSlots(
                    filters=decision.filters,
                    last_prices=prices,
                    last_areas=areas,
                    history=new_history,
                )

        finally:
            # Clean up single-flight
            if sf_future is not None:
                self._single_flight.pop(exact_key, None)
                if not sf_future.done():
                    sf_future.set_result(None)

    # ------------------------------------------------------------------
    # Helpers for specific answer modes
    # ------------------------------------------------------------------

    def _load_system_prompt(self) -> str:
        """Return the system prompt text. Cached after first load."""
        if not hasattr(self, "_system_prompt_text"):
            import pathlib
            prompt_path = pathlib.Path(__file__).parent / "prompts" / "system_v1.txt"
            self._system_prompt_text = prompt_path.read_text(encoding="utf-8").strip()
        return self._system_prompt_text

    async def _serve_cached(
        self,
        request_id: str,
        decision: RouteDecision,
        cached: CachedAnswer,
        cache_tier: AnswerCacheTier,
        cache_hits: list[InnerCacheTier],
        t0: float,
        t_cache_lookup: float | None,
        t_embed: float | None,
        t_sparse: float | None,
    ) -> AsyncIterator[dict[str, str]]:
        yield ChatEventStart(
            request_id=request_id,
            route=decision.route,
            language=decision.language,
            filters=decision.filters,
            cache_tier=cache_tier,
            sources=cached.sources,
        ).to_sse()

        yield ChatEventToken(
            request_id=request_id,
            delta=cached.answer,
        ).to_sse()

        total_ms = (time.perf_counter() - t0) * 1000
        timings = StageTimings(
            cache_lookup_ms=t_cache_lookup,
            embed_ms=t_embed,
            sparse_ms=t_sparse,
            total_ms=total_ms,
        )
        yield ChatEventEnd(
            request_id=request_id,
            answer_mode=AnswerMode.cache,
            cache_tier=cache_tier,
            cache_hits=list(cache_hits),
            provider=cached.provider,
            model=cached.model,
            tokens=TokenUsage(),
            finish_reason=FinishReason.cached,
            timings=timings,
            grounding=cached.grounding,
            flags=AnswerFlags(truncated=False, unverified=not cached.grounding.grounded),
            prompt_version=self._prompt_version,
            index_version=self._index_version,
        ).to_sse()

        record = RequestRecord(
            request_id=request_id,
            route=decision.route,
            cache_tier=cache_tier,
            cache_hits=tuple(cache_hits),
            provider=cached.provider,
            model=cached.model,
            prompt_tokens=0,
            completion_tokens=0,
            timings=timings,
            ts=time.time(),
        )
        self._ring.push(record)
        log_request(record)

    async def _serve_refusal(
        self,
        request_id: str,
        decision: RouteDecision,
        cache_tier: AnswerCacheTier,
        cache_hits: list[InnerCacheTier],
        t0: float,
        t_cache_lookup: float | None,
        t_embed: float | None = None,
        t_sparse: float | None = None,
        t_retrieve: float | None = None,
    ) -> AsyncIterator[dict[str, str]]:
        refusal = REFUSAL_AR if decision.language == Language.ar else REFUSAL_EN

        yield ChatEventStart(
            request_id=request_id,
            route=decision.route,
            language=decision.language,
            filters=decision.filters,
            cache_tier=cache_tier,
            sources=[],
        ).to_sse()

        yield ChatEventToken(
            request_id=request_id,
            delta=refusal,
        ).to_sse()

        total_ms = (time.perf_counter() - t0) * 1000
        timings = StageTimings(
            cache_lookup_ms=t_cache_lookup,
            embed_ms=t_embed,
            sparse_ms=t_sparse,
            retrieve_ms=t_retrieve,
            total_ms=total_ms,
        )
        grounding = GroundingReport()
        yield ChatEventEnd(
            request_id=request_id,
            answer_mode=AnswerMode.refusal,
            cache_tier=cache_tier,
            cache_hits=list(cache_hits),
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(),
            finish_reason=FinishReason.no_llm,
            timings=timings,
            grounding=grounding,
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=self._prompt_version,
            index_version=self._index_version,
        ).to_sse()

        record = RequestRecord(
            request_id=request_id,
            route=decision.route,
            cache_tier=cache_tier,
            cache_hits=tuple(cache_hits),
            provider=Provider.none,
            model=None,
            prompt_tokens=0,
            completion_tokens=0,
            timings=timings,
            ts=time.time(),
        )
        self._ring.push(record)
        log_request(record)

    async def _serve_nomatch_relaxed(
        self,
        request_id: str,
        decision: RouteDecision,
        cache_tier: AnswerCacheTier,
        cache_hits: list[InnerCacheTier],
        t0: float,
        t_cache_lookup: float | None,
        t_embed: float | None,
        t_sparse: float | None,
        t_retrieve: float | None,
        qe: QueryEmbedding | None = None,
    ) -> AsyncIterator[dict[str, str]]:
        """Zero-hit lookup with non-empty filters: relax constraints and give a count."""
        f = decision.filters
        is_en = decision.language == Language.en
        _describe = describe_filters_en if is_en else describe_filters_ar

        # Build kwargs for the full original description
        desc_kw: dict = {
            "property_type": f.property_type.value if f.property_type else None,
            "purpose": f.purpose.value if f.purpose else None,
            "city": f.city,
            "district": f.district,
            "bedrooms": (
                str(f.bedrooms_min)
                if f.bedrooms_min is not None and f.bedrooms_min == f.bedrooms_max
                else None
            ),
            "price_min": f.price_min_sar,
            "price_max": f.price_max_sar,
            "area_min": f.area_min_sqm,
            "area_max": f.area_max_sqm,
        }
        orig_desc = _describe(**desc_kw)

        # Try relaxing one constraint at a time: bedrooms, price, area, district
        relaxation_steps: list[tuple[str, SearchFilters, dict]] = []

        if f.bedrooms_min is not None or f.bedrooms_max is not None:
            relaxation_steps.append((
                "bedrooms",
                SearchFilters(
                    kinds=f.kinds, source=f.source, purpose=f.purpose,
                    property_type=f.property_type, city=f.city, district=f.district,
                    country=f.country, project_id=f.project_id,
                    price_min_sar=f.price_min_sar, price_max_sar=f.price_max_sar,
                    area_min_sqm=f.area_min_sqm, area_max_sqm=f.area_max_sqm,
                ),
                {**desc_kw, "bedrooms": None},
            ))

        if f.price_min_sar is not None or f.price_max_sar is not None:
            relaxation_steps.append((
                "price",
                SearchFilters(
                    kinds=f.kinds, source=f.source, purpose=f.purpose,
                    property_type=f.property_type, city=f.city, district=f.district,
                    country=f.country, project_id=f.project_id,
                    bedrooms_min=f.bedrooms_min, bedrooms_max=f.bedrooms_max,
                    area_min_sqm=f.area_min_sqm, area_max_sqm=f.area_max_sqm,
                ),
                {**desc_kw, "price_min": None, "price_max": None},
            ))

        if f.area_min_sqm is not None or f.area_max_sqm is not None:
            relaxation_steps.append((
                "area",
                SearchFilters(
                    kinds=f.kinds, source=f.source, purpose=f.purpose,
                    property_type=f.property_type, city=f.city, district=f.district,
                    country=f.country, project_id=f.project_id,
                    bedrooms_min=f.bedrooms_min, bedrooms_max=f.bedrooms_max,
                    price_min_sar=f.price_min_sar, price_max_sar=f.price_max_sar,
                ),
                {**desc_kw, "area_min": None, "area_max": None},
            ))

        if f.district is not None:
            relaxation_steps.append((
                "district",
                SearchFilters(
                    kinds=f.kinds, source=f.source, purpose=f.purpose,
                    property_type=f.property_type, city=f.city,
                    country=f.country, project_id=f.project_id,
                    bedrooms_min=f.bedrooms_min, bedrooms_max=f.bedrooms_max,
                    price_min_sar=f.price_min_sar, price_max_sar=f.price_max_sar,
                    area_min_sqm=f.area_min_sqm, area_max_sqm=f.area_max_sqm,
                ),
                {**desc_kw, "district": None},
            ))

        # Try each relaxation (at most 4 DuckDB calls)
        relaxed_text: str | None = None
        relaxed_filters: SearchFilters | None = None
        t_sql_relax: float = 0.0

        for field_name, trial_filters, trial_desc_kw in relaxation_steps:
            t_sq0 = time.perf_counter()
            try:
                result = await self._sql.aggregate(
                    AggregateQuery(op=AggregateOp.count, filters=trial_filters, example_limit=0)
                )
            except StructuredQueryError:
                continue
            finally:
                t_sql_relax += (time.perf_counter() - t_sq0) * 1000

            if result.n_matched > 0:
                relaxed_desc = (
                    describe_filters_en(**trial_desc_kw)
                    if is_en
                    else describe_filters_ar(**trial_desc_kw)
                )
                if is_en:
                    relaxed_text = nomatch_relaxed_en(
                        description=orig_desc,
                        relaxed_field=field_name,
                        relaxed_count=result.n_matched,
                        relaxed_description=relaxed_desc,
                    )
                else:
                    relaxed_text = nomatch_relaxed_ar(
                        description=orig_desc,
                        relaxed_field=field_name,
                        relaxed_count=result.n_matched,
                        relaxed_description=relaxed_desc,
                    )
                relaxed_filters = trial_filters
                break

        if relaxed_text is None:
            relaxed_text = nomatch_none_en(orig_desc) if is_en else nomatch_none_ar(orig_desc)

        # Try retrieval with relaxed filters for cards (reuse existing embedding)
        cards: list[Card] = []
        t_retrieve_relax: float | None = None
        if relaxed_filters is not None and qe is not None:
            try:
                t_rr0 = time.perf_counter()
                relaxed_result = await self._retriever.search(
                    qe, relaxed_filters, limit=self._settings.PREFETCH_LIMIT,
                )
                t_retrieve_relax = (time.perf_counter() - t_rr0) * 1000
                if relaxed_result.hits:
                    hits_for_collapse = [
                        (hit.payload, hit.score) for hit in relaxed_result.hits
                    ]
                    doc_hits = collapse_to_docs(hits_for_collapse, top_k=TOP_K_DOCS)
                    doc_hits = dedup_docs(doc_hits, top_k=TOP_K_DOCS)
                    cards = to_cards(doc_hits)
            except RetrievalError:
                pass  # no cards is fine

        yield ChatEventStart(
            request_id=request_id,
            route=decision.route,
            language=decision.language,
            filters=decision.filters,
            cache_tier=cache_tier,
            sources=cards,
        ).to_sse()

        yield ChatEventToken(request_id=request_id, delta=relaxed_text).to_sse()

        total_ms = (time.perf_counter() - t0) * 1000
        timings = StageTimings(
            cache_lookup_ms=t_cache_lookup,
            embed_ms=t_embed,
            sparse_ms=t_sparse,
            retrieve_ms=(t_retrieve or 0) + (t_retrieve_relax or 0) if t_retrieve is not None or t_retrieve_relax is not None else None,
            sql_ms=t_sql_relax if t_sql_relax > 0 else None,
            total_ms=total_ms,
        )
        grounding = GroundingReport()
        yield ChatEventEnd(
            request_id=request_id,
            answer_mode=AnswerMode.refusal,
            cache_tier=cache_tier,
            cache_hits=list(cache_hits),
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(),
            finish_reason=FinishReason.no_llm,
            timings=timings,
            grounding=grounding,
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=self._prompt_version,
            index_version=self._index_version,
        ).to_sse()

        record = RequestRecord(
            request_id=request_id,
            route=decision.route,
            cache_tier=cache_tier,
            cache_hits=tuple(cache_hits),
            provider=Provider.none,
            model=None,
            prompt_tokens=0,
            completion_tokens=0,
            timings=timings,
            ts=time.time(),
        )
        self._ring.push(record)
        log_request(record)

    async def _serve_retrieval_only(
        self,
        request_id: str,
        decision: RouteDecision,
        cards: list[Card],
        cache_tier: AnswerCacheTier,
        cache_hits: list[InnerCacheTier],
        t0: float,
        t_cache_lookup: float | None,
        t_embed: float | None,
        t_sparse: float | None,
        t_retrieve: float | None,
        t_rerank: float | None,
        t_sql: float | None,
    ) -> AsyncIterator[dict[str, str]]:
        text = degrade_text_retrieval_only(cards, decision.language)
        yield ChatEventToken(request_id=request_id, delta=text).to_sse()

        total_ms = (time.perf_counter() - t0) * 1000
        timings = StageTimings(
            cache_lookup_ms=t_cache_lookup,
            embed_ms=t_embed,
            sparse_ms=t_sparse,
            retrieve_ms=t_retrieve,
            rerank_ms=t_rerank,
            sql_ms=t_sql,
            total_ms=total_ms,
        )
        grounding = GroundingReport()
        yield ChatEventEnd(
            request_id=request_id,
            answer_mode=AnswerMode.retrieval_only,
            cache_tier=cache_tier,
            cache_hits=list(cache_hits),
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(),
            finish_reason=FinishReason.no_llm,
            timings=timings,
            grounding=grounding,
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=self._prompt_version,
            index_version=self._index_version,
        ).to_sse()

        record = RequestRecord(
            request_id=request_id,
            route=decision.route,
            cache_tier=cache_tier,
            cache_hits=tuple(cache_hits),
            provider=Provider.none,
            model=None,
            prompt_tokens=0,
            completion_tokens=0,
            timings=timings,
            ts=time.time(),
        )
        self._ring.push(record)
        log_request(record)

    async def _serve_static_busy_after_error(
        self,
        request_id: str,
        decision: RouteDecision,
        cache_hits: list[InnerCacheTier],
        t0: float,
        t_cache_lookup: float | None,
    ) -> AsyncIterator[dict[str, str]]:
        """Emit start + busy token + end after a non-fatal error that degrades to static_busy."""
        busy = BUSY_AR if decision.language == Language.ar else BUSY_EN
        yield ChatEventStart(
            request_id=request_id,
            route=decision.route,
            language=decision.language,
            filters=decision.filters,
            cache_tier=AnswerCacheTier.miss,
            sources=[],
        ).to_sse()
        yield ChatEventToken(request_id=request_id, delta=busy).to_sse()

        total_ms = (time.perf_counter() - t0) * 1000
        timings = StageTimings(cache_lookup_ms=t_cache_lookup, total_ms=total_ms)
        grounding = GroundingReport()
        yield ChatEventEnd(
            request_id=request_id,
            answer_mode=AnswerMode.static_busy,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=list(cache_hits),
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(),
            finish_reason=FinishReason.no_llm,
            timings=timings,
            grounding=grounding,
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=self._prompt_version,
            index_version=self._index_version,
        ).to_sse()

        record = RequestRecord(
            request_id=request_id,
            route=decision.route,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=tuple(cache_hits),
            provider=Provider.none,
            model=None,
            prompt_tokens=0,
            completion_tokens=0,
            timings=timings,
            ts=time.time(),
        )
        self._ring.push(record)
        log_request(record)

    async def _static_busy(
        self,
        request_id: str,
        t0: float,
        lang_hint: Language | None,
    ) -> AsyncIterator[dict[str, str]]:
        """Admission-control busy: level 4."""
        lang = lang_hint or Language.en
        busy = BUSY_AR if lang == Language.ar else BUSY_EN

        yield ChatEventStart(
            request_id=request_id,
            route=Route.out_of_scope,
            language=lang,
            filters=SearchFilters(),
            cache_tier=AnswerCacheTier.miss,
            sources=[],
        ).to_sse()
        yield ChatEventToken(request_id=request_id, delta=busy).to_sse()

        total_ms = (time.perf_counter() - t0) * 1000
        timings = StageTimings(total_ms=total_ms)
        grounding = GroundingReport()
        yield ChatEventEnd(
            request_id=request_id,
            answer_mode=AnswerMode.static_busy,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(),
            finish_reason=FinishReason.no_llm,
            timings=timings,
            grounding=grounding,
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=self._prompt_version,
            index_version=self._index_version,
        ).to_sse()
