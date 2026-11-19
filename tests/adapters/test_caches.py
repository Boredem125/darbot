"""Tests for app/adapters/caches.py (L0-L4 cache tiers).

Tests are written first, before the implementation, per the brief.
Uses fake clock and fake adapters; no network, no keys.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.domain.ports import QueryEmbedding, RetrievedHit
from common.schemas import (
    CachedAnswer,
    ChunkPayload,
    DocKind,
    GroundingReport,
    Language,
    Provider,
    Route,
    SearchFilters,
    Source,
    SparseVector,
    WarmCacheRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDX_VERSION = "0123456789abcdef"
_PROMPT_VERSION = "system_v1@1a2b3c4d"
_SCRAPED = "2026-11-01T00:00:00+00:00"


def _make_cached_answer(
    answer: str = "Some answer",
    index_version: str = _IDX_VERSION,
    prompt_version: str = _PROMPT_VERSION,
    provider: Provider = Provider.openrouter,
    model: str = "meta-llama/llama-4-scout",
) -> CachedAnswer:
    return CachedAnswer(
        answer=answer,
        route=Route.lookup,
        language=Language.en,
        filters=SearchFilters(),
        sources=[],
        provider=provider,
        model=model,
        grounding=GroundingReport(),
        prompt_version=prompt_version,
        index_version=index_version,
        created_at="2026-11-18T10:00:00+00:00",
    )


def _warm_record(
    key: str = "ck2:k:abcdef1234567890abcdef1234567890",
    query: str = "test query",
    **kw,
) -> WarmCacheRecord:
    return WarmCacheRecord(
        key=key,
        query=query,
        entry=_make_cached_answer(**kw),
    )


def _unit_vec(dim: int = 384, seed: int = 0) -> tuple[float, ...]:
    """Deterministic L2-normalised vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float64)
    v /= np.linalg.norm(v)
    return tuple(v.tolist())


def _rotated_vec(base: tuple[float, ...], angle_deg: float = 5.0, seed: int = 42) -> tuple[float, ...]:
    """Rotate base slightly to get a cosine near 1.0 (small angle) or lower."""
    rng = np.random.default_rng(seed)
    dim = len(base)
    base_arr = np.array(base, dtype=np.float64)
    # Random orthogonal direction
    rand = rng.standard_normal(dim)
    rand -= rand.dot(base_arr) * base_arr  # project out base
    rand /= np.linalg.norm(rand)
    # Rotate
    angle_rad = math.radians(angle_deg)
    rotated = math.cos(angle_rad) * base_arr + math.sin(angle_rad) * rand
    rotated /= np.linalg.norm(rotated)
    return tuple(rotated.tolist())


class FakeClock:
    """Clock for testing TTL-based caches."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self._t = t

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def _make_chunk(doc_id: str = "wasalt:1") -> ChunkPayload:
    return ChunkPayload(
        doc_id=doc_id,
        kind=DocKind.listing,
        source=Source.wasalt,
        title="Test",
        url="https://example.com/1",
        scraped_at=_SCRAPED,
        chunk_index=0,
        text="Some chunk text",
    )


# ---------------------------------------------------------------------------
# L0: warm cache load
# ---------------------------------------------------------------------------


class TestL0WarmCacheLoad:
    """L0: read-only dict loaded from warm_cache.jsonl."""

    @pytest.mark.asyncio
    async def test_l0_load_and_hit(self, tmp_path):
        from app.adapters.caches import AnswerCache

        rec = _warm_record()
        wc = tmp_path / "warm_cache.jsonl"
        wc.write_text(rec.model_dump_json() + "\n", encoding="utf-8")

        cache = AnswerCache(
            warm_cache_path=wc,
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
        )
        hit = await cache.get_exact(rec.key)
        assert hit is not None
        assert hit.tier == "L0"

    @pytest.mark.asyncio
    async def test_l0_absent_file_is_empty(self, tmp_path):
        from app.adapters.caches import AnswerCache

        cache = AnswerCache(
            warm_cache_path=tmp_path / "missing.jsonl",
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
        )
        hit = await cache.get_exact("nonexistent")
        assert hit is None

    @pytest.mark.asyncio
    async def test_l0_version_mismatch_skipped(self, tmp_path):
        from app.adapters.caches import AnswerCache

        rec = _warm_record(index_version="aaaaaaaaaaaaaaaa")
        wc = tmp_path / "warm_cache.jsonl"
        wc.write_text(rec.model_dump_json() + "\n", encoding="utf-8")

        cache = AnswerCache(
            warm_cache_path=wc,
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
        )
        hit = await cache.get_exact(rec.key)
        assert hit is None
        assert cache.stats()["l0_skipped"] == 1


# ---------------------------------------------------------------------------
# L1: TTL cache with fake clock
# ---------------------------------------------------------------------------


class TestL1TTLCache:
    @pytest.mark.asyncio
    async def test_l1_put_and_get(self):
        from app.adapters.caches import AnswerCache

        clock = FakeClock()
        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            clock=clock,
        )
        entry = _make_cached_answer()
        await cache.put("key1", entry)
        hit = await cache.get_exact("key1")
        assert hit is not None
        assert hit.tier == "L1"
        assert hit.entry.answer == entry.answer

    @pytest.mark.asyncio
    async def test_l1_ttl_expiry(self):
        from app.adapters.caches import AnswerCache

        clock = FakeClock()
        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            l1_ttl=60,
            clock=clock,
        )
        entry = _make_cached_answer()
        await cache.put("key1", entry)
        clock.advance(61)
        hit = await cache.get_exact("key1")
        assert hit is None


# ---------------------------------------------------------------------------
# L2: semantic cache
# ---------------------------------------------------------------------------


class TestL2SemanticCache:
    @pytest.mark.asyncio
    async def test_l2_hit_at_tau(self):
        """Entry with cosine >= tau returns a hit."""
        from app.adapters.caches import AnswerCache

        tau = 0.93
        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            tau=tau,
        )
        vec = _unit_vec(384, seed=10)
        scope = "scope1"
        entry = _make_cached_answer()
        await cache.put("key1", entry, dense=vec, scope=scope)

        # Query with same vector -> cosine 1.0 >= tau
        hit = await cache.get_semantic(vec, scope=scope)
        assert hit is not None
        assert hit.tier == "L2"
        assert hit.similarity is not None
        assert hit.similarity >= tau

    @pytest.mark.asyncio
    async def test_l2_miss_below_tau(self):
        """Entry with cosine < tau returns None."""
        from app.adapters.caches import AnswerCache

        tau = 0.93
        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            tau=tau,
        )
        vec = _unit_vec(384, seed=10)
        scope = "scope1"
        entry = _make_cached_answer()
        await cache.put("key1", entry, dense=vec, scope=scope)

        # Rotated vector with large angle -> low cosine
        far_vec = _rotated_vec(vec, angle_deg=30.0, seed=99)
        hit = await cache.get_semantic(far_vec, scope=scope)
        assert hit is None

    @pytest.mark.asyncio
    async def test_l2_miss_across_scopes(self):
        """Different scopes never match."""
        from app.adapters.caches import AnswerCache

        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            tau=0.93,
        )
        vec = _unit_vec(384, seed=10)
        entry = _make_cached_answer()
        await cache.put("key1", entry, dense=vec, scope="scopeA")

        # Same vector but different scope
        hit = await cache.get_semantic(vec, scope="scopeB")
        assert hit is None

    @pytest.mark.asyncio
    async def test_l2_version_mismatch_miss(self):
        """Entry with wrong index_version is a miss."""
        from app.adapters.caches import AnswerCache

        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            tau=0.93,
        )
        vec = _unit_vec(384, seed=10)
        old_entry = _make_cached_answer(index_version="aaaaaaaaaaaaaaaa")
        await cache.put("key1", old_entry, dense=vec, scope="scope1")

        hit = await cache.get_semantic(vec, scope="scope1")
        assert hit is None

    @pytest.mark.asyncio
    async def test_l2_capacity_eviction(self):
        """When capacity is exceeded, the oldest entries are evicted."""
        from app.adapters.caches import AnswerCache

        clock = FakeClock()
        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            tau=0.93,
            l2_capacity=5,
            clock=clock,
        )
        # Fill with 5 entries
        for i in range(5):
            vec = _unit_vec(384, seed=100 + i)
            entry = _make_cached_answer(answer=f"answer {i}")
            await cache.put(f"key{i}", entry, dense=vec, scope="scope1")
            clock.advance(1)

        # Add one more -> oldest should be evicted
        vec6 = _unit_vec(384, seed=200)
        entry6 = _make_cached_answer(answer="answer 5")
        await cache.put("key5", entry6, dense=vec6, scope="scope1")

        assert cache.stats()["l2_size"] <= 5

    @pytest.mark.asyncio
    async def test_l2_disabled_flag(self):
        """When semantic_enabled=False, get_semantic always returns None."""
        from app.adapters.caches import AnswerCache

        cache = AnswerCache(
            index_version=_IDX_VERSION,
            prompt_version=_PROMPT_VERSION,
            tau=0.93,
            semantic_enabled=False,
        )
        vec = _unit_vec(384, seed=10)
        entry = _make_cached_answer()
        await cache.put("key1", entry, dense=vec, scope="scope1")

        hit = await cache.get_semantic(vec, scope="scope1")
        assert hit is None


# ---------------------------------------------------------------------------
# L3: CachedEmbedder
# ---------------------------------------------------------------------------


class TestL3CachedEmbedder:
    @pytest.mark.asyncio
    async def test_l3_hit_returns_cached_true(self):
        from app.adapters.caches import CachedEmbedder
        from tests.fakes.fake_adapters import FakeEmbedder

        inner = FakeEmbedder()
        embedder = CachedEmbedder(inner)

        # First call: not cached
        r1 = await embedder.embed_query("test query")
        assert r1.cached is False

        # Second call: should be cached
        r2 = await embedder.embed_query("test query")
        assert r2.cached is True
        assert r2.sparse_ms is None
        assert r1.dense == r2.dense

    @pytest.mark.asyncio
    async def test_l3_different_queries_not_shared(self):
        from app.adapters.caches import CachedEmbedder
        from tests.fakes.fake_adapters import FakeEmbedder

        inner = FakeEmbedder()
        embedder = CachedEmbedder(inner)

        r1 = await embedder.embed_query("query A")
        r2 = await embedder.embed_query("query B")
        # Both should be fresh (not cached)
        assert r1.cached is False
        assert r2.cached is False

    @pytest.mark.asyncio
    async def test_l3_properties(self):
        from app.adapters.caches import CachedEmbedder
        from tests.fakes.fake_adapters import FakeEmbedder

        inner = FakeEmbedder(dim=384, dense_model="test-model", sparse_model="test-sparse")
        embedder = CachedEmbedder(inner)
        assert embedder.dim == 384
        assert embedder.dense_model == "test-model"
        assert embedder.sparse_model == "test-sparse"


# ---------------------------------------------------------------------------
# L4: CachedRetriever
# ---------------------------------------------------------------------------


class TestL4CachedRetriever:
    @pytest.mark.asyncio
    async def test_l4_hit_returns_cached_true(self):
        from app.adapters.caches import CachedRetriever
        from tests.fakes.fake_adapters import FakeRetriever

        inner = FakeRetriever()
        inner.set_hits(
            [RetrievedHit(payload=_make_chunk(), score=0.9)],
            top_dense_score=0.8,
        )
        retriever = CachedRetriever(inner, index_version=_IDX_VERSION)

        emb = QueryEmbedding(
            dense=_unit_vec(384, seed=1),
            sparse=SparseVector(indices=[1], values=[0.5]),
        )
        filters = SearchFilters()

        # First call: not cached
        r1 = await retriever.search(emb, filters)
        assert r1.cached is False

        # Second call: should be cached
        r2 = await retriever.search(emb, filters)
        assert r2.cached is True

    @pytest.mark.asyncio
    async def test_l4_different_filters_not_shared(self):
        from app.adapters.caches import CachedRetriever
        from tests.fakes.fake_adapters import FakeRetriever

        inner = FakeRetriever()
        inner.set_hits([RetrievedHit(payload=_make_chunk(), score=0.9)])
        retriever = CachedRetriever(inner, index_version=_IDX_VERSION)

        emb = QueryEmbedding(
            dense=_unit_vec(384, seed=1),
            sparse=SparseVector(indices=[1], values=[0.5]),
        )
        # Different filters
        r1 = await retriever.search(emb, SearchFilters(city="Riyadh"))
        r2 = await retriever.search(emb, SearchFilters(city="Jeddah"))
        assert r1.cached is False
        assert r2.cached is False

    @pytest.mark.asyncio
    async def test_3bed_vs_4bed_not_shared(self):
        """3-bed vs 4-bed must never share a retrieval result."""
        from app.adapters.caches import CachedRetriever
        from tests.fakes.fake_adapters import FakeRetriever

        inner = FakeRetriever()
        inner.set_hits([RetrievedHit(payload=_make_chunk(), score=0.9)])
        retriever = CachedRetriever(inner, index_version=_IDX_VERSION)

        emb = QueryEmbedding(
            dense=_unit_vec(384, seed=1),
            sparse=SparseVector(indices=[1], values=[0.5]),
        )
        r3 = await retriever.search(
            emb, SearchFilters(bedrooms_min=3, bedrooms_max=3), limit=5,
        )
        r4 = await retriever.search(
            emb, SearchFilters(bedrooms_min=4, bedrooms_max=4), limit=5,
        )
        # Both should be fresh (not cached from each other)
        assert r3.cached is False
        assert r4.cached is False
