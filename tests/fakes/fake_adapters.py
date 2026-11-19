"""Fake adapters implementing the Protocol ports for testing.

Fakes may import common and app.domain.ports only.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Sequence
from typing import Literal

from app.domain.ports import (
    AggregateOp,
    AggregateQuery,
    CacheHit,
    ChatMessage,
    LLMStatus,
    LLMStreamItem,
    QueryEmbedding,
    RetrievalResult,
    RetrievedHit,
    SqlResult,
)
from common.schemas import (
    CachedAnswer,
    SearchFilters,
    SparseVector,
)

# ---------------------------------------------------------------------------
# FakeEmbedder
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embedder for testing. Returns a fixed vector."""

    def __init__(
        self,
        dim: int = 384,
        dense_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        sparse_model: str = "Qdrant/bm25",
    ) -> None:
        self._dim = dim
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self._vectors: dict[str, tuple[float, ...]] = {}

    @property
    def dense_model(self) -> str:
        return self._dense_model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def sparse_model(self) -> str:
        return self._sparse_model

    def set_vector(self, text: str, vector: tuple[float, ...]) -> None:
        """Pre-set a dense vector for a given text."""
        self._vectors[text] = vector

    async def embed_query(self, text: str) -> QueryEmbedding:
        if text in self._vectors:
            dense = self._vectors[text]
        else:
            # Deterministic vector based on hash
            h = hash(text) & 0xFFFFFFFF
            vec = []
            for i in range(self._dim):
                val = ((h * (i + 1)) % 1000) / 1000.0 - 0.5
                vec.append(val)
            # L2-normalise
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            dense = tuple(vec)

        return QueryEmbedding(
            dense=dense,
            sparse=SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.1]),
            cached=False,
            sparse_ms=0.5,
        )


# ---------------------------------------------------------------------------
# FakeRetriever
# ---------------------------------------------------------------------------


class FakeRetriever:
    """Retriever that returns pre-configured hits."""

    def __init__(self) -> None:
        self._hits: tuple[RetrievedHit, ...] = ()
        self._top_dense_score: float | None = None

    def set_hits(
        self,
        hits: Sequence[RetrievedHit],
        top_dense_score: float | None = None,
    ) -> None:
        self._hits = tuple(hits)
        self._top_dense_score = top_dense_score

    async def search(
        self,
        embedding: QueryEmbedding,
        filters: SearchFilters,
        *,
        limit: int = 20,
    ) -> RetrievalResult:
        return RetrievalResult(
            hits=self._hits[:limit],
            cached=False,
            top_dense_score=self._top_dense_score,
        )


# ---------------------------------------------------------------------------
# FakeReranker
# ---------------------------------------------------------------------------


class FakeReranker:
    """Reranker that returns the top_k hits in the same order."""

    async def rerank(
        self, query: str, hits: Sequence[RetrievedHit], *, top_k: int
    ) -> list[RetrievedHit]:
        return list(hits[:top_k])


# ---------------------------------------------------------------------------
# FakeStructuredQuery
# ---------------------------------------------------------------------------


class FakeStructuredQuery:
    """Returns pre-configured SQL results."""

    def __init__(self) -> None:
        self._results: dict[AggregateOp, SqlResult] = {}
        self._callback: object | None = None  # Callable[[AggregateQuery], SqlResult | None]
        self._calls: list[AggregateQuery] = []

    def set_result(self, op: AggregateOp, result: SqlResult) -> None:
        self._results[op] = result

    def set_callback(self, fn: object) -> None:
        """Set a callback(query) -> SqlResult | None. Called before the op dict."""
        self._callback = fn

    async def aggregate(self, query: AggregateQuery) -> SqlResult:
        self._calls.append(query)
        if self._callback is not None:
            result = self._callback(query)  # type: ignore[operator]
            if result is not None:
                return result
        if query.op in self._results:
            return self._results[query.op]
        return SqlResult(
            op=query.op,
            value=None,
            unit="count",
            n_matched=0,
        )


# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


class FakeLLM:
    """Scriptable LLM: takes a sequence of items to yield."""

    def __init__(self) -> None:
        self._script: list[LLMStreamItem | Exception] = []
        self._call_count = 0

    def set_script(self, items: list[LLMStreamItem | Exception]) -> None:
        self._script = items

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMStreamItem]:
        self._call_count += 1
        for item in self._script:
            if isinstance(item, Exception):
                raise item
            yield item

    def status(self) -> LLMStatus:
        return LLMStatus(
            openrouter_quota_remaining=None,
            providers=(),
        )

    @property
    def call_count(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# FakeCache
# ---------------------------------------------------------------------------


class FakeCache:
    """In-memory cache for testing. Exact match + optional cosine over given vectors."""

    def __init__(self) -> None:
        self._exact: dict[str, tuple[Literal["L0", "L1"], CachedAnswer]] = {}
        self._semantic: dict[str, list[tuple[Sequence[float], CachedAnswer]]] = {}

    def put_exact(self, key: str, entry: CachedAnswer, tier: Literal["L0", "L1"] = "L1") -> None:
        self._exact[key] = (tier, entry)

    async def get_exact(self, key: str) -> CacheHit | None:
        if key in self._exact:
            tier, entry = self._exact[key]
            return CacheHit(tier=tier, entry=entry)
        return None

    async def get_semantic(self, dense: Sequence[float], *, scope: str) -> CacheHit | None:
        if scope not in self._semantic:
            return None
        best_sim = -1.0
        best_entry: CachedAnswer | None = None
        for vec, entry in self._semantic[scope]:
            sim = _cosine(dense, vec)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_entry is not None and best_sim >= 0.92:
            return CacheHit(tier="L2", entry=best_entry, similarity=best_sim)
        return None

    async def put(
        self,
        key: str,
        entry: CachedAnswer,
        *,
        dense: Sequence[float] | None = None,
        scope: str | None = None,
    ) -> None:
        self._exact[key] = ("L1", entry)
        if dense is not None and scope is not None:
            if scope not in self._semantic:
                self._semantic[scope] = []
            self._semantic[scope].append((dense, entry))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
