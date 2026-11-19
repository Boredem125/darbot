"""Cache adapters: L0-L4 (caching.md, ADR-0006).

L0: warm answers loaded from ``data/index/warm_cache.jsonl`` (read-only).
L1: in-process exact-key TTL cache (``cachetools.TTLCache``).
L2: in-process numpy semantic cache, partitioned by scope.
L3: ``CachedEmbedder`` wrapping any ``Embedder`` with an LRU on the query text.
L4: ``CachedRetriever`` wrapping any ``Retriever`` with an LRU on a composite key.

Concurrency: all reads and writes run on the event-loop thread.  The cachetools
caches are not thread-safe, but the single-worker uvicorn (ADR-0008) and the
"event loop only" rule make a lock unnecessary.

Imports only ``app.domain.ports``, ``common`` and stdlib (dependency rule R3).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cachetools import LRUCache, TTLCache

from app.domain.ports import (
    CacheHit,
    QueryEmbedding,
    RetrievalResult,
)
from common.schemas import CachedAnswer, SearchFilters, WarmCacheRecord
from common.text import normalise_for_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_L1_SIZE = 2_000
_DEFAULT_L1_TTL = 86_400  # 24 h
_DEFAULT_L2_CAPACITY = 2_000
_DEFAULT_L2_TTL = 86_400  # 24 h
_DEFAULT_L3_SIZE = 1_024
_DEFAULT_L4_SIZE = 512
_DEFAULT_TAU = 0.93
_DIM = 384


# ---------------------------------------------------------------------------
# L2 internal structures
# ---------------------------------------------------------------------------


@dataclass
class _L2Entry:
    entry_id: int
    cached_answer: CachedAnswer
    created_at: float
    last_hit_at: float


# ---------------------------------------------------------------------------
# AnswerCache (L0 + L1 + L2)
# ---------------------------------------------------------------------------


class AnswerCache:
    """Implements the ``Cache`` port (L0, L1, L2).

    Constructor signature (for the service agent to wire):

    .. code-block:: python

        AnswerCache(
            *,
            warm_cache_path: Path | None = None,
            index_version: str,
            prompt_version: str,
            tau: float = 0.93,
            l1_size: int = 2000,
            l1_ttl: int = 86400,
            l2_capacity: int = 2000,
            l2_ttl: int = 86400,
            semantic_enabled: bool = True,
            clock: Callable[[], float] | None = None,
        )
    """

    def __init__(
        self,
        *,
        warm_cache_path: Path | None = None,
        index_version: str,
        prompt_version: str,
        tau: float = _DEFAULT_TAU,
        l1_size: int = _DEFAULT_L1_SIZE,
        l1_ttl: int = _DEFAULT_L1_TTL,
        l2_capacity: int = _DEFAULT_L2_CAPACITY,
        l2_ttl: int = _DEFAULT_L2_TTL,
        semantic_enabled: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._index_version = index_version
        self._prompt_version = prompt_version
        self._tau = tau
        self._semantic_enabled = semantic_enabled
        self._clock: Callable[[], float] = clock or time.monotonic

        # L0: read-only warm cache
        self._l0: dict[str, CachedAnswer] = {}
        self._l0_skipped = 0
        self._l0_hits = 0
        self._l0_misses = 0

        # L1: exact-key TTL cache
        self._l1: TTLCache[str, CachedAnswer] = TTLCache(maxsize=l1_size, ttl=l1_ttl, timer=self._clock)
        self._l1_hits = 0
        self._l1_misses = 0

        # L2: semantic cache
        self._l2_capacity = l2_capacity
        self._l2_ttl = l2_ttl
        self._l2_scopes: dict[str, tuple[np.ndarray, list[int]]] = {}  # scope -> (matrix, entry_ids)
        self._l2_entries: dict[int, _L2Entry] = {}
        self._l2_next_id = 0
        self._l2_hits = 0
        self._l2_misses = 0

        # Load L0
        if warm_cache_path is not None:
            self._load_warm_cache(warm_cache_path)

    def _load_warm_cache(self, path: Path) -> None:
        if not path.exists():
            logger.info("Warm cache file absent: %s (L0 empty)", path)
            return
        loaded = 0
        skipped = 0
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = WarmCacheRecord.model_validate_json(line)
            except (ValueError, TypeError, KeyError):
                logger.warning("Warm cache line %d: validation error, skipping", line_no)
                skipped += 1
                continue
            if rec.entry.index_version != self._index_version:
                skipped += 1
                continue
            if rec.entry.prompt_version != self._prompt_version:
                skipped += 1
                continue
            self._l0[rec.key] = rec.entry
            loaded += 1
        self._l0_skipped = skipped
        logger.info("L0: loaded %d, skipped %d from %s", loaded, skipped, path)

    # --- L0 + L1: exact ---

    async def get_exact(self, key: str) -> CacheHit | None:
        # L0 first
        if key in self._l0:
            self._l0_hits += 1
            return CacheHit(tier="L0", entry=self._l0[key])

        # L1
        entry = self._l1.get(key)
        if entry is not None:
            self._l1_hits += 1
            return CacheHit(tier="L1", entry=entry)

        self._l0_misses += 1
        self._l1_misses += 1
        return None

    # --- L2: semantic ---

    async def get_semantic(self, dense: Sequence[float], *, scope: str) -> CacheHit | None:
        if not self._semantic_enabled:
            return None

        scope_data = self._l2_scopes.get(scope)
        if scope_data is None:
            self._l2_misses += 1
            return None

        matrix, entry_ids = scope_data
        if matrix.shape[0] == 0:
            self._l2_misses += 1
            return None

        query_vec = np.array(dense, dtype=np.float32)
        # matrix rows are L2-normalised, query_vec should be too
        cosines = matrix @ query_vec
        best_idx = int(np.argmax(cosines))
        best_cos = float(cosines[best_idx])

        if best_cos < self._tau:
            self._l2_misses += 1
            return None

        entry_id = entry_ids[best_idx]
        l2_entry = self._l2_entries.get(entry_id)
        if l2_entry is None:
            self._l2_misses += 1
            return None

        # Version check (belt-and-braces)
        if (l2_entry.cached_answer.index_version != self._index_version
                or l2_entry.cached_answer.prompt_version != self._prompt_version):
            self._l2_misses += 1
            return None

        # TTL check
        now = self._clock()
        if now - l2_entry.created_at > self._l2_ttl:
            self._l2_misses += 1
            return None

        # Hit
        l2_entry.last_hit_at = now
        self._l2_hits += 1
        return CacheHit(tier="L2", entry=l2_entry.cached_answer, similarity=best_cos)

    # --- put (L1 + L2) ---

    async def put(
        self,
        key: str,
        entry: CachedAnswer,
        *,
        dense: Sequence[float] | None = None,
        scope: str | None = None,
    ) -> None:
        # L1
        self._l1[key] = entry

        # L2
        if dense is not None and scope is not None and self._semantic_enabled:
            self._l2_put(entry, dense, scope)

    def _l2_put(self, entry: CachedAnswer, dense: Sequence[float], scope: str) -> None:
        now = self._clock()
        # Evict expired entries first
        self._l2_evict_expired(now)
        # Evict oldest if at capacity
        while len(self._l2_entries) >= self._l2_capacity:
            self._l2_evict_oldest()

        entry_id = self._l2_next_id
        self._l2_next_id += 1
        self._l2_entries[entry_id] = _L2Entry(
            entry_id=entry_id,
            cached_answer=entry,
            created_at=now,
            last_hit_at=now,
        )

        vec = np.array(dense, dtype=np.float32).reshape(1, -1)
        if scope in self._l2_scopes:
            matrix, ids = self._l2_scopes[scope]
            matrix = np.vstack([matrix, vec])
            ids.append(entry_id)
            self._l2_scopes[scope] = (matrix, ids)
        else:
            self._l2_scopes[scope] = (vec, [entry_id])

    def _l2_evict_expired(self, now: float) -> None:
        expired_ids = [
            eid for eid, e in self._l2_entries.items()
            if now - e.created_at > self._l2_ttl
        ]
        for eid in expired_ids:
            self._l2_remove_entry(eid)

    def _l2_evict_oldest(self) -> None:
        if not self._l2_entries:
            return
        oldest_id = min(self._l2_entries, key=lambda eid: self._l2_entries[eid].last_hit_at)
        self._l2_remove_entry(oldest_id)

    def _l2_remove_entry(self, entry_id: int) -> None:
        self._l2_entries.pop(entry_id, None)
        # Remove from scope matrices
        for scope, (matrix, ids) in list(self._l2_scopes.items()):
            if entry_id in ids:
                idx = ids.index(entry_id)
                ids.pop(idx)
                matrix = np.delete(matrix, idx, axis=0)
                if matrix.shape[0] == 0:
                    del self._l2_scopes[scope]
                else:
                    self._l2_scopes[scope] = (matrix, ids)
                break

    # --- stats ---

    def stats(self) -> dict:
        return {
            "l0_size": len(self._l0),
            "l0_hits": self._l0_hits,
            "l0_misses": self._l0_misses,
            "l0_skipped": self._l0_skipped,
            "l1_size": len(self._l1),
            "l1_hits": self._l1_hits,
            "l1_misses": self._l1_misses,
            "l2_size": len(self._l2_entries),
            "l2_hits": self._l2_hits,
            "l2_misses": self._l2_misses,
            "l2_scopes": len(self._l2_scopes),
        }


# ---------------------------------------------------------------------------
# CachedEmbedder (L3)
# ---------------------------------------------------------------------------


class CachedEmbedder:
    """L3: wraps any ``Embedder``, ``LRUCache(1024)`` keyed by
    ``normalise_for_embedding(query)``.

    Constructor signature:

    .. code-block:: python

        CachedEmbedder(inner: Embedder, *, capacity: int = 1024)
    """

    def __init__(self, inner: object, *, capacity: int = _DEFAULT_L3_SIZE) -> None:
        self._inner = inner  # Embedder protocol
        self._cache: LRUCache[str, QueryEmbedding] = LRUCache(maxsize=capacity)
        self._hits = 0
        self._misses = 0

    @property
    def dense_model(self) -> str:
        return self._inner.dense_model  # type: ignore[attr-defined]

    @property
    def dim(self) -> int:
        return self._inner.dim  # type: ignore[attr-defined]

    @property
    def sparse_model(self) -> str:
        return self._inner.sparse_model  # type: ignore[attr-defined]

    async def embed_query(self, text: str) -> QueryEmbedding:
        key = normalise_for_embedding(text)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            # Return with cached=True and sparse_ms=None
            return QueryEmbedding(
                dense=cached.dense,
                sparse=cached.sparse,
                cached=True,
                sparse_ms=None,
            )
        self._misses += 1
        result = await self._inner.embed_query(text)  # type: ignore[attr-defined]
        self._cache[key] = result
        return result

    def stats(self) -> dict:
        return {
            "l3_size": len(self._cache),
            "l3_hits": self._hits,
            "l3_misses": self._misses,
        }


# ---------------------------------------------------------------------------
# CachedRetriever (L4)
# ---------------------------------------------------------------------------


class CachedRetriever:
    """L4: wraps any ``Retriever``, ``LRUCache(512)`` keyed by
    ``sha256(index_version, normalise_for_embedding(query_text), filters.canonical_json(), limit)``.

    The ``Retriever.search`` signature takes a ``QueryEmbedding`` not query text.
    We key the cache on the dense vector bytes since the dense vector is a
    deterministic function of the query text (through the embedder), and two
    identical dense vectors always come from the same normalised text.

    Constructor signature:

    .. code-block:: python

        CachedRetriever(inner: Retriever, *, index_version: str, capacity: int = 512)
    """

    def __init__(self, inner: object, *, index_version: str, capacity: int = _DEFAULT_L4_SIZE) -> None:
        self._inner = inner  # Retriever protocol
        self._index_version = index_version
        self._cache: LRUCache[str, RetrievalResult] = LRUCache(maxsize=capacity)
        self._hits = 0
        self._misses = 0

    async def search(
        self,
        embedding: QueryEmbedding,
        filters: SearchFilters,
        *,
        limit: int = 20,
    ) -> RetrievalResult:
        # Build the cache key from the dense vector bytes + filters + limit
        # The dense vector uniquely represents the normalised query text
        dense_bytes = np.array(embedding.dense, dtype=np.float32).tobytes()
        preimage = (
            self._index_version
            + "\x1f" + dense_bytes.hex()
            + "\x1f" + filters.canonical_json()
            + "\x1f" + str(limit)
        )
        cache_key = hashlib.sha256(preimage.encode("utf-8", errors="surrogateescape")).hexdigest()

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._hits += 1
            return RetrievalResult(
                hits=cached.hits,
                cached=True,
                top_dense_score=cached.top_dense_score,
            )
        self._misses += 1
        result = await self._inner.search(embedding, filters, limit=limit)  # type: ignore[attr-defined]
        self._cache[cache_key] = result
        return result

    def stats(self) -> dict:
        return {
            "l4_size": len(self._cache),
            "l4_hits": self._hits,
            "l4_misses": self._misses,
        }
