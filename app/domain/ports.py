"""Ports: the only seams between ChatService and the outside world (ADR-0001).

Frozen in Wave 1a by dg-lead-architect. Imports only `common` and the stdlib.

Adapters in `app/adapters/` implement these Protocols structurally (no inheritance
needed). Fakes for tests implement the same Protocols. Value types that cross a port
but are never serialised to a file or the wire live here as frozen dataclasses;
anything serialised lives in `common.schemas`.

Concurrency contract for every port:
- Methods are `async` and must not block the event loop. CPU-bound ONNX work runs
  on the adapter's dedicated `ThreadPoolExecutor(max_workers=2)` (PLAN section 5).
- Adapters are safe to call concurrently from multiple requests.
- Timing of a port call is measured by the caller (service) around the await,
  except where a result type carries its own timing fields (`QueryEmbedding.sparse_ms`).

Amended in WP-2-contracts (18 Nov 2026, see `.orchestration/CONTRACT_CHANGES.md`):
`QueryEmbedding.sparse_ms` and `RetrievalResult.top_dense_score` added, both defaulted,
so existing constructors and fakes keep working.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from common.schemas import (
    CachedAnswer,
    ChunkPayload,
    ErrorCode,
    FinishReason,
    Listing,
    Provider,
    SearchFilters,
    SparseVector,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Errors raised across ports
# ---------------------------------------------------------------------------


class PortError(Exception):
    """Base class for adapter failures the service is expected to handle."""


class RetrievalError(PortError):
    """Vector search failed. Service degrades (PLAN section 5, ladder)."""


class StructuredQueryError(PortError):
    """SQL template execution failed."""


class EmbeddingError(PortError):
    """Query embedding failed."""


class LLMUnavailableError(PortError):
    """No provider in the chain produced a first token. Carries every attempt."""

    def __init__(self, attempts: Sequence["LLMFailover"]) -> None:
        self.attempts = tuple(attempts)
        super().__init__(f"all LLM providers unavailable ({len(self.attempts)} attempts)")


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueryEmbedding:
    """Result of `Embedder.embed_query`.

    Timing rule (fills `StageTimings.embed_ms` / `sparse_ms` in the `end` event):
    - `sparse_ms` is the wall time of the bm25 query embedding, measured by the adapter
      inside its executor job. It is None when `cached` is True (L3 hit) or when the
      adapter does not measure it.
    - The service measures the wall time `W` around `await embed_query(...)` and sets
      `StageTimings.embed_ms = max(W - (sparse_ms or 0), 0)` and
      `StageTimings.sparse_ms = sparse_ms`. So `embed_ms` covers the dense model plus the
      executor hop and queue wait, and `embed_ms + sparse_ms == W`.
    - On an L3 hit both stage timings are null (the stage did not run); the lookup
      cost stays inside `total_ms`.
    """

    dense: tuple[float, ...]  # len == Embedder.dim, L2-normalised (cosine)
    sparse: SparseVector  # bm25 query-side vector
    cached: bool = False  # True when served from the L3 query-embedding LRU
    sparse_ms: float | None = None  # >= 0; bm25 part, measured by the adapter (see above)


@runtime_checkable
class Embedder(Protocol):
    """Query-side embedding (dense + sparse). Build-time document embedding lives in
    `organize/` and does not use this port."""

    @property
    def dense_model(self) -> str: ...  # must equal Manifest.embed_model

    @property
    def dim(self) -> int: ...  # must equal Manifest.dim

    @property
    def sparse_model(self) -> str: ...  # must equal Manifest.sparse_model

    async def embed_query(self, text: str) -> QueryEmbedding:
        """Embed one normalised query. Raises EmbeddingError."""
        ...


# ---------------------------------------------------------------------------
# Retriever (+ optional Reranker)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievedHit:
    payload: ChunkPayload
    score: float  # RRF-fused score, or the rerank score after Reranker.rerank


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Result of `Retriever.search`.

    `top_dense_score` is the raw cosine similarity (in [-1, 1]) of the top-1 point of the
    *filtered* dense candidate list, before fusion, or None when that list is empty
    (no point passes the filters). It is the only input to the score-floor refusal
    (`docs/specs/retrieval.md` §3.6); `RetrievedHit.score` is rank-based (RRF) and must
    not be compared with the floor. An L4 cache hit returns the stored value unchanged.
    Rerank does not change it: the service keeps the value from `search()`.
    """

    hits: tuple[RetrievedHit, ...]  # sorted by score desc; may hold several chunks per doc
    cached: bool = False  # True when served from the L4 retrieval-result LRU
    top_dense_score: float | None = None  # filtered dense top-1 cosine; None if no candidate


@runtime_checkable
class Retriever(Protocol):
    async def search(
        self,
        embedding: QueryEmbedding,
        filters: SearchFilters,
        *,
        limit: int = 20,
    ) -> RetrievalResult:
        """Hybrid search: dense prefetch(limit) + bm25 prefetch(limit), RRF fusion,
        payload filters applied to both prefetches (mapping on SearchFilters).
        Returns at most `limit` hits. `filters.sort` is ignored. Sets
        `RetrievalResult.top_dense_score` (see its docstring). Raises RetrievalError.

        The runtime adapter is an in-process numpy engine with Qdrant Query API
        semantics (ADR-0002, amendment 1); Qdrant local mode is the parity test oracle.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Optional cross-encoder rerank (English only, flag RERANK_ENABLED). The service
    decides whether to call it (skipped for Arabic)."""

    async def rerank(
        self, query: str, hits: Sequence[RetrievedHit], *, top_k: int
    ) -> list[RetrievedHit]:
        """Return at most `top_k` hits, re-scored and sorted desc. Raises RetrievalError."""
        ...


# ---------------------------------------------------------------------------
# StructuredQuery (DuckDB whitelisted templates over listings.parquet)
# ---------------------------------------------------------------------------


class AggregateOp(StrEnum):
    """Whitelisted templates. Price ops use `annual_rent_sar` when
    filters.purpose == rent, else `price_sar`; rows with a null target are excluded."""

    count = "count"
    avg_price = "avg_price"
    median_price = "median_price"
    min_price = "min_price"  # examples = the cheapest listings
    max_price = "max_price"  # examples = the most expensive listings
    avg_area = "avg_area"


@dataclass(frozen=True, slots=True)
class AggregateQuery:
    op: AggregateOp
    filters: SearchFilters  # `sort` decides example order for count/avg ops
    example_limit: int = 3  # example listings returned as cards (0..5)


@dataclass(frozen=True, slots=True)
class SqlResult:
    op: AggregateOp
    value: int | float | None  # None when n_matched == 0; prices are int SAR (rounded)
    unit: Literal["SAR", "sqm", "count"]
    n_matched: int  # rows matching the filters (before op)
    examples: tuple[Listing, ...] = field(default_factory=tuple)


@runtime_checkable
class StructuredQuery(Protocol):
    async def aggregate(self, query: AggregateQuery) -> SqlResult:
        """Run one whitelisted template with bound parameters (never string-built SQL
        from user text). Raises StructuredQueryError."""
        ...


# ---------------------------------------------------------------------------
# LLM (implemented by the LLMRouter adapter, ADR-0005)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LLMDelta:
    text: str  # non-empty


@dataclass(frozen=True, slots=True)
class LLMFailover:
    """A provider attempt failed before its first token; the router moved on."""

    provider: Provider
    model: str
    code: ErrorCode  # provider_rate_limited | provider_timeout | provider_error | provider_skipped
    detail: str  # log-safe, no keys


@dataclass(frozen=True, slots=True)
class LLMDone:
    provider: Provider  # openrouter | groq
    model: str
    finish_reason: FinishReason  # stop | length | content_filter | error
    usage: TokenUsage
    ttft_ms: float
    total_ms: float


LLMStreamItem = LLMDelta | LLMFailover | LLMDone


@dataclass(frozen=True, slots=True)
class ProviderState:
    provider: Provider
    model: str
    available: bool
    cooldown_remaining_s: float  # 0 when available


@dataclass(frozen=True, slots=True)
class LLMStatus:
    openrouter_quota_remaining: int | None  # last polled value; None if never polled
    providers: tuple[ProviderState, ...]


@runtime_checkable
class LLM(Protocol):
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMStreamItem]:
        """Stream one completion through the provider chain.

        Item order: LLMFailover* (only before the first delta), LLMDelta*, exactly one
        LLMDone, then the iterator ends.
        - Failure before the first token: fail over silently to the next provider
          (yielding an LLMFailover). If every provider fails: raise LLMUnavailableError
          and yield nothing else.
        - Failure after the first token: no failover; yield LLMDone with
          finish_reason=error and the usage known so far.
        Implemented as `async def stream(...)` with `yield`.
        """
        ...

    def status(self) -> LLMStatus:
        """Cheap, synchronous snapshot for /ready and /stats."""
        ...


# ---------------------------------------------------------------------------
# Cache (answer tiers L0-L2; L3/L4 are wrappers around Embedder/Retriever)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheHit:
    tier: Literal["L0", "L1", "L2"]
    entry: CachedAnswer
    similarity: float | None = None  # cosine for L2, None for exact tiers


@runtime_checkable
class Cache(Protocol):
    """Answer cache. L3/L4 hits are reported via QueryEmbedding.cached and
    RetrievalResult.cached and surface in `end.cache_hits`.

    `key` is the exact key built by the domain from index_version + prompt_version +
    normalised query + SearchFilters.canonical_json() + extracted numbers (ADR-0006);
    it is opaque here. `scope` is the L2 partition: the same string minus the
    normalised query, so a semantic hit can only match identical filters and numbers.
    """

    async def get_exact(self, key: str) -> CacheHit | None:
        """L0 (warm, read-only) then L1 (TTL 24 h). No embedding needed."""
        ...

    async def get_semantic(self, dense: Sequence[float], *, scope: str) -> CacheHit | None:
        """L2: nearest entry in `scope` with cosine >= tau. Returns None when the
        semantic cache is disabled. Callers only use it for standalone turns."""
        ...

    async def put(
        self,
        key: str,
        entry: CachedAnswer,
        *,
        dense: Sequence[float] | None = None,
        scope: str | None = None,
    ) -> None:
        """Store in L1, and in L2 when `dense` and `scope` are given. The caller only
        passes grounded, untruncated answers (CachedAnswer enforces grounded)."""
        ...
