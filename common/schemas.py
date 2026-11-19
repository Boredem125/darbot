"""Shared contracts (pydantic v2). Frozen in Wave 1a by dg-lead-architect.

Rules for this module:
- Contracts only. No I/O, no network, no filesystem, no environment reads.
- Imports nothing internal; third-party imports are limited to pydantic.
- Changing anything here requires a note in `.orchestration/CONTRACT_CHANGES.md`
  plus dg-lead-architect approval (PLAN section 14).

Units and conventions used throughout:
- Listing prices are SAR as `int` (whole riyals, no halalas, no floats).
- Areas are square metres as `float` (`*_sqm`).
- Durations are milliseconds as `float` (`*_ms`).
- Datetimes are timezone-aware (`AwareDatetime`), serialised as ISO 8601; producers use UTC.
- Place names (city, district, country) are the canonical English names from
  `data/index/entities.json` (`Entity.canonical`). Filters match them exactly.
- Document ids:
    listing  -> "wasalt:<source_id>"
    project  -> "darglobal:project:<slug>"
    article  -> "darglobal:<article_kind>:<slug>"

Amended in WP-2-contracts (18 Nov 2026, see `.orchestration/CONTRACT_CHANGES.md`):
model-id env names and defaults (`MODEL_ENV_DEFAULTS`), `Project.body`.
"""

import hashlib
import json
import uuid
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

# ---------------------------------------------------------------------------
# Constants shared by the offline (organize) and online (app) halves
# ---------------------------------------------------------------------------

SSE_VERSION: Literal[1] = 1
MANIFEST_SCHEMA_VERSION: Literal[1] = 1
ENTITIES_SCHEMA_VERSION: Literal[1] = 1

# Qdrant names. Since ADR-0002 amendment 1 the runtime search and the L2 cache are numpy;
# these names are used only by the Qdrant parity oracle (tests) and the optional
# `RETRIEVER_BACKEND=qdrant_local` adapter.
KB_COLLECTION = "kb"  # Qdrant collection name for all documents
QA_CACHE_COLLECTION = "qa_cache"  # reserved; L2 is numpy at runtime (no Qdrant collection)
DENSE_VECTOR_NAME = "dense"  # named dense vector in KB_COLLECTION
SPARSE_VECTOR_NAME = "bm25"  # named sparse vector in KB_COLLECTION

# Model ids (exact fastembed ids). The env var names are shared by the runtime image
# (Dockerfile ARG/ENV), `app/config.py` and `organize/build_index.py`; each reads
# `os.environ.get(name, default)` itself (this module does no env reads).
# EMBED_MODEL must equal Manifest.embed_model and SPARSE_MODEL Manifest.sparse_model,
# or startup aborts (manifest.md fail-fast rule 2). RERANK_MODEL is query-side only and
# is not recorded in or checked against the manifest.
EMBED_MODEL_DEFAULT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM_DEFAULT = 384
SPARSE_MODEL_DEFAULT = "Qdrant/bm25"
RERANK_MODEL_DEFAULT = "Xenova/ms-marco-MiniLM-L-6-v2"
MODEL_ENV_DEFAULTS: dict[str, str] = {
    "EMBED_MODEL": EMBED_MODEL_DEFAULT,
    "SPARSE_MODEL": SPARSE_MODEL_DEFAULT,
    "RERANK_MODEL": RERANK_MODEL_DEFAULT,
}

DESCRIPTION_MAX_CHARS = 400  # sanitised description cap (PLAN section 5, Security)
MAX_QUERY_CHARS = 500  # max user message length accepted by /api/chat
MAX_CARDS = 10  # max cards in one `start` event
MAX_SHARD_BYTES = 10_000_000  # every file listed in the manifest is below this (HF 10 MB)

# uuid5 namespace for Qdrant point ids; never change once an index is built.
POINT_ID_NAMESPACE = uuid.UUID("6f1c2a52-6a0e-4c43-9a57-2d1b8f6c9e10")

_SLUG = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_HTTP_URL = r"^https?://\S+$"
_HEX64 = r"^[0-9a-f]{64}$"
_REQUEST_ID = r"^[0-9a-f]{32}$"  # uuid4().hex
_PROMPT_VERSION = r"^[a-z0-9_]+@[0-9a-f]{8}$"  # e.g. "system_v1@1a2b3c4d"
_INDEX_VERSION = r"^[0-9a-f]{16}$"
_CURRENCY = r"^[A-Z]{3}$"  # ISO 4217

HttpUrlStr = Annotated[str, Field(pattern=_HTTP_URL, max_length=2048)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
ShortText = Annotated[str, Field(min_length=1, max_length=DESCRIPTION_MAX_CHARS)]
Ms = Annotated[float, Field(ge=0)]


class _Contract(BaseModel):
    """Base for every contract: unknown fields are rejected so typos surface early."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Source(StrEnum):
    wasalt = "wasalt"
    darglobal = "darglobal"


class DocKind(StrEnum):
    listing = "listing"
    project = "project"
    article = "article"


class Purpose(StrEnum):
    sale = "sale"
    rent = "rent"


class RentPeriod(StrEnum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"
    yearly = "yearly"


# Multiplier from a rent amount per period to an annual amount.
ANNUAL_RENT_FACTOR: dict[RentPeriod, int] = {
    RentPeriod.daily: 365,
    RentPeriod.weekly: 52,
    RentPeriod.monthly: 12,
    RentPeriod.yearly: 1,
}


class PropertyType(StrEnum):
    """Normalised property type. Anything unmapped becomes `other`; the raw
    source value is kept in `Listing.property_type_raw`."""

    apartment = "apartment"
    villa = "villa"
    townhouse = "townhouse"
    duplex = "duplex"
    floor = "floor"
    land = "land"
    building = "building"
    office = "office"
    shop = "shop"
    warehouse = "warehouse"
    chalet = "chalet"
    farm = "farm"
    other = "other"


class ArticleKind(StrEnum):
    press = "press"
    insight = "insight"
    blog = "blog"


class Language(StrEnum):
    en = "en"
    ar = "ar"


class Route(StrEnum):
    lookup = "lookup"
    aggregate = "aggregate"
    project_info = "project_info"
    out_of_scope = "out_of_scope"


class SortKey(StrEnum):
    relevance = "relevance"
    price_asc = "price_asc"
    price_desc = "price_desc"
    newest = "newest"


class AnswerCacheTier(StrEnum):
    """Tier that served the whole answer. `miss` means the answer was produced now."""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    miss = "miss"


class InnerCacheTier(StrEnum):
    """Sub-answer caches whose hits are reported in `end.cache_hits`."""

    L3 = "L3"  # query-embedding LRU
    L4 = "L4"  # retrieval-result LRU


class Provider(StrEnum):
    openrouter = "openrouter"
    groq = "groq"
    none = "none"  # no LLM authored the text (refusal, retrieval-only, busy)


class FinishReason(StrEnum):
    stop = "stop"
    length = "length"  # hit max_tokens -> truncated, never cached
    content_filter = "content_filter"
    error = "error"  # stream died after the first token
    cached = "cached"  # served from L0/L1/L2
    no_llm = "no_llm"  # refusal, retrieval-only or static busy


class AnswerMode(StrEnum):
    """How the answer text was produced (degradation ladder, PLAN section 5)."""

    llm = "llm"  # levels 1-2: OpenRouter or Groq (see `provider`)
    cache = "cache"  # L0/L1/L2 hit
    retrieval_only = "retrieval_only"  # level 3: cards + templated summary
    static_busy = "static_busy"  # level 4: fixed busy message
    refusal = "refusal"  # out_of_scope or below score floor; fixed refusal line


class DegradedTo(StrEnum):
    fallback_llm = "fallback_llm"  # next provider in the LLM chain
    retrieval_only = "retrieval_only"
    static_busy = "static_busy"


class ErrorCode(StrEnum):
    provider_rate_limited = "provider_rate_limited"  # 429 from a provider
    provider_timeout = "provider_timeout"  # no first token within the TTFT deadline
    provider_error = "provider_error"  # 5xx or transport failure
    provider_skipped = "provider_skipped"  # cooldown / quota guard, not attempted
    all_providers_unavailable = "all_providers_unavailable"
    llm_interrupted = "llm_interrupted"  # failed after first token; no failover
    retrieval_failed = "retrieval_failed"
    sql_failed = "sql_failed"
    internal = "internal"


class EntityKind(StrEnum):
    city = "city"
    district = "district"
    country = "country"
    project = "project"
    brand = "brand"


# ---------------------------------------------------------------------------
# Pure helpers (deterministic, no I/O)
# ---------------------------------------------------------------------------


def listing_id(source_id: str) -> str:
    return f"wasalt:{source_id}"


def project_id(slug: str) -> str:
    return f"darglobal:project:{slug}"


def article_id(kind: ArticleKind, slug: str) -> str:
    return f"darglobal:{ArticleKind(kind).value}:{slug}"


def point_id(doc_id: str, chunk_index: int) -> str:
    """Qdrant point id for chunk `chunk_index` of `doc_id` (uuid5, string form)."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{doc_id}#{chunk_index}"))


def compute_prompt_version(prompt_name: str, prompt_bytes: bytes) -> str:
    """`<file stem>@<first 8 hex of sha256(file bytes)>`, e.g. `system_v1@1a2b3c4d`.

    Any edit to the prompt file changes the version, even without a rename.
    """
    return f"{prompt_name}@{hashlib.sha256(prompt_bytes).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# Source records (data/processed/*.jsonl, one JSON object per line)
# ---------------------------------------------------------------------------


class Listing(_Contract):
    """A Wasalt listing after normalise + dedupe + sanitise. PII already dropped."""

    id: NonEmptyStr
    source: Literal["wasalt"] = "wasalt"
    source_id: Annotated[str, Field(pattern=_SLUG, max_length=64)]
    url: HttpUrlStr
    title: Annotated[str, Field(min_length=1, max_length=300)]
    description: ShortText | None = None  # sanitised: no URLs/phones/emails, <= 400 chars
    purpose: Purpose
    property_type: PropertyType
    property_type_raw: str | None = None  # source value before mapping
    city: NonEmptyStr  # canonical English name
    district: str | None = None  # canonical English name
    # sale: asking price. rent: rent amount per `rent_period`, as listed.
    price_sar: Annotated[int, Field(gt=0)] | None = None
    rent_period: RentPeriod | None = None  # rent only
    # rent only: price_sar * ANNUAL_RENT_FACTOR[rent_period]; rent price filters use this.
    annual_rent_sar: Annotated[int, Field(gt=0)] | None = None
    bedrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    bathrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    area_sqm: Annotated[float, Field(gt=0)] | None = None
    lat: Annotated[float, Field(ge=-90, le=90)] | None = None
    lon: Annotated[float, Field(ge=-180, le=180)] | None = None
    amenities: list[NonEmptyStr] = Field(default_factory=list)
    published_at: AwareDatetime | None = None
    scraped_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> "Listing":
        if self.id != listing_id(self.source_id):
            raise ValueError(f"id must be {listing_id(self.source_id)!r}")
        if self.purpose is Purpose.sale:
            if self.rent_period is not None or self.annual_rent_sar is not None:
                raise ValueError("sale listings must not set rent_period/annual_rent_sar")
        elif self.rent_period is not None and self.price_sar is not None:
            expected = self.price_sar * ANNUAL_RENT_FACTOR[self.rent_period]
            if self.annual_rent_sar != expected:
                raise ValueError(f"annual_rent_sar must be {expected}")
        elif self.annual_rent_sar is not None:
            raise ValueError("annual_rent_sar requires price_sar and rent_period")
        return self


class Project(_Contract):
    """A DarGlobal project page after LLM extraction + verbatim check.

    Every extracted field is nullable: a value that does not appear verbatim on its
    page is set to null (PLAN section 6). Project prices stay in the source currency;
    they are never FX-converted.

    `body` is not an extracted field. It is the main page text (trafilatura output),
    sanitised with the same rules as `Article.body` (no URLs, phones, emails, `<` or
    `>`); organize re-applies its sanitiser before chunking. It is chunked at build time
    into project chunks 1..n (`docs/specs/retrieval.md` §2.3), never shown on a card,
    and null when the page text could not be extracted (the project then has chunk 0
    only).
    """

    id: NonEmptyStr
    source: Literal["darglobal"] = "darglobal"
    slug: Annotated[str, Field(pattern=_SLUG, max_length=120)]
    url: HttpUrlStr
    name: Annotated[str, Field(min_length=1, max_length=200)]
    brand: str | None = None  # partner brand, e.g. "Trump", "Missoni"
    country: str | None = None  # canonical English
    city: str | None = None  # canonical English
    location: str | None = None  # verbatim area/address text
    status: str | None = None  # verbatim, e.g. "Off-plan"
    handover: str | None = None  # verbatim, e.g. "Q4 2027"
    unit_types: list[NonEmptyStr] = Field(default_factory=list)  # verbatim
    bedrooms_min: Annotated[int, Field(ge=0, le=50)] | None = None
    bedrooms_max: Annotated[int, Field(ge=0, le=50)] | None = None
    area_sqm_min: Annotated[float, Field(gt=0)] | None = None
    area_sqm_max: Annotated[float, Field(gt=0)] | None = None
    price_from_amount: Annotated[int, Field(gt=0)] | None = None
    price_from_currency: Annotated[str, Field(pattern=_CURRENCY)] | None = None
    payment_plan: Annotated[str, Field(max_length=200)] | None = None  # verbatim
    amenities: list[NonEmptyStr] = Field(default_factory=list)
    description: ShortText | None = None
    body: NonEmptyStr | None = None  # sanitised main page text, chunked at build time
    scraped_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> "Project":
        if self.id != project_id(self.slug):
            raise ValueError(f"id must be {project_id(self.slug)!r}")
        if (self.price_from_amount is None) != (self.price_from_currency is None):
            raise ValueError("price_from_amount and price_from_currency are set together")
        _check_range(self.bedrooms_min, self.bedrooms_max, "bedrooms")
        _check_range(self.area_sqm_min, self.area_sqm_max, "area_sqm")
        return self


class Article(_Contract):
    """A DarGlobal press release, insight or blog post (sanitised full text)."""

    id: NonEmptyStr
    source: Literal["darglobal"] = "darglobal"
    kind: ArticleKind
    slug: Annotated[str, Field(pattern=_SLUG, max_length=200)]
    url: HttpUrlStr
    title: Annotated[str, Field(min_length=1, max_length=300)]
    body: NonEmptyStr  # chunked at build time
    published_at: AwareDatetime | None = None
    scraped_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> "Article":
        if self.id != article_id(self.kind, self.slug):
            raise ValueError(f"id must be {article_id(self.kind, self.slug)!r}")
        return self


def _check_range(lo: float | None, hi: float | None, name: str) -> None:
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"{name}: min > max")


# ---------------------------------------------------------------------------
# Index payloads and cards
# ---------------------------------------------------------------------------


class DocFields(_Contract):
    """Display + filter fields shared by Qdrant payloads and UI cards.

    Fields not relevant to `kind` are null. Filter field mapping (SearchFilters ->
    payload key) is documented on `SearchFilters`.
    """

    doc_id: NonEmptyStr
    kind: DocKind
    source: Source
    title: NonEmptyStr
    url: HttpUrlStr
    scraped_at: AwareDatetime
    # listing fields
    purpose: Purpose | None = None
    property_type: PropertyType | None = None
    city: str | None = None
    district: str | None = None
    price_sar: Annotated[int, Field(gt=0)] | None = None
    rent_period: RentPeriod | None = None
    annual_rent_sar: Annotated[int, Field(gt=0)] | None = None
    bedrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    bathrooms: Annotated[int, Field(ge=0, le=50)] | None = None
    area_sqm: Annotated[float, Field(gt=0)] | None = None
    # project fields
    country: str | None = None
    brand: str | None = None
    handover: str | None = None
    price_from_amount: Annotated[int, Field(gt=0)] | None = None
    price_from_currency: Annotated[str, Field(pattern=_CURRENCY)] | None = None
    # article fields
    article_kind: ArticleKind | None = None
    published_at: AwareDatetime | None = None


class ChunkPayload(DocFields):
    """Qdrant point payload (JSON string column `payload` in points_*.parquet)."""

    chunk_index: Annotated[int, Field(ge=0)]
    text: NonEmptyStr  # embed-mode rendering of this chunk (no [n], URL or id)


class Card(DocFields):
    """A source card shown in the UI and cited as `[n]` in the answer."""

    n: Annotated[int, Field(ge=1, le=MAX_CARDS)]  # citation number, 1-based
    score: float | None = None  # fused/rerank score; null for SQL example cards
    snippet: ShortText | None = None


class SparseVector(_Contract):
    indices: list[Annotated[int, Field(ge=0)]]
    values: list[float]

    @model_validator(mode="after")
    def _check(self) -> "SparseVector":
        if len(self.indices) != len(self.values):
            raise ValueError("indices and values must have equal length")
        return self


# ---------------------------------------------------------------------------
# Filters and request
# ---------------------------------------------------------------------------


class SearchFilters(_Contract):
    """Structured constraints extracted by the rule-based router.

    Payload / column mapping (exact match unless noted):
      kinds          -> kind (match any)
      source         -> source
      purpose        -> purpose
      property_type  -> property_type
      city/district/country -> same-named field (canonical English names)
      project_id     -> doc_id (project_info route)
      bedrooms_min/max -> bedrooms (inclusive range; exact "3-bed" = min=max=3)
      area_min/max_sqm -> area_sqm (inclusive)
      price_min/max_sar -> annual_rent_sar when purpose == rent, else price_sar (inclusive)
    `sort` is used by StructuredQuery and ignored by the hybrid retriever.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    kinds: tuple[DocKind, ...] | None = None
    source: Source | None = None
    purpose: Purpose | None = None
    property_type: PropertyType | None = None
    city: str | None = None
    district: str | None = None
    country: str | None = None
    project_id: str | None = None
    bedrooms_min: Annotated[int, Field(ge=0, le=50)] | None = None
    bedrooms_max: Annotated[int, Field(ge=0, le=50)] | None = None
    price_min_sar: Annotated[int, Field(ge=0)] | None = None
    price_max_sar: Annotated[int, Field(ge=0)] | None = None
    area_min_sqm: Annotated[float, Field(ge=0)] | None = None
    area_max_sqm: Annotated[float, Field(ge=0)] | None = None
    sort: SortKey = SortKey.relevance

    @model_validator(mode="after")
    def _check(self) -> "SearchFilters":
        _check_range(self.bedrooms_min, self.bedrooms_max, "bedrooms")
        _check_range(self.price_min_sar, self.price_max_sar, "price_sar")
        _check_range(self.area_min_sqm, self.area_max_sqm, "area_sqm")
        return self

    def canonical_json(self) -> str:
        """Deterministic serialisation for cache keys: non-null fields, sorted keys."""
        data = self.model_dump(mode="json", exclude_none=True)
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def is_empty(self) -> bool:
        return self.model_dump(exclude_none=True, exclude={"sort"}) == {}


class ChatRequest(_Contract):
    """Body of `POST /api/chat`. The session is an HttpOnly cookie, not a field."""

    message: Annotated[str, Field(min_length=1, max_length=MAX_QUERY_CHARS)]
    lang: Language | None = None  # optional client hint; the router decides


# ---------------------------------------------------------------------------
# SSE contract v1 (see docs/contracts/sse-v1.md)
# ---------------------------------------------------------------------------


class StageTimings(_Contract):
    """Per-stage wall time in ms. A stage that did not run is null.

    `embed_ms` + `sparse_ms` together equal the wall time of the query-embedding call:
    `sparse_ms` is the bm25 part reported by the adapter (`QueryEmbedding.sparse_ms`),
    `embed_ms` is the rest (dense model, executor hop and queue wait). Both are null on
    an L3 hit. See `app/domain/ports.py` `QueryEmbedding`.
    """

    cache_lookup_ms: Ms | None = None
    embed_ms: Ms | None = None
    sparse_ms: Ms | None = None
    retrieve_ms: Ms | None = None
    rerank_ms: Ms | None = None
    sql_ms: Ms | None = None
    llm_ttft_ms: Ms | None = None
    llm_total_ms: Ms | None = None
    grounding_ms: Ms | None = None
    total_ms: Ms


class TokenUsage(_Contract):
    """Tokens spent by this request (0/0 when no LLM call was made)."""

    prompt_tokens: Annotated[int, Field(ge=0)] = 0
    completion_tokens: Annotated[int, Field(ge=0)] = 0


class GroundingReport(_Contract):
    """Deterministic grounding check result. null = check not applicable."""

    numbers_ok: bool | None = None
    entities_ok: bool | None = None
    citations_ok: bool | None = None
    failures: list[Annotated[str, Field(max_length=200)]] = Field(
        default_factory=list, max_length=20
    )

    @property
    def grounded(self) -> bool:
        return False not in (self.numbers_ok, self.entities_ok, self.citations_ok)


class AnswerFlags(_Contract):
    truncated: bool = False
    unverified: bool = False


class _EventBase(_Contract):
    v: Literal[1] = SSE_VERSION
    request_id: Annotated[str, Field(pattern=_REQUEST_ID)]

    def to_sse(self) -> dict[str, str]:
        """`{"event": name, "data": json}`, accepted by sse-starlette as a dict."""
        return {"event": self.event, "data": self.model_dump_json()}  # type: ignore[attr-defined]


class ChatEventStart(_EventBase):
    event: Literal["start"] = "start"
    route: Route
    language: Language
    filters: SearchFilters
    cache_tier: AnswerCacheTier
    sources: list[Card] = Field(default_factory=list, max_length=MAX_CARDS)

    @model_validator(mode="after")
    def _check(self) -> "ChatEventStart":
        if [c.n for c in self.sources] != list(range(1, len(self.sources) + 1)):
            raise ValueError("sources[].n must be 1..k in order")
        return self


class ChatEventToken(_EventBase):
    event: Literal["token"] = "token"
    delta: Annotated[str, Field(min_length=1)]

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class ChatEventEnd(_EventBase):
    event: Literal["end"] = "end"
    answer_mode: AnswerMode
    cache_tier: AnswerCacheTier
    cache_hits: list[InnerCacheTier] = Field(default_factory=list)
    provider: Provider  # who authored the text (for cache hits: the original author)
    model: str | None = None  # provider model id; null iff provider == none
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: FinishReason
    timings: StageTimings
    grounding: GroundingReport
    flags: AnswerFlags
    prompt_version: Annotated[str, Field(pattern=_PROMPT_VERSION)]
    index_version: Annotated[str, Field(pattern=_INDEX_VERSION)]

    @model_validator(mode="after")
    def _check(self) -> "ChatEventEnd":
        if (self.provider is Provider.none) != (self.model is None):
            raise ValueError("model is null iff provider == none")
        is_cache = self.answer_mode is AnswerMode.cache
        if is_cache != (self.cache_tier is not AnswerCacheTier.miss):
            raise ValueError("cache_tier is L0-L2 iff answer_mode == cache")
        if is_cache != (self.finish_reason is FinishReason.cached):
            raise ValueError("finish_reason == cached iff answer_mode == cache")
        if self.answer_mode is AnswerMode.llm and self.provider is Provider.none:
            raise ValueError("llm answers need provider openrouter|groq")
        no_llm = self.answer_mode in _NO_LLM_MODES
        if no_llm != (self.finish_reason is FinishReason.no_llm):
            raise ValueError("finish_reason == no_llm iff answer_mode is a no-LLM mode")
        if no_llm and self.provider is not Provider.none:
            raise ValueError("no-LLM answers need provider == none")
        truncating = self.finish_reason in (FinishReason.length, FinishReason.error)
        if truncating and not self.flags.truncated:
            raise ValueError("finish_reason length/error implies flags.truncated")
        if self.flags.unverified != (not self.grounding.grounded):
            raise ValueError("flags.unverified must equal (not grounding.grounded)")
        return self


_NO_LLM_MODES = (AnswerMode.retrieval_only, AnswerMode.static_busy, AnswerMode.refusal)


class ChatEventError(_EventBase):
    event: Literal["error"] = "error"
    code: ErrorCode
    message: Annotated[str, Field(min_length=1, max_length=300)]  # user-safe, no internals
    degraded_to: DegradedTo | None = None
    fatal: bool = False  # true: no further events; the stream closes

    @model_validator(mode="after")
    def _check(self) -> "ChatEventError":
        if self.fatal and self.degraded_to is not None:
            raise ValueError("a fatal error cannot degrade")
        return self


ChatEvent = Annotated[
    ChatEventStart | ChatEventToken | ChatEventEnd | ChatEventError,
    Field(discriminator="event"),
]
CHAT_EVENT_ADAPTER: TypeAdapter[ChatEvent] = TypeAdapter(ChatEvent)


# ---------------------------------------------------------------------------
# Answer cache records (L0 warm cache file, L1/L2 values)
# ---------------------------------------------------------------------------


class CachedAnswer(_Contract):
    """A cacheable answer. Only grounded, untruncated answers are ever stored."""

    answer: NonEmptyStr
    route: Route
    language: Language
    filters: SearchFilters
    sources: list[Card] = Field(default_factory=list, max_length=MAX_CARDS)
    provider: Provider
    model: str | None = None
    grounding: GroundingReport
    prompt_version: Annotated[str, Field(pattern=_PROMPT_VERSION)]
    index_version: Annotated[str, Field(pattern=_INDEX_VERSION)]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> "CachedAnswer":
        if not self.grounding.grounded:
            raise ValueError("ungrounded answers are never cached")
        return self


class WarmCacheRecord(_Contract):
    """One line of `data/index/warm_cache.jsonl` (L0)."""

    key: NonEmptyStr  # exact cache key, same function as L1
    query: NonEmptyStr  # the original query, for audit
    entry: CachedAnswer


# ---------------------------------------------------------------------------
# Entities dictionary (data/index/entities.json)
# ---------------------------------------------------------------------------


class Entity(_Contract):
    kind: EntityKind
    canonical: NonEmptyStr  # English canonical name used in payloads and filters
    aliases: list[NonEmptyStr] = Field(default_factory=list)  # incl. Arabic spellings
    parent: str | None = None  # district -> its city; city -> its country (if known)
    doc_id: str | None = None  # project -> "darglobal:project:<slug>"


class Entities(_Contract):
    schema_version: Literal[1] = ENTITIES_SCHEMA_VERSION
    entities: list[Entity]

    @model_validator(mode="after")
    def _check(self) -> "Entities":
        seen: set[tuple[EntityKind, str, str | None]] = set()
        for e in self.entities:
            key = (e.kind, e.canonical, e.parent)
            if key in seen:
                raise ValueError(f"duplicate entity {key}")
            seen.add(key)
        return self


# ---------------------------------------------------------------------------
# Manifest (data/index/manifest.json, see docs/contracts/manifest.md)
# ---------------------------------------------------------------------------

_EXCLUDED_FROM_FILES = {"manifest.json", "warm_cache.jsonl"}


class ManifestMismatchError(Exception):
    """Raised at startup when the runtime config is incompatible with the index."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class ManifestCounts(_Contract):
    listings_sale: Annotated[int, Field(ge=0)]
    listings_rent: Annotated[int, Field(ge=0)]
    projects: Annotated[int, Field(ge=0)]
    articles: Annotated[int, Field(ge=0)]
    points: Annotated[int, Field(ge=0)]  # Qdrant points (chunks) across all shards

    @model_validator(mode="after")
    def _check(self) -> "ManifestCounts":
        docs = self.listings_sale + self.listings_rent + self.projects + self.articles
        if self.points < docs:
            raise ValueError("every document has at least one point")
        return self


class ManifestFile(_Contract):
    path: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$")]  # flat name in data/index/
    sha256: Annotated[str, Field(pattern=_HEX64)]
    size_bytes: Annotated[int, Field(ge=0, lt=MAX_SHARD_BYTES)]
    rows: Annotated[int, Field(ge=0)] | None = None  # parquet row count; null for json


def compute_index_version(files: Iterable[ManifestFile]) -> str:
    """First 16 hex of sha256 over "path\\tsha256\\n" lines sorted by path."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda f: f.path):
        h.update(f"{f.path}\t{f.sha256}\n".encode())
    return h.hexdigest()[:16]


class Manifest(_Contract):
    schema_version: Literal[1]
    index_version: Annotated[str, Field(pattern=_INDEX_VERSION)]
    built_at: AwareDatetime
    embed_model: NonEmptyStr
    dim: Annotated[int, Field(gt=0)]
    sparse_model: NonEmptyStr
    distance: Literal["cosine"]
    vector_dtype: Literal["float16"]
    counts: ManifestCounts
    scraped_at_min: AwareDatetime
    scraped_at_max: AwareDatetime
    git_sha: Annotated[str, Field(pattern=r"^([0-9a-f]{7,40}(-dirty)?|unknown)$")]
    # Required key, nullable value: null until scripts/warm_cache.py has run.
    prompt_version: Annotated[str, Field(pattern=_PROMPT_VERSION)] | None
    files: Annotated[list[ManifestFile], Field(min_length=1)]

    @model_validator(mode="after")
    def _check(self) -> "Manifest":
        if self.scraped_at_min > self.scraped_at_max:
            raise ValueError("scraped_at_min > scraped_at_max")
        names = [f.path for f in self.files]
        if len(names) != len(set(names)):
            raise ValueError("duplicate file paths")
        if bad := _EXCLUDED_FROM_FILES.intersection(names):
            raise ValueError(f"files must not list {sorted(bad)}")
        missing = [n for n in ("listings.parquet", "entities.json") if n not in names]
        if not any(n.startswith("points_") and n.endswith(".parquet") for n in names):
            missing.append("points_*.parquet")
        if missing:
            raise ValueError(f"files missing {missing}")
        expected = compute_index_version(self.files)
        if self.index_version != expected:
            raise ValueError(f"index_version must be {expected!r} for the listed files")
        return self

    def assert_compatible(
        self,
        *,
        embed_model: str,
        dim: int,
        sparse_model: str,
        prompt_version: str,
        warm_cache_present: bool,
    ) -> None:
        """Fail fast at startup (PLAN section 4). Raises ManifestMismatchError."""
        reasons: list[str] = []
        if embed_model != self.embed_model:
            reasons.append(f"embed_model {embed_model!r} != manifest {self.embed_model!r}")
        if dim != self.dim:
            reasons.append(f"dim {dim} != manifest {self.dim}")
        if sparse_model != self.sparse_model:
            reasons.append(f"sparse_model {sparse_model!r} != manifest {self.sparse_model!r}")
        if warm_cache_present and self.prompt_version != prompt_version:
            reasons.append(
                f"warm cache prompt_version {self.prompt_version!r} != current {prompt_version!r}"
            )
        if reasons:
            raise ManifestMismatchError(reasons)
