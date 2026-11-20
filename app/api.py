"""FastAPI application: /api/chat SSE, /health, /ready, /stats, /metrics, static UI.

Imports: common, service, adapters, telemetry, config, domain.ports.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse

from app.config import Settings, get_settings
from app.service import ChatService
from app.telemetry import RingBuffer, build_stats
from common.schemas import (
    ChatRequest,
    ManifestMismatchError,
    compute_prompt_version,
)

logger = logging.getLogger("app.api")

# ---------------------------------------------------------------------------
# Globals wired at startup
# ---------------------------------------------------------------------------

_chat_service: ChatService | None = None
_ring: RingBuffer = RingBuffer(2000)
_ready: bool = False
_ready_detail: str = "not started"
_settings: Settings | None = None


# ---------------------------------------------------------------------------
# App wiring (api layer is allowed to import service, adapters, config, telemetry, domain.ports)
# ---------------------------------------------------------------------------


def _build_app(settings: Settings, ring: RingBuffer) -> tuple[ChatService, str]:
    """Wire the full app: read side, LLM, caches, entities, ChatService.

    Returns (ChatService, detail_string). Raises ManifestMismatchError on fail-fast.
    The api layer builds adapters and passes them to the service constructor.
    """
    from concurrent.futures import ThreadPoolExecutor

    from app.adapters.factory import build_llm, build_read_side
    from common.schemas import Entities

    index_dir = Path(settings.INDEX_DIR)

    # Compute prompt version
    prompt_path = Path(__file__).parent / "prompts" / "system_v1.txt"
    prompt_bytes = prompt_path.read_bytes()
    prompt_version = compute_prompt_version("system_v1", prompt_bytes)

    # Check warm cache
    warm_cache_path = index_dir / "warm_cache.jsonl"
    warm_cache_present = warm_cache_path.exists() and warm_cache_path.stat().st_size > 0

    # Build read side (adapters layer)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="onnx")
    rs = build_read_side(
        index_dir,
        rerank=bool(settings.RERANK_ENABLED),
        rrf_k=settings.RRF_K,
        dense_weight=settings.DENSE_WEIGHT,
        sparse_weight=settings.SPARSE_WEIGHT,
        prefetch_limit=settings.PREFETCH_LIMIT,
        embed_model=settings.EMBED_MODEL,
        sparse_model=settings.SPARSE_MODEL,
        rerank_model=settings.RERANK_MODEL,
        prompt_version=prompt_version,
        warm_cache_present=warm_cache_present,
        executor=executor,
    )

    # Build LLM (adapters layer)
    llm = build_llm(max_concurrency=settings.LLM_MAX_CONCURRENCY)

    # Load entities JSON (api layer loads data, service builds domain objects)
    entities_path = index_dir / "entities.json"
    entities_data = json.loads(entities_path.read_text(encoding="utf-8"))
    entities = Entities(**entities_data)

    # Load manifest for index_version
    manifest_data = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    index_version = manifest_data["index_version"]

    # Cache: stub if caches.py is not available yet (WP-4-caches writes it)
    cache: object
    embedder = rs.embedder
    retriever = rs.retriever
    try:
        from app.adapters.caches import AnswerCache, CachedEmbedder, CachedRetriever

        warm_cache_path = index_dir / "warm_cache.jsonl"
        cache = AnswerCache(
            warm_cache_path=warm_cache_path if warm_cache_present else None,
            index_version=index_version,
            prompt_version=prompt_version,
            tau=settings.TAU,
            semantic_enabled=bool(settings.SEMANTIC_CACHE_ENABLED),
        )
        embedder = CachedEmbedder(rs.embedder, capacity=1024)
        retriever = CachedRetriever(rs.retriever, index_version=index_version, capacity=512)
        logger.info("Caches wired from app.adapters.caches")
    except (ImportError, TypeError) as exc:
        logger.warning("Caches not wired (using stub): %s", exc)

        class _StubCache:
            async def get_exact(self, key: str):
                return None

            async def get_semantic(self, dense, *, scope: str):
                return None

            async def put(self, key: str, entry, *, dense=None, scope=None):
                pass

        cache = _StubCache()

    # ChatService constructor builds EntityIndex from Entities (domain-layer work)
    svc = ChatService(
        embedder=embedder,
        retriever=retriever,
        reranker=rs.reranker,
        sql=rs.sql,
        llm=llm,
        cache=cache,
        entities=entities,
        settings=settings,
        ring=ring,
        index_version=index_version,
        prompt_version=prompt_version,
    )

    detail = f"index_version={index_version}, prompt_version={prompt_version}"
    return svc, detail


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def _get_client_ip(request: Request) -> str:
    """Extract client IP from X-Forwarded-For based on RATE_LIMIT_XFF_HOP setting."""
    s = _settings or get_settings()
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            if s.RATE_LIMIT_XFF_HOP == "last":
                return parts[-1]
            return parts[0]
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_get_client_ip)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _chat_service, _ready, _ready_detail, _settings

    _settings = get_settings()
    logger.info("Starting app with INDEX_DIR=%s", _settings.INDEX_DIR)

    try:
        svc, detail = _build_app(_settings, _ring)
        # Warm-up: one embed_query and one trivial DuckDB query before /ready
        await svc.warm_up()
        _chat_service = svc
        _ready = True
        _ready_detail = detail
        logger.info("App ready: %s", detail)
    except ManifestMismatchError as e:
        _ready = False
        _ready_detail = f"manifest mismatch: {e}"
        logger.error("Startup fail-fast: %s", e)
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as e:
        _ready = False
        _ready_detail = f"startup error: {e}"
        logger.error("Startup error: %s", e)

    yield

    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.state.limiter = limiter


# Rate limit error handler
@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "rate_limited", "retry_after_s": 60},
    )


# CORS: same-origin only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # same-origin: no cross-origin allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health / ready / stats
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not _ready or _chat_service is None:
        return JSONResponse(status_code=503, content={"detail": _ready_detail})
    return {"status": "ready", "detail": _ready_detail}


@app.get("/stats")
async def stats():
    llm_status = None
    if _chat_service is not None:
        llm_status = _chat_service._llm.status()
    return build_stats(_ring, llm_status)


# ---------------------------------------------------------------------------
# Metrics (prometheus, flag-gated)
# ---------------------------------------------------------------------------


@app.get("/metrics")
async def metrics():
    s = _settings or get_settings()
    if not s.METRICS_ENABLED:
        return Response(status_code=404)
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return Response(status_code=501)


# ---------------------------------------------------------------------------
# Chat SSE endpoint
# ---------------------------------------------------------------------------


def _check_origin(request: Request) -> str | None:
    """S2: return an error message if the origin is cross-origin, else None."""
    origin = request.headers.get("origin")
    if not origin:
        return None  # no Origin header (curl, bench) is allowed
    # Parse origin host
    from urllib.parse import urlparse
    try:
        origin_host = urlparse(origin).hostname or ""
    except (ValueError, AttributeError):
        return "bad_origin"
    # Compare with request Host header
    request_host = request.headers.get("host", "")
    # Strip port from host
    if ":" in request_host:
        request_host = request_host.rsplit(":", 1)[0]
    if origin_host == request_host:
        return None  # same origin
    # Check ALLOWED_ORIGINS setting
    s = _settings or get_settings()
    allowed = getattr(s, "ALLOWED_ORIGINS", "")
    if allowed:
        allowed_list = [h.strip() for h in allowed.split(",") if h.strip()]
        if origin_host in allowed_list:
            return None
    return "cross_origin"


@app.post("/api/chat")
@limiter.limit(lambda: f"{(_settings or get_settings()).RATE_LIMIT_PER_MIN}/minute")
@limiter.limit(lambda: f"{(_settings or get_settings()).RATE_LIMIT_PER_DAY}/day")
async def chat(request: Request):
    # S2: reject cross-origin requests
    origin_err = _check_origin(request)
    if origin_err:
        return JSONResponse(status_code=403, content={"detail": origin_err})

    # S2: require Content-Type: application/json
    ct = request.headers.get("content-type", "")
    if not ct.startswith("application/json"):
        return JSONResponse(status_code=415, content={"detail": "Content-Type must be application/json"})

    if not _ready or _chat_service is None:
        return JSONResponse(status_code=503, content={"detail": "not_ready"})

    # Parse body
    try:
        body = await request.json()
        chat_req = ChatRequest(**body)
    except (ValueError, TypeError, KeyError):
        return JSONResponse(status_code=422, content={"detail": "Invalid request body"})

    # Session cookie
    session_id = request.cookies.get("sid")
    if not session_id:
        session_id = uuid.uuid4().hex

    async def event_generator():
        async for event in _chat_service.handle(
            chat_req.message,
            session_id=session_id,
            lang_hint=chat_req.lang,
        ):
            yield event

    response = EventSourceResponse(event_generator())
    response.set_cookie(
        key="sid",
        value=session_id,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=1800,
    )
    return response


# ---------------------------------------------------------------------------
# Static files (UI)
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():

    from starlette.middleware.base import BaseHTTPMiddleware

    class _NoCacheStaticMiddleware(BaseHTTPMiddleware):
        """Add Cache-Control: no-cache to static file responses."""

        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            path = request.url.path
            if path == "/" or path.endswith((".js", ".css", ".html")):
                response.headers["Cache-Control"] = "no-cache"
            return response

    app.add_middleware(_NoCacheStaticMiddleware)
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
