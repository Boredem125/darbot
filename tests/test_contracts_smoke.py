"""Smoke tests for the Wave 1a contracts (common/schemas.py, app/domain/ports.py).

Minimal by design: models import, one example of each round-trips through JSON,
key invariants reject bad input, and the documented SSE example stream validates.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from app.domain import ports
from common.schemas import (
    CHAT_EVENT_ADAPTER,
    EMBED_DIM_DEFAULT,
    EMBED_MODEL_DEFAULT,
    MODEL_ENV_DEFAULTS,
    SPARSE_MODEL_DEFAULT,
    Article,
    ArticleKind,
    CachedAnswer,
    Card,
    ChatEventEnd,
    ChatEventError,
    ChatEventStart,
    ChatEventToken,
    ChatRequest,
    ChunkPayload,
    Entities,
    GroundingReport,
    Listing,
    Manifest,
    ManifestFile,
    ManifestMismatchError,
    Project,
    SearchFilters,
    SparseVector,
    WarmCacheRecord,
    compute_index_version,
    compute_prompt_version,
    point_id,
)

REPO = Path(__file__).resolve().parents[1]
T = datetime(2026, 11, 19, 8, 0, tzinfo=UTC)
RID = "9f2c4e0b7a1d4c3e8b6a5d4c3b2a1f00"
PV = compute_prompt_version("system_v1", b"prompt text")
FILES = [
    ManifestFile(path="points_000.parquet", sha256="a" * 64, size_bytes=9_000_000, rows=8000),
    ManifestFile(path="listings.parquet", sha256="b" * 64, size_bytes=400_000, rows=3000),
    ManifestFile(path="entities.json", sha256="c" * 64, size_bytes=20_000),
]
IV = compute_index_version(FILES)


def listing() -> Listing:
    return Listing(
        id="wasalt:123456", source_id="123456", url="https://wasalt.sa/en/property/123456",
        title="Apartment for rent in Al Rawdah", purpose="rent", property_type="apartment",
        city="Jeddah", district="Al Rawdah", price_sar=5000, rent_period="monthly",
        annual_rent_sar=60000, bedrooms=2, area_sqm=120.0, scraped_at=T,
    )


def card(n: int = 1) -> Card:
    return Card(
        n=n, doc_id="wasalt:123456", kind="listing", source="wasalt", title="Villa",
        url="https://wasalt.sa/en/property/123456", scraped_at=T, purpose="sale",
        property_type="villa", city="Riyadh", price_sar=1_950_000, bedrooms=3, score=0.03,
    )


def grounded() -> GroundingReport:
    return GroundingReport(numbers_ok=True, entities_ok=True, citations_ok=True)


def cached_answer() -> CachedAnswer:
    return CachedAnswer(
        answer="SAR 1,950,000 [1].", route="lookup", language="en",
        filters=SearchFilters(city="Riyadh"), sources=[card()], provider="groq",
        model="openai/gpt-oss-120b", grounding=grounded(), prompt_version=PV,
        index_version=IV, created_at=T,
    )


def manifest_dict() -> dict:
    return {
        "schema_version": 1, "index_version": IV, "built_at": T.isoformat(),
        "embed_model": EMBED_MODEL_DEFAULT,
        "dim": EMBED_DIM_DEFAULT, "sparse_model": SPARSE_MODEL_DEFAULT, "distance": "cosine",
        "vector_dtype": "float16",
        "counts": {"listings_sale": 2000, "listings_rent": 1000, "projects": 15,
                   "articles": 100, "points": 3400},
        "scraped_at_min": T.isoformat(), "scraped_at_max": T.isoformat(),
        "git_sha": "unknown", "prompt_version": None,
        "files": [f.model_dump() for f in FILES],
    }


def end_event(**over) -> ChatEventEnd:
    base = {
        "request_id": RID, "answer_mode": "llm", "cache_tier": "miss", "provider": "groq",
        "model": "openai/gpt-oss-120b", "finish_reason": "stop",
        "timings": {"total_ms": 900.0}, "grounding": grounded(),
        "flags": {"truncated": False, "unverified": False},
        "prompt_version": PV, "index_version": IV,
    }
    base.update(over)
    return ChatEventEnd(**base)


def _examples() -> list[BaseModel]:
    return [
        listing(),
        Project(id="darglobal:project:trump-tower-jeddah", slug="trump-tower-jeddah",
                url="https://www.darglobal.co.uk/projects/trump-tower-jeddah",
                name="Trump Tower Jeddah", brand="Trump", city="Jeddah", handover="Q4 2027",
                price_from_amount=2_500_000, price_from_currency="SAR",
                body="Trump Tower Jeddah offers branded residences.", scraped_at=T),
        Article(id="darglobal:press:q3-results", kind=ArticleKind.press, slug="q3-results",
                url="https://www.darglobal.co.uk/press/q3-results", title="Q3 results",
                body="Dar Global reported...", scraped_at=T),
        card(),
        ChunkPayload(doc_id="wasalt:123456", kind="listing", source="wasalt", title="Villa",
                     url="https://wasalt.sa/en/property/123456", scraped_at=T,
                     chunk_index=0, text="Villa for sale in Riyadh, 3 bedrooms"),
        SearchFilters(city="Riyadh", property_type="villa", bedrooms_min=3, bedrooms_max=3,
                      price_max_sar=2_000_000, kinds=("listing",)),
        ChatRequest(message="3-bed villas in Riyadh under 2M SAR"),
        cached_answer(),
        WarmCacheRecord(key="k1", query="villas in riyadh", entry=cached_answer()),
        Entities(entities=[{"kind": "city", "canonical": "Riyadh", "aliases": ["الرياض"]}]),
        Manifest.model_validate(manifest_dict()),
    ]


@pytest.mark.parametrize("model", _examples(), ids=lambda m: type(m).__name__)
def test_round_trip(model: BaseModel):
    again = type(model).model_validate_json(model.model_dump_json())
    assert again == model


def test_chat_events_round_trip_through_union():
    events = [
        ChatEventStart(request_id=RID, route="lookup", language="en",
                       filters=SearchFilters(), cache_tier="miss", sources=[card(1), card(2)]),
        ChatEventToken(request_id=RID, delta=" SAR 1,950,000 [1]. "),
        end_event(),
        ChatEventError(request_id=RID, code="provider_rate_limited", message="Switching model.",
                       degraded_to="fallback_llm"),
    ]
    for ev in events:
        frame = ev.to_sse()
        assert frame["event"] == ev.event
        assert CHAT_EVENT_ADAPTER.validate_json(frame["data"]) == ev
    assert events[1].delta == " SAR 1,950,000 [1]. "  # whitespace preserved in tokens


def test_documented_example_stream_validates():
    doc = (REPO / "docs" / "contracts" / "sse-v1.md").read_text(encoding="utf-8")
    pairs = re.findall(r"^event: (\w+)\ndata: (\{.*\})$", doc, flags=re.MULTILINE)
    assert [name for name, _ in pairs] == ["start", "error", "token", "token", "end"]
    for name, data in pairs:
        assert CHAT_EVENT_ADAPTER.validate_json(data).event == name


@pytest.mark.parametrize("missing", ["embed_model", "dim", "index_version", "prompt_version",
                                     "files", "git_sha"])
def test_manifest_rejects_missing_required_field(missing: str):
    data = manifest_dict()
    del data[missing]
    with pytest.raises(ValidationError):
        Manifest.model_validate_json(json.dumps(data))


def test_manifest_rejects_wrong_index_version_and_big_file():
    data = manifest_dict()
    data["index_version"] = "0" * 16
    with pytest.raises(ValidationError):
        Manifest.model_validate(data)
    with pytest.raises(ValidationError):
        ManifestFile(path="points_001.parquet", sha256="d" * 64, size_bytes=10_000_000)


def test_manifest_fail_fast_compatibility():
    m = Manifest.model_validate(manifest_dict())
    ok = {"embed_model": m.embed_model, "dim": 384, "sparse_model": "Qdrant/bm25",
          "prompt_version": PV}
    m.assert_compatible(**ok, warm_cache_present=False)  # null prompt_version, no L0: fine
    with pytest.raises(ManifestMismatchError):
        m.assert_compatible(**ok, warm_cache_present=True)  # L0 present, version null
    with pytest.raises(ManifestMismatchError) as exc:
        m.assert_compatible(**{**ok, "dim": 768}, warm_cache_present=False)
    assert exc.value.reasons == ["dim 768 != manifest 384"]


def test_key_invariants():
    with pytest.raises(ValidationError):  # annual rent must equal monthly * 12
        Listing(**{**listing().model_dump(), "annual_rent_sar": 50000})
    with pytest.raises(ValidationError):  # cache tier without answer_mode=cache
        end_event(cache_tier="L1")
    with pytest.raises(ValidationError):  # length finish must set truncated
        end_event(finish_reason="length")
    with pytest.raises(ValidationError):  # unverified must mirror grounding
        end_event(grounding={"numbers_ok": False})
    with pytest.raises(ValidationError):  # ungrounded answers are never cached
        CachedAnswer(**{**cached_answer().model_dump(), "grounding": {"numbers_ok": False}})
    with pytest.raises(ValidationError):  # card numbering must be 1..k
        ChatEventStart(request_id=RID, route="lookup", language="en", filters={},
                       cache_tier="miss", sources=[card(2)])
    assert SearchFilters(bedrooms_min=3).canonical_json() != SearchFilters(
        bedrooms_min=4).canonical_json()
    assert point_id("wasalt:1", 0) == point_id("wasalt:1", 0) != point_id("wasalt:1", 1)


def test_ports_are_protocols():
    for name in ("Retriever", "StructuredQuery", "LLM", "Cache", "Embedder", "Reranker"):
        assert getattr(ports, name)._is_protocol


# --- WP-2-contracts amendments (CONTRACT_CHANGES.md CC-1 to CC-6) ---------------------


def test_retrieval_result_top_dense_score_defaults_to_none():
    hit = ports.RetrievedHit(payload=_examples()[4], score=0.03)
    assert ports.RetrievalResult(hits=(hit,)).top_dense_score is None
    assert ports.RetrievalResult(hits=(hit,), top_dense_score=0.52).top_dense_score == 0.52


def test_query_embedding_sparse_ms_defaults_to_none():
    qe = ports.QueryEmbedding(dense=(1.0,), sparse=SparseVector(indices=[1], values=[1.0]))
    assert qe.sparse_ms is None and qe.cached is False
    assert ports.QueryEmbedding(dense=(1.0,), sparse=qe.sparse, sparse_ms=0.02).sparse_ms == 0.02


def test_project_body_is_optional_and_non_empty():
    base = _examples()[1].model_dump()
    assert Project(**{**base, "body": None}).body is None
    with pytest.raises(ValidationError):
        Project(**{**base, "body": "   "})


def test_model_env_names_are_consistent_with_manifest_contract():
    assert set(MODEL_ENV_DEFAULTS) == {"EMBED_MODEL", "SPARSE_MODEL", "RERANK_MODEL"}
    doc = (REPO / "docs" / "contracts" / "manifest.md").read_text(encoding="utf-8")
    for name, default in MODEL_ENV_DEFAULTS.items():
        assert f"`{name}`" in doc and default in doc


def test_sse_doc_places_cache_lookup_after_routing():
    doc = (REPO / "docs" / "contracts" / "sse-v1.md").read_text(encoding="utf-8")
    assert "cache lookup runs before routing" not in doc
    assert "**after routing**" in doc
