"""Mock SSE server for Wasalt / DarGlobal RAG chatbot UI development.

Usage:
    .venv\\Scripts\\python.exe scripts/mock_sse.py --port 7861
    .venv\\Scripts\\python.exe scripts/mock_sse.py --port 7861 --fast

Stdlib only (http.server.ThreadingHTTPServer), plus common.schemas to build
every event as a model and serialise it with to_sse().  That way the mock
cannot emit an invalid event.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

# Ensure the project root is on sys.path so we can import common.schemas.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from common.schemas import (
    AnswerCacheTier,
    AnswerFlags,
    AnswerMode,
    Card,
    ChatEventEnd,
    ChatEventError,
    ChatEventStart,
    ChatEventToken,
    ChatRequest,
    DegradedTo,
    ErrorCode,
    FinishReason,
    GroundingReport,
    Language,
    Provider,
    Route,
    SearchFilters,
    StageTimings,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Type alias for the event list
# ---------------------------------------------------------------------------

ChatEvent = ChatEventStart | ChatEventToken | ChatEventEnd | ChatEventError


@dataclass
class HttpError:
    """Signals a non-200 HTTP response (not an SSE stream)."""

    status: int
    body: dict


# ---------------------------------------------------------------------------
# Shared fake data
# ---------------------------------------------------------------------------

REQUEST_ID = uuid.uuid4().hex
PROMPT_VERSION = "system_v1@1a2b3c4d"
INDEX_VERSION = "0123456789abcdef"
SCRAPED_AT = "2026-11-19T08:00:00Z"


def _rid() -> str:
    """Return a fixed request_id. Tests can predict it."""
    return REQUEST_ID


def _card_listing_sale_villa() -> Card:
    return Card(
        n=1,
        doc_id="wasalt:123456",
        kind="listing",
        source="wasalt",
        title="Villa for sale in Al Narjis",
        url="https://wasalt.sa/en/property/123456",
        scraped_at=SCRAPED_AT,
        purpose="sale",
        property_type="villa",
        city="Riyadh",
        district="Al Narjis",
        price_sar=1950000,
        bedrooms=3,
        bathrooms=4,
        area_sqm=312.5,
        score=0.032,
        snippet="Modern villa near King Salman Road.",
    )


def _card_listing_sale_apartment() -> Card:
    return Card(
        n=2,
        doc_id="wasalt:234567",
        kind="listing",
        source="wasalt",
        title="Apartment for sale in Jeddah",
        url="https://wasalt.sa/en/property/234567",
        scraped_at=SCRAPED_AT,
        purpose="sale",
        property_type="apartment",
        city="Jeddah",
        district="Al Hamra",
        price_sar=850000,
        bedrooms=2,
        bathrooms=2,
        area_sqm=145.0,
        score=0.028,
        snippet="Spacious apartment with sea view.",
    )


def _card_listing_rental() -> Card:
    return Card(
        n=3,
        doc_id="wasalt:345678",
        kind="listing",
        source="wasalt",
        title="Apartment for rent in Riyadh",
        url="https://wasalt.sa/en/property/345678",
        scraped_at=SCRAPED_AT,
        purpose="rent",
        property_type="apartment",
        city="Riyadh",
        district="Al Olaya",
        price_sar=7500,
        rent_period="monthly",
        annual_rent_sar=90000,
        bedrooms=2,
        bathrooms=1,
        area_sqm=110.0,
        score=0.025,
        snippet="Furnished apartment near business district.",
    )


def _card_project(n: int = 1) -> Card:
    return Card(
        n=n,
        doc_id="darglobal:project:ocean-breeze",
        kind="project",
        source="darglobal",
        title="Ocean Breeze Residences",
        url="https://darglobal.co.uk/projects/ocean-breeze",
        scraped_at=SCRAPED_AT,
        country="UAE",
        city="Dubai",
        brand="Trump",
        handover="Q4 2027",
        price_from_amount=450000,
        price_from_currency="USD",
        score=0.045,
        snippet="Luxury waterfront residences in Dubai Marina.",
    )


def _card_article(n: int = 2) -> Card:
    return Card(
        n=n,
        doc_id="darglobal:press:expansion-2026",
        kind="article",
        source="darglobal",
        title="DarGlobal announces expansion plans for 2026",
        url="https://darglobal.co.uk/press/expansion-2026",
        scraped_at=SCRAPED_AT,
        article_kind="press",
        published_at="2026-10-15T12:00:00Z",
        score=0.020,
        snippet="Major new projects across the GCC region.",
    )


def _grounded() -> GroundingReport:
    return GroundingReport(
        numbers_ok=True, entities_ok=True, citations_ok=True, failures=[]
    )


def _null_grounding() -> GroundingReport:
    return GroundingReport(
        numbers_ok=None, entities_ok=None, citations_ok=None, failures=[]
    )


def _ungrounded() -> GroundingReport:
    return GroundingReport(
        numbers_ok=False,
        entities_ok=True,
        citations_ok=True,
        failures=["Price SAR 1,900,000 not found in sources (expected SAR 1,950,000)"],
    )


def _full_timings() -> StageTimings:
    return StageTimings(
        cache_lookup_ms=0.4,
        embed_ms=11.2,
        sparse_ms=1.3,
        retrieve_ms=9.8,
        rerank_ms=None,
        sql_ms=None,
        llm_ttft_ms=640.0,
        llm_total_ms=910.5,
        grounding_ms=0.6,
        total_ms=941.7,
    )


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------


def build_scenario(name: str, message: str) -> list[ChatEvent] | HttpError:
    """Build a list of SSE events (validated models) for the named scenario.

    Returns an HttpError when the scenario calls for a non-200 response.
    """
    rid = _rid()

    if name == "happy":
        return _scenario_happy(rid)
    elif name == "failover":
        return _scenario_failover(rid)
    elif name == "cache":
        return _scenario_cache(rid)
    elif name == "aggregate":
        return _scenario_aggregate(rid)
    elif name == "project":
        return _scenario_project(rid)
    elif name == "arabic":
        return _scenario_arabic(rid)
    elif name == "retrieval_only":
        return _scenario_retrieval_only(rid)
    elif name == "busy":
        return _scenario_busy(rid)
    elif name == "refusal":
        return _scenario_refusal(rid)
    elif name == "unverified":
        return _scenario_unverified(rid)
    elif name == "truncated":
        return _scenario_truncated(rid)
    elif name == "interrupted":
        return _scenario_interrupted(rid)
    elif name == "fatal":
        return _scenario_fatal(rid)
    elif name == "chunked":
        return _scenario_happy(rid)  # Same events; the HTTP handler writes in 7-byte pieces
    elif name == "rate_limited":
        return HttpError(status=429, body={"detail": "rate_limited", "retry_after_s": 30})
    elif name == "not_ready":
        return HttpError(status=503, body={"detail": "not_ready"})
    else:
        return _scenario_happy(rid)


def _scenario_happy(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
                bedrooms_min=3,
                bedrooms_max=3,
                price_max_sar=2000000,
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[
                _card_listing_sale_villa(),
                _card_listing_sale_apartment(),
                _card_listing_rental(),
            ],
        ),
        ChatEventToken(request_id=rid, delta="There is one 3-bedroom "),
        ChatEventToken(request_id=rid, delta="villa in Al Narjis, "),
        ChatEventToken(request_id=rid, delta="Riyadh for "),
        ChatEventToken(request_id=rid, delta="SAR 1,950,000 [1]. "),
        ChatEventToken(request_id=rid, delta="A 2-bedroom apartment "),
        ChatEventToken(request_id=rid, delta="is available in Al Hamra, Jeddah "),
        ChatEventToken(request_id=rid, delta="for SAR 850,000 [2]. "),
        ChatEventToken(
            request_id=rid,
            delta="For rentals, there is a furnished apartment in Al Olaya, Riyadh at SAR 7,500 / month [3].",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            tokens=TokenUsage(prompt_tokens=812, completion_tokens=48),
            finish_reason=FinishReason.stop,
            timings=_full_timings(),
            grounding=_grounded(),
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_failover(rid: str) -> list[ChatEvent]:
    """The example stream from sse-v1.md: soft provider_rate_limited then Groq."""
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
                bedrooms_min=3,
                bedrooms_max=3,
                price_max_sar=2000000,
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[_card_listing_sale_villa()],
        ),
        ChatEventError(
            request_id=rid,
            code=ErrorCode.provider_rate_limited,
            message="Primary model is busy; switching to a fallback model.",
            degraded_to=DegradedTo.fallback_llm,
            fatal=False,
        ),
        ChatEventToken(
            request_id=rid,
            delta="There is one 3-bedroom villa in Al Narjis, Riyadh for ",
        ),
        ChatEventToken(request_id=rid, delta="SAR 1,950,000 [1]."),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.groq,
            model="openai/gpt-oss-120b",
            tokens=TokenUsage(prompt_tokens=812, completion_tokens=24),
            finish_reason=FinishReason.stop,
            timings=_full_timings(),
            grounding=_grounded(),
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_cache(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
                bedrooms_min=3,
                bedrooms_max=3,
                price_max_sar=2000000,
            ),
            cache_tier=AnswerCacheTier.L1,
            sources=[_card_listing_sale_villa()],
        ),
        ChatEventToken(
            request_id=rid,
            delta="There is one 3-bedroom villa in Al Narjis, Riyadh for SAR 1,950,000 [1].",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.cache,
            cache_tier=AnswerCacheTier.L1,
            cache_hits=[],
            provider=Provider.groq,
            model="openai/gpt-oss-120b",
            tokens=TokenUsage(prompt_tokens=0, completion_tokens=0),
            finish_reason=FinishReason.cached,
            timings=StageTimings(cache_lookup_ms=0.3, total_ms=1.2),
            grounding=_grounded(),
            flags=AnswerFlags(truncated=False, unverified=False),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_aggregate(rid: str) -> list[ChatEvent]:
    card1 = Card(
        n=1,
        doc_id="wasalt:555001",
        kind="listing",
        source="wasalt",
        title="Example villa in Riyadh",
        url="https://wasalt.sa/en/property/555001",
        scraped_at=SCRAPED_AT,
        purpose="sale",
        property_type="villa",
        city="Riyadh",
        district="Al Narjis",
        price_sar=1800000,
        bedrooms=3,
        bathrooms=3,
        area_sqm=280.0,
        score=None,
        snippet="Example listing for aggregate query.",
    )
    card2 = Card(
        n=2,
        doc_id="wasalt:555002",
        kind="listing",
        source="wasalt",
        title="Example villa in Al Malqa",
        url="https://wasalt.sa/en/property/555002",
        scraped_at=SCRAPED_AT,
        purpose="sale",
        property_type="villa",
        city="Riyadh",
        district="Al Malqa",
        price_sar=2100000,
        bedrooms=4,
        bathrooms=4,
        area_sqm=350.0,
        score=None,
        snippet="Another example listing.",
    )
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.aggregate,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[card1, card2],
        ),
        ChatEventToken(
            request_id=rid,
            delta="The average price of villas for sale in Riyadh is ",
        ),
        ChatEventToken(request_id=rid, delta="SAR 1,950,000 "),
        ChatEventToken(request_id=rid, delta="based on 2 listings. "),
        ChatEventToken(
            request_id=rid,
            delta="Prices range from SAR 1,800,000 [1] to SAR 2,100,000 [2].",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            tokens=TokenUsage(prompt_tokens=600, completion_tokens=35),
            finish_reason=FinishReason.stop,
            timings=StageTimings(
                cache_lookup_ms=0.3,
                sql_ms=4.2,
                llm_ttft_ms=580.0,
                llm_total_ms=820.0,
                grounding_ms=0.5,
                total_ms=860.0,
            ),
            grounding=_grounded(),
            flags=AnswerFlags(),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_project(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.project_info,
            language=Language.en,
            filters=SearchFilters(
                project_id="darglobal:project:ocean-breeze",
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[_card_project(n=1), _card_article(n=2)],
        ),
        ChatEventToken(
            request_id=rid,
            delta="Ocean Breeze Residences is a DarGlobal project in Dubai, UAE. ",
        ),
        ChatEventToken(
            request_id=rid,
            delta="Developed in partnership with Trump, prices start from USD 450,000 [1]. ",
        ),
        ChatEventToken(
            request_id=rid,
            delta="Handover is expected in Q4 2027. ",
        ),
        ChatEventToken(
            request_id=rid,
            delta="For recent news, see the DarGlobal expansion announcement [2].",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            tokens=TokenUsage(prompt_tokens=700, completion_tokens=42),
            finish_reason=FinishReason.stop,
            timings=_full_timings(),
            grounding=_grounded(),
            flags=AnswerFlags(),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_arabic(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.ar,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[
                Card(
                    n=1,
                    doc_id="wasalt:789012",
                    kind="listing",
                    source="wasalt",
                    title="فيلا للبيع في النرجس",
                    url="https://wasalt.sa/ar/property/789012",
                    scraped_at=SCRAPED_AT,
                    purpose="sale",
                    property_type="villa",
                    city="Riyadh",
                    district="النرجس",
                    price_sar=1950000,
                    bedrooms=3,
                    bathrooms=4,
                    area_sqm=312.5,
                    score=0.032,
                    snippet="فيلا حديثة بالقرب من طريق الملك سلمان.",
                ),
                Card(
                    n=2,
                    doc_id="wasalt:789013",
                    kind="listing",
                    source="wasalt",
                    title="فيلا للبيع في الملقا",
                    url="https://wasalt.sa/ar/property/789013",
                    scraped_at=SCRAPED_AT,
                    purpose="sale",
                    property_type="villa",
                    city="Riyadh",
                    district="الملقا",
                    price_sar=2200000,
                    bedrooms=4,
                    bathrooms=5,
                    area_sqm=400.0,
                    score=0.028,
                    snippet="فيلا فاخرة في حي الملقا.",
                ),
            ],
        ),
        ChatEventToken(
            request_id=rid,
            delta="يوجد فيلا للبيع في حي النرجس، ",
        ),
        ChatEventToken(
            request_id=rid,
            delta="الرياض بسعر ١٬٩٥٠٬٠٠٠ ر.س [1]. ",
        ),
        ChatEventToken(
            request_id=rid,
            delta="كما توجد فيلا أخرى في حي الملقا ",
        ),
        ChatEventToken(
            request_id=rid,
            delta="بسعر SAR 2,200,000 [2].",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            tokens=TokenUsage(prompt_tokens=850, completion_tokens=55),
            finish_reason=FinishReason.stop,
            timings=_full_timings(),
            grounding=_grounded(),
            flags=AnswerFlags(),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_retrieval_only(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[_card_listing_sale_villa(), _card_listing_sale_apartment()],
        ),
        ChatEventError(
            request_id=rid,
            code=ErrorCode.all_providers_unavailable,
            message="All AI models are currently unavailable. Showing matching sources.",
            degraded_to=DegradedTo.retrieval_only,
            fatal=False,
        ),
        ChatEventToken(
            request_id=rid,
            delta="Here are the most relevant properties matching your query. Please review the source cards above for details.",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.retrieval_only,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(prompt_tokens=0, completion_tokens=0),
            finish_reason=FinishReason.no_llm,
            timings=StageTimings(
                cache_lookup_ms=0.3,
                embed_ms=10.0,
                sparse_ms=1.1,
                retrieve_ms=8.5,
                total_ms=55.0,
            ),
            grounding=_null_grounding(),
            flags=AnswerFlags(),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_busy(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(),
            cache_tier=AnswerCacheTier.miss,
            sources=[],
        ),
        ChatEventError(
            request_id=rid,
            code=ErrorCode.retrieval_failed,
            message="Retrieval is currently unavailable. Please try again shortly.",
            degraded_to=DegradedTo.static_busy,
            fatal=False,
        ),
        ChatEventToken(
            request_id=rid,
            delta="The service is currently experiencing high demand. Please try again in a few moments.",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.static_busy,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(prompt_tokens=0, completion_tokens=0),
            finish_reason=FinishReason.no_llm,
            timings=StageTimings(total_ms=12.0),
            grounding=_null_grounding(),
            flags=AnswerFlags(),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_refusal(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.out_of_scope,
            language=Language.en,
            filters=SearchFilters(),
            cache_tier=AnswerCacheTier.miss,
            sources=[],
        ),
        ChatEventToken(
            request_id=rid,
            delta="I can only help with property listings from Wasalt and projects from DarGlobal. Please ask about those topics.",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.refusal,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.none,
            model=None,
            tokens=TokenUsage(prompt_tokens=0, completion_tokens=0),
            finish_reason=FinishReason.no_llm,
            timings=StageTimings(total_ms=5.0),
            grounding=_null_grounding(),
            flags=AnswerFlags(),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_unverified(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
                bedrooms_min=3,
                bedrooms_max=3,
                price_max_sar=2000000,
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[_card_listing_sale_villa()],
        ),
        ChatEventToken(
            request_id=rid,
            delta="There is one 3-bedroom villa in Al Narjis, Riyadh for ",
        ),
        ChatEventToken(request_id=rid, delta="SAR 1,900,000 [1]. "),
        ChatEventToken(
            request_id=rid,
            delta="This is a great property with 4 bathrooms and 312.5 sqm.",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            tokens=TokenUsage(prompt_tokens=812, completion_tokens=30),
            finish_reason=FinishReason.stop,
            timings=_full_timings(),
            grounding=_ungrounded(),
            flags=AnswerFlags(truncated=False, unverified=True),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_truncated(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[_card_listing_sale_villa()],
        ),
        ChatEventToken(
            request_id=rid,
            delta="There is one 3-bedroom villa in Al Narjis, Riyadh for SAR 1,950,000 [1]. ",
        ),
        ChatEventToken(
            request_id=rid,
            delta="The property features 4 bathrooms, 312.5 sqm of living space, and is located near",
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            tokens=TokenUsage(prompt_tokens=812, completion_tokens=500),
            finish_reason=FinishReason.length,
            timings=_full_timings(),
            grounding=_grounded(),
            flags=AnswerFlags(truncated=True, unverified=False),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_interrupted(rid: str) -> list[ChatEvent]:
    return [
        ChatEventStart(
            request_id=rid,
            route=Route.lookup,
            language=Language.en,
            filters=SearchFilters(
                purpose="sale",
                property_type="villa",
                city="Riyadh",
            ),
            cache_tier=AnswerCacheTier.miss,
            sources=[_card_listing_sale_villa()],
        ),
        ChatEventToken(
            request_id=rid,
            delta="There is one 3-bedroom villa in ",
        ),
        ChatEventToken(request_id=rid, delta="Al Narjis, Riyadh"),
        ChatEventError(
            request_id=rid,
            code=ErrorCode.llm_interrupted,
            message="The model stream was interrupted. Showing partial answer.",
            degraded_to=None,
            fatal=False,
        ),
        ChatEventEnd(
            request_id=rid,
            answer_mode=AnswerMode.llm,
            cache_tier=AnswerCacheTier.miss,
            cache_hits=[],
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            tokens=TokenUsage(prompt_tokens=812, completion_tokens=12),
            finish_reason=FinishReason.error,
            timings=StageTimings(
                cache_lookup_ms=0.4,
                embed_ms=11.2,
                sparse_ms=1.3,
                retrieve_ms=9.8,
                llm_ttft_ms=640.0,
                llm_total_ms=450.0,
                total_ms=500.0,
            ),
            grounding=_grounded(),
            flags=AnswerFlags(truncated=True, unverified=False),
            prompt_version=PROMPT_VERSION,
            index_version=INDEX_VERSION,
        ),
    ]


def _scenario_fatal(rid: str) -> list[ChatEvent]:
    return [
        ChatEventError(
            request_id=rid,
            code=ErrorCode.internal,
            message="An unexpected error occurred. Please try again later.",
            degraded_to=None,
            fatal=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Scenario name extraction
# ---------------------------------------------------------------------------

_SCENARIO_PATTERN = re.compile(r"#(\w+)")
ALL_SCENARIOS = {
    "happy",
    "failover",
    "cache",
    "aggregate",
    "project",
    "arabic",
    "retrieval_only",
    "busy",
    "refusal",
    "unverified",
    "truncated",
    "interrupted",
    "fatal",
    "chunked",
    "rate_limited",
    "not_ready",
}


def extract_scenario(message: str) -> str:
    """Extract #scenario tag from message; default to 'happy'."""
    m = _SCENARIO_PATTERN.search(message)
    if m and m.group(1) in ALL_SCENARIOS:
        return m.group(1)
    return "happy"


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _Delays(NamedTuple):
    before_start: float
    before_first_token: float
    between_tokens_min: float
    between_tokens_max: float


_NORMAL_DELAYS = _Delays(0.15, 0.40, 0.03, 0.08)
_FAST_DELAYS = _Delays(0.0, 0.0, 0.0, 0.0)


def make_handler(delays: _Delays, static_dir: str):
    """Factory: returns a request handler class bound to the given delays and static dir."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # quieter logging
            sys.stderr.write(f"[mock] {fmt % args}\n")

        def _cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")

        def do_OPTIONS(self):
            self.send_response(200)
            self._cors_headers()
            self.end_headers()

        def do_GET(self):
            """Serve static files from app/static/."""
            path = self.path.split("?")[0].split("#")[0]
            if path == "/":
                path = "/index.html"
            # Security: resolve relative paths
            safe = os.path.normpath(path.lstrip("/"))
            if safe.startswith(".."):
                self.send_error(403)
                return
            fpath = os.path.join(static_dir, safe)
            if not os.path.isfile(fpath):
                self.send_error(404)
                return
            ctype = mimetypes.guess_type(fpath)[0] or "application/octet-stream"
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.rstrip("/") != "/api/chat":
                self.send_error(404)
                return

            # Read body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            # Validate with ChatRequest
            try:
                raw = json.loads(body)
                req = ChatRequest.model_validate(raw)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                self.send_response(422)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                detail = str(exc)[:300]
                self.wfile.write(json.dumps({"detail": detail}).encode("utf-8"))
                return

            scenario_name = extract_scenario(req.message)
            result = build_scenario(scenario_name, req.message)

            # Non-200 response (rate_limited, not_ready)
            if isinstance(result, HttpError):
                self.send_response(result.status)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result.body).encode("utf-8"))
                return

            # SSE stream
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header(
                "Set-Cookie", "sid=mock-session-123; HttpOnly; SameSite=Lax; Path=/"
            )
            self._cors_headers()
            self.end_headers()

            is_chunked_scenario = scenario_name == "chunked"
            sent_first_token = False
            ping_inserted = False

            for i, event in enumerate(result):
                sse = event.to_sse()
                frame = f"event: {sse['event']}\ndata: {sse['data']}\n\n"
                frame_bytes = frame.encode("utf-8")

                if is_chunked_scenario:
                    # Insert a single `: ping` comment between the first and second frame
                    if i == 1 and not ping_inserted:
                        try:
                            self.wfile.write(b": ping\n")
                            self.wfile.flush()
                        except BrokenPipeError:
                            return
                        ping_inserted = True

                    # Write in 7-byte pieces
                    offset = 0
                    while offset < len(frame_bytes):
                        chunk = frame_bytes[offset : offset + 7]
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except BrokenPipeError:
                            return
                        offset += 7
                else:
                    try:
                        self.wfile.write(frame_bytes)
                        self.wfile.flush()
                    except BrokenPipeError:
                        return

                # Pacing
                if event.event == "start":
                    time.sleep(delays.before_start)
                elif event.event == "token":
                    if not sent_first_token:
                        time.sleep(delays.before_first_token)
                        sent_first_token = True
                    else:
                        import random

                        time.sleep(
                            random.uniform(
                                delays.between_tokens_min, delays.between_tokens_max
                            )
                        )

    return Handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Mock SSE server for the RAG chatbot")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--fast", action="store_true", help="Disable delays (for tests)"
    )
    args = parser.parse_args()

    static_dir = str(Path(__file__).resolve().parent.parent / "app" / "static")
    delays = _FAST_DELAYS if args.fast else _NORMAL_DELAYS
    handler_cls = make_handler(delays, static_dir)

    server = ThreadingHTTPServer(("", args.port), handler_cls)
    print(f"Mock SSE server on http://localhost:{args.port}  (static: {static_dir})")
    print("Scenarios: include #<name> in message. Available:", ", ".join(sorted(ALL_SCENARIOS)))
    if args.fast:
        print("Fast mode: delays disabled.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    main()
