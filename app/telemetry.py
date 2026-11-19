"""Telemetry: per-request stage timings, ring buffer, /stats payload, JSON log line.

Imports only common.* and stdlib (dependency rule R8 for telemetry layer).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from common.schemas import (
    AnswerCacheTier,
    InnerCacheTier,
    Provider,
    Route,
    StageTimings,
)

logger = logging.getLogger("app.telemetry")

# ---------------------------------------------------------------------------
# RequestRecord: one per finished chat request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestRecord:
    request_id: str
    route: Route
    cache_tier: AnswerCacheTier
    cache_hits: tuple[InnerCacheTier, ...]
    provider: Provider
    model: str | None
    prompt_tokens: int
    completion_tokens: int
    timings: StageTimings
    ts: float  # time.time()


# ---------------------------------------------------------------------------
# RingBuffer
# ---------------------------------------------------------------------------


class RingBuffer:
    """Thread-unsafe ring buffer (event-loop only). Holds up to `maxsize` records."""

    def __init__(self, maxsize: int = 2000) -> None:
        self._buf: deque[RequestRecord] = deque(maxlen=maxsize)

    def push(self, record: RequestRecord) -> None:
        self._buf.append(record)

    def records(self) -> list[RequestRecord]:
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Percentile helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    idx = int(len(values) * pct / 100)
    idx = min(idx, len(values) - 1)
    return sorted(values)[idx]


def _tier_stats(values: list[float], n: int) -> dict[str, Any]:
    """p50/p95 with the 'max (n<20)' rule from concurrency-slo section 2."""
    if not values:
        return {"p50": None, "p95": None, "n": 0}
    result: dict[str, Any] = {
        "p50": round(_percentile(values, 50), 2),
        "n": n,
    }
    if n < 20:
        result["p95_max"] = round(max(values), 2)
    else:
        result["p95"] = round(_percentile(values, 95), 2)
    return result


# ---------------------------------------------------------------------------
# Stats builder
# ---------------------------------------------------------------------------


def build_stats(
    buf: RingBuffer,
    llm_status: Any | None = None,
) -> dict[str, Any]:
    """Build the /stats JSON payload.

    p50/p95 per route and tier with n; 'max (n<20)' rule; cache hit rate per tier
    including L3/L4; provider/model mix; quota remaining.
    """
    records = buf.records()
    if not records:
        return {"n": 0, "routes": {}, "tiers": {}, "providers": {}, "cache_rates": {}}

    # Per-route total_ms
    by_route: dict[str, list[float]] = {}
    for r in records:
        by_route.setdefault(r.route.value, []).append(r.timings.total_ms)

    route_stats = {
        route: _tier_stats(vals, len(vals)) for route, vals in by_route.items()
    }

    # Per cache tier
    by_tier: dict[str, list[float]] = {}
    for r in records:
        by_tier.setdefault(r.cache_tier.value, []).append(r.timings.total_ms)

    tier_stats = {
        tier: _tier_stats(vals, len(vals)) for tier, vals in by_tier.items()
    }

    # Provider/model mix
    provider_counts: dict[str, int] = {}
    for r in records:
        key = f"{r.provider.value}:{r.model or 'none'}"
        provider_counts[key] = provider_counts.get(key, 0) + 1

    # Cache hit rates (L0, L1, L2 from cache_tier; L3, L4 from cache_hits)
    total = len(records)
    l0_hits = sum(1 for r in records if r.cache_tier == AnswerCacheTier.L0)
    l1_hits = sum(1 for r in records if r.cache_tier == AnswerCacheTier.L1)
    l2_hits = sum(1 for r in records if r.cache_tier == AnswerCacheTier.L2)
    l3_hits = sum(1 for r in records if InnerCacheTier.L3 in r.cache_hits)
    l4_hits = sum(1 for r in records if InnerCacheTier.L4 in r.cache_hits)

    cache_rates = {
        "L0": round(l0_hits / total, 4) if total else 0,
        "L1": round(l1_hits / total, 4) if total else 0,
        "L2": round(l2_hits / total, 4) if total else 0,
        "L3": round(l3_hits / total, 4) if total else 0,
        "L4": round(l4_hits / total, 4) if total else 0,
    }

    result: dict[str, Any] = {
        "n": total,
        "routes": route_stats,
        "tiers": tier_stats,
        "providers": provider_counts,
        "cache_rates": cache_rates,
    }

    if llm_status is not None:
        result["llm_status"] = {
            "openrouter_quota_remaining": llm_status.openrouter_quota_remaining,
            "providers": [
                {
                    "provider": ps.provider.value,
                    "model": ps.model,
                    "available": ps.available,
                    "cooldown_remaining_s": ps.cooldown_remaining_s,
                }
                for ps in llm_status.providers
            ],
        }

    return result


# ---------------------------------------------------------------------------
# JSON log
# ---------------------------------------------------------------------------


def log_request(record: RequestRecord) -> None:
    """Emit a structured JSON log line to stdout."""
    entry = {
        "request_id": record.request_id,
        "route": record.route.value,
        "cache_tier": record.cache_tier.value,
        "cache_hits": [h.value for h in record.cache_hits],
        "provider": record.provider.value,
        "model": record.model,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_ms": round(record.timings.total_ms, 2),
    }
    # Stage timings (only non-null)
    for stage in (
        "cache_lookup_ms", "embed_ms", "sparse_ms", "retrieve_ms",
        "rerank_ms", "sql_ms", "llm_ttft_ms", "llm_total_ms", "grounding_ms",
    ):
        val = getattr(record.timings, stage, None)
        if val is not None:
            entry[stage] = round(val, 2)
    logger.info(json.dumps(entry, ensure_ascii=False))
