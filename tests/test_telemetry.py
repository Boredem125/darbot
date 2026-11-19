"""Tests for the telemetry module: ring buffer, stats builder, logging."""

from __future__ import annotations

import time

from app.telemetry import (
    RequestRecord,
    RingBuffer,
    build_stats,
    log_request,
)
from common.schemas import (
    AnswerCacheTier,
    InnerCacheTier,
    Provider,
    Route,
    StageTimings,
)


def _make_record(
    *,
    route: Route = Route.lookup,
    cache_tier: AnswerCacheTier = AnswerCacheTier.miss,
    cache_hits: tuple[InnerCacheTier, ...] = (),
    provider: Provider = Provider.openrouter,
    model: str | None = "google/gemma-4-31b-it:free",
    total_ms: float = 100.0,
) -> RequestRecord:
    return RequestRecord(
        request_id="a" * 32,
        route=route,
        cache_tier=cache_tier,
        cache_hits=cache_hits,
        provider=provider,
        model=model,
        prompt_tokens=100,
        completion_tokens=20,
        timings=StageTimings(total_ms=total_ms),
        ts=time.time(),
    )


class TestRingBuffer:
    def test_push_and_len(self):
        buf = RingBuffer(maxsize=5)
        for i in range(3):
            buf.push(_make_record(total_ms=float(i)))
        assert len(buf) == 3

    def test_overflow(self):
        buf = RingBuffer(maxsize=3)
        for i in range(5):
            buf.push(_make_record(total_ms=float(i)))
        assert len(buf) == 3
        # Oldest should have been evicted
        records = buf.records()
        assert records[0].timings.total_ms == 2.0

    def test_records_returns_list(self):
        buf = RingBuffer()
        buf.push(_make_record())
        assert isinstance(buf.records(), list)


class TestBuildStats:
    def test_empty_buffer(self):
        buf = RingBuffer()
        stats = build_stats(buf)
        assert stats["n"] == 0

    def test_basic_stats(self):
        buf = RingBuffer()
        for i in range(5):
            buf.push(_make_record(total_ms=10.0 * (i + 1)))
        stats = build_stats(buf)
        assert stats["n"] == 5
        assert "lookup" in stats["routes"]
        assert "miss" in stats["tiers"]
        assert stats["cache_rates"]["L0"] == 0
        assert stats["cache_rates"]["L3"] == 0

    def test_cache_rates(self):
        buf = RingBuffer()
        buf.push(_make_record(cache_tier=AnswerCacheTier.L1))
        buf.push(_make_record(cache_tier=AnswerCacheTier.miss,
                              cache_hits=(InnerCacheTier.L3,)))
        stats = build_stats(buf)
        assert stats["cache_rates"]["L1"] == 0.5
        assert stats["cache_rates"]["L3"] == 0.5

    def test_max_n_less_than_20_rule(self):
        """When n < 20, stats should use p95_max instead of p95."""
        buf = RingBuffer()
        for i in range(10):
            buf.push(_make_record(total_ms=float(i)))
        stats = build_stats(buf)
        route_stats = stats["routes"]["lookup"]
        assert "p95_max" in route_stats  # n < 20, so max reported
        assert "p95" not in route_stats

    def test_p95_when_n_ge_20(self):
        buf = RingBuffer()
        for i in range(25):
            buf.push(_make_record(total_ms=float(i)))
        stats = build_stats(buf)
        route_stats = stats["routes"]["lookup"]
        assert "p95" in route_stats
        assert "p95_max" not in route_stats


class TestLogRequest:
    def test_log_does_not_raise(self):
        """log_request should not raise."""
        record = _make_record()
        log_request(record)  # Should not raise
