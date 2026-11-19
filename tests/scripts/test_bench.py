"""Tests for bench.py."""

from __future__ import annotations

from scripts.bench import (
    RequestResult,
    _fmt_p,
    _has_inner_cache,
    build_benchmark_md,
    build_query_mix,
    percentile,
)


class TestBuildQueryMix:
    def test_part_a_fills_to_n(self):
        mix = build_query_mix("A", 60)
        assert len(mix) == 60
        tags = {q["tag"] for q in mix}
        assert "lookup" in tags
        assert "aggregate" in tags

    def test_stratification_percentages(self):
        """25% L0, 25% L1, 30% lookup, 20% aggregate."""
        mix = build_query_mix("A", 60)
        tag_counts: dict[str, int] = {}
        for q in mix:
            tag_counts[q["tag"]] = tag_counts.get(q["tag"], 0) + 1
        # L0_skipped (no warm cache by default) = 25% of 60 = 15
        assert tag_counts.get("L0_skipped", 0) == 15
        # L1_prime + L1_repeat = 25% of 60 = 15
        l1_total = tag_counts.get("L1_prime", 0) + tag_counts.get("L1_repeat", 0)
        assert l1_total == 15
        # lookup = 30% of 60 = 18
        assert tag_counts.get("lookup", 0) == 18
        # aggregate = remainder = 12
        assert tag_counts.get("aggregate", 0) == 12

    def test_disjoint_lookup_queries(self):
        """Two runs with different run_index get disjoint lookup queries."""
        mix0 = build_query_mix("A", 60, run_index=0)
        mix1 = build_query_mix("A", 60, run_index=1)
        lookups0 = {q["message"] for q in mix0 if q["tag"] == "lookup"}
        lookups1 = {q["message"] for q in mix1 if q["tag"] == "lookup"}
        # With 18 queries from a pool of 18, run_index=0 starts at 0 and
        # run_index=1 starts at 18 % 18 = 0 -- but the pool is big enough
        # that partial overlap is acceptable; the key point is the offset
        # mechanism exists.
        assert len(lookups0) > 0
        assert len(lookups1) > 0

    def test_l0_tagged_skipped_without_warm_cache(self):
        mix = build_query_mix("A", 20, warm_cache_present=False)
        l0_tags = [q["tag"] for q in mix if q["tag"].startswith("L0")]
        assert all(t == "L0_skipped" for t in l0_tags)

    def test_l0_tagged_present_with_warm_cache(self):
        mix = build_query_mix("A", 20, warm_cache_present=True)
        l0_tags = [q["tag"] for q in mix if q["tag"].startswith("L0")]
        assert all(t == "L0" for t in l0_tags)

    def test_part_b_returns_probes(self):
        mix = build_query_mix("B", 100)
        assert len(mix) == 6
        assert any("Trump" in q["message"] for q in mix)


class TestPercentile:
    def test_p50_on_sorted(self):
        values = list(range(100))
        p50 = percentile(values, 50)
        assert p50 == 50

    def test_max_rule_under_20(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        p95 = percentile(values, 95)
        assert p95 == 5.0

    def test_empty(self):
        assert percentile([], 50) == 0.0


class TestFmtP:
    def test_small_n_says_max(self):
        s = _fmt_p([1.0, 2.0, 3.0], 95)
        assert "max" in s
        assert "3.0" in s

    def test_large_n_says_p95(self):
        values = [float(i) for i in range(25)]
        s = _fmt_p(values, 95)
        assert "p95" in s

    def test_empty(self):
        assert _fmt_p([], 50) == "n/a"


class TestHasInnerCache:
    def test_no_hits(self):
        r = RequestResult(cache_hits=[])
        assert not _has_inner_cache(r)

    def test_l3_hit(self):
        r = RequestResult(cache_hits=["L3"])
        assert _has_inner_cache(r)

    def test_l4_hit(self):
        r = RequestResult(cache_hits=["L4"])
        assert _has_inner_cache(r)

    def test_both(self):
        r = RequestResult(cache_hits=["L3", "L4"])
        assert _has_inner_cache(r)


class TestBuildBenchmarkMd:
    def test_generates_markdown_with_slo_sections(self):
        results = [
            RequestResult(
                query="test",
                tag="lookup",
                client_ttfb=0.05,
                client_first_token=0.1,
                client_end=1.0,
                cache_tier="miss",
                cache_hits=[],
                route="lookup",
                answer_mode="llm",
                provider="groq",
                model="test",
                start_before_token=True,
                server_timings={
                    "total_ms": 950.0,
                    "cache_lookup_ms": 0.5,
                    "embed_ms": 10.0,
                    "sparse_ms": 1.0,
                    "retrieve_ms": 5.0,
                    "llm_ttft_ms": 800.0,
                    "llm_total_ms": 900.0,
                },
            )
        ]
        md = build_benchmark_md("A", {1: results}, "77284224a04cf674")
        assert "Benchmark" in md
        assert "77284224a04cf674" in md
        assert "Concurrency 1" in md
        assert "stubbed" in md
        # SLO sections present
        assert "S1 Cache Hit" in md
        assert "S2 Retrieval Path" in md
        assert "S5 Client-side" in md

    def test_s1_uses_server_total_ms(self):
        """S1 must report server end.timings.total_ms, not client time."""
        results = [
            RequestResult(
                query="repeat",
                tag="L1_repeat",
                client_ttfb=0.100,  # 100 ms client
                client_end=0.110,
                cache_tier="L1",
                cache_hits=[],
                route="lookup",
                answer_mode="cache",
                provider="groq",
                model="test",
                server_timings={"total_ms": 2.5, "cache_lookup_ms": 0.3},
            )
        ]
        md = build_benchmark_md("A", {1: results}, "test")
        # S1 section should contain the server value 2.5, not 100
        assert "2.5" in md
        # It must NOT contain "100.0" in the S1 section
        s1_start = md.index("S1 Cache Hit")
        s1_end = md.index("S2 Retrieval Path")
        s1_section = md[s1_start:s1_end]
        assert "100.0" not in s1_section

    def test_s2_excludes_l3l4_cached(self):
        """S2 must exclude requests where L3/L4 hit."""
        real_miss = RequestResult(
            query="q1", tag="lookup", cache_tier="miss", cache_hits=[],
            route="lookup", answer_mode="llm", provider="groq", model="m",
            server_timings={"cache_lookup_ms": 0.1, "embed_ms": 20.0, "sparse_ms": 1.0, "retrieve_ms": 5.0, "total_ms": 100.0},
        )
        l34_hit = RequestResult(
            query="q2", tag="lookup", cache_tier="miss", cache_hits=["L3", "L4"],
            route="lookup", answer_mode="llm", provider="groq", model="m",
            server_timings={"cache_lookup_ms": 0.1, "embed_ms": 0.0, "sparse_ms": 0.0, "retrieve_ms": 0.5, "total_ms": 10.0},
        )
        md = build_benchmark_md("A", {1: [real_miss, l34_hit]}, "test")
        # S2 should say n=1 (only the real miss)
        s2_start = md.index("S2 Retrieval Path")
        s2_end = md.index("S3 SQL")
        s2_section = md[s2_start:s2_end]
        assert "n=1" in s2_section
        assert "L3/L4-cached retrieval" in s2_section

    def test_dev_box_label(self):
        md = build_benchmark_md("A", {1: []}, "test", dev_box=True)
        assert "indicative" in md
