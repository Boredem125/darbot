"""Unit tests for eval metrics with hand-computed values."""

from __future__ import annotations

from scripts.evallib.metrics import (
    MRRResult,
    aggregate_exact,
    canary_leak,
    constraint_precision_at_k,
    entity_hit_at_k,
    entity_mrr,
    false_refusal,
    filter_accuracy,
    metric,
    numeric_grounding,
    refusal_accuracy,
    route_accuracy,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_of_zero(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_zero_of_n(self):
        lo, hi = wilson_ci(0, 10)
        assert lo == 0.0
        assert hi > 0.0
        # With z=1.96, 0/10 upper should be about 0.308
        assert 0.25 < hi < 0.35

    def test_n_of_n(self):
        lo, hi = wilson_ci(10, 10)
        assert hi == 1.0
        assert lo > 0.0
        # With z=1.96, 10/10 lower should be about 0.692
        assert 0.60 < lo < 0.80

    def test_eight_of_ten(self):
        lo, hi = wilson_ci(8, 10)
        # p_hat = 0.8
        # Manual: z=1.96, z2=3.8416
        # denom = 1 + 3.8416/10 = 1.38416
        # centre = (0.8 + 3.8416/20) / 1.38416 = (0.8 + 0.19208) / 1.38416 = 0.71717
        # spread = (1.96 / 1.38416) * sqrt(0.8*0.2/10 + 3.8416/400)
        #        = 1.4160 * sqrt(0.016 + 0.009604) = 1.4160 * sqrt(0.025604) = 1.4160 * 0.16001
        #        = 0.22658
        # lo = 0.71717 - 0.22658 = 0.4906
        # hi = 0.71717 + 0.22658 = 0.9438
        assert 0.44 < lo < 0.52
        assert 0.93 < hi < 0.98

    def test_symmetry(self):
        """wilson_ci(k, n) and wilson_ci(n-k, n) have mirrored bounds."""
        lo1, hi1 = wilson_ci(3, 10)
        lo2, hi2 = wilson_ci(7, 10)
        assert abs((1 - hi2) - lo1) < 0.001
        assert abs((1 - lo2) - hi1) < 0.001

    def test_monotonic(self):
        """More successes -> higher lower bound."""
        _, _ = wilson_ci(1, 10)
        lo2, _ = wilson_ci(5, 10)
        lo3, _ = wilson_ci(9, 10)
        assert lo2 > wilson_ci(1, 10)[0]
        assert lo3 > lo2


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


class TestMetricResult:
    def test_render(self):
        m = metric(8, 10)
        s = m.render()
        assert "8/10" in s
        assert "80.0%" in s
        assert "CI" in s

    def test_render_zero(self):
        m = metric(0, 0)
        assert "no cases" in m.render()

    def test_rate(self):
        m = metric(3, 10)
        assert abs(m.rate - 0.3) < 0.001


# ---------------------------------------------------------------------------
# Route accuracy
# ---------------------------------------------------------------------------


class TestRouteAccuracy:
    def test_all_correct(self):
        cases = [
            {"expect": {"route": "lookup"}, "predicted": {"route": "lookup"}},
            {"expect": {"route": "aggregate"}, "predicted": {"route": "aggregate"}},
        ]
        m = route_accuracy(cases)
        assert m.successes == 2
        assert m.total == 2

    def test_one_wrong(self):
        cases = [
            {"expect": {"route": "lookup"}, "predicted": {"route": "lookup"}},
            {"expect": {"route": "aggregate"}, "predicted": {"route": "lookup"}},
        ]
        m = route_accuracy(cases)
        assert m.successes == 1
        assert m.total == 2

    def test_no_route_expected(self):
        cases = [{"expect": {}, "predicted": {"route": "lookup"}}]
        m = route_accuracy(cases)
        assert m.total == 0


# ---------------------------------------------------------------------------
# Filter accuracy
# ---------------------------------------------------------------------------


class TestFilterAccuracy:
    def test_subset_match(self):
        cases = [
            {
                "expect": {"filters": {"city": "Riyadh", "purpose": "sale"}},
                "predicted": {"filters": {"city": "Riyadh", "purpose": "sale", "property_type": "villa"}},
            }
        ]
        m = filter_accuracy(cases)
        assert m.successes == 1

    def test_missing_field(self):
        cases = [
            {
                "expect": {"filters": {"city": "Riyadh", "purpose": "sale"}},
                "predicted": {"filters": {"city": "Riyadh"}},
            }
        ]
        m = filter_accuracy(cases)
        assert m.successes == 0
        assert m.total == 1


# ---------------------------------------------------------------------------
# Constraint precision@5
# ---------------------------------------------------------------------------


class TestConstraintPrecision:
    def test_all_match(self):
        cases = [
            {
                "expect": {"filters": {"city": "Riyadh"}},
                "predicted": {
                    "cards": [
                        {"city": "Riyadh"},
                        {"city": "Riyadh"},
                    ]
                },
            }
        ]
        m = constraint_precision_at_k(cases, k=5)
        assert m.successes == 2
        assert m.total == 2

    def test_partial_match(self):
        cases = [
            {
                "expect": {"filters": {"city": "Riyadh"}},
                "predicted": {
                    "cards": [
                        {"city": "Riyadh"},
                        {"city": "Jeddah"},
                    ]
                },
            }
        ]
        m = constraint_precision_at_k(cases, k=5)
        assert m.successes == 1
        assert m.total == 2


# ---------------------------------------------------------------------------
# Entity hit@5 and MRR
# ---------------------------------------------------------------------------


class TestEntityHit:
    def test_hit(self):
        cases = [
            {
                "expect": {"entity_ids": ["wasalt:123"]},
                "predicted": {"cards": [{"doc_id": "wasalt:123"}]},
            }
        ]
        m = entity_hit_at_k(cases, k=5)
        assert m.successes == 1

    def test_miss(self):
        cases = [
            {
                "expect": {"entity_ids": ["wasalt:123"]},
                "predicted": {"cards": [{"doc_id": "wasalt:999"}]},
            }
        ]
        m = entity_hit_at_k(cases, k=5)
        assert m.successes == 0
        assert m.total == 1

    def test_empty_entity_ids_skipped(self):
        cases = [
            {
                "expect": {"entity_ids": []},
                "predicted": {"cards": []},
            }
        ]
        m = entity_hit_at_k(cases, k=5)
        assert m.total == 0


class TestMRR:
    def test_returns_mrr_result(self):
        cases = [
            {
                "expect": {"entity_ids": ["wasalt:1"]},
                "predicted": {"cards": [{"doc_id": "wasalt:1"}]},
            }
        ]
        m = entity_mrr(cases, k=5)
        assert isinstance(m, MRRResult)

    def test_first_position(self):
        cases = [
            {
                "expect": {"entity_ids": ["wasalt:1"]},
                "predicted": {"cards": [{"doc_id": "wasalt:1"}, {"doc_id": "wasalt:2"}]},
            }
        ]
        m = entity_mrr(cases, k=5)
        assert abs(m.value - 1.0) < 1e-9  # rank 1 -> RR = 1.0
        assert m.total == 1

    def test_second_position(self):
        cases = [
            {
                "expect": {"entity_ids": ["wasalt:1"]},
                "predicted": {"cards": [{"doc_id": "wasalt:2"}, {"doc_id": "wasalt:1"}]},
            }
        ]
        m = entity_mrr(cases, k=5)
        assert abs(m.value - 0.5) < 1e-9  # rank 2 -> RR = 0.5

    def test_hand_computed_ranks_1_2_miss(self):
        """Ranks 1, 2, miss -> RR = 1 + 0.5 + 0 = 1.5, MRR = 1.5/3 = 0.5."""
        cases = [
            {
                "expect": {"entity_ids": ["wasalt:1"]},
                "predicted": {"cards": [{"doc_id": "wasalt:1"}, {"doc_id": "wasalt:2"}]},
            },
            {
                "expect": {"entity_ids": ["wasalt:3"]},
                "predicted": {"cards": [{"doc_id": "wasalt:2"}, {"doc_id": "wasalt:3"}]},
            },
            {
                "expect": {"entity_ids": ["wasalt:5"]},
                "predicted": {"cards": [{"doc_id": "wasalt:7"}, {"doc_id": "wasalt:8"}]},
            },
        ]
        m = entity_mrr(cases, k=5)
        assert m.total == 3
        assert abs(m.value - 0.5) < 1e-9  # (1 + 0.5 + 0) / 3 = 0.5

    def test_empty(self):
        m = entity_mrr([], k=5)
        assert m.total == 0
        assert m.value == 0.0

    def test_render(self):
        cases = [
            {
                "expect": {"entity_ids": ["wasalt:1"]},
                "predicted": {"cards": [{"doc_id": "wasalt:1"}]},
            }
        ]
        m = entity_mrr(cases, k=5)
        assert "MRR=" in m.render()
        assert "n=1" in m.render()


# ---------------------------------------------------------------------------
# Aggregate exact
# ---------------------------------------------------------------------------


class TestAggregateExact:
    def test_exact_count(self):
        cases = [
            {
                "expect": {"op": "count", "sql_truth": 42, "tolerance_pct": 0.5},
                "predicted": {"sql_value": 42},
            }
        ]
        m = aggregate_exact(cases)
        assert m.successes == 1

    def test_count_mismatch(self):
        cases = [
            {
                "expect": {"op": "count", "sql_truth": 42, "tolerance_pct": 0.5},
                "predicted": {"sql_value": 43},
            }
        ]
        m = aggregate_exact(cases)
        assert m.successes == 0

    def test_avg_within_tolerance(self):
        cases = [
            {
                "expect": {"op": "avg_price", "sql_truth": 1000000, "tolerance_pct": 0.5},
                "predicted": {"sql_value": 1004000},  # 0.4% off
            }
        ]
        m = aggregate_exact(cases)
        assert m.successes == 1

    def test_avg_outside_tolerance(self):
        cases = [
            {
                "expect": {"op": "avg_price", "sql_truth": 1000000, "tolerance_pct": 0.5},
                "predicted": {"sql_value": 1006000},  # 0.6% off
            }
        ]
        m = aggregate_exact(cases)
        assert m.successes == 0


# ---------------------------------------------------------------------------
# Refusal / false refusal
# ---------------------------------------------------------------------------


class TestRefusal:
    def test_correct_refusal(self):
        cases = [
            {"expect": {"refuse": True}, "predicted": {"refused": True}},
        ]
        m = refusal_accuracy(cases)
        assert m.successes == 1
        assert m.total == 1

    def test_missed_refusal(self):
        cases = [
            {"expect": {"refuse": True}, "predicted": {"refused": False}},
        ]
        m = refusal_accuracy(cases)
        assert m.successes == 0

    def test_false_refusal(self):
        cases = [
            {"expect": {"refuse": False}, "predicted": {"refused": True}},
            {"expect": {"refuse": False}, "predicted": {"refused": False}},
        ]
        m = false_refusal(cases)
        assert m.successes == 1  # 1 false refusal
        assert m.total == 2


# ---------------------------------------------------------------------------
# Canary leak
# ---------------------------------------------------------------------------


class TestCanaryLeak:
    def test_no_leak(self):
        cases = [
            {"expect": {"canary": "CANARY_TOKEN"}, "predicted": {"answer": "Safe answer"}},
        ]
        m = canary_leak(cases)
        assert m.successes == 0

    def test_leak(self):
        cases = [
            {"expect": {"canary": "CANARY_TOKEN"}, "predicted": {"answer": "Look CANARY_TOKEN here"}},
        ]
        m = canary_leak(cases)
        assert m.successes == 1


# ---------------------------------------------------------------------------
# Numeric grounding
# ---------------------------------------------------------------------------


class TestNumericGrounding:
    def test_numbers_present(self):
        cases = [
            {
                "llm": True,
                "expect": {"answer_numbers": ["1950000"]},
                "predicted": {"answer": "The price is 1950000 SAR"},
            }
        ]
        m = numeric_grounding(cases)
        assert m.successes == 1

    def test_non_llm_skipped(self):
        cases = [
            {
                "llm": False,
                "expect": {"answer_numbers": ["1950000"]},
                "predicted": {"answer": "No number here"},
            }
        ]
        m = numeric_grounding(cases)
        assert m.total == 0

    def test_h1_formatted_number_matches(self):
        """H-1: 'SAR 249,000' must match answer_numbers=['249000'].

        The old metric did str(n) in answer, which failed because '249000'
        is not literally in 'SAR 249,000'. The fix uses canonical_number +
        parse_numbers so that the comma-formatted form matches.
        """
        cases = [
            {
                "llm": True,
                "expect": {"answer_numbers": ["249000"]},
                "predicted": {"answer": "The price is SAR 249,000 [1]."},
            }
        ]
        m = numeric_grounding(cases)
        assert m.successes == 1
        assert m.total == 1

    def test_h1_magnitude_number_matches(self):
        """H-1: '1.95M' in the answer must match answer_numbers=['1950000']."""
        cases = [
            {
                "llm": True,
                "expect": {"answer_numbers": ["1950000"]},
                "predicted": {"answer": "The price is 1.95M SAR [1]."},
            }
        ]
        m = numeric_grounding(cases)
        assert m.successes == 1
        assert m.total == 1

    def test_h1_uses_grounding_numbers_ok(self):
        """H-1: When grounding report has numbers_ok, use that as the check."""
        cases = [
            {
                "llm": True,
                "expect": {"answer_numbers": ["249000"]},
                "predicted": {
                    "answer": "SAR 249,000 [1].",
                    "grounding": {"numbers_ok": True, "entities_ok": True, "citations_ok": True},
                    "llm_runs": [{"answer": "SAR 249,000 [1].", "skipped_budget": False, "llm_unavailable": False}],
                },
            }
        ]
        m = numeric_grounding(cases)
        assert m.successes == 1
