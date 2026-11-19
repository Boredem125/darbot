"""Eval metrics: Wilson CI, precision@k, hit@k, MRR, aggregate exact, refusal.

All metric functions return (successes, total, lower_ci, upper_ci).
Wilson CI uses 95% confidence (z = 1.96).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricResult:
    """A metric value reported as x/n with a Wilson 95% CI."""
    successes: int
    total: int
    lower: float
    upper: float

    @property
    def rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successes / self.total

    def render(self) -> str:
        if self.total == 0:
            return "0/0 (no cases)"
        return f"{self.successes}/{self.total} ({self.rate:.1%}) CI [{self.lower:.3f}, {self.upper:.3f}]"


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (lower, upper). Handles edge cases:
    - 0/n -> (0.0, upper)
    - n/n -> (lower, 1.0)
    - 0/0 -> (0.0, 0.0)
    """
    if total == 0:
        return (0.0, 0.0)

    p_hat = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    centre = (p_hat + z2 / (2 * total)) / denom
    spread = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / total + z2 / (4 * total * total))

    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return (lower, upper)


def metric(successes: int, total: int) -> MetricResult:
    lo, hi = wilson_ci(successes, total)
    return MetricResult(successes=successes, total=total, lower=lo, upper=hi)


# ---------------------------------------------------------------------------
# Domain metrics
# ---------------------------------------------------------------------------


def route_accuracy(cases: list[dict]) -> MetricResult:
    """Fraction of cases where predicted route matches expected route."""
    ok = 0
    total = 0
    for c in cases:
        exp = c.get("expect", {})
        pred = c.get("predicted", {})
        if "route" not in exp:
            continue
        total += 1
        if pred.get("route") == exp["route"]:
            ok += 1
    return metric(ok, total)


def filter_accuracy(cases: list[dict]) -> MetricResult:
    """Subset match: every expected filter field matches the predicted value."""
    ok = 0
    total = 0
    for c in cases:
        exp_filters = c.get("expect", {}).get("filters")
        if not exp_filters:
            continue
        pred_filters = c.get("predicted", {}).get("filters", {})
        total += 1
        match = True
        for key, val in exp_filters.items():
            if pred_filters.get(key) != val:
                match = False
                break
        if match:
            ok += 1
    return metric(ok, total)


def constraint_precision_at_k(cases: list[dict], k: int = 5) -> MetricResult:
    """For filter queries: fraction of top-k results satisfying all filters.

    Only scored on cases that have expect.filters with at least one field.
    """
    ok = 0
    total = 0
    for c in cases:
        exp_filters = c.get("expect", {}).get("filters")
        if not exp_filters:
            continue
        cards = c.get("predicted", {}).get("cards", [])[:k]
        for card in cards:
            total += 1
            if _card_matches_filters(card, exp_filters):
                ok += 1
    return metric(ok, total)


def _card_matches_filters(card: dict, filters: dict) -> bool:
    """Check if a card satisfies all filter constraints."""
    for key, val in filters.items():
        if val is None:
            continue
        if key == "purpose" and card.get("purpose") != val:
            return False
        if key == "property_type" and card.get("property_type") != val:
            return False
        if key == "city" and card.get("city") != val:
            return False
        if key == "district" and card.get("district") != val:
            return False
        if key == "country" and card.get("country") != val:
            return False
        if key == "bedrooms_min" and card.get("bedrooms") is not None and card["bedrooms"] < val:
            return False
        if key == "bedrooms_max" and card.get("bedrooms") is not None and card["bedrooms"] > val:
            return False
        if key == "price_max_sar":
            price = card.get("annual_rent_sar") if card.get("purpose") == "rent" else card.get("price_sar")
            if price is not None and price > val:
                return False
        if key == "price_min_sar":
            price = card.get("annual_rent_sar") if card.get("purpose") == "rent" else card.get("price_sar")
            if price is not None and price < val:
                return False
    return True


def entity_hit_at_k(cases: list[dict], k: int = 5) -> MetricResult:
    """Any expected entity_id in top-k cards is a hit."""
    ok = 0
    total = 0
    for c in cases:
        entity_ids = c.get("expect", {}).get("entity_ids", [])
        if not entity_ids:
            continue
        total += 1
        cards = c.get("predicted", {}).get("cards", [])[:k]
        card_ids = {cd.get("doc_id") for cd in cards}
        if card_ids & set(entity_ids):
            ok += 1
    return metric(ok, total)


@dataclass(frozen=True)
class MRRResult:
    """MRR reported as a float with n (not x/n with Wilson CI)."""

    value: float  # mean reciprocal rank in [0, 1]
    total: int  # number of scored cases

    @property
    def rate(self) -> float:
        return self.value

    def render(self) -> str:
        if self.total == 0:
            return "0/0 (no cases)"
        return f"MRR={self.value:.4f} (n={self.total})"


def entity_mrr(cases: list[dict], k: int = 5) -> MRRResult:
    """Mean Reciprocal Rank for entity queries.

    Returns the mean of 1/rank across all scored cases (0 when no expected
    entity_id appears in the top k). Reported as a float with n, not x/n.
    """
    total = 0
    rr_sum = 0.0
    for c in cases:
        entity_ids = c.get("expect", {}).get("entity_ids", [])
        if not entity_ids:
            continue
        total += 1
        cards = c.get("predicted", {}).get("cards", [])[:k]
        for i, cd in enumerate(cards):
            if cd.get("doc_id") in entity_ids:
                rr_sum += 1.0 / (i + 1)
                break
        # If not found, RR contribution is 0
    if total == 0:
        return MRRResult(value=0.0, total=0)
    return MRRResult(value=rr_sum / total, total=total)


def aggregate_exact(cases: list[dict]) -> MetricResult:
    """SQL aggregate exact match (tolerance for avg/median only)."""
    ok = 0
    total = 0
    for c in cases:
        exp = c.get("expect", {})
        if exp.get("op") is None:
            continue
        total += 1
        pred = c.get("predicted", {})
        exp_val = exp.get("sql_truth")
        pred_val = pred.get("sql_value")
        tol = exp.get("tolerance_pct", 0.5)
        op = exp.get("op", "")

        if exp_val is None and pred_val is None:
            ok += 1
            continue
        if exp_val is None or pred_val is None:
            continue

        # Exact for count, min_price, max_price; tolerance for avg and median
        if op in ("avg_price", "median_price", "avg_area"):
            if exp_val == 0:
                if pred_val == 0:
                    ok += 1
            elif abs(pred_val - exp_val) / abs(exp_val) <= tol / 100.0:
                ok += 1
        else:
            if pred_val == exp_val:
                ok += 1
    return metric(ok, total)


def refusal_accuracy(cases: list[dict]) -> MetricResult:
    """Out-of-scope cases that were correctly refused."""
    ok = 0
    total = 0
    for c in cases:
        exp = c.get("expect", {})
        if not exp.get("refuse"):
            continue
        total += 1
        pred = c.get("predicted", {})
        if pred.get("refused"):
            ok += 1
    return metric(ok, total)


def false_refusal(cases: list[dict]) -> MetricResult:
    """Cases that should NOT be refused but were refused (false refusal)."""
    falsely_refused = 0
    total = 0
    for c in cases:
        exp = c.get("expect", {})
        if exp.get("refuse"):
            continue  # out-of-scope cases excluded
        total += 1
        pred = c.get("predicted", {})
        if pred.get("refused"):
            falsely_refused += 1
    return metric(falsely_refused, total)


def canary_leak(cases: list[dict]) -> MetricResult:
    """Cases where the canary token appeared in the answer. 0 is passing."""
    leaked = 0
    total = 0
    for c in cases:
        canary = c.get("expect", {}).get("canary")
        if canary is None:
            continue
        total += 1
        answer = c.get("predicted", {}).get("answer", "")
        if canary in answer:
            leaked += 1
    return metric(leaked, total)


def numeric_grounding(cases: list[dict]) -> MetricResult:
    """LLM cases: grounding report numbers_ok, falling back to canonical
    number matching on answer_numbers.

    Uses ``canonical_number(parse_numbers(answer))`` exactly as the service's
    ``app.domain.grounding`` does, instead of the old ``str(n) in answer``
    which failed on comma-formatted numbers like "SAR 249,000" (H-1).

    Excludes budget-skipped and LLM-unavailable cases.
    """
    from common.text import canonical_number, parse_numbers
    from scripts.evallib.llm_step import llm_case_scored

    ok = 0
    total = 0
    for c in cases:
        exp = c.get("expect", {})
        answer_nums = exp.get("answer_numbers", [])
        if not answer_nums:
            continue
        if not c.get("llm"):
            continue
        if not llm_case_scored(c):
            continue
        total += 1

        # Prefer the grounding report's numbers_ok when available (H-1)
        grounding = c.get("predicted", {}).get("grounding")
        if grounding is not None and grounding.get("numbers_ok") is not None:
            if grounding["numbers_ok"]:
                ok += 1
            continue

        # Fallback: canonical number matching on the answer text
        answer = c.get("predicted", {}).get("answer", "")
        answer_canons = {canonical_number(n) for n in parse_numbers(answer)}
        all_found = all(str(num) in answer_canons for num in answer_nums)
        if all_found:
            ok += 1
    return metric(ok, total)


def entity_grounding(cases: list[dict]) -> MetricResult:
    """LLM cases: grounding report says entities_ok.

    Excludes budget-skipped and LLM-unavailable cases.
    """
    from scripts.evallib.llm_step import llm_case_scored

    ok = 0
    total = 0
    for c in cases:
        if not c.get("llm"):
            continue
        if not llm_case_scored(c):
            continue
        grounding = c.get("predicted", {}).get("grounding")
        if grounding is None:
            continue
        total += 1
        if grounding.get("entities_ok") is not False:
            ok += 1
    return metric(ok, total)
