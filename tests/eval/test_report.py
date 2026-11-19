"""Tests for compare_with_baseline in report.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.evallib.report import compare_with_baseline


def _report(metrics: dict) -> dict:
    return {"metrics": metrics}


def _write_baseline(tmp_path: Path, metrics: dict) -> Path:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({"metrics": metrics}), encoding="utf-8")
    return p


class TestCompareWithBaseline:
    def test_equal_to_baseline_no_regression(self, tmp_path: Path):
        """Equal to baseline must exit 0 (no regressions), even if below
        the absolute gate."""
        metrics = {
            "entity_hit_at_5": {"rate": 0.706, "successes": 12, "total": 17},
            "refusal_accuracy": {"rate": 0.857, "successes": 6, "total": 7},
            "false_refusal": {"rate": 0.017, "successes": 1, "total": 59},
        }
        bp = _write_baseline(tmp_path, metrics)
        gates = {
            "entity_hit_at_5": 0.80,
            "refusal_accuracy": 1.00,
            "false_refusal": 0.05,
        }
        regressions = compare_with_baseline(_report(metrics), bp, gates)
        assert regressions == []

    def test_metric_worse_than_baseline_is_regression(self, tmp_path: Path):
        """A metric that drops below the baseline by more than tolerance
        is a regression."""
        baseline = {"entity_hit_at_5": {"rate": 0.80, "successes": 16, "total": 20}}
        current = {"entity_hit_at_5": {"rate": 0.60, "successes": 12, "total": 20}}
        bp = _write_baseline(tmp_path, baseline)
        regressions = compare_with_baseline(_report(current), bp, {"entity_hit_at_5": 0.80})
        assert len(regressions) == 1
        assert "entity_hit_at_5" in regressions[0]

    def test_false_refusal_improvement_no_regression(self, tmp_path: Path):
        """false_refusal is lower-is-better: going from 0.05 to 0.02 is an
        improvement, not a regression."""
        baseline = {"false_refusal": {"rate": 0.05, "successes": 3, "total": 60}}
        current = {"false_refusal": {"rate": 0.02, "successes": 1, "total": 60}}
        bp = _write_baseline(tmp_path, baseline)
        regressions = compare_with_baseline(_report(current), bp, {"false_refusal": 0.05})
        assert regressions == []

    def test_false_refusal_worsening_is_regression(self, tmp_path: Path):
        """false_refusal going from 0.02 to 0.08 is a regression
        (worse than baseline)."""
        baseline = {"false_refusal": {"rate": 0.02, "successes": 1, "total": 60}}
        current = {"false_refusal": {"rate": 0.08, "successes": 5, "total": 60}}
        bp = _write_baseline(tmp_path, baseline)
        regressions = compare_with_baseline(_report(current), bp, {"false_refusal": 0.05})
        assert len(regressions) == 1

    def test_canary_leak_zero_stays_zero(self, tmp_path: Path):
        """canary_leak is lower-is-better: 0 == 0 is not a regression."""
        baseline = {"canary_leak": {"rate": 0.0, "successes": 0, "total": 3}}
        current = {"canary_leak": {"rate": 0.0, "successes": 0, "total": 3}}
        bp = _write_baseline(tmp_path, baseline)
        regressions = compare_with_baseline(_report(current), bp, {"canary_leak": 0.0})
        assert regressions == []

    def test_non_gated_metric_drop_is_regression(self, tmp_path: Path):
        """A metric with no gate that drops from baseline is still flagged,
        but only for higher-is-better metrics."""
        baseline = {"route_accuracy": {"rate": 0.95, "successes": 19, "total": 20}}
        current = {"route_accuracy": {"rate": 0.80, "successes": 16, "total": 20}}
        bp = _write_baseline(tmp_path, baseline)
        regressions = compare_with_baseline(_report(current), bp, {})
        assert len(regressions) == 1
        assert "route_accuracy" in regressions[0]
