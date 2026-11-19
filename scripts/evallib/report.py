"""Report generation: JSON + Markdown summary."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.evallib.metrics import MetricResult, MRRResult


def generate_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def build_report(
    *,
    run_id: str,
    split: str,
    config: dict[str, Any],
    cases: list[dict[str, Any]],
    metrics: dict[str, MetricResult | MRRResult],
    index_version: str = "unknown",
    prompt_version: str = "unknown",
) -> dict[str, Any]:
    """Build the full JSON report."""
    metric_summary = {}
    for name, m in metrics.items():
        if isinstance(m, MRRResult):
            metric_summary[name] = {
                "kind": "mrr",
                "value": round(m.value, 4),
                "total": m.total,
                "rate": round(m.rate, 4),
            }
        else:
            metric_summary[name] = {
                "successes": m.successes,
                "total": m.total,
                "rate": round(m.rate, 4),
                "ci_lower": round(m.lower, 4),
                "ci_upper": round(m.upper, 4),
            }

    return {
        "run_id": run_id,
        "split": split,
        "timestamp": datetime.now(UTC).isoformat(),
        "config": config,
        "index_version": index_version,
        "prompt_version": prompt_version,
        "n_cases": len(cases),
        "metrics": metric_summary,
        "cases": cases,
    }


def write_json_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{report['run_id']}.json"
    fp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return fp


def write_md_summary(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{report['run_id']}.md"
    lines = [
        f"# Eval report {report['run_id']}",
        "",
        f"Split: {report['split']}  ",
        f"Cases: {report['n_cases']}  ",
        f"Index: {report['index_version']}  ",
        f"Prompt: {report['prompt_version']}  ",
        f"Time: {report['timestamp']}  ",
        "",
        "## Metrics",
        "",
        "| Metric | x/n | Rate | CI 95% |",
        "|--------|-----|------|--------|",
    ]
    for name, vals in report["metrics"].items():
        if vals.get("kind") == "mrr":
            lines.append(f"| {name} | MRR={vals['value']:.4f} | n={vals['total']} | -- |")
        else:
            rate_pct = f"{vals['rate']:.1%}"
            ci = f"[{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}]"
            lines.append(f"| {name} | {vals['successes']}/{vals['total']} | {rate_pct} | {ci} |")
    lines.append("")
    fp.write_text("\n".join(lines), encoding="utf-8")
    return fp


def write_baseline(report: dict[str, Any], baseline_path: Path) -> Path:
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return baseline_path


# Metrics where lower is better.  A *regression* on these means the rate
# went UP compared to the baseline, not down.  The absolute §7 gates for
# these metrics are ceilings (e.g. false_refusal <= 5%, canary_leak == 0).
_LOWER_IS_BETTER: frozenset[str] = frozenset({"false_refusal", "canary_leak"})

# Tolerance: a metric must worsen by more than this fraction of the total
# count to be flagged as a regression.  One case out of 60 is about 0.017.
_REGRESSION_TOLERANCE = 0.02


def compare_with_baseline(
    current: dict[str, Any],
    baseline_path: Path,
    gates: dict[str, float] | None = None,
) -> list[str]:
    """Compare current metrics against the baseline.  Returns regressions.

    ``--compare`` fails CI only on a **regression** against the stored
    baseline (worse than the baseline by more than the tolerance).  The
    absolute section 7 gates are informational: they are printed as a
    PASS/FAIL table by eval.py but do not cause ``--compare`` to fail.

    Direction: ``false_refusal`` and ``canary_leak`` are lower-is-better.
    For those metrics, an *increase* beyond the baseline is a regression.
    """
    if not baseline_path.exists():
        return [f"baseline not found: {baseline_path}"]

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline.get("metrics", {})
    current_metrics = current.get("metrics", {})
    regressions: list[str] = []

    if gates is None:
        gates = {}

    all_names = set(baseline_metrics) | set(gates)

    for name in sorted(all_names):
        cur = current_metrics.get(name, {})
        cur_rate = cur.get("rate", 0.0)
        base = baseline_metrics.get(name, {})
        base_rate = base.get("rate", 0.0)

        lower_better = name in _LOWER_IS_BETTER

        if lower_better:
            # Regression = current is *higher* than baseline + tolerance
            if cur_rate > base_rate + _REGRESSION_TOLERANCE:
                regressions.append(
                    f"{name}: {cur_rate:.3f} > baseline {base_rate:.3f} (higher is worse)"
                )
        else:
            # Regression = current is *lower* than baseline - tolerance
            if cur_rate < base_rate - _REGRESSION_TOLERANCE:
                regressions.append(
                    f"{name}: {cur_rate:.3f} < baseline {base_rate:.3f} (lower is worse)"
                )

    return regressions
