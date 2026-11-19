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


def compare_with_baseline(
    current: dict[str, Any],
    baseline_path: Path,
    gates: dict[str, float] | None = None,
) -> list[str]:
    """Compare current metrics against baseline. Returns list of regressions.

    Gates are metric_name -> minimum_rate. If a gated metric drops below its gate,
    that is a regression.
    """
    if not baseline_path.exists():
        return [f"baseline not found: {baseline_path}"]

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline.get("metrics", {})
    current_metrics = current.get("metrics", {})
    regressions: list[str] = []

    if gates is None:
        gates = {}

    for name, gate in gates.items():
        cur = current_metrics.get(name, {})
        cur_rate = cur.get("rate", 0.0)
        if cur_rate < gate:
            base_rate = baseline_metrics.get(name, {}).get("rate", 0.0)
            regressions.append(
                f"{name}: {cur_rate:.3f} < gate {gate:.3f} (baseline {base_rate:.3f})"
            )

    # Also check for any metric that dropped from baseline
    for name in baseline_metrics:
        if name in current_metrics:
            base_rate = baseline_metrics[name].get("rate", 0.0)
            cur_rate = current_metrics[name].get("rate", 0.0)
            if cur_rate < base_rate and name not in gates:
                regressions.append(
                    f"{name}: dropped {base_rate:.3f} -> {cur_rate:.3f}"
                )

    return regressions
