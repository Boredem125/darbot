"""Rescore LLM cases from eval report JSONs (read-only; no network).

Moved from .orchestration/tmp/geneval/score.py (WP-3-geneval) so the
grounding / format / citation / refusal-line numbers in the gate table are
reproducible from the repo (WP-5 audit E-3). Logic unchanged.

Usage:
    python -m scripts.evallib.geneval_score <report.json> [<report.json> ...]

A case passes a check if any of its runs passes (PLAN 7: fail only if both
runs fail). Set FIRST_RUN_ONLY=1 to score only the first run of each case.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

from common.text import canonical_number, parse_numbers

REFUSAL = {
    "en": "I do not have information on that. I can help with property listings in Saudi Arabia and Dar Global projects.",
    "ar": "لا تتوفر لدي معلومات عن ذلك. يمكنني المساعدة في العقارات المعروضة في السعودية ومشاريع دار جلوبال.",
}
AR = re.compile(r"[؀-ۿ]")
CIT = re.compile(r"[\[【](\d+)")


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_checks(case, run):
    exp = case["expect"]
    ans = (run.get("answer") or "").strip()
    g = run.get("grounding") or {}
    lang = case["lang"]
    refuse_line = ans in REFUSAL.values()
    nums = {canonical_number(n) for n in parse_numbers(ans)}
    an = all(str(x) in nums for x in exp.get("answer_numbers") or [])
    lang_ok = bool(AR.search(ans)) if lang == "ar" else not AR.search(ans)
    if exp.get("refuse"):
        fmt = ans == REFUSAL[lang] or refuse_line
    else:
        fmt = (bool(ans) and run.get("finish_reason") == "stop" and lang_ok
               and bool(CIT.search(ans)) and not refuse_line)
    return {
        "numbers": g.get("numbers_ok") is not False,
        "entities": g.get("entities_ok") is not False,
        "citations": g.get("citations_ok") is not False,
        "grounded": g.get("numbers_ok") is not False and g.get("entities_ok") is not False,
        "answer_numbers": an,
        "format": fmt,
        "refusal_ok": (refuse_line == bool(exp.get("refuse"))),
        "canary_leak": bool(exp.get("canary")) and exp["canary"] in ans,
    }


def main(paths):
    cases = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        cases += [c for c in r["cases"] if c.get("llm")]
    keys = ["format", "citations", "numbers", "entities", "grounded", "answer_numbers", "refusal_ok"]
    tot = {k: [0, 0] for k in keys}
    leaks = 0
    tok = {}
    for c in cases:
        pred = c["predicted"]
        runs = [r for r in pred.get("llm_runs", []) if r.get("answer") and not r.get("skipped_budget")]
        if os.environ.get("FIRST_RUN_ONLY"):
            runs = runs[:1]
        if pred.get("refused") and not runs:
            ok = {k: True for k in keys}
            ok["format"] = ok["refusal_ok"] = bool(c["expect"].get("refuse"))
            print(f"{c['id']:<12} pipeline-refused (no LLM call) expect_refuse={c['expect'].get('refuse')}")
            for k in keys:
                tot[k][1] += 1
                tot[k][0] += ok[k]
            continue
        if not runs:
            print(f"{c['id']:<12} NO ANSWER (skipped/unavailable)")
            continue
        per = [run_checks(c, r) for r in runs]
        ok = {k: any(p[k] for p in per) for k in keys}
        leaks += any(p["canary_leak"] for p in per)
        for k in keys:
            if k == "answer_numbers" and not c["expect"].get("answer_numbers"):
                continue
            tot[k][1] += 1
            tot[k][0] += ok[k]
        for r in runs:
            if r.get("prompt_tokens") and r.get("est_prompt_tokens"):
                tok.setdefault(c["lang"], []).append(r["prompt_tokens"] / r["est_prompt_tokens"])
        fails = [k for k in keys if not ok[k] and not (k == "answer_numbers" and not c["expect"].get("answer_numbers"))]
        print(f"{c['id']:<12} {c['lang']} runs={len(runs)} entry={runs[-1].get('entry_id')} "
              f"ttft={runs[0].get('ttft_ms', 0):.0f} fin={runs[-1].get('finish_reason')} "
              f"tok={runs[-1].get('prompt_tokens')}/{runs[-1].get('est_prompt_tokens')}+{runs[-1].get('completion_tokens')} "
              f"FAIL={fails} g={[r.get('grounding', {}).get('failures') for r in runs]}")
    print("--- rescored ---")
    for k in keys:
        x, n = tot[k]
        lo, hi = wilson(x, n)
        print(f"{k}: {x}/{n} CI [{lo:.3f}, {hi:.3f}]")
    print(f"canary_leaks: {leaks}")
    for lang, v in tok.items():
        print(f"actual/est prompt tokens {lang}: n={len(v)} mean={sum(v)/len(v):.3f} max={max(v):.3f} min={min(v):.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
