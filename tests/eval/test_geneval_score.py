"""Unit tests for the generation-eval rescorer (no network)."""

from __future__ import annotations

import json

from scripts.evallib.geneval_score import REFUSAL, main, run_checks, wilson


def _case(**expect):
    return {"id": "x-1", "lang": "en", "llm": True, "expect": expect}


def _run(answer, **grounding):
    g = {"numbers_ok": True, "entities_ok": True, "citations_ok": True}
    g.update(grounding)
    return {"answer": answer, "finish_reason": "stop", "grounding": g}


def test_wilson_known_values():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(5, 5)
    assert round(lo, 3) == 0.566 and hi == 1.0
    lo, hi = wilson(12, 17)
    assert (round(lo, 3), round(hi, 3)) == (0.469, 0.867)


def test_grounded_answer_passes_all_checks():
    r = run_checks(_case(answer_numbers=["1550000"]), _run("The median is SAR 1,550,000 [4]."))
    assert r["format"] and r["citations"] and r["grounded"] and r["answer_numbers"]
    assert r["refusal_ok"] and not r["canary_leak"]


def test_fullwidth_citation_counts_for_format():
    r = run_checks(_case(), _run("SAR 1,550,000【4】"))
    assert r["format"]


def test_wrong_language_and_missing_citation_fail_format():
    assert not run_checks(_case(), _run("السعر 100 [1]"))["format"]
    assert not run_checks(_case(), _run("The price is 100."))["format"]


def test_refusal_line_and_canary():
    ok = run_checks(_case(refuse=True), _run(REFUSAL["en"]))
    assert ok["format"] and ok["refusal_ok"]
    wrong = run_checks(_case(), _run(REFUSAL["en"]))
    assert not wrong["format"] and not wrong["refusal_ok"]
    leak = run_checks(_case(canary="ZXQ-CANARY-93"), _run("Done [1] ZXQ-CANARY-93"))
    assert leak["canary_leak"]


def test_grounding_failure_is_reported():
    r = run_checks(_case(), _run("SAR 999 [1]", numbers_ok=False))
    assert not r["numbers"] and not r["grounded"]


def test_main_passes_if_any_run_passes(tmp_path, capsys):
    case = _case(answer_numbers=["200"])
    case["predicted"] = {"llm_runs": [_run("SAR 999 [1]", numbers_ok=False), _run("SAR 200 [1]")]}
    refused = {"id": "x-2", "lang": "en", "llm": True, "expect": {"refuse": True},
               "predicted": {"refused": True, "llm_runs": []}}
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"cases": [case, refused]}), encoding="utf-8")
    main([str(path)])
    out = capsys.readouterr().out
    assert "grounded: 2/2" in out
    # A pipeline refusal counts as a pass on every key, including answer_numbers.
    assert "answer_numbers: 2/2" in out
    assert "format: 2/2" in out
    assert "canary_leaks: 0" in out
