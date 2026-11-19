"""Validate eval/sets/*.jsonl against the WP-3 eval-set format (no LLM, no network).

Checks: every line parses; keys and types follow the format; filters validate as
SearchFilters fields; op is an AggregateOp; ids are unique across all sets; no query
text appears in both dev and test; no phone numbers or emails; canary listings are
valid Listing records. Prints counts per set, split and language.

Usage: .venv/Scripts/python.exe eval/sets/check_sets.py
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.domain.ports import AggregateOp
from common.schemas import Listing, SearchFilters
from common.text import normalise_for_key

SETS = Path(__file__).resolve().parent
CASE_FILES = ["templated_dev", "templated_test", "handwritten", "projects", "multiturn", "arabic", "oos",
              "provider_subset", "injection_fixtures"]
EXPECT_KEYS = {"route", "filters", "op", "sql_truth", "tolerance_pct", "entity_ids", "refuse",
               "answer_numbers", "kind", "canary"}
ROUTES = {"lookup", "aggregate", "project_info", "out_of_scope"}
KINDS = {None, "lookup", "aggregate", "project", "arabic", "refusal"}
PII = [re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), re.compile(r"(?:\+?966|\b05)\d[\d\s-]{6,}"),
       re.compile(r"\+?\d[\d\s-]{9,}\d")]


def main() -> int:
    errors: list[str] = []
    ids: set[str] = set()
    texts: dict[str, set[str]] = {"dev": set(), "test": set()}
    counts: collections.Counter = collections.Counter()
    llm = 0

    def err(msg: str) -> None:
        errors.append(msg)

    for name in CASE_FILES:
        for ln, line in enumerate((SETS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines(), 1):
            where = f"{name}:{ln}"
            r = json.loads(line)
            if set(r) != {"id", "set", "split", "lang", "turns", "llm", "expect"}:
                err(f"{where} top-level keys {sorted(r)}")
            e = r["expect"]
            if set(e) != EXPECT_KEYS:
                err(f"{where} expect keys {sorted(e)}")
            if r["id"] in ids:
                err(f"{where} duplicate id {r['id']}")
            ids.add(r["id"])
            if r["split"] not in texts or r["lang"] not in ("en", "ar") or not isinstance(r["llm"], bool):
                err(f"{where} split/lang/llm")
            if not r["turns"] or not all(isinstance(t, str) and t for t in r["turns"]):
                err(f"{where} turns")
            if name == "multiturn" and len(r["turns"]) != 2:
                err(f"{where} multiturn needs 2 turns")
            if e["route"] not in ROUTES or e["kind"] not in KINDS:
                err(f"{where} route/kind")
            SearchFilters(**e["filters"])  # raises on unknown field or bad value
            if e["op"] is not None:
                AggregateOp(e["op"])
                if e["sql_truth"] is None or e["route"] != "aggregate":
                    err(f"{where} op without sql_truth/aggregate route")
            if e["refuse"] and e["route"] == "aggregate":
                err(f"{where} refuse on aggregate")
            if r["llm"] is False and e["answer_numbers"] and name != "projects":
                err(f"{where} answer_numbers without llm")
            if name == "provider_subset" and e["kind"] is None:
                err(f"{where} provider_subset needs kind")
            if name == "injection_fixtures" and not e["canary"]:
                err(f"{where} injection needs canary")
            for t in r["turns"]:
                if any(p.search(t) for p in PII):
                    err(f"{where} PII-like text")
                texts[r["split"]].add(normalise_for_key(t))
            counts[(r["set"], r["split"], r["lang"])] += 1
            llm += r["llm"]
    overlap = texts["dev"] & texts["test"]
    if overlap:
        err(f"dev/test overlap: {sorted(overlap)[:5]}")

    cp = [json.loads(x) for x in (SETS / "cache_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in cp:
        if set(r) != {"id", "split", "a", "b", "same_answer", "note"} or r["id"] in ids:
            err(f"cache_pairs {r.get('id')} format/id")
        ids.add(r["id"])
    cpc = collections.Counter((r["split"], r["same_answer"]) for r in cp)
    same_scope = sum(1 for r in cp if not r["same_answer"] and r["note"].startswith("same scope"))
    if sum(1 for r in cp if r["same_answer"]) != 15 or sum(1 for r in cp if not r["same_answer"]) != 15:
        err("cache_pairs must be 15 + 15")
    if same_scope < 10:
        err("cache_pairs needs >= 10 same-scope near-misses")

    for line in (SETS / "injection_listings.jsonl").read_text(encoding="utf-8").splitlines():
        rec = Listing.model_validate_json(line)
        if any(p.search(rec.description or "") for p in PII):
            err(f"canary listing {rec.id} PII-like text")

    for k in sorted(counts):
        print(*k, counts[k])
    print("cache_pairs", dict(sorted(cpc.items())), "same-scope near-misses", same_scope)
    print("llm cases", llm, "| case ids", len(ids) - len(cp), "| errors", len(errors))
    for m in errors:
        print("ERROR", m)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
