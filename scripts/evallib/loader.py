"""Load eval sets from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cases(
    sets_dir: Path,
    *,
    split: str | None = None,
    llm_only: bool = False,
) -> list[dict[str, Any]]:
    """Load all cases from eval/sets/*.jsonl, optionally filtering by split and llm flag."""
    cases: list[dict[str, Any]] = []
    if not sets_dir.is_dir():
        return cases

    for fp in sorted(sets_dir.glob("*.jsonl")):
        if fp.name == "cache_pairs.jsonl":
            continue  # separate loader
        if fp.name == "injection_listings.jsonl":
            continue  # listing fixtures, not cases
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if split and case.get("split") != split:
                continue
            if llm_only and not case.get("llm"):
                continue
            cases.append(case)
    return cases


def load_cache_pairs(
    sets_dir: Path,
    *,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Load cache_pairs.jsonl."""
    fp = sets_dir / "cache_pairs.jsonl"
    if not fp.exists():
        return []
    pairs: list[dict[str, Any]] = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        pair = json.loads(line)
        if split and pair.get("split") != split:
            continue
        pairs.append(pair)
    return pairs
