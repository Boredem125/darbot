"""Tests for warm_cache.py."""

from __future__ import annotations

from pathlib import Path

from scripts.warm_cache import _get_prompt_version, _load_queries


class TestLoadQueries:
    def test_loads_jsonl(self, tmp_path: Path):
        p = tmp_path / "queries.jsonl"
        p.write_text(
            '{"query": "villas in Riyadh", "lang": "en"}\n'
            '{"query": "apartments in Jeddah", "lang": "en"}\n',
            encoding="utf-8",
        )
        queries = _load_queries(p)
        assert len(queries) == 2
        assert queries[0]["query"] == "villas in Riyadh"

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert _load_queries(p) == []

    def test_missing_file(self, tmp_path: Path):
        p = tmp_path / "missing.jsonl"
        assert _load_queries(p) == []


class TestGetPromptVersion:
    def test_returns_stem_at_hash(self):
        pv = _get_prompt_version("system_v1")
        assert pv.startswith("system_v1@")
        assert len(pv.split("@")[1]) == 8
