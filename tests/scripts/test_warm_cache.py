"""Tests for warm_cache.py."""

from __future__ import annotations

import asyncio
import contextlib
import csv
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from app.domain.ports import (
    ChatMessage,
    ErrorCode,
    LLMFailover,
    LLMStreamItem,
    LLMUnavailableError,
)
from common.schemas import Provider
from scripts.evallib.llm_step import QuotaLogger
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


# ---------------------------------------------------------------------------
# 429 quota logging
# ---------------------------------------------------------------------------


class _FakeLLM429:
    """Yields one LLMFailover(429) then raises LLMUnavailableError."""

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMStreamItem]:
        fo = LLMFailover(
            provider=Provider.openrouter,
            model="google/gemma-4-31b-it:free",
            code=ErrorCode.provider_rate_limited,
            detail="429 from upstream",
        )
        yield fo
        raise LLMUnavailableError([fo])


class _FakeLLMUnavailable:
    """Raises LLMUnavailableError immediately (no items yielded)."""

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMStreamItem]:
        fo = LLMFailover(
            provider=Provider.groq,
            model="openai/gpt-oss-120b",
            code=ErrorCode.provider_rate_limited,
            detail="429 from upstream",
        )
        raise LLMUnavailableError([fo])
        yield  # make it an async generator


def _read_csv_rows(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


class TestCitationNormalisation:
    """Warm cache must store answers with ASCII [n], never fullwidth."""

    def test_no_fullwidth_citations_in_cache(self):
        """Every record in data/index/warm_cache.jsonl uses [n], not fullwidth."""
        import re

        from common.schemas import WarmCacheRecord

        wc = Path(__file__).resolve().parents[2] / "data" / "index" / "warm_cache.jsonl"
        if not wc.exists():
            pytest.skip("warm_cache.jsonl not present")
        fullwidth_re = re.compile(r"[【】〔〕［］]")
        for i, line in enumerate(wc.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            rec = WarmCacheRecord.model_validate_json(line)
            assert not fullwidth_re.search(rec.entry.answer), (
                f"Record {i} contains fullwidth citation brackets"
            )

    def test_no_wrong_script_answers(self):
        """Every record's answer script matches its language (en/ar)."""
        from common.schemas import WarmCacheRecord
        from common.text import arabic_ratio

        wc = Path(__file__).resolve().parents[2] / "data" / "index" / "warm_cache.jsonl"
        if not wc.exists():
            pytest.skip("warm_cache.jsonl not present")
        for i, line in enumerate(wc.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            rec = WarmCacheRecord.model_validate_json(line)
            ar = arabic_ratio(rec.entry.answer)
            lang = rec.entry.language.value
            if lang == "en":
                assert ar <= 0.5, (
                    f"Record {i} language=en but arabic_ratio={ar:.2f}"
                )
            elif lang == "ar":
                assert ar >= 0.2, (
                    f"Record {i} language=ar but arabic_ratio={ar:.2f}"
                )


class TestQuotaLoggingOn429:
    """Prove that a 429 from the LLM logs a row to quota_log.csv.

    Exercises the same stream-processing loop that warm_cache.py uses,
    with no network, no real adapters.
    """

    def test_failover_in_stream_logs_row(self, tmp_path: Path):
        """When the stream yields LLMFailover(429), a row is logged."""
        csv_path = tmp_path / "quota_log.csv"
        logger = QuotaLogger(csv_path)
        llm = _FakeLLM429()

        async def run():
            try:
                async with contextlib.aclosing(
                    llm.stream([], max_tokens=50, temperature=0.0)
                ) as stream:
                    async for item in stream:
                        if isinstance(item, LLMFailover):
                            code_val = item.code.value
                            if code_val != "provider_skipped":
                                logger.log(
                                    provider=item.provider.value,
                                    model=item.model,
                                    purpose="warm_cache",
                                    wp="WP-test",
                                    ok=False,
                                    prompt_tokens=0,
                                    completion_tokens=0,
                                )
                            if "rate_limited" in code_val:
                                break
            except LLMUnavailableError:
                pass

        asyncio.run(run())
        rows = _read_csv_rows(csv_path)
        # header + 1 data row
        assert len(rows) == 2, f"Expected header + 1 row, got {len(rows)}: {rows}"
        data = rows[1]
        assert data[1] == "openrouter"  # provider
        assert data[2] == "google/gemma-4-31b-it:free"  # model
        assert data[5] == "0"  # ok=False

    def test_unavailable_exception_logs_row(self, tmp_path: Path):
        """When LLMUnavailableError is raised (no items yielded), a row is logged."""
        csv_path = tmp_path / "quota_log.csv"
        logger = QuotaLogger(csv_path)
        llm = _FakeLLMUnavailable()

        async def run():
            try:
                async with contextlib.aclosing(
                    llm.stream([], max_tokens=50, temperature=0.0)
                ) as stream:
                    async for item in stream:
                        if isinstance(item, LLMFailover):
                            code_val = item.code.value
                            if code_val != "provider_skipped":
                                logger.log(
                                    provider=item.provider.value,
                                    model=item.model,
                                    purpose="warm_cache",
                                    wp="WP-test",
                                    ok=False,
                                    prompt_tokens=0,
                                    completion_tokens=0,
                                )
            except LLMUnavailableError as exc:
                for fa in exc.attempts:
                    code_val = fa.code.value
                    if code_val != "provider_skipped":
                        logger.log(
                            provider=fa.provider.value,
                            model=fa.model,
                            purpose="warm_cache",
                            wp="WP-test",
                            ok=False,
                            prompt_tokens=0,
                            completion_tokens=0,
                        )

        asyncio.run(run())
        rows = _read_csv_rows(csv_path)
        assert len(rows) == 2, f"Expected header + 1 row, got {len(rows)}: {rows}"
        data = rows[1]
        assert data[1] == "groq"  # provider
        assert data[5] == "0"  # ok=False
