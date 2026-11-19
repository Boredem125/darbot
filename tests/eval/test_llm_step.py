"""Tests for the LLM generation step in the eval harness.

Written tests-first per WP-3-harness-llm Phase A.
"""

from __future__ import annotations

import asyncio
import csv
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from app.domain.ports import (
    ChatMessage,
    LLMDelta,
    LLMDone,
    LLMFailover,
    LLMStreamItem,
)
from app.domain.router import EntityIndex
from common.schemas import (
    Entities,
    Entity,
    EntityKind,
    ErrorCode,
    FinishReason,
    Provider,
    TokenUsage,
)
from scripts.evallib.llm_step import (
    BudgetTracker,
    LLMCaseResult,
    QuotaLogger,
    TPMPacer,
    filter_cases,
    llm_case_scored,
    llm_skip_counts,
    plan_summary,
    render_card_text,
    run_llm_step,
    should_rerun,
)
from scripts.evallib.metrics import entity_grounding, numeric_grounding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MultiScriptFakeLLM:
    """Fake LLM that yields different scripts on successive calls."""

    def __init__(self) -> None:
        self._scripts: list[list[LLMStreamItem]] = []
        self._call_idx = 0

    def add_script(self, items: list[LLMStreamItem]) -> None:
        self._scripts.append(items)

    @property
    def call_count(self) -> int:
        return self._call_idx

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMStreamItem]:
        script = self._scripts[self._call_idx % len(self._scripts)]
        self._call_idx += 1
        for item in script:
            yield item


def _make_entities() -> EntityIndex:
    return EntityIndex(
        Entities(
            entities=[
                Entity(kind=EntityKind.city, canonical="Riyadh", aliases=["riyadh"]),
                Entity(kind=EntityKind.city, canonical="Jeddah", aliases=["jeddah"]),
            ]
        )
    )


SYSTEM_PROMPT = "You answer questions about property listings. Cite [n] after each fact."


def _listing_card(
    n: int = 1,
    doc_id: str = "wasalt:1",
    title: str = "Test Villa",
    city: str = "Riyadh",
    price_sar: int = 1_500_000,
    bedrooms: int = 3,
) -> dict[str, Any]:
    return {
        "n": n,
        "doc_id": doc_id,
        "kind": "listing",
        "source": "wasalt",
        "title": title,
        "url": "https://example.com/1",
        "scraped_at": "2024-01-01T00:00:00Z",
        "purpose": "sale",
        "property_type": "villa",
        "city": city,
        "bedrooms": bedrooms,
        "price_sar": price_sar,
        "area_sqm": 350.0,
    }


def _case(
    case_id: str = "test-llm-001",
    query: str = "villas in Riyadh",
    llm: bool = True,
    lang: str = "en",
    **expect_kw: Any,
) -> dict[str, Any]:
    expect: dict[str, Any] = {
        "route": "lookup",
        "filters": {},
        "refuse": False,
        "answer_numbers": [],
        "kind": None,
        **expect_kw,
    }
    return {
        "id": case_id,
        "set": "handwritten",
        "split": "dev",
        "lang": lang,
        "turns": [query],
        "llm": llm,
        "expect": expect,
    }


def _predicted(
    route: str = "lookup",
    refused: bool = False,
    cards: list[dict[str, Any]] | None = None,
    sql_value: Any = None,
) -> dict[str, Any]:
    return {
        "case_id": "test-llm-001",
        "route": route,
        "filters": {},
        "cards": cards if cards is not None else [_listing_card()],
        "refused": refused,
        "sql_value": sql_value,
        "answer": "",
        "grounding": None,
    }


# ---------------------------------------------------------------------------
# Tests: render_card_text
# ---------------------------------------------------------------------------


class TestRenderCardText:
    def test_listing_card(self) -> None:
        card = _listing_card()
        text = render_card_text(card)
        assert "Riyadh" in text
        assert "1,500,000" in text
        assert "3 bedrooms" in text

    def test_project_card(self) -> None:
        card: dict[str, Any] = {
            "n": 1,
            "doc_id": "darglobal:project:trump-tower",
            "kind": "project",
            "source": "darglobal",
            "title": "Trump Tower Jeddah",
            "url": "https://example.com/trump",
            "scraped_at": "2024-01-01T00:00:00Z",
            "city": "Jeddah",
            "country": "Saudi Arabia",
            "brand": "Trump",
            "handover": "Q4 2027",
            "price_from_amount": 2_000_000,
            "price_from_currency": "SAR",
        }
        text = render_card_text(card)
        assert "Trump Tower" in text
        assert "Jeddah" in text
        assert "Trump" in text
        assert "Q4 2027" in text

    def test_h2_snippet_included(self) -> None:
        """H-2: render_card_text must include snippet/description in context."""
        card: dict[str, Any] = {
            "n": 1,
            "doc_id": "darglobal:project:test",
            "kind": "project",
            "source": "darglobal",
            "title": "Test Project",
            "url": "https://example.com/test",
            "scraped_at": "2024-01-01T00:00:00Z",
            "city": "Jeddah",
            "snippet": "Only 15 minutes from Marbella beach.",
        }
        text = render_card_text(card)
        assert "15 minutes from Marbella" in text

    def test_h2_uses_shared_renderer_format(self) -> None:
        """H-2: render_card_text uses context mode with <source> delimiters."""
        card = _listing_card()
        text = render_card_text(card)
        # Context mode includes [n], URL and As of
        assert "[1]" in text
        assert "URL:" in text
        assert "As of:" in text


# ---------------------------------------------------------------------------
# Tests: BudgetTracker
# ---------------------------------------------------------------------------


class TestBudgetTracker:
    def test_zero_budget_always_blocks(self) -> None:
        bt = BudgetTracker(limit=0)
        assert not bt.can_attempt(1)

    def test_allows_within_limit(self) -> None:
        bt = BudgetTracker(limit=5)
        bt.used = 2
        assert bt.can_attempt(2)  # 2+2=4 <=5

    def test_blocks_exceeding_limit(self) -> None:
        bt = BudgetTracker(limit=3)
        bt.used = 2
        assert not bt.can_attempt(2)  # 2+2=4 >3

    def test_exact_limit(self) -> None:
        bt = BudgetTracker(limit=3)
        bt.used = 1
        assert bt.can_attempt(2)  # 1+2=3 <=3

    def test_record_increments(self) -> None:
        bt = BudgetTracker(limit=10)
        assert bt.used == 0
        bt.record_attempt()
        assert bt.used == 1
        bt.record_attempt()
        assert bt.used == 2


# ---------------------------------------------------------------------------
# Tests: QuotaLogger
# ---------------------------------------------------------------------------


class TestQuotaLogger:
    def test_creates_file_with_header(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "quota.csv"
        QuotaLogger(csv_path)
        assert csv_path.exists()
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "ts_utc,provider,model,purpose,wp,ok,prompt_tokens,completion_tokens"

    def test_appends_row(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "quota.csv"
        logger = QuotaLogger(csv_path)
        logger.log(
            provider="groq",
            model="test-model",
            purpose="geneval",
            wp="WP-3",
            ok=True,
            prompt_tokens=100,
            completion_tokens=20,
        )
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2  # header + 1 row
        row = lines[1].split(",")
        assert row[1] == "groq"
        assert row[2] == "test-model"
        assert row[5] == "1"  # ok

    def test_multiple_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "quota.csv"
        logger = QuotaLogger(csv_path)
        for i in range(3):
            logger.log(
                provider="groq",
                model=f"model-{i}",
                purpose="geneval",
                wp="WP-3",
                ok=i % 2 == 0,
                prompt_tokens=100 * i,
                completion_tokens=20 * i,
            )
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4  # header + 3 rows


# ---------------------------------------------------------------------------
# Tests: run_llm_step (the core)
# ---------------------------------------------------------------------------


class TestRunLLMStep:
    def test_two_cases_end_to_end(self, tmp_path: Path) -> None:
        """Done-when item 1: Run 2 LLM cases with FakeLLM, check answer,
        grounding, token fields, and quota-log rows."""
        entities = _make_entities()
        csv_path = tmp_path / "quota.csv"
        quota_logger = QuotaLogger(csv_path)
        budget = BudgetTracker(limit=10)

        fake_llm = MultiScriptFakeLLM()
        # Case 1 answer: references card data (Riyadh, 1,500,000)
        fake_llm.add_script(
            [
                LLMDelta(text="A villa in Riyadh "),
                LLMDelta(text="costs SAR 1,500,000 [1]."),
                LLMDone(
                    provider=Provider.groq,
                    model="test-model",
                    finish_reason=FinishReason.stop,
                    usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                    ttft_ms=50.0,
                    total_ms=200.0,
                ),
            ]
        )
        # Case 2 answer: references card data (Jeddah, 800,000)
        fake_llm.add_script(
            [
                LLMDelta(text="An apartment in Jeddah "),
                LLMDelta(text="is listed for SAR 800,000 [1]."),
                LLMDone(
                    provider=Provider.groq,
                    model="test-model",
                    finish_reason=FinishReason.stop,
                    usage=TokenUsage(prompt_tokens=110, completion_tokens=25),
                    ttft_ms=60.0,
                    total_ms=250.0,
                ),
            ]
        )

        case1 = _case("llm-001", "villas in Riyadh")
        pred1 = _predicted(
            cards=[_listing_card(1, "wasalt:1", "Test Villa", "Riyadh", 1_500_000, 3)]
        )

        case2 = _case("llm-002", "apartments in Jeddah")
        pred2 = _predicted(
            cards=[_listing_card(1, "wasalt:2", "City Apartment", "Jeddah", 800_000, 2)]
        )

        r1 = asyncio.run(
            run_llm_step(
                case=case1,
                predicted=pred1,
                llm=fake_llm,
                system_prompt=SYSTEM_PROMPT,
                entities=entities,
                budget=budget,
                chain_length=1,
                quota_logger=quota_logger,
                purpose="geneval",
                wp="WP-3",
            )
        )

        r2 = asyncio.run(
            run_llm_step(
                case=case2,
                predicted=pred2,
                llm=fake_llm,
                system_prompt=SYSTEM_PROMPT,
                entities=entities,
                budget=budget,
                chain_length=1,
                quota_logger=quota_logger,
                purpose="geneval",
                wp="WP-3",
            )
        )

        # -- Case 1 checks --
        assert r1.answer == "A villa in Riyadh costs SAR 1,500,000 [1]."
        assert r1.grounding is not None
        assert r1.grounding["numbers_ok"] is True
        assert r1.grounding["citations_ok"] is True
        assert r1.entry_id == "groq:test-model"
        assert r1.prompt_tokens == 100
        assert r1.completion_tokens == 20
        assert r1.ttft_ms == 50.0
        assert r1.total_ms == 200.0
        assert r1.finish_reason == "stop"
        assert r1.est_prompt_tokens > 0
        assert not r1.skipped_budget

        # -- Case 2 checks --
        assert r2.answer == "An apartment in Jeddah is listed for SAR 800,000 [1]."
        assert r2.grounding is not None
        assert r2.grounding["numbers_ok"] is True
        assert r2.grounding["citations_ok"] is True
        assert r2.entry_id == "groq:test-model"
        assert r2.prompt_tokens == 110
        assert r2.completion_tokens == 25

        # -- Quota log: 1 row per attempt (1 per case, since no failovers) --
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        assert rows[1][1] == "groq"  # provider col
        assert rows[1][5] == "1"  # ok col
        assert rows[2][1] == "groq"

        # -- Budget consumed --
        assert budget.used == 2

    def test_budget_stops_at_limit(self, tmp_path: Path) -> None:
        """Done-when item 2: chain length 2 with budget 3 stops after the
        first request that could exceed the budget."""
        entities = _make_entities()
        csv_path = tmp_path / "quota.csv"
        quota_logger = QuotaLogger(csv_path)
        budget = BudgetTracker(limit=3)

        fake_llm = MultiScriptFakeLLM()
        for _ in range(5):
            fake_llm.add_script(
                [
                    LLMDelta(text="Answer [1]."),
                    LLMDone(
                        provider=Provider.groq,
                        model="test-model",
                        finish_reason=FinishReason.stop,
                        usage=TokenUsage(prompt_tokens=50, completion_tokens=10),
                        ttft_ms=30.0,
                        total_ms=100.0,
                    ),
                ]
            )

        results: list[LLMCaseResult] = []
        for i in range(3):
            case = _case(f"budget-{i}", "villas in Riyadh")
            pred = _predicted(cards=[_listing_card()])
            r = asyncio.run(
                run_llm_step(
                    case=case,
                    predicted=pred,
                    llm=fake_llm,
                    system_prompt=SYSTEM_PROMPT,
                    entities=entities,
                    budget=budget,
                    chain_length=2,  # chain of 2 entries
                    quota_logger=quota_logger,
                    purpose="geneval",
                    wp="WP-3",
                )
            )
            results.append(r)

        # Case 0: used=0, 0+2=2 <=3 -> proceed (1 attempt used) -> used=1
        assert results[0].answer == "Answer [1]."
        assert not results[0].skipped_budget
        # Case 1: used=1, 1+2=3 <=3 -> proceed (1 attempt used) -> used=2
        assert results[1].answer == "Answer [1]."
        assert not results[1].skipped_budget
        # Case 2: used=2, 2+2=4 >3 -> SKIP
        assert results[2].skipped_budget
        assert results[2].answer == ""

        assert budget.used == 2
        assert budget.skipped_count == 1

    def test_refused_case_skips_llm(self, tmp_path: Path) -> None:
        """When the pipeline already refused, the LLM step is skipped."""
        entities = _make_entities()
        budget = BudgetTracker(limit=10)
        fake_llm = MultiScriptFakeLLM()
        fake_llm.add_script([])  # should never be reached

        case = _case("refused-001", "weather today", refuse=True)
        pred = _predicted(refused=True)

        r = asyncio.run(
            run_llm_step(
                case=case,
                predicted=pred,
                llm=fake_llm,
                system_prompt=SYSTEM_PROMPT,
                entities=entities,
                budget=budget,
                chain_length=1,
            )
        )

        assert r.answer == ""
        assert r.grounding is None
        assert not r.skipped_budget
        assert budget.used == 0

    def test_failover_counts_attempts(self, tmp_path: Path) -> None:
        """A failover followed by success counts 2 attempts."""
        entities = _make_entities()
        csv_path = tmp_path / "quota.csv"
        quota_logger = QuotaLogger(csv_path)
        budget = BudgetTracker(limit=10)

        fake_llm = MultiScriptFakeLLM()
        fake_llm.add_script(
            [
                LLMFailover(
                    provider=Provider.openrouter,
                    model="failing-model",
                    code=ErrorCode.provider_rate_limited,
                    detail="429",
                ),
                LLMDelta(text="Fallback answer [1]."),
                LLMDone(
                    provider=Provider.groq,
                    model="backup-model",
                    finish_reason=FinishReason.stop,
                    usage=TokenUsage(prompt_tokens=90, completion_tokens=15),
                    ttft_ms=100.0,
                    total_ms=300.0,
                ),
            ]
        )

        case = _case("failover-001", "villas in Riyadh")
        pred = _predicted(cards=[_listing_card()])

        r = asyncio.run(
            run_llm_step(
                case=case,
                predicted=pred,
                llm=fake_llm,
                system_prompt=SYSTEM_PROMPT,
                entities=entities,
                budget=budget,
                chain_length=2,
                quota_logger=quota_logger,
                purpose="geneval",
                wp="WP-3",
            )
        )

        assert r.answer == "Fallback answer [1]."
        assert r.entry_id == "groq:backup-model"
        assert len(r.attempts) == 1  # one failover record
        assert r.attempts[0]["provider"] == "openrouter"
        assert budget.used == 2  # 1 failover + 1 done

        # Quota log should have 2 rows (1 fail + 1 ok)
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2

    def test_to_dict(self) -> None:
        """LLMCaseResult.to_dict produces all expected keys."""
        r = LLMCaseResult(answer="hello", finish_reason="stop")
        d = r.to_dict()
        for key in (
            "answer",
            "grounding",
            "entry_id",
            "ttft_ms",
            "total_ms",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "est_prompt_tokens",
            "attempts",
            "skipped_budget",
            "llm_unavailable",
            "run_index",
        ):
            assert key in d

    def test_h4d_breaker_skip_not_logged(self, tmp_path: Path) -> None:
        """H-4d: provider_skipped failovers must not be logged to quota CSV."""
        entities = _make_entities()
        csv_path = tmp_path / "quota.csv"
        quota_logger = QuotaLogger(csv_path)
        budget = BudgetTracker(limit=10)

        fake_llm = MultiScriptFakeLLM()
        fake_llm.add_script(
            [
                LLMFailover(
                    provider=Provider.openrouter,
                    model="skipped-model",
                    code=ErrorCode.provider_skipped,
                    detail="breaker_cooldown",
                ),
                LLMDelta(text="Fallback answer [1]."),
                LLMDone(
                    provider=Provider.groq,
                    model="backup-model",
                    finish_reason=FinishReason.stop,
                    usage=TokenUsage(prompt_tokens=90, completion_tokens=15),
                    ttft_ms=100.0,
                    total_ms=300.0,
                ),
            ]
        )

        case = _case("breaker-001", "villas in Riyadh")
        pred = _predicted(cards=[_listing_card()])

        r = asyncio.run(
            run_llm_step(
                case=case,
                predicted=pred,
                llm=fake_llm,
                system_prompt=SYSTEM_PROMPT,
                entities=entities,
                budget=budget,
                chain_length=2,
                quota_logger=quota_logger,
                purpose="geneval",
                wp="WP-4",
            )
        )

        assert r.answer == "Fallback answer [1]."
        # Only 1 attempt logged (the LLMDone), not the breaker skip
        assert budget.used == 1  # breaker skip not counted
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2  # header + 1 row (LLMDone only)
        assert rows[1][1] == "groq"  # only the successful provider logged

    def test_h3_sql_result_in_prompt(self, tmp_path: Path) -> None:
        """H-3: when sql_result is passed, the prompt includes <sql_result>."""
        from app.domain.ports import AggregateOp, SqlResult

        entities = _make_entities()
        budget = BudgetTracker(limit=10)

        messages_seen: list[Any] = []

        class CaptureLLM:
            async def stream(self, messages, *, max_tokens, temperature=0.0):
                messages_seen.extend(messages)
                yield LLMDelta(text="There are 42 villas [1].")
                yield LLMDone(
                    provider=Provider.groq,
                    model="test",
                    finish_reason=FinishReason.stop,
                    usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
                    ttft_ms=50.0,
                    total_ms=200.0,
                )

        sql_result = SqlResult(
            op=AggregateOp.count,
            value=42,
            unit="count",
            n_matched=42,
        )

        case = _case("agg-001", "how many villas in Riyadh")
        pred = _predicted(route="aggregate", cards=[_listing_card()], sql_value=42)

        asyncio.run(
            run_llm_step(
                case=case,
                predicted=pred,
                llm=CaptureLLM(),
                system_prompt=SYSTEM_PROMPT,
                entities=entities,
                budget=budget,
                chain_length=1,
                sql_result=sql_result,
            )
        )

        # Check that the user message contains <sql_result>
        user_msg = [m for m in messages_seen if m.role == "user"]
        assert len(user_msg) == 1
        assert "<sql_result" in user_msg[0].content
        assert "42" in user_msg[0].content


# ---------------------------------------------------------------------------
# Tests: should_rerun
# ---------------------------------------------------------------------------


class TestShouldRerun:
    def test_grounding_number_failure_needs_rerun(self) -> None:
        result = LLMCaseResult(
            answer="Bad answer with 999",
            grounding={"numbers_ok": False, "entities_ok": True, "citations_ok": True},
        )
        assert should_rerun(result, _case()) is True

    def test_grounding_entity_failure_needs_rerun(self) -> None:
        result = LLMCaseResult(
            answer="Something in Dubai",
            grounding={"numbers_ok": True, "entities_ok": False, "citations_ok": True},
        )
        assert should_rerun(result, _case()) is True

    def test_grounding_citation_failure_needs_rerun(self) -> None:
        result = LLMCaseResult(
            answer="A villa [99].",
            grounding={"numbers_ok": True, "entities_ok": True, "citations_ok": False},
        )
        assert should_rerun(result, _case()) is True

    def test_answer_numbers_mismatch_needs_rerun(self) -> None:
        result = LLMCaseResult(
            answer="A villa costs SAR 2,000,000 [1].",
            grounding={"numbers_ok": True, "entities_ok": True, "citations_ok": True},
        )
        case = _case(answer_numbers=["1500000"])
        assert should_rerun(result, case) is True

    def test_all_pass_no_rerun(self) -> None:
        result = LLMCaseResult(
            answer="A villa costs SAR 1,500,000 [1].",
            grounding={"numbers_ok": True, "entities_ok": True, "citations_ok": True},
        )
        case = _case(answer_numbers=["1500000"])
        assert should_rerun(result, case) is False

    def test_skipped_budget_no_rerun(self) -> None:
        result = LLMCaseResult(skipped_budget=True)
        assert should_rerun(result, _case()) is False

    def test_empty_answer_no_rerun(self) -> None:
        result = LLMCaseResult(answer="")
        assert should_rerun(result, _case()) is False

    def test_no_expected_numbers_grounding_pass(self) -> None:
        """When no answer_numbers expected and grounding passes, no rerun."""
        result = LLMCaseResult(
            answer="A villa in Riyadh [1].",
            grounding={"numbers_ok": True, "entities_ok": True, "citations_ok": True},
        )
        assert should_rerun(result, _case()) is False


# ---------------------------------------------------------------------------
# Tests: filter_cases
# ---------------------------------------------------------------------------


class TestFilterCases:
    def test_filter_by_case_ids(self) -> None:
        cases = [_case("a"), _case("b"), _case("c")]
        result = filter_cases(cases, case_ids=["a", "c"])
        assert [c["id"] for c in result] == ["a", "c"]

    def test_filter_by_sets(self) -> None:
        cases = [
            {**_case("a"), "set": "handwritten"},
            {**_case("b"), "set": "provider_subset"},
            {**_case("c"), "set": "handwritten"},
        ]
        result = filter_cases(cases, sets=["provider_subset"])
        assert len(result) == 1
        assert result[0]["id"] == "b"

    def test_h4b_injection_fixtures_alias(self) -> None:
        """H-4b: --sets injection_fixtures must match cases with set='injection'."""
        cases = [
            {**_case("a"), "set": "injection"},
            {**_case("b"), "set": "handwritten"},
            {**_case("c"), "set": "injection"},
        ]
        result = filter_cases(cases, sets=["injection_fixtures"])
        assert len(result) == 2
        assert {c["id"] for c in result} == {"a", "c"}

    def test_h4b_injection_direct_still_works(self) -> None:
        """H-4b: --sets injection still works directly."""
        cases = [
            {**_case("a"), "set": "injection"},
            {**_case("b"), "set": "handwritten"},
        ]
        result = filter_cases(cases, sets=["injection"])
        assert len(result) == 1
        assert result[0]["id"] == "a"

    def test_filter_by_kind(self) -> None:
        cases = [
            _case("a", kind="lookup"),
            _case("b", kind="aggregate"),
            _case("c", kind="lookup"),
        ]
        result = filter_cases(cases, kind="lookup")
        assert len(result) == 2

    def test_filter_llm_only(self) -> None:
        cases = [
            _case("a", llm=True),
            _case("b", llm=False),
            _case("c", llm=True),
        ]
        result = filter_cases(cases, llm_only=True)
        assert len(result) == 2

    def test_no_filters_returns_all(self) -> None:
        cases = [_case("a"), _case("b")]
        result = filter_cases(cases)
        assert len(result) == 2

    def test_combined_filters(self) -> None:
        cases = [
            {**_case("a", llm=True, kind="lookup"), "set": "handwritten"},
            {**_case("b", llm=True, kind="aggregate"), "set": "handwritten"},
            {**_case("c", llm=False, kind="lookup"), "set": "provider_subset"},
        ]
        result = filter_cases(cases, llm_only=True, kind="lookup")
        assert len(result) == 1
        assert result[0]["id"] == "a"


# ---------------------------------------------------------------------------
# Tests: plan_summary
# ---------------------------------------------------------------------------


class TestPlanSummary:
    def test_counts_llm_cases(self) -> None:
        cases = [
            _case("a", llm=True),
            _case("b", llm=False),
            _case("c", llm=True),
            _case("d", llm=True, refuse=True),
        ]
        summary = plan_summary(cases, chain_length=3)
        assert summary["total_cases"] == 4
        assert summary["llm_cases"] == 3
        assert summary["llm_non_refused"] == 2
        assert summary["worst_case_attempts"] == 2 * 3  # 2 non-refused * chain 3
        assert summary["chain_length"] == 3

    def test_zero_llm_cases(self) -> None:
        cases = [
            _case("a", llm=False),
            _case("b", llm=False),
        ]
        summary = plan_summary(cases, chain_length=2)
        assert summary["llm_cases"] == 0
        assert summary["worst_case_attempts"] == 0

    def test_all_refused(self) -> None:
        cases = [
            _case("a", llm=True, refuse=True),
            _case("b", llm=True, refuse=True),
        ]
        summary = plan_summary(cases, chain_length=5)
        assert summary["llm_cases"] == 2
        assert summary["llm_non_refused"] == 0
        assert summary["worst_case_attempts"] == 0


# ---------------------------------------------------------------------------
# Tests: skipped cases excluded from grounding metrics (T0 fix 1)
# ---------------------------------------------------------------------------


class TestSkippedCasesExcluded:
    """Verify that budget-skipped and unavailable cases are excluded
    from numeric_grounding and entity_grounding."""

    def _merged(
        self,
        case_id: str,
        answer: str = "",
        skipped_budget: bool = False,
        llm_unavailable: bool = False,
        grounding: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a merged result dict as eval.py produces."""
        llm_run = {
            "answer": answer,
            "grounding": grounding,
            "entry_id": entry_id,
            "skipped_budget": skipped_budget,
            "llm_unavailable": llm_unavailable,
        }
        return {
            **_case(case_id, answer_numbers=["1500000"]),
            "predicted": {
                "answer": answer,
                "grounding": grounding,
                "llm_runs": [llm_run],
            },
        }

    def test_skipped_budget_excluded_from_numeric(self) -> None:
        cases = [self._merged("s1", skipped_budget=True)]
        m = numeric_grounding(cases)
        assert m.total == 0  # not counted

    def test_skipped_budget_excluded_from_entity(self) -> None:
        cases = [
            self._merged(
                "s1",
                skipped_budget=True,
                grounding={"entities_ok": False},
            )
        ]
        m = entity_grounding(cases)
        assert m.total == 0

    def test_unavailable_excluded_from_numeric(self) -> None:
        cases = [self._merged("u1", llm_unavailable=True)]
        m = numeric_grounding(cases)
        assert m.total == 0

    def test_scored_case_counted(self) -> None:
        cases = [
            self._merged(
                "ok1",
                answer="Price is 1500000 SAR [1].",
                entry_id="groq:m",
                grounding={"numbers_ok": True, "entities_ok": True, "citations_ok": True},
            )
        ]
        m = numeric_grounding(cases)
        assert m.total == 1
        assert m.successes == 1

    def test_llm_case_scored_helper(self) -> None:
        assert not llm_case_scored(self._merged("s1", skipped_budget=True))
        assert not llm_case_scored(self._merged("u1", llm_unavailable=True))
        assert llm_case_scored(
            self._merged("ok1", answer="hello", entry_id="groq:m")
        )

    def test_llm_skip_counts(self) -> None:
        cases = [
            self._merged("s1", skipped_budget=True),
            self._merged("s2", skipped_budget=True),
            self._merged("u1", llm_unavailable=True),
            self._merged(
                "ok1", answer="hello", entry_id="groq:m",
                grounding={"numbers_ok": True, "entities_ok": True, "citations_ok": True},
            ),
        ]
        sc = llm_skip_counts(cases)
        assert sc["llm_skipped_budget"] == 2
        assert sc["llm_unavailable"] == 1


# ---------------------------------------------------------------------------
# Tests: TPMPacer (T0 fix 2)
# ---------------------------------------------------------------------------


class TestTPMPacer:
    def test_no_wait_under_threshold(self) -> None:
        """When window + estimate <= 0.9 * tpm, no wait."""
        clock_time = [0.0]
        pacer = TPMPacer(tpm=8000, _clock=lambda: clock_time[0])
        waited = asyncio.run(pacer.wait_if_needed(1000))
        assert waited == 0.0

    def test_wait_when_over_threshold(self) -> None:
        """Fill the window to 6500 tokens.  Adding 1500 would hit 8000
        which exceeds 0.9 * 8000 = 7200.  The pacer should compute a wait."""
        clock_time = [10.0]

        pacer = TPMPacer(tpm=8000, _clock=lambda: clock_time[0])
        # Record 6500 tokens at t=10
        pacer.record(6500, now=10.0)

        # At t=15, window still has 6500 (oldest is at t=10, cutoff at t=-45)
        clock_time[0] = 15.0
        # 6500 + 1500 = 8000 > 7200 -> needs wait
        # wait until oldest (t=10) leaves window (t=10+60=70)
        # wait_s = 70 - 15 = 55s, but we do not actually sleep in test
        # We test the computed wait by patching asyncio.sleep
        sleeps: list[float] = []

        async def _run() -> float:
            import scripts.evallib.llm_step as mod
            orig_sleep = asyncio.sleep

            async def fake_sleep(s: float) -> None:
                sleeps.append(s)
                # Advance clock past the window so the oldest entry expires
                clock_time[0] += s

            mod.asyncio.sleep = fake_sleep  # type: ignore[attr-defined]
            try:
                return await pacer.wait_if_needed(1500)
            finally:
                mod.asyncio.sleep = orig_sleep  # type: ignore[attr-defined]

        waited = asyncio.run(_run())
        assert waited > 0
        assert len(sleeps) == 1
        assert abs(sleeps[0] - 55.0) < 0.01

    def test_record_and_window_total(self) -> None:
        clock_time = [0.0]
        pacer = TPMPacer(tpm=8000, _clock=lambda: clock_time[0])
        pacer.record(1000, now=0.0)
        pacer.record(2000, now=30.0)
        clock_time[0] = 50.0
        assert pacer.window_total() == 3000
        # Advance past first entry
        clock_time[0] = 61.0
        assert pacer.window_total() == 2000

    def test_prune_old_entries(self) -> None:
        clock_time = [0.0]
        pacer = TPMPacer(tpm=8000, _clock=lambda: clock_time[0])
        pacer.record(5000, now=0.0)
        clock_time[0] = 61.0  # 1 second past the 60s window
        assert pacer.window_total() == 0

    def test_no_wait_when_window_empty(self) -> None:
        clock_time = [100.0]
        pacer = TPMPacer(tpm=8000, _clock=lambda: clock_time[0])
        waited = asyncio.run(pacer.wait_if_needed(7500))
        assert waited == 0.0  # under 7200 threshold? No, 7500 > 7200
        # but window is empty so current=0, 0+7500 > 7200...
        # wait, the check is current + est <= threshold
        # 0 + 7500 = 7500 > 7200, so we need to wait
        # but _entries is empty, so the function returns 0.0
        # (nothing to wait for to expire)
