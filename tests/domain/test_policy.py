"""Tests for policy (cacheability and degradation text)."""

from datetime import UTC, datetime

from app.domain.messages import BUSY_AR, BUSY_EN
from app.domain.policy import degrade_text_retrieval_only, is_cacheable
from common.schemas import (
    AnswerMode,
    Card,
    DocKind,
    FinishReason,
    GroundingReport,
    Language,
    Route,
    Source,
)

# ---------------------------------------------------------------------------
# is_cacheable (caching.md section 5)
# ---------------------------------------------------------------------------


class TestIsCacheable:
    def _grounded(self):
        return GroundingReport(numbers_ok=True, entities_ok=True, citations_ok=True)

    def _ungrounded(self):
        return GroundingReport(numbers_ok=False, entities_ok=True, citations_ok=True, failures=["bad"])

    def test_all_conditions_met(self):
        assert is_cacheable(
            standalone=True,
            route=Route.lookup,
            answer_mode=AnswerMode.llm,
            finish_reason=FinishReason.stop,
            grounding=self._grounded(),
        ) is True

    def test_not_standalone(self):
        assert is_cacheable(
            standalone=False,
            route=Route.lookup,
            answer_mode=AnswerMode.llm,
            finish_reason=FinishReason.stop,
            grounding=self._grounded(),
        ) is False

    def test_oos_route(self):
        assert is_cacheable(
            standalone=True,
            route=Route.out_of_scope,
            answer_mode=AnswerMode.llm,
            finish_reason=FinishReason.stop,
            grounding=self._grounded(),
        ) is False

    def test_retrieval_only(self):
        assert is_cacheable(
            standalone=True,
            route=Route.lookup,
            answer_mode=AnswerMode.retrieval_only,
            finish_reason=FinishReason.no_llm,
            grounding=self._grounded(),
        ) is False

    def test_truncated(self):
        assert is_cacheable(
            standalone=True,
            route=Route.lookup,
            answer_mode=AnswerMode.llm,
            finish_reason=FinishReason.length,
            grounding=self._grounded(),
        ) is False

    def test_ungrounded(self):
        assert is_cacheable(
            standalone=True,
            route=Route.lookup,
            answer_mode=AnswerMode.llm,
            finish_reason=FinishReason.stop,
            grounding=self._ungrounded(),
        ) is False

    def test_aggregate_route(self):
        assert is_cacheable(
            standalone=True,
            route=Route.aggregate,
            answer_mode=AnswerMode.llm,
            finish_reason=FinishReason.stop,
            grounding=self._grounded(),
        ) is True

    def test_project_info_route(self):
        assert is_cacheable(
            standalone=True,
            route=Route.project_info,
            answer_mode=AnswerMode.llm,
            finish_reason=FinishReason.stop,
            grounding=self._grounded(),
        ) is True


# ---------------------------------------------------------------------------
# degrade_text_retrieval_only
# ---------------------------------------------------------------------------


class TestDegradeText:
    def _make_card(self, n: int, title: str, price_sar: int | None = None) -> Card:
        return Card(
            n=n,
            score=0.5,
            snippet="text",
            doc_id=f"wasalt:{n}",
            kind=DocKind.listing,
            source=Source.wasalt,
            title=title,
            url=f"https://wasalt.sa/{n}",
            scraped_at=datetime(2026, 11, 18, tzinfo=UTC),
            price_sar=price_sar,
        )

    def test_with_price(self):
        cards = [self._make_card(1, "Villa in Riyadh", 1_950_000)]
        text = degrade_text_retrieval_only(cards, Language.en)
        assert "[1] Villa in Riyadh" in text
        assert "SAR 1,950,000" in text

    def test_without_price(self):
        cards = [self._make_card(1, "Project Tower", None)]
        text = degrade_text_retrieval_only(cards, Language.en)
        assert "[1] Project Tower" in text
        assert "SAR" not in text

    def test_multiple_cards(self):
        cards = [
            self._make_card(1, "Villa 1", 1_000_000),
            self._make_card(2, "Villa 2", 2_000_000),
        ]
        text = degrade_text_retrieval_only(cards, Language.en)
        lines = text.strip().split("\n")
        assert len(lines) == 2

    def test_empty_cards_en(self):
        text = degrade_text_retrieval_only([], Language.en)
        assert text == BUSY_EN

    def test_empty_cards_ar(self):
        text = degrade_text_retrieval_only([], Language.ar)
        assert text == BUSY_AR
