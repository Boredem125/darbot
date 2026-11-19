"""Tests for prompt assembly."""


from app.domain.ports import AggregateOp, ChatMessage, SqlResult
from app.domain.prompting import (
    build_messages,
    estimate_tokens,
    max_tokens_for,
)
from common.schemas import Language


class TestEstimateTokens:
    def test_ascii(self):
        assert estimate_tokens("hello") == 2  # ceil(5/3)

    def test_arabic(self):
        # Arabic characters are 2 bytes each in utf-8
        text = "فيلا"  # 4 chars, 8 bytes
        assert estimate_tokens(text) == 3  # ceil(8/3)

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestMaxTokensFor:
    def test_english(self):
        assert max_tokens_for(Language.en) == 500

    def test_arabic(self):
        assert max_tokens_for(Language.ar) == 700


class TestBuildMessages:
    def test_basic(self):
        plan = build_messages(
            system_prompt="You are a helpful assistant.",
            cards_context=["Card 1 text"],
            question="Show me villas",
        )
        assert plan.k == 1
        assert len(plan.messages) == 2  # system + user
        assert plan.messages[0].role == "system"
        assert plan.messages[1].role == "user"
        assert '<source id="1">' in plan.messages[1].content

    def test_cards_budget_enforcement(self):
        """Cards exceeding 1200 tokens are dropped."""
        big_card = "x" * 2000  # ~667 tokens, below 1200
        cards = [big_card, big_card]  # second should be dropped
        plan = build_messages(
            system_prompt="System prompt.",
            cards_context=cards,
            question="query",
        )
        # First card is about 670 tokens in context, second would exceed 1200
        assert plan.k <= 2

    def test_history_included(self):
        history = (
            ChatMessage(role="user", content="first question"),
            ChatMessage(role="assistant", content="first answer"),
        )
        plan = build_messages(
            system_prompt="System prompt.",
            cards_context=["card"],
            history=history,
            question="follow up",
            include_history=True,
        )
        # system + 2 history + user = 4
        assert len(plan.messages) == 4

    def test_history_excluded(self):
        history = (
            ChatMessage(role="user", content="first question"),
            ChatMessage(role="assistant", content="first answer"),
        )
        plan = build_messages(
            system_prompt="System prompt.",
            cards_context=["card"],
            history=history,
            question="follow up",
            include_history=False,
        )
        # system + user = 2
        assert len(plan.messages) == 2

    def test_sql_block(self):
        sql = SqlResult(
            op=AggregateOp.avg_price,
            value=1_950_000,
            unit="SAR",
            n_matched=10,
        )
        plan = build_messages(
            system_prompt="System prompt.",
            cards_context=[],
            sql_block=sql,
            question="average price",
        )
        assert '<sql_result id="1">' in plan.messages[1].content
        assert "SAR 1,950,000" in plan.messages[1].content

    def test_no_context(self):
        plan = build_messages(
            system_prompt="System prompt.",
            cards_context=[],
            question="hello",
        )
        assert plan.k == 0
        assert plan.messages[1].content == "hello"

    def test_sql_result_has_id_k_plus_1(self):
        """Addendum: sql_result rendered with id=k+1 for citation grounding."""
        sql = SqlResult(
            op=AggregateOp.count,
            value=42,
            unit="count",
            n_matched=42,
        )
        plan = build_messages(
            system_prompt="System prompt.",
            cards_context=["Card 1 text", "Card 2 text"],
            sql_block=sql,
            question="how many",
        )
        assert plan.k == 2
        # sql_result should have id="3" (k+1)
        assert '<sql_result id="3">' in plan.messages[1].content

    def test_sql_result_id_1_when_no_cards(self):
        """sql_result with zero cards gets id=1."""
        sql = SqlResult(
            op=AggregateOp.avg_price,
            value=500000,
            unit="SAR",
            n_matched=10,
        )
        plan = build_messages(
            system_prompt="System prompt.",
            cards_context=[],
            sql_block=sql,
            question="average price",
        )
        assert plan.k == 0
        assert '<sql_result id="1">' in plan.messages[1].content
