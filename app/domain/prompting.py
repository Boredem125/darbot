"""Prompt assembly and token estimation.

Imports only common.* and app.domain.* (R4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.domain.ports import ChatMessage, SqlResult
from common.schemas import Language

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Conservative token estimate: ceil(utf8_bytes / 3)."""
    return math.ceil(len(text.encode("utf-8")) / 3)


def max_tokens_for(language: Language) -> int:
    """Max completion tokens by language (llm-router.md section 2)."""
    if language == Language.ar:
        return 700
    return 500


# ---------------------------------------------------------------------------
# PromptPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PromptPlan:
    messages: tuple[ChatMessage, ...]
    k: int  # number of cards kept
    est_prompt_tokens: int


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_card(n: int, text: str) -> str:
    """Wrap a card as <source id="n">...</source>."""
    return f'<source id="{n}">{text}</source>'


def _format_sql(sql_result: SqlResult, *, sql_id: int) -> str:
    """Wrap an SqlResult as <sql_result id="sql_id">...</sql_result>.

    sql_id is k+1 (number of cards + 1), so the LLM can cite it as [k+1].
    """
    if sql_result.value is None:
        value_str = "no results"
    elif sql_result.unit == "SAR":
        # Format as SAR with comma separators
        value_str = f"SAR {sql_result.value:,}"
    elif sql_result.unit == "sqm":
        value_str = f"{sql_result.value} sqm"
    elif sql_result.unit == "count":
        value_str = str(sql_result.value)
    else:
        value_str = str(sql_result.value)

    lines = [
        f"operation: {sql_result.op.value}",
        f"value: {value_str}",
        f"matched: {sql_result.n_matched}",
    ]
    return f'<sql_result id="{sql_id}">' + "\n".join(lines) + "</sql_result>"


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


MAX_CONTEXT_TOKENS = 1200
MAX_HISTORY_TOKENS = 300


def build_messages(
    *,
    system_prompt: str,
    cards_context: list[str],
    sql_block: SqlResult | None = None,
    history: tuple[ChatMessage, ...] = (),
    question: str,
    include_history: bool = False,
    language: Language | None = None,
) -> PromptPlan:
    """Assemble the prompt messages within token budgets.

    Cards are kept whole in rank order while total <= 1200 estimated tokens.
    Drop from the bottom, never truncate inside a card.
    History included only when include_history is True, at most 300 tokens,
    dropping the oldest turn first.
    """
    # Build context: cards in rank order within budget
    kept_cards: list[str] = []
    total_card_tokens = 0
    for i, card_text in enumerate(cards_context):
        wrapped = _format_card(i + 1, card_text)
        card_tokens = estimate_tokens(wrapped)
        if total_card_tokens + card_tokens > MAX_CONTEXT_TOKENS:
            break
        kept_cards.append(wrapped)
        total_card_tokens += card_tokens

    k = len(kept_cards)

    # Build context string
    context_parts: list[str] = kept_cards
    if sql_block is not None:
        context_parts.append(_format_sql(sql_block, sql_id=k + 1))

    context_text = "\n\n".join(context_parts) if context_parts else ""

    # Build messages list
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
    ]

    # History (if included), dropping oldest first to stay within budget
    if include_history and history:
        history_messages: list[ChatMessage] = list(history)
        total_hist_tokens = sum(estimate_tokens(m.content) for m in history_messages)
        while total_hist_tokens > MAX_HISTORY_TOKENS and history_messages:
            removed = history_messages.pop(0)
            total_hist_tokens -= estimate_tokens(removed.content)
        messages.extend(history_messages)

    # User message with context
    if context_text:
        user_content = f"{context_text}\n\nQuestion: {question}"
    else:
        user_content = question

    # Explicit language instruction (router's decision, not the system prompt)
    if language == Language.ar:
        user_content += "\nأجب باللغة العربية."
    elif language == Language.en:
        user_content += "\nAnswer in English."

    messages.append(ChatMessage(role="user", content=user_content))

    # Estimate total prompt tokens
    est = sum(estimate_tokens(m.content) for m in messages)

    return PromptPlan(
        messages=tuple(messages),
        k=k,
        est_prompt_tokens=est,
    )
