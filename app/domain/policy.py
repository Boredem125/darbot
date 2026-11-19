"""Cacheability policy, degradation fallback text.

Imports only common.* and app.domain.* (R4).
"""

from __future__ import annotations

from app.domain.messages import BUSY_AR, BUSY_EN
from common.schemas import (
    AnswerMode,
    Card,
    FinishReason,
    GroundingReport,
    Language,
    Route,
)


def is_cacheable(
    *,
    standalone: bool,
    route: Route,
    answer_mode: AnswerMode,
    finish_reason: FinishReason,
    grounding: GroundingReport,
) -> bool:
    """Exactly caching.md section 5: all conditions must hold."""
    if not standalone:
        return False
    if route not in (Route.lookup, Route.aggregate, Route.project_info):
        return False
    if answer_mode != AnswerMode.llm:
        return False
    if finish_reason != FinishReason.stop:
        return False
    return grounding.grounded


def degrade_text_retrieval_only(cards: list[Card], language: Language) -> str:
    """Level 3 templated summary: one line per card, no LLM.

    Format: [n] {title}, SAR 1,950,000
    """
    lines: list[str] = []
    for card in cards:
        line = f"[{card.n}] {card.title}"
        if card.price_sar is not None:
            line += f", SAR {card.price_sar:,}"
        lines.append(line)

    if not lines:
        if language == Language.ar:
            return BUSY_AR
        return BUSY_EN

    return "\n".join(lines)
