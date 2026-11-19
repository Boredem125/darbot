"""LLM generation step for the eval harness.

Runs the LLM generation on cases that have ``llm=true`` and a non-refusal
route.  Handles budget tracking, quota logging, re-run on grounding failure,
and case filtering.

New in WP-3-harness-llm Phase A.  Does not modify ``eval.py`` or
``pipeline.py``; integration happens in Phase B.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain import grounding, prompting
from app.domain.ports import (
    LLMDelta,
    LLMDone,
    LLMFailover,
    LLMUnavailableError,
    SqlResult,
)
from common.cards import render_card
from common.schemas import Card, Language

# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def _card_dict_to_card(card: dict[str, Any]) -> Card:
    """Convert an eval card dict to a ``Card`` for the shared renderer.

    Eval pipeline's ``_card_to_dict`` produces card dicts via
    ``Card.model_dump(exclude_none=True)``.  Here we reconstruct a ``Card``
    from that dict, filling in any required fields that may be missing
    (such as ``scraped_at``) with safe defaults.
    """
    # Ensure required fields have defaults for dict-to-Card round-trip
    defaults: dict[str, Any] = {}
    if "scraped_at" not in card:
        defaults["scraped_at"] = "2024-01-01T00:00:00Z"
    if "n" not in card:
        defaults["n"] = 1
    return Card.model_validate({**defaults, **card})


def render_card_text(card: dict[str, Any]) -> str:
    """Render a card dict to context-mode text using the shared renderer.

    Uses ``common.cards.render_card(doc, mode="context")`` -- the same
    renderer that ``ChatService`` uses -- so the description/snippet
    reaches the LLM prompt (H-2 fix).
    """
    card_obj = _card_dict_to_card(card)
    return render_card(card_obj, mode="context", n=card_obj.n, snippet=card_obj.snippet)


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    """Tracks LLM attempt budget.

    An *attempt* is any HTTP request to an LLM provider: each ``LLMFailover``,
    each ``LLMUnavailableError`` attempt, and the final ``LLMDone``.
    """

    limit: int
    used: int = 0
    skipped_count: int = 0

    def can_attempt(self, chain_length: int) -> bool:
        """Return True when the worst case ``used + chain_length <= limit``."""
        if self.limit <= 0:
            return False
        return self.used + chain_length <= self.limit

    def record_attempt(self) -> None:
        """Record one HTTP attempt."""
        self.used += 1


# ---------------------------------------------------------------------------
# TPM pacer (Groq per-entry throttle)
# ---------------------------------------------------------------------------


@dataclass
class TPMPacer:
    """Token-per-minute sliding-window pacer for Groq entries.

    Keeps a 60 s window of tokens consumed.  Before a request, if
    ``window_total + estimate > 0.9 * tpm``, sleeps exactly until the
    oldest entry leaves the window (one computed sleep, no polling).
    Skipped when *tpm* is absent (OpenRouter entries).
    """

    tpm: int
    window_s: float = 60.0
    _entries: list[tuple[float, int]] = field(default_factory=list)
    _clock: Callable[[], float] = field(default_factory=lambda: time.monotonic)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        self._entries = [(t, tok) for t, tok in self._entries if t > cutoff]

    def window_total(self, now: float | None = None) -> int:
        """Current token count inside the window."""
        now = now if now is not None else self._clock()
        self._prune(now)
        return sum(tok for _, tok in self._entries)

    async def wait_if_needed(self, est_tokens: int) -> float:
        """Wait if adding *est_tokens* would exceed 0.9 * tpm.

        Returns the seconds actually waited (0.0 when no wait needed).
        """
        now = self._clock()
        self._prune(now)
        threshold = 0.9 * self.tpm
        current = sum(tok for _, tok in self._entries)
        if current + est_tokens <= threshold:
            return 0.0
        if not self._entries:
            return 0.0
        oldest_t = self._entries[0][0]
        wait_s = (oldest_t + self.window_s) - now
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        return max(wait_s, 0.0)

    def record(self, tokens: int, now: float | None = None) -> None:
        """Record *tokens* consumed at *now*."""
        now = now if now is not None else self._clock()
        self._entries.append((now, tokens))


# ---------------------------------------------------------------------------
# Quota logger
# ---------------------------------------------------------------------------

QUOTA_HEADER = "ts_utc,provider,model,purpose,wp,ok,prompt_tokens,completion_tokens\n"


class QuotaLogger:
    """Appends one row per LLM attempt to a CSV file.

    Never logs keys or prompt text (PLAN section 9).
    """

    def __init__(self, csv_path: Path) -> None:
        self._path = csv_path
        self._ensure_header()

    def _ensure_header(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists() or self._path.stat().st_size == 0:
            self._path.write_text(QUOTA_HEADER, encoding="utf-8")

    def log(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        wp: str,
        ok: bool,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Append one row."""
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now(UTC).isoformat(),
                    provider,
                    model,
                    purpose,
                    wp,
                    "1" if ok else "0",
                    prompt_tokens,
                    completion_tokens,
                ]
            )


# ---------------------------------------------------------------------------
# LLM case result
# ---------------------------------------------------------------------------


@dataclass
class LLMCaseResult:
    """Per-case LLM generation result."""

    answer: str = ""
    grounding: dict[str, Any] | None = None
    entry_id: str | None = None
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_prompt_tokens: int = 0
    attempts: list[dict[str, str]] = field(default_factory=list)
    skipped_budget: bool = False
    llm_unavailable: bool = False
    run_index: int = 0  # 0 = first run, 1 = rerun

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the JSON report."""
        return {
            "answer": self.answer,
            "grounding": self.grounding,
            "entry_id": self.entry_id,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "est_prompt_tokens": self.est_prompt_tokens,
            "attempts": self.attempts,
            "skipped_budget": self.skipped_budget,
            "llm_unavailable": self.llm_unavailable,
            "run_index": self.run_index,
        }


# ---------------------------------------------------------------------------
# LLM step execution
# ---------------------------------------------------------------------------


def _provider_str(obj: Any) -> str:
    return obj.value if hasattr(obj, "value") else str(obj)


def _finish_ok(done: LLMDone) -> bool:
    fr = done.finish_reason
    val = fr.value if hasattr(fr, "value") else str(fr)
    return val != "error"


async def run_llm_step(
    *,
    case: dict[str, Any],
    predicted: dict[str, Any],
    llm: Any,
    system_prompt: str,
    entities: Any,
    budget: BudgetTracker,
    chain_length: int = 1,
    quota_logger: QuotaLogger | None = None,
    purpose: str = "geneval",
    wp: str = "",
    sql_result: SqlResult | None = None,
    run_index: int = 0,
    pacer: TPMPacer | None = None,
) -> LLMCaseResult:
    """Run the LLM generation step for one eval case.

    Takes the pipeline-produced *predicted* dict (with route, cards,
    refused, etc.) and adds LLM generation, grounding check, and
    attempt tracking.

    Returns an ``LLMCaseResult``; never raises for LLM failures.
    """
    result = LLMCaseResult(run_index=run_index)

    # --- Budget gate ---
    if not budget.can_attempt(chain_length):
        result.skipped_budget = True
        budget.skipped_count += 1
        return result

    # --- Skip if already refused (out_of_scope, score-floor) ---
    if predicted.get("refused"):
        return result

    # --- Language and max tokens ---
    lang_str = case.get("lang", "en")
    language = Language(lang_str) if lang_str in ("en", "ar") else Language.en
    max_tokens = prompting.max_tokens_for(language)

    # --- Build card context texts ---
    cards: list[dict[str, Any]] = predicted.get("cards", [])
    cards_context = [render_card_text(c) for c in cards]

    # --- Build the prompt ---
    question: str = (case.get("turns") or [""])[-1]
    plan = prompting.build_messages(
        system_prompt=system_prompt,
        cards_context=cards_context,
        sql_block=sql_result,
        question=question,
    )
    result.est_prompt_tokens = plan.est_prompt_tokens

    # --- TPM pacing ---
    if pacer is not None:
        await pacer.wait_if_needed(result.est_prompt_tokens + max_tokens)

    # --- Stream ---
    failovers: list[LLMFailover] = []
    answer_parts: list[str] = []
    done: LLMDone | None = None

    try:
        async with contextlib.aclosing(
            llm.stream(plan.messages, max_tokens=max_tokens, temperature=0.0)
        ) as stream:
            async for item in stream:
                if isinstance(item, LLMFailover):
                    failovers.append(item)
                    # H-4d: breaker-skipped attempts (provider_skipped)
                    # never reach the network; do not log them to the
                    # quota CSV, and do not count them against the budget.
                    _code = _provider_str(item.code)
                    is_breaker_skip = _code in (
                        "provider_skipped",
                        "provider_skipped:breaker_cooldown",
                    )
                    if not is_breaker_skip:
                        budget.record_attempt()
                        if quota_logger:
                            quota_logger.log(
                                provider=_provider_str(item.provider),
                                model=item.model,
                                purpose=purpose,
                                wp=wp,
                                ok=False,
                                prompt_tokens=0,
                                completion_tokens=0,
                            )
                elif isinstance(item, LLMDelta):
                    answer_parts.append(item.text)
                elif isinstance(item, LLMDone):
                    done = item
                    budget.record_attempt()
                    if quota_logger:
                        quota_logger.log(
                            provider=_provider_str(done.provider),
                            model=done.model,
                            purpose=purpose,
                            wp=wp,
                            ok=_finish_ok(done),
                            prompt_tokens=done.usage.prompt_tokens,
                            completion_tokens=done.usage.completion_tokens,
                        )
    except LLMUnavailableError:
        # All providers failed.  The failover items were already yielded
        # and counted above, so nothing extra to record here.
        pass

    # --- Populate result from stream items ---
    result.answer = "".join(answer_parts)
    result.attempts = [
        {
            "provider": _provider_str(f.provider),
            "model": f.model,
            "code": _provider_str(f.code),
            "detail": f.detail,
        }
        for f in failovers
    ]

    if done is not None:
        result.entry_id = f"{_provider_str(done.provider)}:{done.model}"
        result.ttft_ms = done.ttft_ms
        result.total_ms = done.total_ms
        result.finish_reason = _provider_str(done.finish_reason)
        result.prompt_tokens = done.usage.prompt_tokens
        result.completion_tokens = done.usage.completion_tokens
        # Record actual tokens in the pacer window
        if pacer is not None:
            pacer.record(done.usage.prompt_tokens + done.usage.completion_tokens)
    elif failovers and not result.answer:
        # All providers failed before producing an answer
        result.llm_unavailable = True

    # --- Grounding check ---
    if result.answer:
        context_texts = cards_context[: plan.k]
        grounding_report = grounding.check(
            result.answer,
            context_texts=context_texts,
            sql=sql_result,
            query=question,
            entities=entities,
            k=plan.k,
            sql_n_matched=sql_result.n_matched if sql_result is not None else None,
        )
        result.grounding = grounding_report.model_dump()

    return result


# ---------------------------------------------------------------------------
# Re-run rule
# ---------------------------------------------------------------------------


def should_rerun(llm_result: LLMCaseResult, case: dict[str, Any]) -> bool:
    """Return True when a case should be re-run.

    Triggers:
    - Any grounding sub-check is False.
    - Any expected ``answer_numbers`` is missing from the answer.
    """
    if llm_result.skipped_budget:
        return False
    if not llm_result.answer:
        return False

    # Grounding failure
    if llm_result.grounding:
        gr = llm_result.grounding
        if (
            gr.get("numbers_ok") is False
            or gr.get("entities_ok") is False
            or gr.get("citations_ok") is False
        ):
            return True

    # Expected answer numbers missing
    expected_numbers: list[str] = case.get("expect", {}).get("answer_numbers", [])
    if expected_numbers:
        from common.text import canonical_number, parse_numbers

        answer_canons = {canonical_number(n) for n in parse_numbers(llm_result.answer)}
        for expected in expected_numbers:
            if str(expected) not in answer_canons:
                return True

    return False


# ---------------------------------------------------------------------------
# Case filtering
# ---------------------------------------------------------------------------


# H-4b: Accept common aliases for set names so that both
# ``--sets injection`` and ``--sets injection_fixtures`` work.
_SET_ALIASES: dict[str, str] = {
    "injection_fixtures": "injection",
}


def filter_cases(
    cases: list[dict[str, Any]],
    *,
    case_ids: list[str] | None = None,
    sets: list[str] | None = None,
    kind: str | None = None,
    llm_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter eval cases by id, set name, expect.kind, or llm flag."""
    result = cases
    if case_ids is not None:
        id_set = set(case_ids)
        result = [c for c in result if c.get("id") in id_set]
    if sets is not None:
        # Expand aliases (H-4b): e.g. "injection_fixtures" also matches "injection"
        set_set: set[str] = set()
        for s in sets:
            set_set.add(s)
            if s in _SET_ALIASES:
                set_set.add(_SET_ALIASES[s])
        result = [c for c in result if c.get("set") in set_set]
    if kind is not None:
        result = [c for c in result if c.get("expect", {}).get("kind") == kind]
    if llm_only:
        result = [c for c in result if c.get("llm")]
    return result


# ---------------------------------------------------------------------------
# Plan summary
# ---------------------------------------------------------------------------


def llm_case_scored(c: dict[str, Any]) -> bool:
    """Return True when the LLM produced an answer that should be graded.

    Excludes budget-skipped, unavailable, and refused cases so that
    ``numeric_grounding`` and ``entity_grounding`` only count genuine
    LLM outputs.  Backward-compatible with results that have no
    ``llm_runs`` (pre-llm_step eval).
    """
    pred = c.get("predicted", {})
    llm_runs = pred.get("llm_runs", [])
    if not llm_runs:
        # No llm_runs: backward compat -- check if there is an answer
        return bool(pred.get("answer"))
    first = llm_runs[0]
    if first.get("skipped_budget"):
        return False
    if first.get("llm_unavailable"):
        return False
    return bool(first.get("answer"))


def llm_skip_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    """Count how many LLM cases were skipped or unavailable."""
    skipped = 0
    unavailable = 0
    for c in cases:
        if not c.get("llm"):
            continue
        pred = c.get("predicted", {})
        runs = pred.get("llm_runs", [])
        if not runs:
            # LLM step never ran but case wanted it (budget 0, no LLM built)
            skipped += 1
            continue
        first = runs[0]
        if first.get("skipped_budget"):
            skipped += 1
        elif first.get("llm_unavailable"):
            unavailable += 1
    return {"llm_skipped_budget": skipped, "llm_unavailable": unavailable}


def plan_summary(
    cases: list[dict[str, Any]],
    chain_length: int,
) -> dict[str, int]:
    """Count LLM cases and worst-case attempts.

    Used by ``--plan`` to preview what a run would do with zero calls.
    """
    llm_cases = [c for c in cases if c.get("llm")]
    non_refused = [
        c for c in llm_cases if not c.get("expect", {}).get("refuse", False)
    ]
    return {
        "total_cases": len(cases),
        "llm_cases": len(llm_cases),
        "llm_non_refused": len(non_refused),
        "worst_case_attempts": len(non_refused) * chain_length,
        "chain_length": chain_length,
    }
