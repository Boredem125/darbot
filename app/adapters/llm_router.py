"""LLMRouter: provider chain with failover, circuit breaker, quota ledger.

Implements the LLM port (ADR-0005, llm-router.md). One AsyncOpenAI client per
distinct base_url, max_retries=0, explicit timeout. Streams SSE and yields
LLMDelta/LLMFailover/LLMDone items as the port contract requires.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

import httpx
import openai
from openai import AsyncOpenAI

from app.domain.ports import (
    ChatMessage,
    ErrorCode,
    FinishReason,
    LLMDelta,
    LLMDone,
    LLMFailover,
    LLMStatus,
    LLMStreamItem,
    LLMUnavailableError,
    Provider,
    ProviderState,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters (llm-router.md section 8)
# ---------------------------------------------------------------------------

TTFT_TIMEOUT_S = 8.0
CHAIN_FIRST_TOKEN_BUDGET_S = 20.0
STREAM_MAX_S = 60.0
OR_DAILY_LIMIT = 50
OR_RESERVE = 3
OR_POLL_INTERVAL_S = 300
OR_RPM = 20
LLM_MAX_CONCURRENCY = 4

_BASE_COOLDOWN = 60.0
_MAX_COOLDOWN = 900.0

# ---------------------------------------------------------------------------
# CircuitBreaker (per chain entry)
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Per-entry circuit breaker with exponential cooldown and half-open."""

    def __init__(self) -> None:
        self._entries: dict[str, dict] = {}

    def _ensure(self, entry_id: str) -> dict:
        if entry_id not in self._entries:
            self._entries[entry_id] = {
                "cooldown_until": 0.0,
                "multiplier": 1,
                "half_open": False,
                "permanent": False,
            }
        return self._entries[entry_id]

    def fail(
        self,
        entry_id: str,
        code: ErrorCode,
        *,
        base_cooldown: float = _BASE_COOLDOWN,
        permanent: bool = False,
    ) -> None:
        e = self._ensure(entry_id)
        if permanent:
            e["permanent"] = True
            e["cooldown_until"] = float("inf")
            return
        cooldown = min(base_cooldown * e["multiplier"], _MAX_COOLDOWN)
        e["cooldown_until"] = time.monotonic() + cooldown
        e["multiplier"] = min(e["multiplier"] * 2, 16)
        e["half_open"] = False

    def fail_group(
        self,
        entry_ids: list[str],
        code: ErrorCode,
        *,
        base_cooldown: float = _BASE_COOLDOWN,
    ) -> None:
        """Fail all entries in a group (e.g., all openrouter_free on account-level 429)."""
        for eid in entry_ids:
            self.fail(eid, code, base_cooldown=base_cooldown)

    def success(self, entry_id: str) -> None:
        e = self._ensure(entry_id)
        e["cooldown_until"] = 0.0
        e["multiplier"] = 1
        e["half_open"] = False

    def cooldown_remaining(self, entry_id: str) -> float:
        e = self._entries.get(entry_id)
        if e is None:
            return 0.0
        if e["permanent"]:
            return float("inf")
        return max(0.0, e["cooldown_until"] - time.monotonic())

    def is_available(self, entry_id: str) -> bool:
        return self.cooldown_remaining(entry_id) == 0.0

    def try_half_open(self, entry_id: str) -> bool:
        """Try to enter half-open state. Returns True if allowed."""
        e = self._ensure(entry_id)
        if e["half_open"]:
            return False  # another trial already in flight
        if e["permanent"]:
            return False
        e["half_open"] = True
        return True

    def release_half_open(self, entry_id: str) -> None:
        e = self._entries.get(entry_id)
        if e:
            e["half_open"] = False


# ---------------------------------------------------------------------------
# QuotaLedger (OpenRouter 50/day)
# ---------------------------------------------------------------------------


class QuotaLedger:
    """Tracks OpenRouter free-model daily quota."""

    def __init__(self) -> None:
        self._utc_day: str = datetime.now(UTC).strftime("%Y-%m-%d")
        self._local_attempts_today: int = 0
        self._polled_remaining: int | None = None
        self._polled_at: float = 0.0
        self._attempts_since_poll: int = 0
        self._rpm_timestamps: list[float] = []

    def _maybe_rollover(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._utc_day:
            self._utc_day = today
            self._local_attempts_today = 0
            self._polled_remaining = None
            self._polled_at = 0.0
            self._attempts_since_poll = 0

    def record_attempt(self) -> None:
        self._maybe_rollover()
        self._local_attempts_today += 1
        self._attempts_since_poll += 1
        self._rpm_timestamps.append(time.monotonic())

    @property
    def remaining_estimate(self) -> int:
        self._maybe_rollover()
        if self._polled_remaining is not None:
            est = self._polled_remaining - self._attempts_since_poll
        else:
            est = OR_DAILY_LIMIT - self._local_attempts_today
        return max(0, est)

    def update_from_poll(self, remaining: int) -> None:
        self._polled_remaining = remaining
        self._polled_at = time.monotonic()
        self._attempts_since_poll = 0

    def should_poll(self) -> bool:
        if self._polled_at == 0.0:
            return True
        return (time.monotonic() - self._polled_at) >= OR_POLL_INTERVAL_S

    def rpm_ok(self) -> bool:
        """Check if under the RPM limit."""
        now = time.monotonic()
        cutoff = now - 60.0
        self._rpm_timestamps = [t for t in self._rpm_timestamps if t > cutoff]
        return len(self._rpm_timestamps) < OR_RPM


# ---------------------------------------------------------------------------
# LLMRouter
# ---------------------------------------------------------------------------


class LLMRouter:
    """Implements the LLM port: provider chain with failover."""

    def __init__(
        self,
        chain: list[dict],
        *,
        _transport_map: dict[str, httpx.MockTransport] | None = None,
    ) -> None:
        self._chain: list[dict] = []
        self._clients: dict[str, AsyncOpenAI] = {}
        self._breaker = CircuitBreaker()
        self._ledger = QuotaLedger()
        self._semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
        self._ttft_timeout_s = TTFT_TIMEOUT_S
        self._chain_budget_s = CHAIN_FIRST_TOKEN_BUDGET_S
        self._or_free_entries: list[str] = []

        timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

        for entry in chain:
            key_env = entry.get("api_key_env", "")
            api_key = os.environ.get(key_env, "")
            if not api_key:
                logger.warning("Dropping chain entry %s: env %s is empty", entry.get("id"), key_env)
                continue
            self._chain.append(entry)

            base_url = entry["base_url"]
            if base_url not in self._clients:
                kwargs: dict = {
                    "api_key": api_key,
                    "base_url": base_url,
                    "max_retries": 0,
                    "timeout": timeout,
                }
                if _transport_map:
                    # Find matching transport by base_url prefix
                    for prefix, transport in _transport_map.items():
                        if base_url.startswith(prefix):
                            kwargs["http_client"] = httpx.AsyncClient(transport=transport)
                            break
                if entry.get("provider") == "openrouter":
                    kwargs["default_headers"] = {"X-Title": "DG RAG demo"}

                self._clients[base_url] = AsyncOpenAI(**kwargs)

            if entry.get("quota") == "openrouter_free":
                self._or_free_entries.append(entry["id"])

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> AsyncIterator[LLMStreamItem]:
        deadline = time.monotonic() + self._chain_budget_s
        attempts: list[LLMFailover] = []

        # Try to acquire semaphore within budget
        budget_left = deadline - time.monotonic()
        if budget_left <= 0.5:
            raise LLMUnavailableError(attempts)
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=max(0, budget_left))
        except TimeoutError:
            raise LLMUnavailableError(attempts)

        try:
            async for item in self._stream_inner(messages, max_tokens=max_tokens, temperature=temperature, deadline=deadline, attempts=attempts):
                yield item
        finally:
            self._semaphore.release()

    async def _stream_inner(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        deadline: float,
        attempts: list[LLMFailover],
    ) -> AsyncIterator[LLMStreamItem]:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        for entry in self._chain:
            entry_id = entry["id"]
            provider_str = entry["provider"]
            model = entry["model"]
            provider = Provider.openrouter if provider_str == "openrouter" else Provider.groq

            # Gate checks
            skip = self._gate(entry)
            if skip:
                fo = LLMFailover(
                    provider=provider,
                    model=model,
                    code=ErrorCode.provider_skipped,
                    detail=skip,
                )
                attempts.append(fo)
                yield fo
                continue

            budget = min(self._ttft_timeout_s, deadline - time.monotonic())
            if budget <= 0.5:
                break

            # Record attempt for OR entries
            if entry.get("quota") == "openrouter_free":
                self._ledger.record_attempt()

            client = self._clients[entry["base_url"]]
            extra_body = entry.get("extra_body", {})

            t_start = time.monotonic()
            try:
                create_kwargs: dict = {
                    "model": model,
                    "messages": openai_messages,
                    "stream": True,
                    "max_tokens": max_tokens + entry.get("max_tokens_extra", 0),
                    "temperature": temperature,
                    "stream_options": {"include_usage": True},
                    "extra_body": extra_body or None,
                    "timeout": budget,
                }
                if extra_body is None or not extra_body:
                    del create_kwargs["extra_body"]

                raw_stream = await asyncio.wait_for(
                    client.chat.completions.create(**create_kwargs),
                    timeout=budget,
                )
            except (TimeoutError, asyncio.CancelledError):
                self._breaker.fail(entry_id, ErrorCode.provider_timeout)
                fo = LLMFailover(provider=provider, model=model, code=ErrorCode.provider_timeout, detail="ttft timeout")
                attempts.append(fo)
                yield fo
                continue
            except openai.RateLimitError as e:
                self._handle_rate_limit(entry, e)
                fo = LLMFailover(provider=provider, model=model, code=ErrorCode.provider_rate_limited, detail=str(e)[:200])
                attempts.append(fo)
                yield fo
                continue
            except openai.AuthenticationError as e:
                self._breaker.fail(entry_id, ErrorCode.provider_error, base_cooldown=1800)
                fo = LLMFailover(provider=provider, model=model, code=ErrorCode.provider_error, detail=f"auth: {e}"[:200])
                attempts.append(fo)
                yield fo
                continue
            except openai.NotFoundError:
                self._breaker.fail(entry_id, ErrorCode.provider_error, permanent=True)
                fo = LLMFailover(provider=provider, model=model, code=ErrorCode.provider_error, detail="model 404 (permanent)")
                attempts.append(fo)
                yield fo
                continue
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                self._breaker.fail(entry_id, ErrorCode.provider_error)
                fo = LLMFailover(provider=provider, model=model, code=ErrorCode.provider_error, detail=str(e)[:200])
                attempts.append(fo)
                yield fo
                continue
            except openai.APIStatusError as e:
                code = ErrorCode.provider_error
                if e.status_code == 429:
                    code = ErrorCode.provider_rate_limited
                    self._handle_rate_limit(entry, e)
                elif e.status_code in (401, 403):
                    self._breaker.fail(entry_id, code, base_cooldown=1800)
                elif e.status_code == 402:
                    # Credits exhausted: cooldown until next UTC midnight
                    secs_to_midnight = _seconds_to_midnight()
                    self._breaker.fail(entry_id, code, base_cooldown=secs_to_midnight)
                elif e.status_code == 404:
                    self._breaker.fail(entry_id, code, permanent=True)
                else:
                    self._breaker.fail(entry_id, code)
                fo = LLMFailover(provider=provider, model=model, code=code, detail=str(e)[:200])
                attempts.append(fo)
                yield fo
                continue

            # Stream tokens
            first_token_seen = False
            ttft_ms = 0.0
            total_text = ""
            usage: TokenUsage | None = None
            finish_reason = FinishReason.stop
            stream_deadline = None

            try:
                async for chunk in raw_stream:
                    if not chunk.choices:
                        # Usage-only chunk at the end
                        if hasattr(chunk, "usage") and chunk.usage:
                            usage = TokenUsage(
                                prompt_tokens=chunk.usage.prompt_tokens or 0,
                                completion_tokens=chunk.usage.completion_tokens or 0,
                            )
                        continue

                    choice = chunk.choices[0]
                    delta = choice.delta

                    # Check for finish_reason
                    if choice.finish_reason:
                        fr = choice.finish_reason
                        if fr == "stop":
                            finish_reason = FinishReason.stop
                        elif fr == "length":
                            finish_reason = FinishReason.length
                        elif fr == "content_filter":
                            finish_reason = FinishReason.content_filter
                        else:
                            finish_reason = FinishReason.stop if total_text else FinishReason.error

                    # Check for usage in this chunk
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            completion_tokens=chunk.usage.completion_tokens or 0,
                        )

                    # Content delta
                    content = getattr(delta, "content", None) if delta else None
                    if content:
                        if not first_token_seen:
                            first_token_seen = True
                            ttft_ms = (time.monotonic() - t_start) * 1000.0
                            stream_deadline = time.monotonic() + STREAM_MAX_S
                            self._breaker.success(entry_id)

                        total_text += content
                        yield LLMDelta(text=content)

                    # Check stream timeout
                    if stream_deadline and time.monotonic() > stream_deadline:
                        finish_reason = FinishReason.error
                        break

            except (openai.APIError, httpx.HTTPError, ValueError, KeyError) as e:
                if first_token_seen:
                    # Error after first token: no failover
                    finish_reason = FinishReason.error
                    logger.warning("Stream error after first token on %s: %s", entry_id, e)
                else:
                    # Error before first token: failover
                    self._breaker.fail(entry_id, ErrorCode.provider_error)
                    fo = LLMFailover(provider=provider, model=model, code=ErrorCode.provider_error, detail=str(e)[:200])
                    attempts.append(fo)
                    yield fo
                    continue

            if not first_token_seen:
                # No content produced at all (e.g., empty stream)
                self._breaker.fail(entry_id, ErrorCode.provider_timeout)
                fo = LLMFailover(provider=provider, model=model, code=ErrorCode.provider_timeout, detail="no content in stream")
                attempts.append(fo)
                yield fo
                continue

            total_ms = (time.monotonic() - t_start) * 1000.0

            # Estimate usage if not provided
            if usage is None:
                msg_text = " ".join(m.content for m in messages)
                est_prompt = math.ceil(len(msg_text.encode("utf-8")) / 3)
                est_completion = math.ceil(len(total_text.encode("utf-8")) / 3)
                usage = TokenUsage(prompt_tokens=est_prompt, completion_tokens=est_completion)

            yield LLMDone(
                provider=provider,
                model=model,
                finish_reason=finish_reason,
                usage=usage,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
            )
            return

        # All entries exhausted
        raise LLMUnavailableError(attempts)

    def _gate(self, entry: dict) -> str | None:
        """Check if entry should be skipped. Returns reason or None."""
        entry_id = entry["id"]

        # Breaker check
        if not self._breaker.is_available(entry_id):
            return "breaker_cooldown"

        # OpenRouter quota guard
        if entry.get("quota") == "openrouter_free":
            if self._ledger.remaining_estimate <= OR_RESERVE:
                return "quota_guard"
            if not self._ledger.rpm_ok():
                return "rpm_guard"

        return None

    def _handle_rate_limit(self, entry: dict, exc: Exception) -> None:
        """Handle a 429 from either provider."""
        entry_id = entry["id"]

        # Check if it's an account-level OpenRouter 429
        is_account_level = False
        if entry.get("provider") == "openrouter":
            # Check if provider_code is absent (account-level)
            err_body = getattr(exc, "body", None)
            if isinstance(err_body, dict):
                metadata = err_body.get("error", {}).get("metadata", {})
                if not metadata.get("provider_code"):
                    is_account_level = True
            else:
                # Assume account-level if we can't parse
                is_account_level = True

        if is_account_level and entry.get("quota") == "openrouter_free":
            # Cool down ALL openrouter_free entries
            self._breaker.fail_group(self._or_free_entries, ErrorCode.provider_rate_limited)
        else:
            self._breaker.fail(entry_id, ErrorCode.provider_rate_limited)

    def status(self) -> LLMStatus:
        providers: list[ProviderState] = []
        for entry in self._chain:
            entry_id = entry["id"]
            provider_str = entry["provider"]
            provider = Provider.openrouter if provider_str == "openrouter" else Provider.groq
            cd = self._breaker.cooldown_remaining(entry_id)
            providers.append(ProviderState(
                provider=provider,
                model=entry["model"],
                available=cd == 0.0,
                cooldown_remaining_s=cd,
            ))
        return LLMStatus(
            openrouter_quota_remaining=self._ledger._polled_remaining,
            providers=tuple(providers),
        )


def _seconds_to_midnight() -> float:
    now = datetime.now(UTC)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    midnight = midnight.replace(day=now.day + 1) if now.hour > 0 or now.minute > 0 else midnight
    return max(60.0, (midnight - now).total_seconds())
