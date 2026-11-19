"""Tests for LLMRouter adapter (llm-router.md section 9).

All tests use httpx MockTransport: no keys, no network.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from app.domain.ports import (
    ChatMessage,
    ErrorCode,
    FinishReason,
    LLMDelta,
    LLMDone,
    LLMFailover,
    LLMStreamItem,
    LLMUnavailableError,
    Provider,
)

# ---------------------------------------------------------------------------
# Helpers: mock SSE responses
# ---------------------------------------------------------------------------


def _sse_chunk(content: str | None = None, finish_reason: str | None = None, usage=None, reasoning: str | None = None) -> str:
    """Build one SSE data line mimicking the OpenAI format."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    choice: dict = {"index": 0, "delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    obj: dict = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "choices": [choice]}
    if usage is not None:
        obj["usage"] = usage
    return f"data: {json.dumps(obj)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _sse_error(message: str = "rate limited", code: int = 429) -> str:
    obj = {"error": {"message": message, "code": code}}
    return f"data: {json.dumps(obj)}\n\n"


def _make_sse_response(chunks: list[str], status: int = 200) -> httpx.Response:
    """Build an httpx.Response with SSE content."""
    body = "".join(chunks)
    return httpx.Response(
        status_code=status,
        headers={"content-type": "text/event-stream"},
        stream=httpx.ByteStream(body.encode()),
    )


# Chain configs for tests
_CHAIN_OR = {
    "id": "or-test1",
    "provider": "openrouter",
    "model": "test/model1:free",
    "base_url": "https://openrouter.test/api/v1",
    "api_key_env": "TEST_OR_KEY",
    "quota": "openrouter_free",
    "extra_body": {},
}

_CHAIN_OR2 = {
    "id": "or-test2",
    "provider": "openrouter",
    "model": "test/model2:free",
    "base_url": "https://openrouter.test/api/v1",
    "api_key_env": "TEST_OR_KEY",
    "quota": "openrouter_free",
    "extra_body": {},
}

_CHAIN_GROQ = {
    "id": "groq-test",
    "provider": "groq",
    "model": "test/groq-model",
    "base_url": "https://api.groq.test/openai/v1",
    "api_key_env": "TEST_GROQ_KEY",
    "quota": None,
    "tpm": 8000,
    "extra_body": {},
}

_MESSAGES = [ChatMessage(role="user", content="Hello")]


def _build_router(chain: list[dict], transport_map: dict[str, httpx.MockTransport] | None = None):
    """Build an LLMRouter with mock transports."""
    from app.adapters.llm_router import LLMRouter

    env = {"TEST_OR_KEY": "sk-test-or", "TEST_GROQ_KEY": "sk-test-groq"}
    with patch.dict("os.environ", env):
        router = LLMRouter(chain, _transport_map=transport_map)
    return router


# ---------------------------------------------------------------------------
# Test 1: 429 on entry 1 -> failover to entry 2
# ---------------------------------------------------------------------------


class TestT1_429Failover:
    """429 on entry 1 gives LLMFailover(provider_rate_limited), then tokens from entry 2.
    Entry 1 is not retried, and all OpenRouter entries are in cooldown."""

    @pytest.mark.asyncio
    async def test_429_failover(self):
        call_count = {"or": 0, "groq": 0}

        def or_handler(request: httpx.Request) -> httpx.Response:
            call_count["or"] += 1
            return httpx.Response(
                status_code=429,
                headers={"content-type": "application/json"},
                json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
            )

        def groq_handler(request: httpx.Request) -> httpx.Response:
            call_count["groq"] += 1
            body = (
                _sse_chunk(content="Hello ")
                + _sse_chunk(content="world", finish_reason="stop",
                             usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
                + _sse_done()
            )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        transports = {
            "https://openrouter.test": httpx.MockTransport(or_handler),
            "https://api.groq.test": httpx.MockTransport(groq_handler),
        }
        router = _build_router([_CHAIN_OR, _CHAIN_GROQ], transports)

        items: list[LLMStreamItem] = []
        async for item in router.stream(_MESSAGES, max_tokens=100):
            items.append(item)

        # First item should be a failover from OR
        failovers = [i for i in items if isinstance(i, LLMFailover)]
        assert len(failovers) >= 1
        assert failovers[0].code == ErrorCode.provider_rate_limited

        # Should have deltas from groq
        deltas = [i for i in items if isinstance(i, LLMDelta)]
        assert len(deltas) >= 1
        assert deltas[0].text == "Hello "

        # Should end with LLMDone
        done = [i for i in items if isinstance(i, LLMDone)]
        assert len(done) == 1
        assert done[0].finish_reason == FinishReason.stop

        # OR entry should not be retried (only 1 call)
        assert call_count["or"] == 1

        # All OR entries should be in cooldown
        status = router.status()
        for ps in status.providers:
            if ps.provider == Provider.openrouter:
                assert not ps.available or ps.cooldown_remaining_s > 0


# ---------------------------------------------------------------------------
# Test 2: SSE error before/after first token
# ---------------------------------------------------------------------------


class TestT2_SSEError:
    """SSE error before first content -> failover. Error after first token -> LLMDone(error)."""

    @pytest.mark.asyncio
    async def test_error_before_first_token_failover(self):
        def or_handler(request: httpx.Request) -> httpx.Response:
            # SSE error before any content
            body = _sse_error("model unavailable") + _sse_done()
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        def groq_handler(request: httpx.Request) -> httpx.Response:
            body = (
                _sse_chunk(content="OK")
                + _sse_chunk(finish_reason="stop",
                             usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6})
                + _sse_done()
            )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        transports = {
            "https://openrouter.test": httpx.MockTransport(or_handler),
            "https://api.groq.test": httpx.MockTransport(groq_handler),
        }
        router = _build_router([_CHAIN_OR, _CHAIN_GROQ], transports)

        items: list[LLMStreamItem] = []
        async for item in router.stream(_MESSAGES, max_tokens=100):
            items.append(item)

        failovers = [i for i in items if isinstance(i, LLMFailover)]
        assert len(failovers) >= 1

        deltas = [i for i in items if isinstance(i, LLMDelta)]
        assert any(d.text == "OK" for d in deltas)

    @pytest.mark.asyncio
    async def test_error_after_first_token_no_failover(self):
        """After the first content delta, an error yields LLMDone(error) with no failover."""

        def or_handler(request: httpx.Request) -> httpx.Response:
            # First token, then error (simulated by abrupt end without finish_reason)
            body = _sse_chunk(content="Hi")
            # Then a malformed chunk that will cause an error
            body += "data: {invalid json}\n\n"
            body += _sse_done()
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        transports = {
            "https://openrouter.test": httpx.MockTransport(or_handler),
        }
        router = _build_router([_CHAIN_OR], transports)

        items: list[LLMStreamItem] = []
        async for item in router.stream(_MESSAGES, max_tokens=100):
            items.append(item)

        deltas = [i for i in items if isinstance(i, LLMDelta)]
        assert any(d.text == "Hi" for d in deltas)

        done = [i for i in items if isinstance(i, LLMDone)]
        assert len(done) == 1
        assert done[0].finish_reason == FinishReason.error


# ---------------------------------------------------------------------------
# Test 3: Timeout before first token
# ---------------------------------------------------------------------------


class TestT3_Timeout:
    """No first token within budget -> provider_timeout."""

    @pytest.mark.asyncio
    async def test_timeout_failover(self):
        """An entry that returns no first content delta -> provider_timeout failover."""

        def empty_handler(request: httpx.Request) -> httpx.Response:
            # SSE stream with only a done marker, no content
            body = _sse_done()
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        def fast_handler(request: httpx.Request) -> httpx.Response:
            body = (
                _sse_chunk(content="Fast")
                + _sse_chunk(finish_reason="stop",
                             usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6})
                + _sse_done()
            )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        transports = {
            "https://openrouter.test": httpx.MockTransport(empty_handler),
            "https://api.groq.test": httpx.MockTransport(fast_handler),
        }
        router = _build_router([_CHAIN_OR, _CHAIN_GROQ], transports)

        items: list[LLMStreamItem] = []
        async for item in router.stream(_MESSAGES, max_tokens=100):
            items.append(item)

        failovers = [i for i in items if isinstance(i, LLMFailover)]
        assert any(f.code == ErrorCode.provider_timeout for f in failovers)
        # Groq should succeed
        deltas = [i for i in items if isinstance(i, LLMDelta)]
        assert any(d.text == "Fast" for d in deltas)


# ---------------------------------------------------------------------------
# Test 4: Ledger quota guard
# ---------------------------------------------------------------------------


class TestT4_QuotaLedger:
    """Ledger counts attempts; guard at <=3 gives provider_skipped/quota_guard; day rollover resets."""

    @pytest.mark.asyncio
    async def test_quota_guard_skips(self):
        def or_handler(request: httpx.Request) -> httpx.Response:
            body = (
                _sse_chunk(content="Hi")
                + _sse_chunk(finish_reason="stop",
                             usage={"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6})
                + _sse_done()
            )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        transports = {
            "https://openrouter.test": httpx.MockTransport(or_handler),
        }
        router = _build_router([_CHAIN_OR], transports)

        # Simulate having used 47 of 50 (remaining=3, guard threshold)
        router._ledger._local_attempts_today = 47
        router._ledger._polled_remaining = None  # no poll

        items: list[LLMStreamItem] = []
        try:
            async for item in router.stream(_MESSAGES, max_tokens=100):
                items.append(item)
        except LLMUnavailableError:
            pass

        failovers = [i for i in items if isinstance(i, LLMFailover)]
        assert any(f.code == ErrorCode.provider_skipped and "quota_guard" in f.detail for f in failovers)

    @pytest.mark.asyncio
    async def test_day_rollover_resets(self):
        from app.adapters.llm_router import QuotaLedger

        ledger = QuotaLedger()
        ledger._local_attempts_today = 50
        ledger._utc_day = "2026-11-17"  # yesterday

        # Calling remaining_estimate should trigger day rollover
        est = ledger.remaining_estimate
        assert est == 50  # reset to full


# ---------------------------------------------------------------------------
# Test 5: All entries fail -> LLMUnavailableError
# ---------------------------------------------------------------------------


class TestT5_AllFail:
    """Every entry failing raises LLMUnavailableError with every attempt."""

    @pytest.mark.asyncio
    async def test_all_fail(self):
        def fail_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                status_code=500,
                headers={"content-type": "application/json"},
                json={"error": {"message": "internal error"}},
            )

        transports = {
            "https://openrouter.test": httpx.MockTransport(fail_handler),
            "https://api.groq.test": httpx.MockTransport(fail_handler),
        }
        router = _build_router([_CHAIN_OR, _CHAIN_GROQ], transports)

        items: list[LLMStreamItem] = []
        with pytest.raises(LLMUnavailableError) as exc_info:
            async for item in router.stream(_MESSAGES, max_tokens=100):
                items.append(item)

        assert len(exc_info.value.attempts) >= 2
        # Only failovers, no deltas
        deltas = [i for i in items if isinstance(i, LLMDelta)]
        assert len(deltas) == 0


# ---------------------------------------------------------------------------
# Test 6: Exponential cooldown + half-open
# ---------------------------------------------------------------------------


class TestT6_ExponentialCooldown:
    """Cooldown doubles on failure, resets on success."""

    def test_cooldown_doubles(self):
        from app.adapters.llm_router import CircuitBreaker

        breaker = CircuitBreaker()
        entry_id = "test-entry"

        breaker.fail(entry_id, ErrorCode.provider_error, base_cooldown=60)
        cd1 = breaker.cooldown_remaining(entry_id)
        assert cd1 > 0
        # After first fail, multiplier doubles: 1 -> 2
        assert breaker._entries[entry_id]["multiplier"] == 2

        # Second failure doubles again: 2 -> 4
        breaker.fail(entry_id, ErrorCode.provider_error, base_cooldown=60)
        assert breaker._entries[entry_id]["multiplier"] == 4

    def test_success_resets(self):
        from app.adapters.llm_router import CircuitBreaker

        breaker = CircuitBreaker()
        entry_id = "test-entry"

        breaker.fail(entry_id, ErrorCode.provider_error, base_cooldown=60)
        breaker.fail(entry_id, ErrorCode.provider_error, base_cooldown=60)
        assert breaker._entries[entry_id]["multiplier"] == 4

        breaker.success(entry_id)
        assert breaker._entries[entry_id]["multiplier"] == 1
        assert breaker.cooldown_remaining(entry_id) == 0.0

    def test_half_open(self):
        from app.adapters.llm_router import CircuitBreaker

        breaker = CircuitBreaker()
        entry_id = "test-entry"

        # Set expired cooldown
        breaker.fail(entry_id, ErrorCode.provider_error, base_cooldown=0)
        # Cooldown is 0 -> entry is available
        assert breaker.cooldown_remaining(entry_id) == 0.0

        # Mark as half-open trial in flight
        ok = breaker.try_half_open(entry_id)
        assert ok
        # Second concurrent attempt should be denied
        ok2 = breaker.try_half_open(entry_id)
        assert not ok2

        breaker.release_half_open(entry_id)


# ---------------------------------------------------------------------------
# Test 7: Reasoning-only chunks don't count as first token
# ---------------------------------------------------------------------------


class TestT7_ReasoningChunks:
    """Reasoning-only chunks do not count as the first token."""

    @pytest.mark.asyncio
    async def test_reasoning_skipped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = (
                # First chunk has only reasoning, no content
                _sse_chunk(reasoning="Let me think...")
                + _sse_chunk(reasoning="Still thinking...")
                + _sse_chunk(content="Answer here")
                + _sse_chunk(finish_reason="stop",
                             usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
                + _sse_done()
            )
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body.encode()),
            )

        transports = {
            "https://openrouter.test": httpx.MockTransport(handler),
        }
        router = _build_router([_CHAIN_OR], transports)

        items: list[LLMStreamItem] = []
        async for item in router.stream(_MESSAGES, max_tokens=100):
            items.append(item)

        # Only "Answer here" should be a delta, not reasoning
        deltas = [i for i in items if isinstance(i, LLMDelta)]
        assert len(deltas) == 1
        assert deltas[0].text == "Answer here"


# ---------------------------------------------------------------------------
# Test 8: max_retries=0 on clients
# ---------------------------------------------------------------------------


class TestT8_MaxRetries:
    """max_retries=0 is asserted on the constructed clients."""

    def test_max_retries_zero(self):
        from app.adapters.llm_router import LLMRouter

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        transports = {
            "https://openrouter.test": httpx.MockTransport(handler),
        }
        env = {"TEST_OR_KEY": "sk-test"}
        with patch.dict("os.environ", env):
            router = LLMRouter([_CHAIN_OR], _transport_map=transports)

        for client in router._clients.values():
            assert client.max_retries == 0
