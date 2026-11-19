"""Tests for scripts/mock_sse.py — mock SSE server and scenario builder.

Run: .venv\\Scripts\\python.exe -m pytest -q tests/test_mock_sse.py
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

# Ensure project root on path.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common.schemas import (
    CHAT_EVENT_ADAPTER,
    AnswerCacheTier,
    AnswerMode,
    ChatEventEnd,
    ChatEventError,
    ChatEventStart,
    ChatEventToken,
    ErrorCode,
    FinishReason,
    Provider,
    Route,
)
from scripts.mock_sse import (
    _FAST_DELAYS,
    ALL_SCENARIOS,
    HttpError,
    build_scenario,
    extract_scenario,
    make_handler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events(name: str) -> list:
    """Build events for a scenario and return them."""
    result = build_scenario(name, f"#{name} test query")
    assert not isinstance(result, HttpError), f"Expected events, got HttpError for {name}"
    return result


def _validate_all(events: list) -> list:
    """Validate each event with CHAT_EVENT_ADAPTER and return the list of validated models."""
    validated = []
    for evt in events:
        raw_json = evt.model_dump_json()
        parsed = CHAT_EVENT_ADAPTER.validate_json(raw_json)
        validated.append(parsed)
    return validated


def _check_grammar(events: list, name: str):
    """Check the grammar rules from the SSE contract.

    stream := start body | fatal
    body   := (token | soft_error)* (end | fatal)
    """
    assert len(events) > 0, f"Scenario {name} produced no events"

    # Single fatal error case
    if len(events) == 1 and isinstance(events[0], ChatEventError) and events[0].fatal:
        return  # Valid: single fatal error

    # Normal case: start must be first
    assert isinstance(events[0], ChatEventStart), (
        f"Scenario {name}: first event must be start, got {events[0].event}"
    )

    # Last event must be end or fatal error
    last = events[-1]
    assert isinstance(last, (ChatEventEnd, ChatEventError)), (
        f"Scenario {name}: last event must be end or fatal error, got {last.event}"
    )
    if isinstance(last, ChatEventError):
        assert last.fatal, f"Scenario {name}: terminal error must be fatal"

    # Only one terminal event; nothing after it
    terminal_count = 0
    for i, evt in enumerate(events):
        if isinstance(evt, ChatEventEnd):
            terminal_count += 1
            assert i == len(events) - 1, (
                f"Scenario {name}: end event at position {i} but there are {len(events)} events"
            )
        elif isinstance(evt, ChatEventError) and evt.fatal:
            terminal_count += 1
            assert i == len(events) - 1, (
                f"Scenario {name}: fatal error at position {i} but there are {len(events)} events"
            )
    assert terminal_count == 1, f"Scenario {name}: expected 1 terminal event, got {terminal_count}"

    # Same request_id throughout
    rid = events[0].request_id
    for evt in events:
        assert evt.request_id == rid, (
            f"Scenario {name}: inconsistent request_id"
        )


def _check_source_numbering(events: list, name: str):
    """Verify sources[i].n == i + 1."""
    for evt in events:
        if isinstance(evt, ChatEventStart):
            for i, card in enumerate(evt.sources):
                assert card.n == i + 1, (
                    f"Scenario {name}: sources[{i}].n should be {i + 1}, got {card.n}"
                )


# ---------------------------------------------------------------------------
# Test: extract_scenario
# ---------------------------------------------------------------------------


class TestExtractScenario:
    def test_happy_default(self):
        assert extract_scenario("show me villas") == "happy"

    def test_hash_tag(self):
        assert extract_scenario("test #failover please") == "failover"

    def test_unknown_tag(self):
        assert extract_scenario("#nonexistent query") == "happy"


# ---------------------------------------------------------------------------
# Test: every scenario validates and obeys the grammar
# ---------------------------------------------------------------------------

_STREAM_SCENARIOS = [
    s for s in ALL_SCENARIOS if s not in ("rate_limited", "not_ready")
]


@pytest.mark.parametrize("name", _STREAM_SCENARIOS)
def test_scenario_validates(name: str):
    """Each event in every scenario validates with CHAT_EVENT_ADAPTER."""
    events = _events(name)
    _validate_all(events)


@pytest.mark.parametrize("name", _STREAM_SCENARIOS)
def test_scenario_grammar(name: str):
    """Each scenario follows the grammar."""
    events = _events(name)
    _check_grammar(events, name)


@pytest.mark.parametrize("name", _STREAM_SCENARIOS)
def test_scenario_source_numbering(name: str):
    """sources[i].n == i + 1."""
    events = _events(name)
    _check_source_numbering(events, name)


# ---------------------------------------------------------------------------
# Test: specific scenario properties
# ---------------------------------------------------------------------------


class TestHappy:
    def test_route_and_cards(self):
        events = _events("happy")
        start = events[0]
        assert start.route == Route.lookup
        assert len(start.sources) == 3
        assert start.sources[0].kind == "listing"
        assert start.sources[0].purpose == "sale"
        assert start.sources[0].property_type == "villa"
        assert start.sources[2].rent_period == "monthly"
        assert start.sources[2].annual_rent_sar == 90000

    def test_token_count(self):
        events = _events("happy")
        tokens = [e for e in events if isinstance(e, ChatEventToken)]
        # Brief: 8 or more tokens
        assert len(tokens) >= 8

    def test_citations(self):
        events = _events("happy")
        text = "".join(
            e.delta for e in events if isinstance(e, ChatEventToken)
        )
        assert "[1]" in text
        assert "[2]" in text
        assert "[3]" in text
        assert "SAR 1,950,000" in text

    def test_end_fields(self):
        events = _events("happy")
        end = events[-1]
        assert isinstance(end, ChatEventEnd)
        assert end.answer_mode == AnswerMode.llm
        assert end.provider == Provider.openrouter
        assert end.model == "google/gemma-4-31b-it:free"
        assert end.grounding.numbers_ok is True
        assert end.timings.sql_ms is None
        assert end.timings.total_ms > 0


class TestFailover:
    def test_soft_error_then_groq(self):
        events = _events("failover")
        # Must have a soft error with provider_rate_limited
        errors = [e for e in events if isinstance(e, ChatEventError)]
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.provider_rate_limited
        assert errors[0].fatal is False
        assert errors[0].degraded_to == "fallback_llm"
        # End with Groq
        end = events[-1]
        assert end.provider == Provider.groq
        assert end.model == "openai/gpt-oss-120b"


class TestCache:
    def test_cache_start_and_end(self):
        events = _events("cache")
        start = events[0]
        assert start.cache_tier == AnswerCacheTier.L1
        end = events[-1]
        assert end.answer_mode == AnswerMode.cache
        assert end.finish_reason == FinishReason.cached
        assert end.tokens.prompt_tokens == 0
        assert end.tokens.completion_tokens == 0
        assert end.timings.cache_lookup_ms is not None
        assert end.timings.total_ms is not None


class TestAggregate:
    def test_route_and_null_score(self):
        events = _events("aggregate")
        start = events[0]
        assert start.route == Route.aggregate
        for card in start.sources:
            assert card.score is None


class TestProject:
    def test_project_card(self):
        events = _events("project")
        start = events[0]
        assert start.route == Route.project_info
        assert len(start.sources) == 2
        proj = start.sources[0]
        assert proj.kind == "project"
        assert proj.price_from_amount == 450000
        assert proj.price_from_currency == "USD"
        assert proj.handover == "Q4 2027"
        art = start.sources[1]
        assert art.kind == "article"


class TestArabic:
    def test_arabic_language(self):
        events = _events("arabic")
        start = events[0]
        assert start.language == "ar"

    def test_arabic_indic_digits(self):
        events = _events("arabic")
        text = "".join(
            e.delta for e in events if isinstance(e, ChatEventToken)
        )
        # At least one Arabic-Indic digit (U+0660..U+0669)
        has_indic = any("٠" <= c <= "٩" for c in text)
        assert has_indic, "Arabic scenario must contain Arabic-Indic digits"


class TestRetrievalOnly:
    def test_degraded(self):
        events = _events("retrieval_only")
        errors = [e for e in events if isinstance(e, ChatEventError)]
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.all_providers_unavailable
        assert errors[0].degraded_to == "retrieval_only"
        end = events[-1]
        assert end.answer_mode == AnswerMode.retrieval_only
        assert end.provider == Provider.none
        assert end.model is None
        assert end.finish_reason == FinishReason.no_llm


class TestBusy:
    def test_busy(self):
        events = _events("busy")
        start = events[0]
        assert len(start.sources) == 0
        errors = [e for e in events if isinstance(e, ChatEventError)]
        assert errors[0].code == ErrorCode.retrieval_failed
        assert errors[0].degraded_to == "static_busy"
        end = events[-1]
        assert end.answer_mode == AnswerMode.static_busy


class TestRefusal:
    def test_refusal(self):
        events = _events("refusal")
        start = events[0]
        assert start.route == Route.out_of_scope
        assert len(start.sources) == 0
        end = events[-1]
        assert end.answer_mode == AnswerMode.refusal
        assert end.finish_reason == FinishReason.no_llm
        assert end.grounding.numbers_ok is None


class TestUnverified:
    def test_unverified(self):
        events = _events("unverified")
        end = events[-1]
        assert end.flags.unverified is True
        assert end.grounding.numbers_ok is False
        assert len(end.grounding.failures) >= 1


class TestTruncated:
    def test_truncated(self):
        events = _events("truncated")
        end = events[-1]
        assert end.finish_reason == FinishReason.length
        assert end.flags.truncated is True


class TestInterrupted:
    def test_interrupted(self):
        events = _events("interrupted")
        errors = [e for e in events if isinstance(e, ChatEventError)]
        assert len(errors) == 1
        assert errors[0].code == ErrorCode.llm_interrupted
        assert errors[0].degraded_to is None
        end = events[-1]
        assert end.finish_reason == FinishReason.error
        assert end.flags.truncated is True


class TestFatal:
    def test_single_fatal(self):
        events = _events("fatal")
        assert len(events) == 1
        assert isinstance(events[0], ChatEventError)
        assert events[0].fatal is True
        assert events[0].code == ErrorCode.internal


class TestChunked:
    def test_same_as_happy(self):
        """The chunked scenario has the same events as happy; only HTTP delivery differs."""
        happy = _events("happy")
        chunked = _events("chunked")
        for a, b in zip(happy, chunked):
            assert a.model_dump_json() == b.model_dump_json()


# ---------------------------------------------------------------------------
# Test: HTTP error scenarios
# ---------------------------------------------------------------------------


class TestHttpErrors:
    def test_rate_limited(self):
        result = build_scenario("rate_limited", "#rate_limited test")
        assert isinstance(result, HttpError)
        assert result.status == 429
        assert result.body["retry_after_s"] == 30

    def test_not_ready(self):
        result = build_scenario("not_ready", "#not_ready test")
        assert isinstance(result, HttpError)
        assert result.status == 503
        assert result.body["detail"] == "not_ready"


# ---------------------------------------------------------------------------
# Test: 422 validation error
# ---------------------------------------------------------------------------


class TestValidation:
    def test_422_on_empty_message(self):
        """Sending an invalid body should return 422."""
        # We test this via the actual HTTP round trip below, but here we test
        # that ChatRequest rejects an empty message.
        from common.schemas import ChatRequest

        with pytest.raises(ValueError):
            ChatRequest.model_validate({"message": "", "lang": None})


# ---------------------------------------------------------------------------
# Test: real HTTP round trip
# ---------------------------------------------------------------------------


class TestHttpRoundTrip:
    @pytest.fixture(autouse=True)
    def _server(self, tmp_path):
        """Start the mock on port 0 (OS-assigned) in a thread."""
        static_dir = str(_ROOT / "app" / "static")
        handler_cls = make_handler(_FAST_DELAYS, static_dir)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        yield
        self.server.shutdown()

    def test_failover_round_trip(self):
        """POST #failover with urllib and verify the raw SSE frames."""
        url = f"http://127.0.0.1:{self.port}/api/chat"
        data = json.dumps({"message": "#failover test"}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert "text/event-stream" in resp.headers.get("Content-Type", "")
            body = resp.read().decode("utf-8")

        # Parse SSE frames
        frames = []
        current_event = ""
        data_lines = []
        for line in body.split("\n"):
            line = line.rstrip("\r")
            if line == "":
                if data_lines:
                    frames.append({"event": current_event, "data": "\n".join(data_lines)})
                    current_event = ""
                    data_lines = []
            elif line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

        # Validate
        assert len(frames) >= 3  # start, error, token(s), end
        events = []
        for frame in frames:
            parsed = CHAT_EVENT_ADAPTER.validate_json(frame["data"])
            events.append(parsed)

        # Check structure: start first, then soft error, tokens, end
        assert events[0].event == "start"
        assert len(events[0].sources) >= 1
        assert any(e.event == "error" for e in events)
        assert events[-1].event == "end"
        assert events[-1].provider == "groq"

    def test_422_round_trip(self):
        """Sending invalid JSON returns 422."""
        url = f"http://127.0.0.1:{self.port}/api/chat"
        data = json.dumps({"message": ""}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req)
            pytest.fail("Expected 422")
        except urllib.error.HTTPError as e:
            assert e.code == 422
