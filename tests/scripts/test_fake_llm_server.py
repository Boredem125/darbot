"""Tests for the fake OpenAI-compatible LLM server."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from scripts.fake_llm_server import _chunk_answer, generate_answer, make_handler

# ---------------------------------------------------------------------------
# Unit tests for answer generation
# ---------------------------------------------------------------------------


class TestGenerateAnswer:
    def test_no_sources(self):
        messages = [{"role": "user", "content": "Tell me about villas"}]
        answer = generate_answer(messages)
        assert len(answer) > 0

    def test_with_sources_has_citations(self):
        content = (
            '<source id="1">Villa for sale. Price SAR 1,500,000. 3 bedrooms.</source>\n'
            '<source id="2">Apartment in Jeddah. Price SAR 800,000.</source>\n'
            "Question: What villas are available?"
        )
        messages = [{"role": "user", "content": content}]
        answer = generate_answer(messages)
        assert "[1]" in answer
        assert "[2]" in answer

    def test_numbers_from_sources_appear(self):
        content = (
            '<source id="1">Villa. Price SAR 1,500,000.</source>\n'
            "Question: price?"
        )
        messages = [{"role": "user", "content": content}]
        answer = generate_answer(messages)
        assert "1,500,000" in answer

    def test_answer_length(self):
        content = (
            '<source id="1">Villa A. SAR 1,000,000.</source>\n'
            '<source id="2">Villa B. SAR 2,000,000.</source>\n'
            "Question: options?"
        )
        messages = [{"role": "user", "content": content}]
        answer = generate_answer(messages)
        words = answer.split()
        assert len(words) >= 20  # at least some content


class TestChunkAnswer:
    def test_chunks_cover_whole_answer(self):
        answer = "word " * 50
        chunks = _chunk_answer(answer.strip(), tokens_per_chunk=5)
        reconstructed = "".join(chunks).strip()
        assert reconstructed == answer.strip()

    def test_chunk_sizes(self):
        answer = "word " * 30
        chunks = _chunk_answer(answer.strip(), tokens_per_chunk=10)
        assert len(chunks) == 3


# ---------------------------------------------------------------------------
# Integration test: actual HTTP streaming
# ---------------------------------------------------------------------------


class TestFakeLLMServerStreaming:
    @pytest.fixture()
    def server(self):
        """Start the fake server on a random port and yield the base URL."""
        from http.server import ThreadingHTTPServer

        handler = make_handler(ttft_ms=0, tok_per_sec=10000)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        yield f"http://127.0.0.1:{port}"
        srv.shutdown()

    def test_streaming_has_valid_citations(self, server):
        """The fake server streams valid [n] citations from <source> blocks."""
        payload = json.dumps({
            "model": "test",
            "stream": True,
            "messages": [{
                "role": "user",
                "content": (
                    '<source id="1">Villa in Riyadh. SAR 1,950,000.</source>\n'
                    '<source id="2">Apartment in Jeddah. SAR 800,000.</source>\n'
                    "Question: What is available?"
                ),
            }],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{server}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            full = resp.read().decode("utf-8")

        # Collect all content deltas
        answer_parts = []
        for line in full.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                delta = data["choices"][0].get("delta", {})
                if "content" in delta:
                    answer_parts.append(delta["content"])

        answer = "".join(answer_parts)
        assert "[1]" in answer
        assert "1,950,000" in answer

    def test_non_streaming(self, server):
        payload = json.dumps({
            "model": "test",
            "stream": False,
            "messages": [{"role": "user", "content": "Hello"}],
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{server}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        assert data["choices"][0]["finish_reason"] == "stop"
        assert len(data["choices"][0]["message"]["content"]) > 0
