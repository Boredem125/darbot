#!/usr/bin/env python3
"""Fake OpenAI-compatible LLM server for benchmarking.

Serves ``POST /v1/chat/completions`` with streaming SSE, emitting
150-word answers with valid ``[n]`` citations taken from the ``<source>``
blocks in the prompt.  Numbers from the sources are copied so grounding
can pass.

Configurable TTFT (default 800 ms) and token rate (default 40 tok/s).
Used by bench part A and by the warm-cache dry run via ``LLM_CHAIN``.

Usage:
    .venv/Scripts/python.exe scripts/fake_llm_server.py --port 8999
    .venv/Scripts/python.exe scripts/fake_llm_server.py --port 8999 --ttft-ms 200 --tok-per-sec 80
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Answer generation helpers
# ---------------------------------------------------------------------------

_SOURCE_PATTERN = re.compile(
    r'<source\s+id="(\d+)">(.*?)</source>', re.DOTALL
)
_NUMBER_PATTERN = re.compile(r"(?:SAR\s+)?([\d,]+(?:\.\d+)?)\s*(?:sqm|SAR)?")


def _extract_sources(messages: list[dict]) -> list[tuple[int, str]]:
    """Extract (id, text) tuples from <source> blocks in the user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return [(int(m.group(1)), m.group(2)) for m in _SOURCE_PATTERN.finditer(content)]
    return []


def _extract_numbers_from_text(text: str) -> list[str]:
    """Extract formatted numbers from text."""
    return [m.group(1) for m in _NUMBER_PATTERN.finditer(text)]


def generate_answer(messages: list[dict]) -> str:
    """Build a 150-word-ish answer with valid citations and numbers."""
    sources = _extract_sources(messages)
    if not sources:
        return "Based on the available information, I can help you with property listings."

    parts: list[str] = []
    for src_id, src_text in sources[:5]:
        numbers = _extract_numbers_from_text(src_text)
        # Build a sentence referencing the source
        if numbers:
            parts.append(
                f"One option is priced at SAR {numbers[0]} [{src_id}]."
            )
        else:
            snippet = src_text[:80].strip().rstrip(".")
            parts.append(f"{snippet} [{src_id}].")

    # Pad to roughly 150 words
    base = " ".join(parts)
    padding_sentences = [
        "These properties offer a range of options for buyers.",
        "The listings are sourced from verified providers.",
        "Prices are in Saudi Riyals unless otherwise noted.",
        "Please review each listing for the most current details.",
        "Contact the listing agent for viewing arrangements.",
        "All areas are approximate and subject to verification.",
        "The market conditions may affect final pricing.",
        "Additional amenities may be available on request.",
    ]
    words = base.split()
    idx = 0
    while len(words) < 150 and idx < len(padding_sentences):
        words.extend(padding_sentences[idx].split())
        idx += 1
    return " ".join(words[:160])


def _chunk_answer(answer: str, tokens_per_chunk: int = 5) -> list[str]:
    """Split answer into chunks of roughly tokens_per_chunk words."""
    words = answer.split()
    chunks: list[str] = []
    for i in range(0, len(words), tokens_per_chunk):
        chunk_words = words[i: i + tokens_per_chunk]
        text = " ".join(chunk_words)
        if i + tokens_per_chunk < len(words):
            text += " "
        chunks.append(text)
    return chunks


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def make_handler(ttft_ms: float, tok_per_sec: float):
    """Factory: returns a handler class with the given timing parameters."""
    inter_token_s = 1.0 / tok_per_sec if tok_per_sec > 0 else 0.0
    ttft_s = ttft_ms / 1000.0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # quiet

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.send_error(404)
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                self.send_error(400)
                return

            messages = payload.get("messages", [])
            model = payload.get("model", "fake-model")
            stream = payload.get("stream", False)

            answer = generate_answer(messages)

            if not stream:
                # Non-streaming response
                resp = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": len(body) // 4,
                        "completion_tokens": len(answer.split()),
                        "total_tokens": len(body) // 4 + len(answer.split()),
                    },
                }
                resp_bytes = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)
                return

            # Streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            chunks = _chunk_answer(answer)
            prompt_tokens = len(body) // 4
            completion_tokens = len(answer.split())

            # TTFT delay
            time.sleep(ttft_s)

            for i, chunk_text in enumerate(chunks):
                delta = {"role": "assistant"} if i == 0 else {}
                delta["content"] = chunk_text

                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }],
                }
                line = f"data: {json.dumps(data)}\n\n"
                try:
                    self.wfile.write(line.encode("utf-8"))
                    self.wfile.flush()
                except BrokenPipeError:
                    return

                if i > 0:
                    time.sleep(inter_token_s)

            # Final chunk with finish_reason and usage
            final = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            try:
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_error(404)

    return Handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fake OpenAI-compatible LLM server")
    parser.add_argument("--port", type=int, default=8999)
    parser.add_argument("--ttft-ms", type=float, default=800.0, help="Time to first token (ms)")
    parser.add_argument("--tok-per-sec", type=float, default=40.0, help="Token rate (tokens/sec)")
    args = parser.parse_args()

    handler_cls = make_handler(args.ttft_ms, args.tok_per_sec)
    server = ThreadingHTTPServer(("", args.port), handler_cls)
    print(f"Fake LLM server on http://localhost:{args.port}")
    print(f"  TTFT: {args.ttft_ms} ms, Rate: {args.tok_per_sec} tok/s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
        server.server_close()


if __name__ == "__main__":
    main()
