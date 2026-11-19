/**
 * Tests for app/static/sse.js — the pure SSE parser module.
 *
 * Run: node --test tests/js/
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

// Import the module under test (relative to project root).
import { createSseParser, parseEvent } from "../../app/static/sse.js";

// ---------------------------------------------------------------------------
// Helper: collect frames from feeding chunks to the parser
// ---------------------------------------------------------------------------

function collectFrames(chunks) {
  const frames = [];
  const parser = createSseParser((frame) => frames.push(frame));
  for (const chunk of chunks) {
    parser.feed(chunk);
  }
  return frames;
}

// ---------------------------------------------------------------------------
// createSseParser
// ---------------------------------------------------------------------------

describe("createSseParser", () => {
  it("parses a simple frame", () => {
    const frames = collectFrames([
      'event: start\ndata: {"v":1,"event":"start","request_id":"aabbccdd00112233aabbccdd00112233"}\n\n',
    ]);
    assert.equal(frames.length, 1);
    assert.equal(frames[0].event, "start");
    const payload = JSON.parse(frames[0].data);
    assert.equal(payload.v, 1);
    assert.equal(payload.event, "start");
  });

  it("handles frame split across multiple chunks", () => {
    const frames = collectFrames([
      'event: tok',
      'en\ndata: {"v":1,"ev',
      'ent":"token","request_id":"aabbccdd00112233aabbccdd00112233","del',
      'ta":"hello"}\n\n',
    ]);
    assert.equal(frames.length, 1);
    assert.equal(frames[0].event, "token");
    const payload = JSON.parse(frames[0].data);
    assert.equal(payload.delta, "hello");
  });

  it("handles \\r\\n line endings", () => {
    const frames = collectFrames([
      'event: start\r\ndata: {"v":1,"event":"start","request_id":"aabbccdd00112233aabbccdd00112233"}\r\n\r\n',
    ]);
    assert.equal(frames.length, 1);
    assert.equal(frames[0].event, "start");
  });

  it("joins multiple data: lines with \\n", () => {
    const frames = collectFrames([
      'event: token\ndata: line1\ndata: line2\n\n',
    ]);
    assert.equal(frames.length, 1);
    assert.equal(frames[0].data, "line1\nline2");
  });

  it("ignores comment lines (starting with :)", () => {
    const frames = collectFrames([
      ': this is a comment\nevent: token\ndata: {"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"hi"}\n\n',
    ]);
    assert.equal(frames.length, 1);
    assert.equal(frames[0].event, "token");
  });

  it("ignores comment frames between events", () => {
    const frames = collectFrames([
      'event: start\ndata: {"v":1,"event":"start","request_id":"aabbccdd00112233aabbccdd00112233"}\n\n',
      ': ping\n',
      'event: end\ndata: {"v":1,"event":"end","request_id":"aabbccdd00112233aabbccdd00112233"}\n\n',
    ]);
    assert.equal(frames.length, 2);
    assert.equal(frames[0].event, "start");
    assert.equal(frames[1].event, "end");
  });

  it("parses multiple frames from one chunk", () => {
    const frames = collectFrames([
      'event: token\ndata: {"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"a"}\n\nevent: token\ndata: {"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"b"}\n\n',
    ]);
    assert.equal(frames.length, 2);
  });

  it("handles empty data lines (no dispatch on blank frame without data)", () => {
    const frames = collectFrames([
      '\n\nevent: token\ndata: {"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"ok"}\n\n',
    ]);
    // The initial blank lines produce no frame (no data lines accumulated)
    assert.equal(frames.length, 1);
  });
});

// ---------------------------------------------------------------------------
// parseEvent
// ---------------------------------------------------------------------------

describe("parseEvent", () => {
  it("parses a valid v=1 event", () => {
    const frame = {
      event: "token",
      data: '{"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"hello"}',
    };
    const payload = parseEvent(frame);
    assert.equal(payload.v, 1);
    assert.equal(payload.event, "token");
    assert.equal(payload.delta, "hello");
  });

  it("throws on unknown v", () => {
    const frame = {
      event: "token",
      data: '{"v":2,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"hello"}',
    };
    assert.throws(() => parseEvent(frame), /Unknown SSE version/);
  });

  it("throws on invalid JSON", () => {
    const frame = { event: "token", data: "not json" };
    assert.throws(() => parseEvent(frame), /Invalid JSON/);
  });

  it("ignores unknown fields in the payload", () => {
    const frame = {
      event: "token",
      data: '{"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"hi","extra_field":true}',
    };
    const payload = parseEvent(frame);
    assert.equal(payload.delta, "hi");
    assert.equal(payload.extra_field, true);
  });

  it("ignores unknown event names (parser does not throw)", () => {
    const frame = {
      event: "unknown_event",
      data: '{"v":1,"event":"unknown_event","request_id":"aabbccdd00112233aabbccdd00112233"}',
    };
    // Should not throw; the caller decides what to do with unknown events
    const payload = parseEvent(frame);
    assert.equal(payload.event, "unknown_event");
  });
});

// ---------------------------------------------------------------------------
// Chunked scenario fixture: simulates 7-byte writes with a ping comment
// ---------------------------------------------------------------------------

describe("chunked scenario", () => {
  it("reassembles frames written in 7-byte chunks with a ping comment between frames", () => {
    // This simulates the mock server's chunked delivery:
    // Frame 1 in 7-byte chunks, then a `: ping` comment, then frame 2 in 7-byte chunks.
    const frame1 =
      'event: start\ndata: {"v":1,"event":"start","request_id":"aabbccdd00112233aabbccdd00112233","route":"lookup","language":"en","filters":{},"cache_tier":"miss","sources":[]}\n\n';
    const frame2 =
      'event: token\ndata: {"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"hello"}\n\n';

    // Build the full byte stream: frame1 + ping + frame2, all in 7-byte chunks
    const fullStream = frame1 + ": ping\n" + frame2;
    const pieces = [];
    for (let i = 0; i < fullStream.length; i += 7) {
      pieces.push(fullStream.slice(i, i + 7));
    }

    const frames = collectFrames(pieces);
    assert.equal(frames.length, 2);
    assert.equal(frames[0].event, "start");
    assert.equal(frames[1].event, "token");

    const payload1 = parseEvent(frames[0]);
    assert.equal(payload1.route, "lookup");

    const payload2 = parseEvent(frames[1]);
    assert.equal(payload2.delta, "hello");
  });

  it("handles the full happy scenario streamed in 7-byte chunks", () => {
    // Build a realistic multi-event stream
    const events = [
      'event: start\ndata: {"v":1,"event":"start","request_id":"aabbccdd00112233aabbccdd00112233","route":"lookup","language":"en","filters":{},"cache_tier":"miss","sources":[]}\n\n',
      ': ping\n',
      'event: token\ndata: {"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":"Hello"}\n\n',
      'event: token\ndata: {"v":1,"event":"token","request_id":"aabbccdd00112233aabbccdd00112233","delta":" world"}\n\n',
      'event: end\ndata: {"v":1,"event":"end","request_id":"aabbccdd00112233aabbccdd00112233"}\n\n',
    ];
    const fullStream = events.join("");

    // Split the entire stream into 7-byte pieces
    const pieces = [];
    for (let i = 0; i < fullStream.length; i += 7) {
      pieces.push(fullStream.slice(i, i + 7));
    }

    const frames = collectFrames(pieces);
    // Should get: start, token, token, end (ping is a comment, not a frame)
    assert.equal(frames.length, 4);
    assert.equal(frames[0].event, "start");
    assert.equal(frames[1].event, "token");
    assert.equal(frames[2].event, "token");
    assert.equal(frames[3].event, "end");

    // Verify each parses
    for (const frame of frames) {
      parseEvent(frame);
    }
  });
});
