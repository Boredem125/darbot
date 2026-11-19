/**
 * SSE parser — pure ES module, no DOM access.
 *
 * Exports:
 *   createSseParser(onFrame) — takes text chunks, emits {event, data} frames.
 *   parseEvent(frame)        — JSON-parses, checks v===1, returns payload or throws.
 */

/**
 * Create a streaming SSE parser.
 *
 * @param {function({event: string, data: string}): void} onFrame
 *   Called for each complete SSE frame (after a blank line).
 * @returns {{ feed(chunk: string): void }}
 */
export function createSseParser(onFrame) {
  let buffer = "";
  let currentEvent = "";
  let dataLines = [];

  function dispatchFrame() {
    if (dataLines.length > 0) {
      const data = dataLines.join("\n");
      onFrame({ event: currentEvent || "message", data: data });
    }
    currentEvent = "";
    dataLines = [];
  }

  return {
    /**
     * Feed a text chunk from the stream. May contain partial lines.
     * @param {string} chunk
     */
    feed(chunk) {
      buffer += chunk;
      // Split on \n or \r\n; keep partial last line in buffer
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop(); // last element is the incomplete line (or "")

      for (const line of lines) {
        if (line === "") {
          // Blank line — end of frame
          dispatchFrame();
        } else if (line.startsWith(":")) {
          // Comment line — ignore
          continue;
        } else if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trimStart();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
        // Other field names (id:, retry:) are ignored per spec
      }
    }
  };
}

/**
 * Parse a raw SSE frame into a validated event payload.
 *
 * @param {{ event: string, data: string }} frame
 * @returns {{ v: number, request_id: string, event: string, [key: string]: any }}
 * @throws {Error} if JSON is invalid, v is not 1, or event field mismatch warning
 */
export function parseEvent(frame) {
  let payload;
  try {
    payload = JSON.parse(frame.data);
  } catch (e) {
    throw new Error("Invalid JSON in SSE data: " + e.message);
  }

  if (payload.v !== 1) {
    throw new Error("Unknown SSE version: " + payload.v + ". Expected v=1.");
  }

  // Warn if the SSE event: line and the JSON event field differ
  if (frame.event !== "message" && frame.event !== payload.event) {
    if (typeof console !== "undefined" && console.warn) {
      console.warn(
        "SSE event: line (" + frame.event + ") differs from JSON event field (" + payload.event + ")"
      );
    }
  }

  return payload;
}
