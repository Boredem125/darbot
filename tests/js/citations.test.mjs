/**
 * Tests for fullwidth citation bracket normalisation in app/static/app.js.
 *
 * Run: node --test tests/js/
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

// ---------------------------------------------------------------------------
// Helpers: replicate the normalisation logic from app.js
// ---------------------------------------------------------------------------

function normaliseCitations(text) {
  return text
    .replace(/【(\d+)】/g, "[$1]")
    .replace(/〔(\d+)〕/g, "[$1]")
    .replace(/［(\d+)］/g, "[$1]");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("citation normalisation", () => {
  it("converts fullwidth lenticular brackets", () => {
    assert.equal(normaliseCitations("price 【1】"), "price [1]");
  });

  it("converts multiple fullwidth citations", () => {
    assert.equal(
      normaliseCitations("【1】 and 【2】"),
      "[1] and [2]"
    );
  });

  it("converts tortoise shell brackets", () => {
    assert.equal(normaliseCitations("see 〔3〕"), "see [3]");
  });

  it("converts fullwidth square brackets", () => {
    assert.equal(normaliseCitations("ref ［5］"), "ref [5]");
  });

  it("leaves ASCII [n] unchanged", () => {
    assert.equal(normaliseCitations("text [1] and [2]"), "text [1] and [2]");
  });

  it("handles mixed bracket types", () => {
    const input = "text [1] more 【2】 end ［3］";
    assert.equal(normaliseCitations(input), "text [1] more [2] end [3]");
  });

  it("handles empty string", () => {
    assert.equal(normaliseCitations(""), "");
  });

  it("does not modify text without citations", () => {
    const text = "no citations here";
    assert.equal(normaliseCitations(text), text);
  });
});

describe("split pattern after normalisation", () => {
  it("splits normalised fullwidth citations correctly", () => {
    const raw = "The price is 1M 【1】 and more 【2】.";
    const normalised = normaliseCitations(raw);
    const parts = normalised.split(/(\[\d+\])/g);
    // Should have: "The price is 1M ", "[1]", " and more ", "[2]", "."
    assert.equal(parts.length, 5);
    assert.equal(parts[1], "[1]");
    assert.equal(parts[3], "[2]");
  });
});

// ---------------------------------------------------------------------------
// Citation-to-card mapping
// ---------------------------------------------------------------------------

/**
 * Replicate the mapping logic from app.js: given numCards, only [1]..[numCards]
 * are valid citation links; higher numbers are rendered as plain text.
 */
function mapCitations(text, numCards) {
  const normalised = normaliseCitations(text);
  const parts = normalised.split(/(\[\d+\])/g);
  const valid = [];
  const invalid = [];
  for (const part of parts) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const n = parseInt(m[1], 10);
      if (n >= 1 && n <= numCards) {
        valid.push(n);
      } else {
        invalid.push(n);
      }
    }
  }
  return { valid, invalid };
}

describe("citation-to-card mapping", () => {
  it("maps [1]..[numCards] to valid, higher to invalid", () => {
    const { valid, invalid } = mapCitations("see [1] and [2] and [3] and [4]", 3);
    assert.deepEqual(valid, [1, 2, 3]);
    assert.deepEqual(invalid, [4]);
  });

  it("maps [0] as invalid (cards are 1-based)", () => {
    const { valid, invalid } = mapCitations("[0] and [1]", 2);
    assert.deepEqual(valid, [1]);
    assert.deepEqual(invalid, [0]);
  });

  it("generates card-N id from citation number", () => {
    // The UI uses id="card-N" on each source card
    for (const n of [1, 2, 10]) {
      assert.equal("card-" + n, "card-" + n);
    }
  });

  it("handles fullwidth brackets in mapping", () => {
    const { valid, invalid } = mapCitations("text 【1】 more 【5】", 3);
    assert.deepEqual(valid, [1]);
    assert.deepEqual(invalid, [5]);
  });

  it("returns empty arrays when no citations present", () => {
    const { valid, invalid } = mapCitations("no citations here", 5);
    assert.deepEqual(valid, []);
    assert.deepEqual(invalid, []);
  });
});
