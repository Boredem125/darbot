/**
 * Tests for app/static/labels.js — answer-mode label mapping and source badges.
 *
 * Run: node --test "tests/js/*.mjs"
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { getAnswerModeLabel, sourceBadgeLabel } from "../../app/static/labels.js";

// ---------------------------------------------------------------------------
// getAnswerModeLabel
// ---------------------------------------------------------------------------

describe("getAnswerModeLabel", () => {
  // --- refusal ---
  it("refusal + out_of_scope -> Outside what I can help with", () => {
    const result = getAnswerModeLabel("out_of_scope", "refusal");
    assert.equal(result.text, "Outside what I can help with");
    assert.equal(result.style, "muted");
  });

  it("refusal + lookup -> No matching listings", () => {
    const result = getAnswerModeLabel("lookup", "refusal");
    assert.equal(result.text, "No matching listings");
    assert.equal(result.style, "neutral");
  });

  it("refusal + aggregate -> No matching listings", () => {
    const result = getAnswerModeLabel("aggregate", "refusal");
    assert.equal(result.text, "No matching listings");
    assert.equal(result.style, "neutral");
  });

  it("refusal + project_info -> No matching listings", () => {
    const result = getAnswerModeLabel("project_info", "refusal");
    assert.equal(result.text, "No matching listings");
    assert.equal(result.style, "neutral");
  });

  // --- retrieval_only ---
  it("retrieval_only shows model-busy label", () => {
    const result = getAnswerModeLabel("lookup", "retrieval_only");
    assert.equal(result.text, "Model busy: showing matching listings without a written summary");
    assert.equal(result.style, "info");
  });

  // --- static_busy ---
  it("static_busy shows high-traffic label", () => {
    const result = getAnswerModeLabel("lookup", "static_busy");
    assert.equal(result.text, "High traffic, please retry shortly");
    assert.equal(result.style, "warn");
  });

  // --- llm ---
  it("llm returns null (no visible label)", () => {
    assert.equal(getAnswerModeLabel("lookup", "llm"), null);
  });

  // --- cache ---
  it("cache returns null (shown inside details only)", () => {
    assert.equal(getAnswerModeLabel("lookup", "cache"), null);
  });
});

// ---------------------------------------------------------------------------
// sourceBadgeLabel
// ---------------------------------------------------------------------------

describe("sourceBadgeLabel", () => {
  it("maps wasalt to Wasalt", () => {
    assert.equal(sourceBadgeLabel("wasalt"), "Wasalt");
  });

  it("maps darglobal to Dar Global", () => {
    assert.equal(sourceBadgeLabel("darglobal"), "Dar Global");
  });

  it("passes through unknown sources", () => {
    assert.equal(sourceBadgeLabel("other"), "other");
  });

  it("returns empty string for null/undefined", () => {
    assert.equal(sourceBadgeLabel(null), "");
    assert.equal(sourceBadgeLabel(undefined), "");
  });
});
