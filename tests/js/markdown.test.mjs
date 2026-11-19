/**
 * Tests for app/static/markdown.js -- safe minimal markdown renderer.
 *
 * Run: node --test "tests/js/*.mjs"
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  escapeHtml,
  normaliseCitations,
  parseInline,
  parseBlocks,
  renderMarkdownToText,
} from "../../app/static/markdown.js";

// ---------------------------------------------------------------------------
// escapeHtml
// ---------------------------------------------------------------------------

describe("escapeHtml", () => {
  it("escapes angle brackets", () => {
    assert.equal(escapeHtml("<script>alert(1)</script>"), "&lt;script&gt;alert(1)&lt;/script&gt;");
  });

  it("escapes ampersands", () => {
    assert.equal(escapeHtml("a&b"), "a&amp;b");
  });

  it("escapes quotes", () => {
    assert.equal(escapeHtml('"hello"'), "&quot;hello&quot;");
  });

  it("XSS: img onerror stays as text", () => {
    var input = '<img src=x onerror="alert(1)">';
    var result = escapeHtml(input);
    assert.ok(!result.includes("<img"));
    assert.ok(result.includes("&lt;img"));
  });

  it("passes through safe text unchanged", () => {
    assert.equal(escapeHtml("SAR 1,950,000"), "SAR 1,950,000");
  });
});

// ---------------------------------------------------------------------------
// normaliseCitations
// ---------------------------------------------------------------------------

describe("normaliseCitations (markdown module)", () => {
  it("normalises fullwidth brackets", () => {
    assert.equal(normaliseCitations("【1】"), "[1]");
  });
});

// ---------------------------------------------------------------------------
// parseInline
// ---------------------------------------------------------------------------

describe("parseInline", () => {
  it("parses plain text", () => {
    var segs = parseInline("hello world");
    assert.equal(segs.length, 1);
    assert.equal(segs[0].type, "text");
    assert.equal(segs[0].text, "hello world");
  });

  it("parses **bold**", () => {
    var segs = parseInline("the **price** is high");
    assert.equal(segs.length, 3);
    assert.equal(segs[0].type, "text");
    assert.equal(segs[1].type, "bold");
    assert.equal(segs[1].text, "price");
    assert.equal(segs[2].type, "text");
  });

  it("parses *italic*", () => {
    var segs = parseInline("*note* here");
    assert.equal(segs[0].type, "italic");
    assert.equal(segs[0].text, "note");
  });

  it("parses [n] citation", () => {
    var segs = parseInline("see [1] and [2]");
    assert.equal(segs[1].type, "citation");
    assert.equal(segs[1].n, 1);
    assert.equal(segs[3].type, "citation");
    assert.equal(segs[3].n, 2);
  });

  it("parses bold with citation inside", () => {
    var segs = parseInline("**SAR 1,950,000** [1]");
    assert.equal(segs[0].type, "bold");
    assert.equal(segs[0].text, "SAR 1,950,000");
    assert.equal(segs[2].type, "citation");
  });
});

// ---------------------------------------------------------------------------
// parseBlocks
// ---------------------------------------------------------------------------

describe("parseBlocks", () => {
  it("parses a paragraph", () => {
    var blocks = parseBlocks("Hello world");
    assert.equal(blocks.length, 1);
    assert.equal(blocks[0].type, "p");
  });

  it("parses unordered list with - ", () => {
    var blocks = parseBlocks("- item one\n- item two");
    assert.equal(blocks.length, 1);
    assert.equal(blocks[0].type, "ul");
    assert.equal(blocks[0].items.length, 2);
  });

  it("parses unordered list with * ", () => {
    var blocks = parseBlocks("* first\n* second");
    assert.equal(blocks[0].type, "ul");
    assert.equal(blocks[0].items.length, 2);
  });

  it("parses ordered list", () => {
    var blocks = parseBlocks("1. first\n2. second\n3. third");
    assert.equal(blocks[0].type, "ol");
    assert.equal(blocks[0].items.length, 3);
  });

  it("parses paragraph then list", () => {
    var blocks = parseBlocks("Intro text\n- item a\n- item b");
    assert.equal(blocks.length, 2);
    assert.equal(blocks[0].type, "p");
    assert.equal(blocks[1].type, "ul");
  });

  it("escapes HTML in input", () => {
    var blocks = parseBlocks("<script>alert(1)</script>");
    var text = renderMarkdownToText("<script>alert(1)</script>");
    assert.ok(text.includes("&lt;script&gt;"));
    assert.ok(!text.includes("<script>"));
  });
});

// ---------------------------------------------------------------------------
// renderMarkdownToText
// ---------------------------------------------------------------------------

describe("renderMarkdownToText", () => {
  it("renders plain text", () => {
    assert.equal(renderMarkdownToText("hello"), "hello");
  });

  it("renders bold as text", () => {
    var result = renderMarkdownToText("the **price** is SAR 1M");
    assert.ok(result.includes("price"));
    assert.ok(result.includes("SAR 1M"));
  });

  it("renders list items", () => {
    var result = renderMarkdownToText("- one\n- two");
    assert.ok(result.includes("- one"));
    assert.ok(result.includes("- two"));
  });

  it("preserves citations", () => {
    var result = renderMarkdownToText("See [1] and **SAR 2M** [2].");
    assert.ok(result.includes("[1]"));
    assert.ok(result.includes("[2]"));
  });

  it("XSS: img onerror is escaped", () => {
    var result = renderMarkdownToText('Price is <img src=x onerror="alert(1)"> SAR');
    assert.ok(!result.includes("<img"));
    assert.ok(result.includes("&lt;img"));
  });

  it("handles fullwidth citations", () => {
    var result = renderMarkdownToText("text 【1】 more");
    assert.ok(result.includes("[1]"));
  });

  it("handles mixed markdown", () => {
    var input = "**Villa for sale** in Riyadh [1].\n\nFeatures:\n- 3 bedrooms\n- 4 bathrooms\n- Pool area";
    var result = renderMarkdownToText(input);
    assert.ok(result.includes("Villa for sale"));
    assert.ok(result.includes("[1]"));
    assert.ok(result.includes("- 3 bedrooms"));
  });
});
