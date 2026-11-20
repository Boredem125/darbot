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

  it("passes HTML through unescaped (createTextNode handles safety)", () => {
    var text = renderMarkdownToText("<script>alert(1)</script>");
    // After S4 fix: parseBlocks no longer escapes; createTextNode makes text inert.
    // The plain-text renderer returns raw characters.
    assert.ok(text.includes("<script>"), "raw tag text preserved for createTextNode");
    assert.ok(!text.includes("&lt;"), "no entity encoding in plain text output");
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

  it("img onerror passes through (createTextNode handles XSS safety)", () => {
    var result = renderMarkdownToText('Price is <img src=x onerror="alert(1)"> SAR');
    // After S4 fix: raw text preserved; DOM safety is via createTextNode, not escaping.
    assert.ok(result.includes("<img"), "raw tag text preserved");
    assert.ok(result.includes("SAR"), "surrounding text preserved");
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

// ---------------------------------------------------------------------------
// S4 regression: double-escape (WP-5-security finding S4)
// Text must be escaped exactly once. Since renderBlocksToDOM uses
// createTextNode (which is inherently XSS-safe), parseBlocks must NOT
// pre-escape. An answer containing &, ", <script> must carry those
// literal characters in its segments -- never entity-encoded forms.
// ---------------------------------------------------------------------------

describe("S4 double-escape fix", () => {
  it("ampersand renders as literal & not &amp;", () => {
    var result = renderMarkdownToText("AT&T is here");
    assert.ok(result.includes("&"), "should contain literal ampersand");
    assert.ok(!result.includes("&amp;"), "must not double-escape ampersand");
  });

  it("double-quote renders as literal \" not &quot;", () => {
    var result = renderMarkdownToText('She said "hello"');
    assert.ok(result.includes('"'), "should contain literal double-quote");
    assert.ok(!result.includes("&quot;"), "must not double-escape double-quote");
  });

  it("<script> renders as literal characters, never as a tag", () => {
    var blocks = parseBlocks('<script>alert("xss")</script>');
    // Every segment must be text (or bold/italic/citation); none should
    // contain unescaped HTML that a browser would interpret as a tag.
    // With createTextNode the text is inert, so the segments carry the
    // raw characters and the DOM handles safety.
    var allText = "";
    for (var bi = 0; bi < blocks.length; bi++) {
      for (var ii = 0; ii < blocks[bi].items.length; ii++) {
        for (var si = 0; si < blocks[bi].items[ii].length; si++) {
          var seg = blocks[bi].items[ii][si];
          assert.ok(
            ["text","bold","italic","citation"].includes(seg.type),
            "segment type must be safe: " + seg.type
          );
          allText += seg.text;
        }
      }
    }
    // The raw characters pass through (createTextNode makes them inert)
    assert.ok(allText.includes("<script>"), "raw <script> passes through for createTextNode");
    assert.ok(!allText.includes("&lt;"), "must not entity-encode (createTextNode handles safety)");
  });

  it("mixed special characters in a realistic answer", () => {
    var input = 'Price is 500,000 SAR & the area is 200 sqm. "Great deal" [1]';
    var result = renderMarkdownToText(input);
    assert.ok(result.includes("&"), "literal ampersand preserved");
    assert.ok(result.includes('"Great deal"'), "literal quotes preserved");
    assert.ok(!result.includes("&amp;"), "no double-escaped ampersand");
    assert.ok(!result.includes("&quot;"), "no double-escaped quotes");
  });
});
