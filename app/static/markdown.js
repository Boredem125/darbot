/**
 * markdown.js -- Safe minimal markdown renderer for assistant text.
 * No external library. No innerHTML with untrusted strings.
 *
 * Supports: **bold**, *italic*, "- " / "* " unordered lists,
 * "1. " ordered lists, [n] citation chips, line breaks.
 *
 * Security: all text is escaped (HTML entities) before any
 * markdown processing. The output is a DOM DocumentFragment
 * built entirely via createElement / createTextNode.
 *
 * Exported for use by app.js and testable in Node (renderMarkdownToText).
 */

/**
 * Escape HTML special chars so <script> etc. become literal text.
 * @param {string} s
 * @returns {string}
 */
export function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Normalise fullwidth citation brackets to ASCII [n].
 * @param {string} s
 * @returns {string}
 */
export function normaliseCitations(s) {
  return s
    .replace(/【(\d+)】/g, "[$1]")
    .replace(/〔(\d+)〕/g, "[$1]")
    .replace(/［(\d+)］/g, "[$1]");
}

/**
 * Parse a single line into inline segments: text, bold, italic, citation.
 * Returns an array of {type, text, n?} objects.
 *
 * @param {string} line - HTML-escaped line
 * @returns {Array<{type: string, text: string, n?: number}>}
 */
export function parseInline(line) {
  // Pattern: **bold**, *italic*, [n] citations
  // Process in order: bold first, then italic (avoid ** matching as two *)
  var segments = [];
  // Split on bold, italic and citations
  // regex: (**...** | *...* | [n])
  var re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\d+\])/g;
  var last = 0;
  var m;
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) {
      segments.push({ type: "text", text: line.slice(last, m.index) });
    }
    var tok = m[0];
    if (tok.startsWith("**") && tok.endsWith("**")) {
      segments.push({ type: "bold", text: tok.slice(2, -2) });
    } else if (tok.startsWith("*") && tok.endsWith("*")) {
      segments.push({ type: "italic", text: tok.slice(1, -1) });
    } else if (/^\[\d+\]$/.test(tok)) {
      var n = parseInt(tok.slice(1, -1), 10);
      segments.push({ type: "citation", text: tok, n: n });
    }
    last = m.index + tok.length;
  }
  if (last < line.length) {
    segments.push({ type: "text", text: line.slice(last) });
  }
  return segments;
}

/**
 * Parse raw markdown text into a block structure.
 * Returns an array of blocks: {type: "p"|"ul"|"ol", lines: [{segments}]}
 *
 * @param {string} raw
 * @returns {Array<{type: string, items: Array<Array<{type:string,text:string}>>}>}
 */
export function parseBlocks(raw) {
  // NOTE: escapeHtml was removed here (S4 fix). All text is inserted via
  // createTextNode in renderBlocksToDOM, which is inherently XSS-safe.
  // Pre-escaping caused double-encoding: users saw &amp; instead of &.
  var normalised = normaliseCitations(raw);
  var lines = normalised.split("\n");
  var blocks = [];
  var currentList = null;

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];

    // Unordered list: "- " or "* " or "• "
    var ulMatch = line.match(/^[-*•]\s+(.*)$/);
    if (ulMatch) {
      if (!currentList || currentList.type !== "ul") {
        currentList = { type: "ul", items: [] };
        blocks.push(currentList);
      }
      currentList.items.push(parseInline(ulMatch[1]));
      continue;
    }

    // Ordered list: "1. ", "2. " etc.
    var olMatch = line.match(/^\d+\.\s+(.*)$/);
    if (olMatch) {
      if (!currentList || currentList.type !== "ol") {
        currentList = { type: "ol", items: [] };
        blocks.push(currentList);
      }
      currentList.items.push(parseInline(olMatch[1]));
      continue;
    }

    // End of list
    currentList = null;

    // Blank line or paragraph text
    if (line.trim() === "") {
      continue; // skip blank lines between blocks
    }

    blocks.push({ type: "p", items: [parseInline(line)] });
  }
  return blocks;
}

/**
 * Render parsed blocks to a DOM DocumentFragment.
 * Citation chips are rendered as clickable buttons when numCards > 0.
 *
 * @param {Array} blocks - from parseBlocks
 * @param {number} numCards
 * @param {function} onCitationClick - click handler for citations
 * @returns {DocumentFragment}
 */
export function renderBlocksToDOM(blocks, numCards, onCitationClick) {
  var frag = document.createDocumentFragment();

  function renderSegments(parent, segments) {
    for (var i = 0; i < segments.length; i++) {
      var seg = segments[i];
      if (seg.type === "bold") {
        var b = document.createElement("strong");
        // Bold can contain citations, re-parse inner
        var inner = parseInline(seg.text);
        renderSegments(b, inner);
        parent.appendChild(b);
      } else if (seg.type === "italic") {
        var em = document.createElement("em");
        em.appendChild(document.createTextNode(seg.text));
        parent.appendChild(em);
      } else if (seg.type === "citation") {
        if (seg.n >= 1 && seg.n <= numCards) {
          var btn = document.createElement("button");
          btn.className = "citation-link";
          btn.textContent = "[" + seg.n + "]";
          btn.setAttribute("aria-label", "Go to source " + seg.n);
          btn.type = "button";
          btn.dataset.card = seg.n;
          if (onCitationClick) btn.addEventListener("click", onCitationClick);
          parent.appendChild(btn);
        } else {
          parent.appendChild(document.createTextNode(seg.text));
        }
      } else {
        parent.appendChild(document.createTextNode(seg.text));
      }
    }
  }

  for (var bi = 0; bi < blocks.length; bi++) {
    var block = blocks[bi];
    if (block.type === "ul" || block.type === "ol") {
      var list = document.createElement(block.type);
      for (var li = 0; li < block.items.length; li++) {
        var item = document.createElement("li");
        renderSegments(item, block.items[li]);
        list.appendChild(item);
      }
      frag.appendChild(list);
    } else {
      // paragraph: render inline segments, add newline between paragraphs
      if (bi > 0) frag.appendChild(document.createTextNode("\n"));
      renderSegments(frag, block.items[0]);
    }
  }
  return frag;
}

/**
 * Pure-text render of blocks (for Node.js testing without DOM).
 * Returns a plain string with **bold** as BOLD, *italic* as text, [n] as [n].
 *
 * @param {string} raw
 * @returns {string}
 */
export function renderMarkdownToText(raw) {
  var blocks = parseBlocks(raw);
  var out = [];
  for (var bi = 0; bi < blocks.length; bi++) {
    var block = blocks[bi];
    if (block.type === "ul") {
      for (var i = 0; i < block.items.length; i++) {
        out.push("- " + segmentsToText(block.items[i]));
      }
    } else if (block.type === "ol") {
      for (var j = 0; j < block.items.length; j++) {
        out.push((j + 1) + ". " + segmentsToText(block.items[j]));
      }
    } else {
      out.push(segmentsToText(block.items[0]));
    }
  }
  return out.join("\n");
}

function segmentsToText(segs) {
  var s = "";
  for (var i = 0; i < segs.length; i++) {
    s += segs[i].text;
  }
  return s;
}
