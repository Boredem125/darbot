/**
 * app.js -- UI for the Dar Bot property assistant.
 * No framework, no build step. Uses only textContent and createElement.
 * Never uses innerHTML with untrusted strings.
 */
import { createSseParser, parseEvent } from "./sse.js";
import { getAnswerModeLabel, sourceBadgeLabel } from "./labels.js";
import { parseBlocks, renderBlocksToDOM } from "./markdown.js";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
var conversationEl = document.getElementById("conversation");
var emptyStateEl   = document.getElementById("empty-state");
var inputEl        = document.getElementById("message-input");
var charCountEl    = document.getElementById("char-count");
var btnSend        = document.getElementById("btn-send");
var btnStop        = document.getElementById("btn-stop");
var srAnnounce     = document.getElementById("sr-announce");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
var abortController = null;
var streaming = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function txt(s) { return document.createTextNode(s); }
function el(tag, cls, text) {
  var e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
function fmtNum(n) { return n == null ? null : Number(n).toLocaleString("en-US"); }
function fmtPrice(a) { return a == null ? null : "SAR " + fmtNum(a); }
function fmtDate(iso) {
  if (!iso) return null;
  try { return new Date(iso).toLocaleDateString("en-US",{year:"numeric",month:"short",day:"numeric"}); }
  catch(_) { return iso; }
}
function fmtMs(ms) { return ms == null ? null : Math.round(ms) + " ms"; }
function isSafeUrl(u) { return typeof u === "string" && (u.startsWith("https://") || u.startsWith("http://")); }

function setStreaming(on) {
  streaming = on;
  inputEl.disabled = on;
  btnSend.disabled = on;
  btnSend.hidden = on;
  btnStop.hidden = !on;
  if (!on) inputEl.focus();
}
function announce(text) {
  srAnnounce.textContent = "";
  requestAnimationFrame(function() { srAnnounce.textContent = text; });
}
function updateCharCount() {
  var len = inputEl.value.length;
  charCountEl.textContent = len + " / 500";
  charCountEl.classList.toggle("over-limit", len > 500);
  btnSend.disabled = streaming || len === 0 || len > 500;
}
function hideEmptyState() { if (emptyStateEl) emptyStateEl.classList.add("hidden"); }

// Property-type SVG icons (inline, tiny)
var ICON_SVG = {
  villa:     '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 21h18"/><path d="M5 21V7l7-4 7 4v14"/><path d="M9 21v-6h6v6"/></svg>',
  apartment: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 22h20"/><path d="M6 22V4a2 2 0 012-2h8a2 2 0 012 2v18"/><path d="M10 6h4M10 10h4M10 14h4"/></svg>',
  project:   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5M2 12l10 5 10-5"/></svg>',
  article:   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4h16a2 2 0 012 2v12a2 2 0 01-2 2H4a2 2 0 01-2-2V6a2 2 0 012-2z"/><path d="M8 2v4M16 2v4M2 10h20"/></svg>',
  _default:  '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 21h18"/><path d="M9 21V13h6v8"/><path d="M5 21V7l7-4 7 4v14"/></svg>',
};
function getIconKey(card) {
  if (card.kind === "project") return "project";
  if (card.kind === "article") return "article";
  if (card.property_type && ICON_SVG[card.property_type]) return card.property_type;
  return "_default";
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------
function createTypingIndicator() {
  var w = el("div", "typing-indicator");
  w.setAttribute("aria-label", "Assistant is typing");
  w.setAttribute("role", "status");
  for (var i = 0; i < 3; i++) w.appendChild(el("span", "typing-dot"));
  return w;
}

// ---------------------------------------------------------------------------
// Assistant header (avatar + label)
// ---------------------------------------------------------------------------
function createAsstHeader() {
  var h = el("div", "asst-header");
  var av = el("div", "asst-avatar", "D");
  h.appendChild(av);
  h.appendChild(el("span", "asst-name", "Dar Bot"));
  return h;
}

// ---------------------------------------------------------------------------
// Compact source card
// ---------------------------------------------------------------------------
function renderCard(card) {
  var c = el("div", "source-card");
  c.setAttribute("dir", "auto");
  c.id = "card-" + card.n;

  // Top row: icon tile + body
  var top = el("div", "card-top");

  // Icon tile
  var iconTile = el("div", "card-icon-tile");
  // We use a safe approach: set innerHTML only on our own static SVG strings
  iconTile.innerHTML = ICON_SVG[getIconKey(card)];
  top.appendChild(iconTile);

  var body = el("div", "card-body");

  // Price (big, single line)
  if (card.kind === "listing" && card.price_sar != null) {
    var priceStr = fmtPrice(card.price_sar);
    if (card.rent_period) {
      // Short period label: "SAR X /mo", "SAR X /yr", etc.
      var periodShort = {daily:"/day",weekly:"/wk",monthly:"/mo",yearly:"/yr"};
      priceStr += " " + (periodShort[card.rent_period] || "/ " + card.rent_period);
    }
    body.appendChild(el("div", "card-price", priceStr));
  } else if (card.kind === "project" && card.price_from_amount != null && card.price_from_currency) {
    body.appendChild(el("div", "card-price",
      "from " + card.price_from_currency + " " + fmtNum(card.price_from_amount)));
  }

  // Title (one line, ellipsis)
  body.appendChild(el("div", "card-title-line", card.title || ""));

  top.appendChild(body);
  c.appendChild(top);

  // Chips
  if (card.kind === "listing") {
    var chips = el("div", "card-chips");
    var hasChips = false;
    if (card.bedrooms != null) { chips.appendChild(el("span","card-chip",card.bedrooms+" bed")); hasChips=true; }
    if (card.bathrooms != null){ chips.appendChild(el("span","card-chip",card.bathrooms+" bath"));hasChips=true; }
    if (card.area_sqm != null) { chips.appendChild(el("span","card-chip",fmtNum(card.area_sqm)+" sqm"));hasChips=true; }
    if (hasChips) c.appendChild(chips);
  }

  // Location
  var locParts = [];
  if (card.kind === "listing") {
    if (card.city) locParts.push(card.city);
    if (card.district) locParts.push(card.district);
  } else if (card.kind === "project") {
    if (card.city) locParts.push(card.city);
    if (card.country) locParts.push(card.country);
    if (card.handover) locParts.push("Handover: " + card.handover);
  } else if (card.kind === "article") {
    if (card.article_kind) locParts.push(card.article_kind);
    var pd = fmtDate(card.published_at);
    if (pd) locParts.push(pd);
  }
  if (locParts.length) c.appendChild(el("div", "card-location", locParts.join(" · ")));

  // Snippet hidden for compact cards (shown only behind View link)

  // Footer: source badge + index + link
  var footer = el("div", "card-footer");
  footer.appendChild(el("span", "card-source-badge", sourceBadgeLabel(card.source)));
  footer.appendChild(el("span", "card-index", String(card.n)));
  if (isSafeUrl(card.url)) {
    var lnk = el("a", "card-link", "View →");
    lnk.href = card.url; lnk.target = "_blank"; lnk.rel = "noopener noreferrer";
    footer.appendChild(lnk);
  }
  c.appendChild(footer);

  return c;
}

function renderCardStrip(sources) {
  if (!sources || sources.length === 0) return null;
  var wrap = document.createDocumentFragment();

  // Header
  var hdr = el("div", "cards-header");
  hdr.appendChild(txt("Sources (" + sources.length + ")"));
  wrap.appendChild(hdr);

  // Strip
  var strip = el("div", "cards-strip");
  strip.setAttribute("role", "list");
  strip.setAttribute("aria-label", "Source cards");
  for (var i = 0; i < sources.length; i++) {
    var card = renderCard(sources[i]);
    card.setAttribute("role", "listitem");
    strip.appendChild(card);
  }
  wrap.appendChild(strip);
  return wrap;
}

// ---------------------------------------------------------------------------
// Markdown-aware answer rendering (safe: HTML escaped first)
// ---------------------------------------------------------------------------
function renderAnswer(raw, numCards) {
  var blocks = parseBlocks(raw);
  return renderBlocksToDOM(blocks, numCards, onCitationClick);
}
function onCitationClick(e) {
  var n = e.currentTarget.dataset.card;
  var t = document.getElementById("card-" + n);
  if (t) {
    t.scrollIntoView({ behavior: "smooth", block: "nearest" });
    t.classList.add("highlight");
    setTimeout(function() { t.classList.remove("highlight"); }, 2000);
  }
}

// ---------------------------------------------------------------------------
// Response details toggle (small, grey)
// ---------------------------------------------------------------------------
function renderResponseDetails(endEvt) {
  var det = document.createElement("details");
  det.className = "response-details";
  var sum = document.createElement("summary");
  var totalMs = endEvt.timings ? endEvt.timings.total_ms : null;
  var sumText = "Details";
  if (totalMs != null) sumText += " · " + fmtMs(totalMs);
  sum.appendChild(txt(sumText));
  var sb = el("span", "summary-badges");
  if (endEvt.answer_mode === "cache" && endEvt.cache_tier && endEvt.cache_tier !== "miss") {
    sb.appendChild(el("span", "summary-tag summary-tag-cached", "cached " + endEvt.cache_tier));
  }
  if (endEvt.grounding && endEvt.grounding.failures && endEvt.grounding.failures.length > 0) {
    sb.appendChild(el("span", "summary-tag summary-tag-warn", "grounding warn"));
  }
  if (sb.childNodes.length > 0) sum.appendChild(sb);
  det.appendChild(sum);

  var content = el("div", "details-content");
  var addD = function(label, val) {
    if (val == null) return;
    var item = el("span", "detail-item");
    item.appendChild(el("span", "detail-label", label + ": "));
    item.appendChild(txt(String(val)));
    content.appendChild(item);
  };
  var ti = endEvt.timings || {};
  addD("total",fmtMs(ti.total_ms)); addD("TTFT",fmtMs(ti.llm_ttft_ms));
  addD("LLM",fmtMs(ti.llm_total_ms)); addD("embed",fmtMs(ti.embed_ms));
  addD("sparse",fmtMs(ti.sparse_ms)); addD("retrieve",fmtMs(ti.retrieve_ms));
  addD("rerank",fmtMs(ti.rerank_ms)); addD("SQL",fmtMs(ti.sql_ms));
  addD("grounding",fmtMs(ti.grounding_ms)); addD("cache lookup",fmtMs(ti.cache_lookup_ms));
  addD("cache tier",endEvt.cache_tier);
  if (endEvt.cache_hits && endEvt.cache_hits.length) addD("cache hits",endEvt.cache_hits.join(", "));
  if (endEvt.provider && endEvt.provider !== "none") {
    var ps = endEvt.provider.charAt(0).toUpperCase()+endEvt.provider.slice(1);
    if (endEvt.model) ps += " / " + endEvt.model;
    addD("model", ps);
  }
  var tok = endEvt.tokens;
  if (tok && (tok.prompt_tokens>0||tok.completion_tokens>0)) addD("tokens",tok.prompt_tokens+" in / "+tok.completion_tokens+" out");
  var g = endEvt.grounding;
  if (g) {
    var gp = [];
    if (g.numbers_ok!=null) gp.push("numbers: "+(g.numbers_ok?"ok":"fail"));
    if (g.entities_ok!=null) gp.push("entities: "+(g.entities_ok?"ok":"fail"));
    if (g.citations_ok!=null) gp.push("citations: "+(g.citations_ok?"ok":"fail"));
    if (gp.length) addD("grounding",gp.join(", "));
  }
  addD("answer mode",endEvt.answer_mode); addD("finish",endEvt.finish_reason);
  det.appendChild(content);
  return det;
}

// ---------------------------------------------------------------------------
// Badges (outside details)
// ---------------------------------------------------------------------------
function renderBadges(endEvt, route, hadDegradedNotice) {
  var frag = document.createDocumentFragment();
  if (endEvt.flags && endEvt.flags.unverified) {
    frag.appendChild(el("span", "badge badge-unverified", " Unverified figures"));
    if (endEvt.grounding && endEvt.grounding.failures && endEvt.grounding.failures.length) {
      var d = document.createElement("details"); d.className = "badge-failures";
      var s = document.createElement("summary");
      s.textContent = "Grounding failures (" + endEvt.grounding.failures.length + ")";
      d.appendChild(s);
      var ul = document.createElement("ul");
      for (var i=0; i<endEvt.grounding.failures.length; i++) {
        var li = document.createElement("li"); li.textContent = endEvt.grounding.failures[i]; ul.appendChild(li);
      }
      d.appendChild(ul); frag.appendChild(d);
    }
  }
  if (endEvt.flags && endEvt.flags.truncated) {
    frag.appendChild(el("span", "badge badge-truncated", " Response cut off"));
  }
  // Skip mode label if a soft-error notice already conveyed the degradation
  if (!hadDegradedNotice) {
    var label = getAnswerModeLabel(route, endEvt.answer_mode);
    if (label) frag.appendChild(el("div", "mode-label mode-label-" + label.style, label.text));
  }
  return frag;
}

// ---------------------------------------------------------------------------
// Main send flow
// ---------------------------------------------------------------------------
async function sendMessage(message) {
  message = (message || inputEl.value).trim();
  if (!message || message.length > 500 || streaming) return;
  hideEmptyState();
  inputEl.value = ""; updateCharCount();

  var turn = el("div", "turn");
  var userBubble = el("div", "bubble-user");
  userBubble.setAttribute("dir", "auto");
  userBubble.textContent = message;
  turn.appendChild(userBubble);

  var asstBubble = el("div", "bubble-assistant");
  asstBubble.setAttribute("aria-busy", "true");
  asstBubble.appendChild(createAsstHeader());
  turn.appendChild(asstBubble);
  conversationEl.appendChild(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "end" });

  setStreaming(true);
  abortController = new AbortController();

  var answerEl = null, rawAnswer = "", numCards = 0;
  var startEvt = null, endEvt = null, hadFatalError = false;
  var typingEl = null, firstTokenReceived = false;
  var cardSources = null; // defer card rendering until after answer
  var hadDegradedNotice = false; // suppresses duplicate mode label

  try {
    var response = await fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify({ message: message, lang: null }),
      credentials: "same-origin",
      signal: abortController.signal,
    });

    if (!response.ok) {
      var errorMsg;
      try {
        var errBody = await response.json();
        if (response.status === 429 && errBody.retry_after_s) errorMsg = "Too many requests. Try again in " + errBody.retry_after_s + " s.";
        else if (response.status === 503) errorMsg = "The service is warming up. Please try again shortly.";
        else if (response.status === 422) errorMsg = "Validation error: " + (errBody.detail || "Invalid request.");
        else errorMsg = errBody.detail || errBody.message || "An error occurred.";
      } catch(_) { errorMsg = "Request failed (HTTP " + response.status + ")."; }
      asstBubble.appendChild(el("div", "notice notice-error", errorMsg));
      asstBubble.removeAttribute("aria-busy");
      setStreaming(false); announce("Error: " + errorMsg); return;
    }

    var reader = response.body.getReader();
    var decoder = new TextDecoder("utf-8", { fatal: false });

    var parser = createSseParser(function(frame) {
      var payload;
      try { payload = parseEvent(frame); }
      catch(e) { asstBubble.appendChild(el("div","notice notice-error",e.message)); hadFatalError=true; abortController.abort(); return; }

      if (payload.event === "start") {
        startEvt = payload;
        numCards = (payload.sources || []).length;
        cardSources = payload.sources || [];
        if (payload.language) {
          asstBubble.setAttribute("lang", payload.language);
          if (payload.language === "ar") asstBubble.setAttribute("dir", "rtl");
        }
        // Show typing indicator; answer + cards come later
        typingEl = createTypingIndicator();
        asstBubble.appendChild(typingEl);
        answerEl = el("div", "answer-text streaming");
        answerEl.setAttribute("dir", "auto");

      } else if (payload.event === "token") {
        if (!firstTokenReceived) {
          firstTokenReceived = true;
          if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
          if (answerEl) asstBubble.appendChild(answerEl);
        }
        if (!answerEl) { answerEl = el("div","answer-text streaming"); answerEl.setAttribute("dir","auto"); asstBubble.appendChild(answerEl); }
        rawAnswer += payload.delta;
        // Re-render whole buffer with markdown on each token
        while (answerEl.firstChild) answerEl.removeChild(answerEl.firstChild);
        answerEl.appendChild(renderAnswer(rawAnswer, numCards));

      } else if (payload.event === "end") {
        endEvt = payload;
        if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
        if (answerEl) answerEl.classList.remove("streaming");

        // If no tokens arrived but answer text element was prepared, attach it
        if (answerEl && !answerEl.parentNode && rawAnswer) asstBubble.appendChild(answerEl);

        // Final render with markdown
        if (answerEl && rawAnswer) {
          while (answerEl.firstChild) answerEl.removeChild(answerEl.firstChild);
          answerEl.appendChild(renderAnswer(rawAnswer, numCards));
        }

        // Cards BELOW the answer
        var cardStrip = renderCardStrip(cardSources);
        if (cardStrip) asstBubble.appendChild(cardStrip);

        // Badges
        var route = startEvt ? startEvt.route : null;
        var badgesNode = renderBadges(payload, route, hadDegradedNotice);
        if (badgesNode.childNodes.length > 0) {
          var br = el("div", "badges-row"); br.appendChild(badgesNode); asstBubble.appendChild(br);
        }

        // Details toggle
        asstBubble.appendChild(renderResponseDetails(payload));
        asstBubble.removeAttribute("aria-busy");

        var annText = "Answer complete.";
        if (payload.flags && payload.flags.unverified) annText += " Warning: unverified figures.";
        if (payload.flags && payload.flags.truncated) annText += " Note: answer was cut off.";
        announce(annText);

      } else if (payload.event === "error") {
        if (payload.fatal) {
          hadFatalError = true;
          if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
          if (answerEl) answerEl.classList.remove("streaming");
          asstBubble.appendChild(el("div","notice notice-fatal",payload.message||"An unexpected error occurred."));
          asstBubble.removeAttribute("aria-busy");
          announce("Error: " + (payload.message||"An unexpected error occurred."));
        } else {
          // Soft error with degraded_to: show as neutral info, suppress duplicate mode label later
          if (payload.degraded_to) hadDegradedNotice = true;
          asstBubble.appendChild(el("div", "notice notice-info", payload.message || "Switching to a fallback."));
        }
      }
    });

    while (true) {
      var result = await reader.read();
      if (result.done) break;
      parser.feed(decoder.decode(result.value, { stream: true }));
    }
    var remaining = decoder.decode();
    if (remaining) parser.feed(remaining);

  } catch(e) {
    if (e.name === "AbortError") {
      if (!hadFatalError) {
        asstBubble.appendChild(el("div","notice notice-stopped","Stopped."));
        asstBubble.removeAttribute("aria-busy"); announce("Generation stopped.");
      }
    } else {
      asstBubble.appendChild(el("div","notice notice-error","Connection error. Please try again."));
      asstBubble.removeAttribute("aria-busy"); announce("Connection error.");
    }
  } finally {
    if (answerEl) answerEl.classList.remove("streaming");
    if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
    setStreaming(false); abortController = null;
    if (!endEvt && !hadFatalError) asstBubble.removeAttribute("aria-busy");
  }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------
inputEl.addEventListener("input", updateCharCount);
inputEl.addEventListener("keydown", function(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
btnSend.addEventListener("click", function() { sendMessage(); });
btnStop.addEventListener("click", function() { if (abortController) abortController.abort(); });
document.querySelectorAll(".example-card").forEach(function(btn) {
  btn.addEventListener("click", function() { var q = btn.dataset.query; if (q) sendMessage(q); });
});
updateCharCount();
