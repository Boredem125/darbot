/**
 * app.js -- UI wiring for the Wasalt / DarGlobal RAG chatbot.
 * No framework, no build step. Uses only textContent and createElement.
 * Never uses innerHTML with untrusted strings.
 */
import { createSseParser, parseEvent } from "./sse.js";
import { getAnswerModeLabel, sourceBadgeLabel } from "./labels.js";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const conversationEl = document.getElementById("conversation");
const emptyStateEl = document.getElementById("empty-state");
const inputEl = document.getElementById("message-input");
const charCountEl = document.getElementById("char-count");
const btnSend = document.getElementById("btn-send");
const btnStop = document.getElementById("btn-stop");
const srAnnounce = document.getElementById("sr-announce");

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let abortController = null;
let streaming = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a text node. */
function txt(str) {
  return document.createTextNode(str);
}

/** Create an element with optional className and textContent. */
function el(tag, className, textContent) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (textContent !== undefined && textContent !== null) e.textContent = textContent;
  return e;
}

/** Format a number with comma separators. */
function fmtNum(n) {
  if (n == null) return null;
  return Number(n).toLocaleString("en-US");
}

/** Format price in SAR. */
function fmtPrice(amount) {
  if (amount == null) return null;
  return "SAR " + fmtNum(amount);
}

/** Format an ISO date string as a short date. */
function fmtDate(iso) {
  if (!iso) return null;
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      year: "numeric", month: "short", day: "numeric",
    });
  } catch { return iso; }
}

/** Format milliseconds. */
function fmtMs(ms) {
  if (ms == null) return null;
  return Math.round(ms) + " ms";
}

/** Check if a URL is safe to link to. */
function isSafeUrl(url) {
  return typeof url === "string" && (url.startsWith("https://") || url.startsWith("http://"));
}

/** Set UI to streaming or idle state. */
function setStreaming(isStreaming) {
  streaming = isStreaming;
  inputEl.disabled = isStreaming;
  btnSend.disabled = isStreaming;
  btnSend.hidden = isStreaming;
  btnStop.hidden = !isStreaming;
  if (!isStreaming) inputEl.focus();
}

/** Screen reader announcement. */
function announce(text) {
  srAnnounce.textContent = "";
  requestAnimationFrame(function () { srAnnounce.textContent = text; });
}

/** Update character counter. */
function updateCharCount() {
  var len = inputEl.value.length;
  charCountEl.textContent = len + " / 500";
  if (len > 500) {
    charCountEl.classList.add("over-limit");
  } else {
    charCountEl.classList.remove("over-limit");
  }
  btnSend.disabled = streaming || len === 0 || len > 500;
}

/** Hide the empty state. */
function hideEmptyState() {
  if (emptyStateEl) emptyStateEl.classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------

function createTypingIndicator() {
  var wrap = el("div", "typing-indicator");
  wrap.setAttribute("aria-label", "Assistant is typing");
  wrap.setAttribute("role", "status");
  for (var i = 0; i < 3; i++) {
    wrap.appendChild(el("span", "typing-dot"));
  }
  return wrap;
}

// ---------------------------------------------------------------------------
// Card rendering
// ---------------------------------------------------------------------------

function renderCard(card) {
  var cardEl = el("div", "source-card");
  cardEl.setAttribute("dir", "auto");
  cardEl.id = "card-" + card.n;

  // Header: index badge + title + source
  var header = el("div", "card-header");
  header.appendChild(el("span", "card-index", String(card.n)));

  var titleWrap = el("div", "card-title-wrap");
  if (isSafeUrl(card.url)) {
    var titleLink = el("a", "card-title");
    titleLink.textContent = card.title || "";
    titleLink.href = card.url;
    titleLink.target = "_blank";
    titleLink.rel = "noopener noreferrer";
    titleWrap.appendChild(titleLink);
  } else {
    titleWrap.appendChild(el("span", "card-title", card.title || ""));
  }
  titleWrap.appendChild(el("span", "card-source-badge", sourceBadgeLabel(card.source)));
  header.appendChild(titleWrap);
  cardEl.appendChild(header);

  // Price
  if (card.kind === "listing") {
    if (card.price_sar != null) {
      var priceStr = fmtPrice(card.price_sar);
      if (card.rent_period) priceStr += " / " + card.rent_period;
      cardEl.appendChild(el("div", "card-price", priceStr));
      if (card.annual_rent_sar != null) {
        cardEl.appendChild(el("div", "card-price-annual", fmtPrice(card.annual_rent_sar) + " / year"));
      }
    }
  } else if (card.kind === "project") {
    if (card.price_from_amount != null && card.price_from_currency) {
      cardEl.appendChild(el("div", "card-price",
        "from " + card.price_from_currency + " " + fmtNum(card.price_from_amount)));
    }
  }

  // Chips (beds, baths, area)
  if (card.kind === "listing") {
    var chips = el("div", "card-chips");
    var hasChips = false;
    if (card.bedrooms != null) {
      chips.appendChild(el("span", "card-chip", card.bedrooms + " bed"));
      hasChips = true;
    }
    if (card.bathrooms != null) {
      chips.appendChild(el("span", "card-chip", card.bathrooms + " bath"));
      hasChips = true;
    }
    if (card.area_sqm != null) {
      chips.appendChild(el("span", "card-chip", fmtNum(card.area_sqm) + " sqm"));
      hasChips = true;
    }
    if (card.purpose) {
      chips.appendChild(el("span", "card-chip", card.purpose));
      hasChips = true;
    }
    if (card.property_type) {
      chips.appendChild(el("span", "card-chip", card.property_type));
      hasChips = true;
    }
    if (hasChips) cardEl.appendChild(chips);
  }

  // Location / project details
  if (card.kind === "listing") {
    var locParts = [];
    if (card.city) locParts.push(card.city);
    if (card.district) locParts.push(card.district);
    if (locParts.length) {
      cardEl.appendChild(el("div", "card-location", locParts.join(", ")));
    }
  } else if (card.kind === "project") {
    var projParts = [];
    if (card.city) projParts.push(card.city);
    if (card.country) projParts.push(card.country);
    if (card.brand) projParts.push(card.brand);
    if (projParts.length) {
      cardEl.appendChild(el("div", "card-location", projParts.join(" · ")));
    }
    if (card.handover) {
      cardEl.appendChild(el("div", "card-location", "Handover: " + card.handover));
    }
  } else if (card.kind === "article") {
    var artParts = [];
    if (card.article_kind) artParts.push(card.article_kind);
    var pubDate = fmtDate(card.published_at);
    if (pubDate) artParts.push(pubDate);
    if (artParts.length) {
      cardEl.appendChild(el("div", "card-location", artParts.join(" · ")));
    }
  }

  // Snippet
  if (card.snippet) {
    cardEl.appendChild(el("div", "card-snippet", card.snippet));
  }

  // Scraped at
  if (card.kind === "listing") {
    var scrapedDate = fmtDate(card.scraped_at);
    if (scrapedDate) {
      cardEl.appendChild(el("div", "card-scraped", "as of " + scrapedDate));
    }
  }

  return cardEl;
}

function renderCards(sources) {
  if (!sources || sources.length === 0) return null;
  var row = el("div", "cards-row");
  row.setAttribute("role", "list");
  row.setAttribute("aria-label", "Source cards");
  for (var i = 0; i < sources.length; i++) {
    var c = renderCard(sources[i]);
    c.setAttribute("role", "listitem");
    row.appendChild(c);
  }
  return row;
}

// ---------------------------------------------------------------------------
// Answer text: post-process [n] citations into clickable links
// ---------------------------------------------------------------------------

function postProcessAnswer(rawText, numCards) {
  // Normalise fullwidth citation brackets to ASCII
  var normalised = rawText
    .replace(/【(\d+)】/g, "[$1]")
    .replace(/〔(\d+)〕/g, "[$1]")
    .replace(/［(\d+)］/g, "[$1]");

  var parts = normalised.split(/(\[\d+\])/g);
  var frag = document.createDocumentFragment();
  var inList = false;
  var currentUl = null;

  for (var pi = 0; pi < parts.length; pi++) {
    var part = parts[pi];
    var citMatch = part.match(/^\[(\d+)\]$/);
    if (citMatch) {
      var n = parseInt(citMatch[1], 10);
      if (n >= 1 && n <= numCards) {
        var btn = el("button", "citation-link", "[" + n + "]");
        btn.setAttribute("aria-label", "Go to source " + n);
        btn.type = "button";
        btn.dataset.card = n;
        btn.addEventListener("click", onCitationClick);
        if (inList && currentUl && currentUl.lastChild) {
          currentUl.lastChild.appendChild(btn);
        } else {
          frag.appendChild(btn);
        }
      } else {
        frag.appendChild(txt(part));
      }
    } else {
      var lines = part.split("\n");
      for (var li = 0; li < lines.length; li++) {
        var line = lines[li];
        if (line.startsWith("- ")) {
          if (!inList) {
            currentUl = document.createElement("ul");
            frag.appendChild(currentUl);
            inList = true;
          }
          var liEl = document.createElement("li");
          liEl.textContent = line.slice(2);
          currentUl.appendChild(liEl);
        } else {
          if (inList) { inList = false; currentUl = null; }
          if (li > 0) frag.appendChild(txt("\n"));
          frag.appendChild(txt(line));
        }
      }
    }
  }
  return frag;
}

function onCitationClick(e) {
  var n = e.currentTarget.dataset.card;
  var target = document.getElementById("card-" + n);
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    target.classList.add("highlight");
    setTimeout(function () { target.classList.remove("highlight"); }, 2000);
  }
}

// ---------------------------------------------------------------------------
// Response details toggle
// ---------------------------------------------------------------------------

function renderResponseDetails(endEvt) {
  var details = document.createElement("details");
  details.className = "response-details";

  var summary = document.createElement("summary");

  // Summary text
  var totalMs = endEvt.timings ? endEvt.timings.total_ms : null;
  var summaryText = "Details";
  if (totalMs != null) summaryText += " · " + fmtMs(totalMs);
  summary.appendChild(txt(summaryText));

  // Summary badges
  var sbadges = el("span", "summary-badges");
  if (endEvt.answer_mode === "cache" && endEvt.cache_tier && endEvt.cache_tier !== "miss") {
    sbadges.appendChild(el("span", "summary-tag summary-tag-cached", "cached " + endEvt.cache_tier));
  }
  if (endEvt.grounding && endEvt.grounding.failures && endEvt.grounding.failures.length > 0) {
    sbadges.appendChild(el("span", "summary-tag summary-tag-warn", "grounding warn"));
  }
  if (sbadges.childNodes.length > 0) summary.appendChild(sbadges);
  details.appendChild(summary);

  // Content
  var content = el("div", "details-content");

  var addDetail = function (label, value) {
    if (value == null) return;
    var item = el("span", "detail-item");
    item.appendChild(el("span", "detail-label", label + ": "));
    item.appendChild(txt(String(value)));
    content.appendChild(item);
  };

  // Timings
  var t = endEvt.timings || {};
  addDetail("total", fmtMs(t.total_ms));
  addDetail("TTFT", fmtMs(t.llm_ttft_ms));
  addDetail("LLM", fmtMs(t.llm_total_ms));
  addDetail("embed", fmtMs(t.embed_ms));
  addDetail("sparse", fmtMs(t.sparse_ms));
  addDetail("retrieve", fmtMs(t.retrieve_ms));
  addDetail("rerank", fmtMs(t.rerank_ms));
  addDetail("SQL", fmtMs(t.sql_ms));
  addDetail("grounding", fmtMs(t.grounding_ms));
  addDetail("cache lookup", fmtMs(t.cache_lookup_ms));

  // Cache
  addDetail("cache tier", endEvt.cache_tier);
  if (endEvt.cache_hits && endEvt.cache_hits.length > 0) {
    addDetail("cache hits", endEvt.cache_hits.join(", "));
  }

  // Provider and model
  if (endEvt.provider && endEvt.provider !== "none") {
    var provStr = endEvt.provider.charAt(0).toUpperCase() + endEvt.provider.slice(1);
    if (endEvt.model) provStr += " / " + endEvt.model;
    addDetail("model", provStr);
  } else if (endEvt.model) {
    addDetail("model", endEvt.model);
  }

  // Tokens
  var tok = endEvt.tokens;
  if (tok && (tok.prompt_tokens > 0 || tok.completion_tokens > 0)) {
    addDetail("tokens", tok.prompt_tokens + " in / " + tok.completion_tokens + " out");
  }

  // Grounding
  var g = endEvt.grounding;
  if (g) {
    var parts = [];
    if (g.numbers_ok != null) parts.push("numbers: " + (g.numbers_ok ? "ok" : "fail"));
    if (g.entities_ok != null) parts.push("entities: " + (g.entities_ok ? "ok" : "fail"));
    if (g.citations_ok != null) parts.push("citations: " + (g.citations_ok ? "ok" : "fail"));
    if (parts.length > 0) addDetail("grounding", parts.join(", "));
  }

  // Answer mode
  addDetail("answer mode", endEvt.answer_mode);
  addDetail("finish", endEvt.finish_reason);

  details.appendChild(content);
  return details;
}

// ---------------------------------------------------------------------------
// Badges (outside the details toggle)
// ---------------------------------------------------------------------------

function renderBadges(endEvt, route) {
  var frag = document.createDocumentFragment();

  // Unverified badge
  if (endEvt.flags && endEvt.flags.unverified) {
    var badge = el("span", "badge badge-unverified", " Unverified figures");
    frag.appendChild(badge);

    if (endEvt.grounding && endEvt.grounding.failures && endEvt.grounding.failures.length > 0) {
      var deets = document.createElement("details");
      deets.className = "badge-failures";
      var sum = document.createElement("summary");
      sum.textContent = "Grounding failures (" + endEvt.grounding.failures.length + ")";
      deets.appendChild(sum);
      var ul = document.createElement("ul");
      for (var fi = 0; fi < endEvt.grounding.failures.length; fi++) {
        var liF = document.createElement("li");
        liF.textContent = endEvt.grounding.failures[fi];
        ul.appendChild(liF);
      }
      deets.appendChild(ul);
      frag.appendChild(deets);
    }
  }

  // Truncated badge
  if (endEvt.flags && endEvt.flags.truncated) {
    frag.appendChild(el("span", "badge badge-truncated", " Response cut off"));
  }

  // Answer mode label
  var label = getAnswerModeLabel(route, endEvt.answer_mode);
  if (label) {
    frag.appendChild(el("div", "mode-label mode-label-" + label.style, label.text));
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

  // Clear input
  inputEl.value = "";
  updateCharCount();

  // Create turn
  var turn = el("div", "turn");
  var userBubble = el("div", "bubble-user");
  userBubble.setAttribute("dir", "auto");
  userBubble.textContent = message;
  turn.appendChild(userBubble);

  var assistantBubble = el("div", "bubble-assistant");
  assistantBubble.setAttribute("aria-busy", "true");
  turn.appendChild(assistantBubble);

  conversationEl.appendChild(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "end" });

  setStreaming(true);
  abortController = new AbortController();

  var answerEl = null;
  var rawAnswer = "";
  var numCards = 0;
  var startEvt = null;
  var endEvt = null;
  var hadFatalError = false;
  var typingEl = null;
  var firstTokenReceived = false;

  try {
    var response = await fetch("api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify({ message: message, lang: null }),
      credentials: "same-origin",
      signal: abortController.signal,
    });

    // Non-200: JSON error
    if (!response.ok) {
      var errorMsg;
      try {
        var errBody = await response.json();
        if (response.status === 429 && errBody.retry_after_s) {
          errorMsg = "Too many requests. Try again in " + errBody.retry_after_s + " s.";
        } else if (response.status === 503) {
          errorMsg = "The service is warming up. Please try again shortly.";
        } else if (response.status === 422) {
          errorMsg = "Validation error: " + (errBody.detail || "Invalid request.");
        } else {
          errorMsg = errBody.detail || errBody.message || "An error occurred.";
        }
      } catch (_) {
        errorMsg = "Request failed (HTTP " + response.status + ").";
      }
      assistantBubble.appendChild(el("div", "notice notice-error", errorMsg));
      assistantBubble.removeAttribute("aria-busy");
      setStreaming(false);
      announce("Error: " + errorMsg);
      return;
    }

    // Stream SSE
    var reader = response.body.getReader();
    var decoder = new TextDecoder("utf-8", { fatal: false });

    var parser = createSseParser(function (frame) {
      var payload;
      try {
        payload = parseEvent(frame);
      } catch (e) {
        assistantBubble.appendChild(el("div", "notice notice-error", e.message));
        hadFatalError = true;
        abortController.abort();
        return;
      }

      if (payload.event === "start") {
        startEvt = payload;
        numCards = (payload.sources || []).length;

        // Language
        if (payload.language) {
          assistantBubble.setAttribute("lang", payload.language);
          if (payload.language === "ar") {
            assistantBubble.setAttribute("dir", "rtl");
          }
        }

        // Cards
        var cardsRow = renderCards(payload.sources);
        if (cardsRow) assistantBubble.appendChild(cardsRow);

        // Typing indicator
        typingEl = createTypingIndicator();
        assistantBubble.appendChild(typingEl);

        // Prepare answer element
        answerEl = el("div", "answer-text streaming");
        answerEl.setAttribute("dir", "auto");

      } else if (payload.event === "token") {
        // Remove typing indicator on first token
        if (!firstTokenReceived) {
          firstTokenReceived = true;
          if (typingEl && typingEl.parentNode) {
            typingEl.parentNode.removeChild(typingEl);
          }
          if (answerEl) assistantBubble.appendChild(answerEl);
        }
        if (!answerEl) {
          answerEl = el("div", "answer-text streaming");
          answerEl.setAttribute("dir", "auto");
          assistantBubble.appendChild(answerEl);
        }
        rawAnswer += payload.delta;
        answerEl.appendChild(txt(payload.delta));

      } else if (payload.event === "end") {
        endEvt = payload;

        // Remove typing indicator if no tokens arrived
        if (typingEl && typingEl.parentNode) {
          typingEl.parentNode.removeChild(typingEl);
        }

        // Remove streaming cursor
        if (answerEl) {
          answerEl.classList.remove("streaming");
        }

        // Post-process citations
        if (answerEl && rawAnswer && numCards > 0) {
          while (answerEl.firstChild) answerEl.removeChild(answerEl.firstChild);
          answerEl.appendChild(postProcessAnswer(rawAnswer, numCards));
        }

        // Badges (unverified, truncated, mode label)
        var route = startEvt ? startEvt.route : null;
        var badgesNode = renderBadges(payload, route);
        if (badgesNode.childNodes.length > 0) {
          var badgeRow = el("div", "badges-row");
          badgeRow.appendChild(badgesNode);
          assistantBubble.appendChild(badgeRow);
        }

        // Response details toggle
        var detailsEl = renderResponseDetails(payload);
        assistantBubble.appendChild(detailsEl);

        assistantBubble.removeAttribute("aria-busy");

        var annText = "Answer complete.";
        if (payload.flags && payload.flags.unverified) annText += " Warning: unverified figures.";
        if (payload.flags && payload.flags.truncated) annText += " Note: answer was cut off.";
        announce(annText);

      } else if (payload.event === "error") {
        if (payload.fatal) {
          hadFatalError = true;
          // Remove typing indicator
          if (typingEl && typingEl.parentNode) {
            typingEl.parentNode.removeChild(typingEl);
          }
          if (answerEl) answerEl.classList.remove("streaming");
          assistantBubble.appendChild(
            el("div", "notice notice-fatal", payload.message || "An unexpected error occurred.")
          );
          assistantBubble.removeAttribute("aria-busy");
          announce("Error: " + (payload.message || "An unexpected error occurred."));
        } else {
          // Soft error: inline notice, stream continues
          var noticeClass = "notice notice-info";
          if (payload.degraded_to) {
            noticeClass = "notice notice-warn";
          }
          assistantBubble.appendChild(
            el("div", noticeClass, payload.message || "Switching to a fallback.")
          );
        }
      }
      // Unknown events silently ignored
    });

    // Read chunks
    while (true) {
      var result = await reader.read();
      if (result.done) break;
      parser.feed(decoder.decode(result.value, { stream: true }));
    }
    var remaining = decoder.decode();
    if (remaining) parser.feed(remaining);

  } catch (e) {
    if (e.name === "AbortError") {
      if (!hadFatalError) {
        assistantBubble.appendChild(el("div", "notice notice-stopped", "Stopped."));
        assistantBubble.removeAttribute("aria-busy");
        announce("Generation stopped.");
      }
    } else {
      assistantBubble.appendChild(
        el("div", "notice notice-error", "Connection error. Please try again.")
      );
      assistantBubble.removeAttribute("aria-busy");
      announce("Connection error.");
    }
  } finally {
    if (answerEl) answerEl.classList.remove("streaming");
    if (typingEl && typingEl.parentNode) typingEl.parentNode.removeChild(typingEl);
    setStreaming(false);
    abortController = null;
    if (!endEvt && !hadFatalError) {
      assistantBubble.removeAttribute("aria-busy");
    }
  }
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

inputEl.addEventListener("input", updateCharCount);

inputEl.addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

btnSend.addEventListener("click", function () { sendMessage(); });

btnStop.addEventListener("click", function () {
  if (abortController) abortController.abort();
});

// Example query buttons
document.querySelectorAll(".example-query").forEach(function (btn) {
  btn.addEventListener("click", function () {
    var query = btn.dataset.query;
    if (query) sendMessage(query);
  });
});

// Initial state
updateCharCount();
