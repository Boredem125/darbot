/**
 * app.js — UI wiring for the Wasalt / DarGlobal RAG chatbot.
 * No framework, no build step. Uses only textContent and createElement — never innerHTML.
 * Imports the pure SSE parser from sse.js.
 */
import { createSseParser, parseEvent } from "./sse.js";

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const conversationEl = document.getElementById("conversation");
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

/** Safely create a text node. */
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

/** Format a number with thousands separators. */
function fmtNum(n) {
  if (n == null) return null;
  return Number(n).toLocaleString("en-US");
}

/** Format price in SAR with thousands separator. */
function fmtPrice(amount) {
  if (amount == null) return null;
  return "SAR " + fmtNum(amount);
}

/** Format an ISO date string as a short date. */
function fmtDate(iso) {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

/** Format milliseconds to a readable string. */
function fmtMs(ms) {
  if (ms == null) return null;
  return Math.round(ms) + " ms";
}

/** Determine if a URL is safe to link to. */
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
  if (!isStreaming) {
    inputEl.focus();
  }
}

/** Announce text for screen readers. */
function announce(text) {
  srAnnounce.textContent = "";
  // Force re-announcement by clearing first, then setting after a tick
  requestAnimationFrame(() => {
    srAnnounce.textContent = text;
  });
}

/** Update character counter. */
function updateCharCount() {
  const len = inputEl.value.length;
  charCountEl.textContent = len + " / 500";
  if (len > 500) {
    charCountEl.classList.add("over-limit");
  } else {
    charCountEl.classList.remove("over-limit");
  }
  btnSend.disabled = streaming || len === 0 || len > 500;
}

// ---------------------------------------------------------------------------
// Card rendering
// ---------------------------------------------------------------------------

function renderCard(card) {
  const cardEl = el("div", "source-card");
  cardEl.setAttribute("dir", "auto");
  cardEl.id = "card-" + card.n;

  // Header: badge + title
  const header = el("div", "card-header");
  const badge = el("span", "card-badge", String(card.n));
  header.appendChild(badge);

  if (isSafeUrl(card.url)) {
    const titleLink = el("a", "card-title");
    titleLink.textContent = card.title || "";
    titleLink.href = card.url;
    titleLink.target = "_blank";
    titleLink.rel = "noopener noreferrer";
    header.appendChild(titleLink);
  } else {
    header.appendChild(el("span", "card-title", card.title || ""));
  }
  cardEl.appendChild(header);

  // Meta info depends on kind
  const meta = el("div", "card-meta");

  if (card.kind === "listing") {
    const parts = [];
    if (card.purpose) parts.push(card.purpose);
    if (card.property_type) parts.push(card.property_type);
    if (parts.length) {
      meta.appendChild(el("div", "card-meta-row", parts.join(" · ")));
    }

    const locParts = [];
    if (card.city) locParts.push(card.city);
    if (card.district) locParts.push(card.district);
    if (locParts.length) {
      meta.appendChild(el("div", "card-meta-row", locParts.join(", ")));
    }

    // Price
    if (card.price_sar != null) {
      const priceEl = el("div", "card-price");
      if (card.rent_period) {
        priceEl.textContent = fmtPrice(card.price_sar) + " / " + card.rent_period;
      } else {
        priceEl.textContent = fmtPrice(card.price_sar);
      }
      cardEl.appendChild(priceEl);

      // Annual rent if present
      if (card.annual_rent_sar != null) {
        const annualEl = el("div", "card-meta-row", fmtPrice(card.annual_rent_sar) + " / year");
        meta.appendChild(annualEl);
      }
    }

    // Details row
    const details = [];
    if (card.bedrooms != null) details.push(card.bedrooms + " bed");
    if (card.bathrooms != null) details.push(card.bathrooms + " bath");
    if (card.area_sqm != null) details.push(fmtNum(card.area_sqm) + " sqm");
    if (details.length) {
      meta.appendChild(el("div", "card-meta-row", details.join(" · ")));
    }

    // Scraped at
    const scrapedDate = fmtDate(card.scraped_at);
    if (scrapedDate) {
      meta.appendChild(el("div", "card-meta-row", "as of " + scrapedDate));
    }
  } else if (card.kind === "project") {
    const parts = [];
    if (card.country) parts.push(card.country);
    if (card.city) parts.push(card.city);
    if (card.brand) parts.push(card.brand);
    if (parts.length) {
      meta.appendChild(el("div", "card-meta-row", parts.join(" · ")));
    }
    if (card.handover) {
      meta.appendChild(el("div", "card-meta-row", "Handover: " + card.handover));
    }
    if (card.price_from_amount != null && card.price_from_currency) {
      const priceEl = el("div", "card-price",
        "from " + card.price_from_currency + " " + fmtNum(card.price_from_amount));
      cardEl.appendChild(priceEl);
    }
  } else if (card.kind === "article") {
    if (card.article_kind) {
      meta.appendChild(el("div", "card-meta-row", card.article_kind));
    }
    const pubDate = fmtDate(card.published_at);
    if (pubDate) {
      meta.appendChild(el("div", "card-meta-row", pubDate));
    }
  }

  cardEl.appendChild(meta);

  // Snippet
  if (card.snippet) {
    cardEl.appendChild(el("div", "card-snippet", card.snippet));
  }

  return cardEl;
}

function renderCards(sources) {
  if (!sources || sources.length === 0) return null;
  const row = el("div", "cards-row");
  row.setAttribute("role", "list");
  row.setAttribute("aria-label", "Source cards");
  for (const card of sources) {
    const c = renderCard(card);
    c.setAttribute("role", "listitem");
    row.appendChild(c);
  }
  return row;
}

// ---------------------------------------------------------------------------
// Chips
// ---------------------------------------------------------------------------

function renderChips(startEvt) {
  const wrapper = el("div", "card-meta");
  wrapper.style.display = "flex";
  wrapper.style.gap = "0.375rem";
  wrapper.style.flexWrap = "wrap";

  // Route chip
  const routeLabel = {
    lookup: "Lookup",
    aggregate: "Aggregate",
    project_info: "Project",
    out_of_scope: "Out of scope",
  };
  const routeChip = el("span", "chip chip-route", routeLabel[startEvt.route] || startEvt.route);
  wrapper.appendChild(routeChip);

  // Cache chip
  if (startEvt.cache_tier && startEvt.cache_tier !== "miss") {
    const cacheChip = el("span", "chip chip-cache", "cached (" + startEvt.cache_tier + ")");
    wrapper.appendChild(cacheChip);
  }

  return wrapper;
}

// ---------------------------------------------------------------------------
// Answer text: post-process [n] citations into clickable links
// ---------------------------------------------------------------------------

function postProcessAnswer(rawText, numCards) {
  // Normalise fullwidth citation brackets to ASCII [n] before splitting.
  // Groq gpt-oss-120b writes 【n】; also handle 〔n〕 and ［n］.
  var normalised = rawText
    .replace(/【(\d+)】/g, "[$1]")
    .replace(/〔(\d+)〕/g, "[$1]")
    .replace(/［(\d+)］/g, "[$1]");
  // Split on [n] patterns, returning an array of text and citation markers
  // Pattern: [1], [2], etc.
  const parts = normalised.split(/(\[\d+\])/g);
  const frag = document.createDocumentFragment();

  let inList = false;
  let currentUl = null;

  for (const part of parts) {
    const citMatch = part.match(/^\[(\d+)\]$/);
    if (citMatch) {
      const n = parseInt(citMatch[1], 10);
      if (n >= 1 && n <= numCards) {
        const btn = el("button", "citation-link", "[" + n + "]");
        btn.setAttribute("aria-label", "Go to source " + n);
        btn.type = "button";
        btn.addEventListener("click", function () {
          const target = document.getElementById("card-" + n);
          if (target) {
            target.scrollIntoView({ behavior: "smooth", block: "nearest" });
            target.classList.add("highlight");
            setTimeout(function () { target.classList.remove("highlight"); }, 2000);
          }
        });
        if (inList && currentUl && currentUl.lastChild) {
          currentUl.lastChild.appendChild(btn);
        } else {
          frag.appendChild(btn);
        }
      } else {
        frag.appendChild(txt(part));
      }
    } else {
      // Handle lines starting with "- " as list items
      const lines = part.split("\n");
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.startsWith("- ")) {
          if (!inList) {
            currentUl = document.createElement("ul");
            frag.appendChild(currentUl);
            inList = true;
          }
          const li = document.createElement("li");
          li.textContent = line.slice(2);
          currentUl.appendChild(li);
        } else {
          if (inList && line.trim() === "") {
            // End list on blank line
            inList = false;
            currentUl = null;
          } else if (inList && line.trim() !== "") {
            inList = false;
            currentUl = null;
          }
          if (i > 0) {
            frag.appendChild(txt("\n"));
          }
          frag.appendChild(txt(line));
        }
      }
    }
  }

  return frag;
}

// ---------------------------------------------------------------------------
// Latency strip
// ---------------------------------------------------------------------------

function renderLatencyStrip(endEvt) {
  const strip = el("div", "latency-strip");
  strip.setAttribute("role", "status");
  strip.setAttribute("aria-label", "Response details");

  // Timing
  const timings = endEvt.timings || {};
  const addTiming = (label, val) => {
    if (val == null) return;
    const s = el("span");
    const lbl = el("span", "label", label + ": ");
    s.appendChild(lbl);
    s.appendChild(txt(fmtMs(val)));
    strip.appendChild(s);
  };

  addTiming("total", timings.total_ms);
  addTiming("TTFT", timings.llm_ttft_ms);
  addTiming("embed", timings.embed_ms);
  addTiming("sparse", timings.sparse_ms);
  addTiming("retrieve", timings.retrieve_ms);
  addTiming("rerank", timings.rerank_ms);
  addTiming("SQL", timings.sql_ms);
  addTiming("grounding", timings.grounding_ms);
  addTiming("cache lookup", timings.cache_lookup_ms);

  // Cache tier
  if (endEvt.cache_tier) {
    const c = el("span");
    c.appendChild(el("span", "label", "cache: "));
    c.appendChild(txt(endEvt.cache_tier));
    strip.appendChild(c);
  }

  // Cache hits
  if (endEvt.cache_hits && endEvt.cache_hits.length > 0) {
    const c = el("span");
    c.appendChild(el("span", "label", "hits: "));
    c.appendChild(txt(endEvt.cache_hits.join(", ")));
    strip.appendChild(c);
  }

  // Provider and model
  if (endEvt.provider || endEvt.model) {
    const c = el("span");
    c.appendChild(el("span", "label", "model: "));
    if (endEvt.answer_mode === "cache" && endEvt.cache_tier && endEvt.cache_tier !== "miss") {
      let modelStr = "cached (" + endEvt.cache_tier + ")";
      if (endEvt.provider && endEvt.provider !== "none") {
        modelStr += " · originally " + endEvt.provider;
      }
      if (endEvt.model) {
        modelStr += " · " + endEvt.model;
      }
      c.appendChild(txt(modelStr));
    } else {
      const parts = [];
      if (endEvt.provider && endEvt.provider !== "none") {
        parts.push(endEvt.provider.charAt(0).toUpperCase() + endEvt.provider.slice(1));
      }
      if (endEvt.model) parts.push(endEvt.model);
      c.appendChild(txt(parts.join(" · ")));
    }
    strip.appendChild(c);
  }

  // Tokens
  const tokens = endEvt.tokens;
  if (tokens && (tokens.prompt_tokens > 0 || tokens.completion_tokens > 0)) {
    const c = el("span");
    c.appendChild(el("span", "label", "tokens: "));
    c.appendChild(txt(tokens.prompt_tokens + " / " + tokens.completion_tokens));
    strip.appendChild(c);
  }

  return strip;
}

// ---------------------------------------------------------------------------
// Badges
// ---------------------------------------------------------------------------

function renderBadges(endEvt) {
  const frag = document.createDocumentFragment();

  // Unverified badge
  if (endEvt.flags && endEvt.flags.unverified) {
    const badge = el("span", "badge badge-unverified", " Unverified figures");
    frag.appendChild(badge);

    // Failures details
    if (endEvt.grounding && endEvt.grounding.failures && endEvt.grounding.failures.length > 0) {
      const details = document.createElement("details");
      details.className = "badge-details";
      const summary = document.createElement("summary");
      summary.textContent = "Grounding failures (" + endEvt.grounding.failures.length + ")";
      details.appendChild(summary);
      const ul = document.createElement("ul");
      for (const f of endEvt.grounding.failures) {
        const li = document.createElement("li");
        li.textContent = f;
        ul.appendChild(li);
      }
      details.appendChild(ul);
      frag.appendChild(details);
    }
  }

  // Truncated badge
  if (endEvt.flags && endEvt.flags.truncated) {
    const badge = el("span", "badge badge-truncated", " Cut off");
    frag.appendChild(badge);
  }

  // Answer mode labels
  const modeLabels = {
    retrieval_only: "AI summary unavailable; showing matching sources",
    static_busy: "The service is currently busy. Please try again shortly.",
    refusal: "This question is outside the scope of this service.",
  };
  if (endEvt.answer_mode && modeLabels[endEvt.answer_mode]) {
    const badge = el("span", "badge badge-mode", modeLabels[endEvt.answer_mode]);
    frag.appendChild(badge);
  }

  return frag;
}

// ---------------------------------------------------------------------------
// Main send flow
// ---------------------------------------------------------------------------

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message || message.length > 500 || streaming) return;

  // Clear input
  inputEl.value = "";
  updateCharCount();

  // Create turn container
  const turn = el("div", "turn");

  // User bubble
  const userBubble = el("div", "bubble-user");
  userBubble.setAttribute("dir", "auto");
  userBubble.textContent = message;
  turn.appendChild(userBubble);

  // Assistant bubble container
  const assistantBubble = el("div", "bubble-assistant");
  assistantBubble.setAttribute("aria-busy", "true");
  turn.appendChild(assistantBubble);

  conversationEl.appendChild(turn);
  turn.scrollIntoView({ behavior: "smooth", block: "end" });

  // Start streaming
  setStreaming(true);
  abortController = new AbortController();

  let answerEl = null;
  let rawAnswer = "";
  let numCards = 0;
  let startEvt = null;
  let endEvt = null;
  let hadFatalError = false;

  try {
    const response = await fetch("api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
      },
      body: JSON.stringify({ message: message, lang: null }),
      credentials: "same-origin",
      signal: abortController.signal,
    });

    // Non-200 responses are JSON, not SSE
    if (!response.ok) {
      let errorMsg;
      try {
        const errBody = await response.json();
        if (response.status === 429 && errBody.retry_after_s) {
          errorMsg = "Too many requests, try again in " + errBody.retry_after_s + " s.";
        } else if (response.status === 503) {
          errorMsg = "The service is warming up. Please try again shortly.";
        } else if (response.status === 422) {
          errorMsg = "Validation error: " + (errBody.detail || "Invalid request.");
        } else {
          errorMsg = errBody.detail || errBody.message || "An error occurred.";
        }
      } catch {
        errorMsg = "Request failed (HTTP " + response.status + ").";
      }
      const notice = el("div", "notice notice-error", errorMsg);
      assistantBubble.appendChild(notice);
      assistantBubble.removeAttribute("aria-busy");
      setStreaming(false);
      announce("Error: " + errorMsg);
      return;
    }

    // Stream SSE
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8", { fatal: false });

    const parser = createSseParser(function (frame) {
      let payload;
      try {
        payload = parseEvent(frame);
      } catch (e) {
        // Unknown version or bad JSON — show client error and end
        const notice = el("div", "notice notice-error", e.message);
        assistantBubble.appendChild(notice);
        hadFatalError = true;
        abortController.abort();
        return;
      }

      const eventName = payload.event;

      if (eventName === "start") {
        startEvt = payload;
        numCards = (payload.sources || []).length;

        // Set language on the assistant bubble
        if (payload.language) {
          assistantBubble.setAttribute("lang", payload.language);
        }

        // Render chips
        const chips = renderChips(payload);
        assistantBubble.appendChild(chips);

        // Render cards immediately
        const cardsRow = renderCards(payload.sources);
        if (cardsRow) {
          assistantBubble.appendChild(cardsRow);
        }

        // Create answer placeholder
        answerEl = el("div", "answer-text");
        answerEl.setAttribute("dir", "auto");
        assistantBubble.appendChild(answerEl);

      } else if (eventName === "token") {
        if (!answerEl) {
          answerEl = el("div", "answer-text");
          answerEl.setAttribute("dir", "auto");
          assistantBubble.appendChild(answerEl);
        }
        rawAnswer += payload.delta;
        // Cheap append: just add a text node for the delta
        answerEl.appendChild(txt(payload.delta));

      } else if (eventName === "end") {
        endEvt = payload;

        // Post-process the answer text to link citations
        if (answerEl && rawAnswer && numCards > 0) {
          // Clear and re-render with citation links
          while (answerEl.firstChild) answerEl.removeChild(answerEl.firstChild);
          answerEl.appendChild(postProcessAnswer(rawAnswer, numCards));
        }

        // Latency strip
        const strip = renderLatencyStrip(payload);
        assistantBubble.appendChild(strip);

        // Badges
        const badges = renderBadges(payload);
        if (badges.childNodes.length > 0) {
          const badgeWrap = el("div");
          badgeWrap.style.display = "flex";
          badgeWrap.style.flexWrap = "wrap";
          badgeWrap.style.gap = "0.375rem";
          badgeWrap.style.alignItems = "flex-start";
          badgeWrap.appendChild(badges);
          assistantBubble.appendChild(badgeWrap);
        }

        assistantBubble.removeAttribute("aria-busy");

        // Screen reader announcement
        let annText = "Answer complete.";
        if (payload.flags && payload.flags.unverified) annText += " Warning: unverified figures.";
        if (payload.flags && payload.flags.truncated) annText += " Note: answer was cut off.";
        announce(annText);

      } else if (eventName === "error") {
        if (payload.fatal) {
          hadFatalError = true;
          const notice = el("div", "notice notice-fatal", payload.message || "An unexpected error occurred.");
          assistantBubble.appendChild(notice);
          assistantBubble.removeAttribute("aria-busy");
          announce("Error: " + (payload.message || "An unexpected error occurred."));
        } else {
          // Soft error: show inline notice, stream continues
          const notice = el("div", "notice notice-soft", payload.message || "Switching to a fallback.");
          assistantBubble.appendChild(notice);
        }
      }
      // Unknown event names are silently ignored
    });

    // Read chunks
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const text = decoder.decode(value, { stream: true });
      parser.feed(text);
    }
    // Flush any remaining bytes
    const remaining = decoder.decode();
    if (remaining) parser.feed(remaining);

  } catch (e) {
    if (e.name === "AbortError") {
      // User stopped the request
      if (!hadFatalError) {
        const notice = el("div", "notice notice-stopped", "Stopped.");
        assistantBubble.appendChild(notice);
        assistantBubble.removeAttribute("aria-busy");
        announce("Generation stopped.");
      }
    } else {
      const notice = el("div", "notice notice-error", "Connection error. Please try again.");
      assistantBubble.appendChild(notice);
      assistantBubble.removeAttribute("aria-busy");
      announce("Connection error.");
    }
  } finally {
    setStreaming(false);
    abortController = null;
    // If end was not received and no fatal error, mark as not busy
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

btnSend.addEventListener("click", sendMessage);

btnStop.addEventListener("click", function () {
  if (abortController) {
    abortController.abort();
  }
});

// Initial state
updateCharCount();
