# SSE contract v1 (`POST /api/chat`)

Status: frozen in Wave 1a (18 Nov 2026); wording amended and re-frozen in WP-2-contracts (18 Nov 2026). Owner: dg-lead-architect. Source of truth: the
`ChatEvent*` models in `common/schemas.py`. If this document and the models disagree, the
models win, and the disagreement is a bug to report to the owner. Changes need a note in
`.orchestration/CONTRACT_CHANGES.md` plus an orchestrator broadcast.

The UI (`app/static/index.html`) depends on this contract and nothing else.

## Request

```
POST /api/chat
Content-Type: application/json
Accept: text/event-stream
Cookie: sid=<opaque>            (optional; HttpOnly, set by the server on first response)

{"message": "3-bed villas in Riyadh under 2M SAR", "lang": null}
```

`ChatRequest`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `message` | string, 1..500 chars after trimming | yes | |
| `lang` | `"en"` \| `"ar"` \| null | no | A client hint only. The router decides the language. |

Responses that happen **before the stream opens** are plain HTTP JSON, not SSE:

| Status | When | Body |
|---|---|---|
| 422 | `ChatRequest` validation failed | FastAPI default |
| 429 | Per-IP rate limit (10/min, 40/day, keyed on the first `X-Forwarded-For` hop) | `{"detail": "rate_limited", "retry_after_s": int}` |
| 503 | `/ready` would fail (index not loaded or manifest invalid) | `{"detail": "not_ready"}` |

When the stream opens, the status is 200 with `Content-Type: text/event-stream` and
`Cache-Control: no-cache`. The server also sends `Server-Timing` only for non-streamed
responses. For streams, timings travel in the `end` event.

## Wire format

Each event is one SSE frame. The `event:` name equals the `event` field in the JSON. The
`data:` line is the model's `model_dump_json()`, which is always a single line.

```
event: <start|token|end|error>
data: <json>

```

`_EventBase.to_sse()` returns `{"event": ..., "data": ...}`, which you can pass straight to
sse-starlette. Every payload carries:

| Field | Type | Notes |
|---|---|---|
| `v` | `1` | Contract version. The UI should ignore events whose `v` it does not know. |
| `request_id` | 32 lowercase hex chars (`uuid4().hex`) | The same value for every event in one stream. It is also logged. |
| `event` | `"start"` \| `"token"` \| `"end"` \| `"error"` | Discriminator. |

Comment frames (`: ping`) may be sent as keep-alives. Clients ignore them.

## Event order (grammar)

```
stream   := start body | fatal
body     := (token | soft_error)* (end | fatal)
soft_err := error with fatal=false
fatal    := error with fatal=true      (last event; the server closes the stream)
```

The rules:
1. `start` is always the first event, unless a fatal error happens before routing finishes.
   In that case the stream is a single fatal `error`.
2. `start` is sent **before the first `token`** and carries the source cards, so the UI can
   render them early.
3. There is exactly one terminal event: `end`, or a fatal `error`. Nothing follows it.
4. A soft `error` never ends the stream. It tells the UI that the server switched level
   (`degraded_to`) or that the LLM stream broke (`llm_interrupted`). An `end` always follows.
5. Provider failover happens only before the first token. After the first `token`, there is
   no switch to another LLM, so the UI never needs to discard tokens it has already shown.

## Events

### `start`, sent once, before any token

| Field | Type | Required | Notes |
|---|---|---|---|
| `route` | `lookup` \| `aggregate` \| `project_info` \| `out_of_scope` | yes | |
| `language` | `en` \| `ar` | yes | Detected reply language. |
| `filters` | `SearchFilters` | yes | Filters after the slot merge. May be empty (`{}` plus `sort`). |
| `cache_tier` | `L0` \| `L1` \| `L2` \| `miss` | yes | Known at start. The answer-cache lookup runs **after routing** (the key needs the route, filters and numbers) and **before retrieval and generation**: L0/L1 exact first, then L2 semantic, which needs the query embedding. |
| `sources` | `Card[]`, 0..10 | yes (may be empty) | `sources[i].n == i+1`. `[n]` in the answer refers to `n`. |

`Card` (all of `DocFields`, plus `n`, `score` and `snippet`):

- Always set: `n` (int, 1..10), `doc_id`, `kind` (`listing`/`project`/`article`), `source` (`wasalt`/`darglobal`), `title`, `url`, `scraped_at` (ISO 8601 with offset).
- Listing fields, null for other kinds: `purpose`, `property_type`, `city`, `district`, `price_sar` (int SAR; for rentals, the rent per `rent_period`), `rent_period`, `annual_rent_sar` (int SAR), `bedrooms`, `bathrooms`, `area_sqm` (float).
- Project fields: `country`, `city`, `brand`, `handover` (verbatim text), `price_from_amount` (int) with `price_from_currency` (ISO 4217, not converted).
- Article fields: `article_kind`, `published_at`.
- `score` (float or null; null for SQL example cards) and `snippet` (≤ 400 chars or null).

The UI shows "as of `scraped_at`" on listing cards (ADR-0003).

### `token`, sent 0..n times

| Field | Type | Notes |
|---|---|---|
| `delta` | string, at least 1 char | Appended verbatim. Whitespace is significant and never trimmed. Markdown is allowed: `[n]` markers, `SAR 1,950,000`, bullet lists. |

Retrieval-only (level 3), static busy (level 4) and refusal answers are also delivered as
`token` events, usually as a single delta. The UI has one rendering path.

### `end`, sent once, after the last token

| Field | Type | Notes |
|---|---|---|
| `answer_mode` | `llm` \| `cache` \| `retrieval_only` \| `static_busy` \| `refusal` | How the text was produced. |
| `cache_tier` | `L0` \| `L1` \| `L2` \| `miss` | Same value as in `start`. |
| `cache_hits` | `("L3"\|"L4")[]` | Inner caches hit on this request. Measurement only. |
| `provider` | `openrouter` \| `groq` \| `none` | Who wrote the text. For cache hits, the original author. |
| `model` | string \| null | Provider model id. Null if and only if `provider == none`. |
| `tokens` | `{prompt_tokens: int, completion_tokens: int}` | Spend for **this request**. 0/0 on cache hits and no-LLM answers. |
| `finish_reason` | `stop` \| `length` \| `content_filter` \| `error` \| `cached` \| `no_llm` | |
| `timings` | `StageTimings` | ms as floats. `total_ms` is required; any stage that did not run is null. Stages: `cache_lookup_ms`, `embed_ms`, `sparse_ms`, `retrieve_ms`, `rerank_ms`, `sql_ms`, `llm_ttft_ms`, `llm_total_ms`, `grounding_ms`, `total_ms`. `embed_ms + sparse_ms` is the wall time of the query-embedding call: `sparse_ms` is the bm25 part measured by the embedder, `embed_ms` the rest (dense model plus executor wait). Both are null on an L3 hit. |
| `grounding` | `{numbers_ok, entities_ok, citations_ok: bool\|null, failures: string[≤20]}` | Null means the check does not apply (for example, a refusal). |
| `flags` | `{truncated: bool, unverified: bool}` | The UI shows an "unverified" badge when `unverified` is true and a "cut off" note when `truncated` is true. |
| `prompt_version` | `"<stem>@<8 hex>"`, e.g. `system_v1@1a2b3c4d` | |
| `index_version` | 16 hex chars | Equals `Manifest.index_version`. |

The model enforces these invariants:
- `answer_mode == cache` holds exactly when `cache_tier` is `L0`, `L1` or `L2`, and exactly when `finish_reason == cached`.
- `answer_mode` is one of `retrieval_only`, `static_busy` or `refusal` exactly when `finish_reason == no_llm`. In that case `provider == none`.
- `answer_mode == llm` requires `provider` to be `openrouter` or `groq`.
- `finish_reason` of `length` or `error` requires `flags.truncated == true`.
- `flags.unverified == not grounded`, where grounded means no check is `false`.

### `error`

| Field | Type | Notes |
|---|---|---|
| `code` | `ErrorCode` | `provider_rate_limited`, `provider_timeout`, `provider_error`, `provider_skipped`, `all_providers_unavailable`, `llm_interrupted`, `retrieval_failed`, `sql_failed`, `internal` |
| `message` | string, 1..300 chars | User-safe. Never includes stack traces, keys, URLs of upstream providers or raw provider error bodies. |
| `degraded_to` | `fallback_llm` \| `retrieval_only` \| `static_busy` \| null | The level the server is moving to. |
| `fatal` | bool | If true, this is the last event and `degraded_to` must be null. |

How each failure maps to events:

| Situation | Events |
|---|---|
| Provider 429, 5xx or TTFT timeout before the first token, and another provider is left | soft `error{code: provider_*, degraded_to: fallback_llm}`, then tokens from the next provider, then `end{provider: groq, ...}` |
| Every LLM provider failed before the first token | soft `error{code: all_providers_unavailable, degraded_to: retrieval_only}`, then templated tokens, then `end{answer_mode: retrieval_only}` |
| Retrieval itself failed | soft `error{code: retrieval_failed, degraded_to: static_busy}`, then the busy text as tokens, then `end{answer_mode: static_busy}` |
| The LLM stream broke after the first token | soft `error{code: llm_interrupted, degraded_to: null}`, then `end{finish_reason: error, flags.truncated: true}` |
| Unexpected exception | fatal `error{code: internal, fatal: true}`, then the stream closes |

The `provider_skipped` code (quota guard or cooldown) is optional on the wire. The server
may emit it, or only log it.

## Example stream (failover to Groq, grounded answer)

```
event: start
data: {"v":1,"request_id":"9f2c4e0b7a1d4c3e8b6a5d4c3b2a1f00","event":"start","route":"lookup","language":"en","filters":{"kinds":null,"source":null,"purpose":"sale","property_type":"villa","city":"Riyadh","district":null,"country":null,"project_id":null,"bedrooms_min":3,"bedrooms_max":3,"price_min_sar":null,"price_max_sar":2000000,"area_min_sqm":null,"area_max_sqm":null,"sort":"relevance"},"cache_tier":"miss","sources":[{"doc_id":"wasalt:123456","kind":"listing","source":"wasalt","title":"Villa for sale in Al Narjis","url":"https://wasalt.sa/en/property/123456","scraped_at":"2026-11-19T08:00:00Z","purpose":"sale","property_type":"villa","city":"Riyadh","district":"Al Narjis","price_sar":1950000,"rent_period":null,"annual_rent_sar":null,"bedrooms":3,"bathrooms":4,"area_sqm":312.5,"country":null,"brand":null,"handover":null,"price_from_amount":null,"price_from_currency":null,"article_kind":null,"published_at":null,"n":1,"score":0.032,"snippet":"Modern villa near King Salman Road."}]}

event: error
data: {"v":1,"request_id":"9f2c4e0b7a1d4c3e8b6a5d4c3b2a1f00","event":"error","code":"provider_rate_limited","message":"Primary model is busy; switching to a fallback model.","degraded_to":"fallback_llm","fatal":false}

event: token
data: {"v":1,"request_id":"9f2c4e0b7a1d4c3e8b6a5d4c3b2a1f00","event":"token","delta":"There is one 3-bedroom villa in Al Narjis, Riyadh for "}

event: token
data: {"v":1,"request_id":"9f2c4e0b7a1d4c3e8b6a5d4c3b2a1f00","event":"token","delta":"SAR 1,950,000 [1]."}

event: end
data: {"v":1,"request_id":"9f2c4e0b7a1d4c3e8b6a5d4c3b2a1f00","event":"end","answer_mode":"llm","cache_tier":"miss","cache_hits":[],"provider":"groq","model":"openai/gpt-oss-120b","tokens":{"prompt_tokens":812,"completion_tokens":24},"finish_reason":"stop","timings":{"cache_lookup_ms":0.4,"embed_ms":11.2,"sparse_ms":1.3,"retrieve_ms":9.8,"rerank_ms":null,"sql_ms":null,"llm_ttft_ms":640.0,"llm_total_ms":910.5,"grounding_ms":0.6,"total_ms":941.7},"grounding":{"numbers_ok":true,"entities_ok":true,"citations_ok":true,"failures":[]},"flags":{"truncated":false,"unverified":false},"prompt_version":"system_v1@1a2b3c4d","index_version":"0123456789abcdef"}
```

A repeat of the same query is served from cache. It gets `start{cache_tier: "L1"}`, one
`token` with the whole cached answer, and
`end{answer_mode: "cache", cache_tier: "L1", finish_reason: "cached", tokens: {0,0}, provider: "groq", model: "openai/gpt-oss-120b"}`
with only `cache_lookup_ms` and `total_ms` set. An L2 hit also sets `embed_ms` and
`sparse_ms` (unless the embedding came from L3), because the semantic lookup needs the
query embedding; `cache_lookup_ms` then covers the exact and the semantic lookups.

## Client parsing

- Python (tests, `bench.py`): `CHAT_EVENT_ADAPTER.validate_json(data_line)` returns the right `ChatEvent*` model.
- JS (UI): switch on `event`. Ignore unknown fields. Treat an unknown `v` as an error.
- A mock SSE server for UI work (Wave 2) must emit only streams that validate against these models.
