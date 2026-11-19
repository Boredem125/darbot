# LLMRouter spec (ADR-0005)

Status: Wave 1b, 18 Nov 2026. Owner: dg-ml-lead. Implementer: dg-backend-engineer (`app/adapters/llm_router.py`, Wave 4). It implements the `LLM` port in `app/domain/ports.py` and does not change it.
Sources: PLAN §3 (OpenRouter, Groq), §5, §9; OpenRouter limits doc (fetched 18 Nov 2026); openai 3.55.0 introspection.

## 0. Verified facts

| # | Fact | Source |
|---|---|---|
| V1 | `:free` models: 50 requests per UTC day per account (accounts with < 10 credits purchased), 20 RPM, shared by all `:free` models | OpenRouter limits doc |
| V2 | `GET /api/v1/key` returns `free_model_daily_requests: {used, limit, remaining}` for "the current UTC day" | OpenRouter limits doc |
| V3 | A rate-limited call returns HTTP 429 with `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`. `Retry-After` is sent only sometimes. Upstream provider 429s carry `error.metadata.provider_code`. **A limit hit mid-stream arrives as an SSE event with `finish_reason: "error"`, not as a 429** | OpenRouter limits doc |
| V4 | `AsyncOpenAI(api_key, base_url, timeout, max_retries, default_headers, http_client, …)`. **`DEFAULT_MAX_RETRIES = 2`**. `DEFAULT_TIMEOUT` read is 600 s | openai 3.55.0 introspection |
| V5 | `chat.completions.create` accepts `stream`, `stream_options`, `max_tokens`, `temperature`, `extra_body`, `extra_headers`, `timeout`, `reasoning_effort`, `seed`. `with_raw_response` exists | openai 3.55.0 introspection |
| V6 | The SDK's stream iterator raises `APIError` when an SSE event carries an `error` object | `openai/_streaming.py` |
| V7 | The exception classes `RateLimitError`, `APIStatusError`, `APIConnectionError`, `APITimeoutError`, `InternalServerError`, `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` and `BadRequestError` exist | openai 3.55.0 |

**Unverified** (needs a live call, so the orchestrator's smoke test resolves it; nothing in this spec depends on these being true without a fallback):
- U1. Whether failed calls count against the 50/day. The doc is silent for free models. PLAN §3 says they count. **We assume they count.**
- U2. Whether `GET /api/v1/key` itself counts. Assume no, and verify by polling twice with no calls in between: `used` must not change.
- U3. The exact reset instant. Assume 00:00 UTC.
- U4. Groq model ids `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`, their `x-ratelimit-remaining-tokens` and `x-ratelimit-reset-tokens` headers, `stream_options.include_usage` support, and `reasoning_effort` / `include_reasoning` support for gpt-oss.
- U5. OpenRouter BYOK with the Groq key (PLAN §5, "verify in build"). It is not used by this spec.
- U6. That `with_raw_response.create(stream=True).parse()` yields an `AsyncStream` with headers readable before iteration. If not, read the headers via `with_streaming_response`.

## 1. Chain configuration

`LLM_CHAIN` is an env JSON list in priority order. The default in `.env.example` is below. API keys come from the env var named in `api_key_env`, never from the JSON.

```json
[
 {"id":"groq-oss",    "provider":"groq","model":"openai/gpt-oss-120b","base_url":"https://api.groq.com/openai/v1","api_key_env":"GROQ_API_KEY","quota":null,"tpm":8000,"extra_body":{"reasoning_effort":"low","include_reasoning":false}},
 {"id":"or-gemma",    "provider":"openrouter","model":"google/gemma-4-31b-it:free",       "base_url":"https://openrouter.ai/api/v1","api_key_env":"OPENROUTER_API_KEY","quota":"openrouter_free","extra_body":{}},
 {"id":"or-free",     "provider":"openrouter","model":"openrouter/free",                  "base_url":"https://openrouter.ai/api/v1","api_key_env":"OPENROUTER_API_KEY","quota":"openrouter_free","extra_body":{}},
 {"id":"groq-qwen",   "provider":"groq","model":"qwen/qwen3.8-27b",   "base_url":"https://api.groq.com/openai/v1","api_key_env":"GROQ_API_KEY","quota":null,"tpm":6000,"extra_body":{}}
]
```

- `or-nemotron` is dropped from the default chain but remains available via `LLM_CHAIN` JSON.
- An entry whose key env var is empty is dropped at startup and logged once. A chain that ends up empty is legal: every turn degrades to retrieval-only, which is how the "all providers disabled" check in PLAN §12 is run.
- There is one `AsyncOpenAI` client per distinct `base_url`, created with **`max_retries=0`** and `timeout=httpx.Timeout(connect=5, read=30, write=10, pool=5)`. The SDK default of 2 retries (V4) would silently resend 429s and 5xx. That burns the 50/day, since failed calls count (U1), and breaks "a 429 is never retried on the same provider".
- OpenRouter `default_headers`: `X-Title: DG RAG demo`. Send no personal data.
- **Order is decided by measurement.** WP-3-geneval probe 19 Nov: or-gemma upstream 429, or-nemotron timeout, or-free wrong language; user decision: Groq first. The chain is groq-oss, or-gemma, or-free, groq-qwen.
- `extra_body` for gpt-oss (U4) keeps reasoning tokens out of the stream and the TTFT short. If Groq rejects a key with a 400, the breaker rule for 400 below applies and the entry is fixed by config, not by code.

## 2. Request parameters (from the service)
`stream=True`, `temperature=0`, `max_tokens` = 500 (English) or 700 (Arabic) (PLAN §6), `stream_options={"include_usage": true}`, `seed=0` where accepted. Messages are the port's `ChatMessage`s as assembled by `app/domain/prompting.py`.

## 3. Algorithm for `stream(messages, max_tokens, temperature)`

```
deadline_first = now + CHAIN_FIRST_TOKEN_BUDGET_S          # 20 s, whole chain
for entry in chain:
    skip_reason = gate(entry)                              # §4 breaker, §5 quota, §6 Groq TPM, §7 RPM
    if skip_reason: yield LLMFailover(entry, provider_skipped, skip_reason); continue
    budget = min(TTFT_TIMEOUT_S, deadline_first - now)     # 8 s per attempt
    if budget <= 0.5: break
    ledger.record_attempt(entry)                           # before sending: failed calls count
    try:
        async with asyncio.timeout(budget):
            stream = await client.chat.completions.create(...)
            first = await next content delta               # see "first token" below
    except <classified error> as e:
        breaker.fail(entry, e); yield LLMFailover(entry, code(e), detail(e)); continue
    breaker.success(entry)
    yield LLMDelta(first); relay remaining deltas (STREAM_MAX_S = 60 s after first token)
    on error after first token: breaker.fail(entry, e); yield LLMDone(finish_reason=error, usage so far); return
    yield LLMDone(provider, model, finish_reason, usage, ttft_ms, total_ms); return
raise LLMUnavailableError(attempts)
```

- **First token** is the first chunk whose `choices[0].delta.content` is a non-empty string. Reasoning fields (`delta.reasoning`, `reasoning_content`) never count and are never relayed. An SSE `error` event before the first token (V3, V6) is a pre-first-token failure, so failover is allowed.
- **No failover after the first token** (port contract; the UI never discards shown tokens). On a mid-stream failure, the iterator yields `LLMDone(finish_reason=error)`. The service sets `flags.truncated`, and the answer is not cached.
- Hitting `STREAM_MAX_S` has the same outcome as a mid-stream failure: `finish_reason=error`.
- `finish_reason` mapping: `stop`, `length` and `content_filter` pass through. Anything else or missing becomes `stop` if text was produced, logged as `finish_reason_raw`.
- **Usage** comes from the final chunk's `usage` when present. Otherwise estimate: prompt = `ceil(utf8_bytes/3)` of the messages, completion = the same over the output. The estimate is flagged `usage_estimated=true` in the log line.
  - `utf8_bytes/3` equals chars/3 for Latin text and chars/1.5 for Arabic (2 bytes per char). That is consistent with the AI-SPEC Arabic ratio and slightly more conservative for Latin (AI-SPEC used 3.5).
  - Calibration (Wave 3, Groq dev eval): the ratio `estimate / usage.prompt_tokens` must lie in [1.0, 1.6], an overestimate within bounds. If it does not, change the divisor once and record it in `docs/tuning.md`.
- **Cancellation.** If the client disconnects, the service cancels the generator, and the router must close the upstream stream (`AsyncStream.close()`). Consume it under `contextlib.aclosing` / `try/finally`. A leaked stream keeps generating tokens and holds a semaphore slot. The AI-SPEC prototype verified the close on cancel and on early `break`.
- **Reasoning tokens and `max_tokens`** (AI-SPEC P10, unverified). Groq may count gpt-oss reasoning tokens against `max_tokens`. Each chain entry has an optional `max_tokens_extra` (default 0), added to the service's `max_tokens` for that entry only. Tuning rule (Wave 3 provider subset, no extra calls): if an entry's `usage.completion_tokens` exceeds `ceil(utf8_bytes(visible text)/3) × 1.5` on ≥ 2 of 5 cases, or it returns `finish_reason=length` on any case with fewer than 150 visible words, set `max_tokens_extra = 300` for that entry.
- **TTFT** is measured from just before `create()` to the first content delta, per attempt. The `LLMDone.ttft_ms` reported is that of the successful attempt. The service's `llm_ttft_ms` stage spans all attempts.

## 4. Circuit breaker (per chain entry)

| Failure (before or after the first token) | ErrorCode | Cooldown |
|---|---|---|
| 429 from OpenRouter **without** `error.metadata.provider_code` (account-level: RPM or daily) | provider_rate_limited | **all `openrouter_free` entries** 60 s, plus an immediate quota poll (§5) |
| 429 with `provider_code` (upstream provider limit) | provider_rate_limited | this entry only, `max(60, Retry-After)` s, capped at 900 |
| 429 from Groq | provider_rate_limited | this entry `max(60, Retry-After or x-ratelimit-reset-*)` s, capped at 900 |
| 5xx, `APIConnectionError`, `APITimeoutError`, SSE error event | provider_error | 60 s |
| No first token within the attempt budget | provider_timeout | 60 s |
| 400 / 422 | provider_error | 60 s; log the body (bad param, e.g. an `extra_body` key) |
| 401 / 403 | provider_error | 1,800 s; log `config_error` once |
| 402 (credits) | provider_error | until the next 00:00 UTC |
| 404 (model gone) | provider_error | for the process lifetime |

- **Exponential cooldown.** Each consecutive failure of the same entry doubles the base cooldown (60, then 120, 240, 480, 900 cap). The first success resets it. This keeps a flapping free model from eating quota.
- **Half-open.** When the cooldown expires, the next request may try the entry. While one trial is in flight, concurrent requests skip it (`provider_skipped`, "half_open_busy").
- A 429 is **never retried on the same entry** within a request (PLAN §5). The chain simply moves on.
- `status()` returns `ProviderState(available, cooldown_remaining_s)` from this table, with no I/O.

## 5. OpenRouter quota ledger (50/day, failed calls count)

State, in memory (ADR-0008; it is lost on restart and restored by the startup poll):
`utc_day`, `local_attempts_today`, `polled_remaining`, `polled_at`, `attempts_since_poll`.

- `record_attempt(entry)` with `quota == "openrouter_free"` increments `local_attempts_today` and `attempts_since_poll` **before** the request is sent. Every attempt counts, whatever its outcome (U1, conservative).
- `remaining_estimate`:
  - `polled_remaining - attempts_since_poll` if a poll has ever succeeded today;
  - otherwise `OR_DAILY_LIMIT (50) - local_attempts_today`.
  - Never below 0.
- **Poll** `GET {base}/key` (field `free_model_daily_requests.remaining`, V2):
  - at startup, before `/ready`;
  - every `OR_POLL_INTERVAL_S = 300` s;
  - immediately after any account-level OpenRouter 429.

  The poll has a 3 s timeout. A failed poll keeps the last value and logs `quota_poll_failed`. A successful poll sets `polled_remaining` and resets `attempts_since_poll = 0`.
- **Guard.** If `remaining_estimate <= OR_RESERVE (3)`, every `openrouter_free` entry is skipped with `LLMFailover(code=provider_skipped, detail="quota_guard")`. The guard lifts when a later poll reports more than 3 remaining, or at the UTC day rollover (`utc_day` changes, which resets the counters).
- **The reserve of 3 exists** so the orchestrator's final public-URL probes (PLAN §12, six probes) can still use OpenRouter after reviewers have drained the day. The orchestrator may lower it to 0 by env for the last probes. The value is not tuned; it is a ledger policy.
- **Local RPM limiter**: at most `OR_RPM = 20` OpenRouter attempts in any rolling 60 s (V1). Beyond that, skip (`detail="rpm_guard"`).
- Every attempt emits one JSON log line: `{"event":"llm_attempt","entry":id,"provider","model","outcome","code","ttft_ms","quota_remaining_est","request_id"}`. The orchestrator reconciles these with `eval/quota_log.csv`. The runtime never writes that CSV, which belongs to the offline scripts.
- `status().openrouter_quota_remaining` is the last `polled_remaining`, or `None` if no poll has ever succeeded. `/stats` shows `remaining_estimate` next to it.

## 6. Groq TPM guard (low TPM, PLAN §3)
- Per Groq entry, a token bucket fed by the response headers `x-ratelimit-remaining-tokens` and `x-ratelimit-reset-tokens` (U4, U6). Until a first response is seen, use the configured `tpm` per 60 s.
- Request cost estimate = `ceil(utf8_bytes(messages)/3) + max_tokens`.
- If the estimate exceeds the remaining tokens and the reset time is in the future, skip the entry (`provider_skipped`, "tpm_guard"). Do not send a request that will 429.

## 7. Global concurrency
- `LLM_MAX_CONCURRENCY = 4` streams in flight across all entries (asyncio.Semaphore).
- Waiting for the semaphore counts against `CHAIN_FIRST_TOKEN_BUDGET_S`. If the budget runs out while waiting, the router raises `LLMUnavailableError` with an empty attempt list, and the service degrades to retrieval-only.

## 8. Parameters

| Name | Value | Rule or evidence |
|---|---|---|
| `TTFT_TIMEOUT_S` | 8 | PLAN §5. Tuning: if Wave 3/4 logs show the chosen primary's TTFT p95 > 6 s on successful calls, raise it to 10 (one change only), so a slow but working free model is not cut off |
| `CHAIN_FIRST_TOKEN_BUDGET_S` | 20 | Bounds the worst-case wait before the UI shows text (5 entries × 8 s would be 40 s). Not tuned |
| `STREAM_MAX_S` | 60 | 700 tokens at a slow 15 tok/s is 47 s. Not tuned |
| base cooldown / cap | 60 s / 900 s | PLAN §5 / ledger policy |
| `OR_DAILY_LIMIT` | 50 | V1 |
| `OR_RESERVE` | 3 | PLAN §5 |
| `OR_POLL_INTERVAL_S` | 300 | PLAN §5 |
| `OR_RPM` | 20 | V1 |
| `LLM_MAX_CONCURRENCY` | 4 | 5-user bench target; below OR_RPM even at 4 req/s bursts for a few seconds. Revisit if the 5-user bench shows a semaphore wait p95 > 1 s |
| openai `max_retries` | 0 | V4 |

## 9. Tests the implementer writes first (httpx `MockTransport`; no keys, no network)
1. 429 on entry 1 gives an `LLMFailover(provider_rate_limited)`, then tokens from entry 2. Entry 1 is not retried, and all OpenRouter entries are in cooldown.
2. An SSE error event before the first content leads to failover. An error after the first token leads to `LLMDone(error)`, with no failover.
3. No first token within 8 s gives `provider_timeout` (use a fake clock or a small budget in the test).
4. The ledger counts failed attempts; the guard at ≤ 3 gives `provider_skipped` with "quota_guard"; day rollover resets it; a poll resets `attempts_since_poll`.
5. Every entry failing raises `LLMUnavailableError` with every attempt and no deltas.
6. Exponential cooldown doubles and resets on success. Half-open lets one trial through.
7. Reasoning-only chunks do not count as the first token.
8. `max_retries=0` is asserted on the constructed clients.
