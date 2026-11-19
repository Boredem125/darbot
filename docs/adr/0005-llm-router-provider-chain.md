# ADR-0005: LLMRouter with an OpenRouter-first chain, Groq fallback, circuit breaker and quota ledger

- Status: Accepted
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0002, ADR-0004, ADR-0008, `docs/contracts/sse-v1.md`, `app/domain/ports.py` (`LLM`), `docs/specs/llm-router.md` (dg-ml-lead)

## Context

The assignment requires a free OpenRouter model. The free tier has hard limits:
- 50 requests per day per account, shared by every `:free` model;
- 20 requests per minute;
- failed calls count against the 50.

`GET /api/v1/key` reports the remaining free requests. Free models also return 429s and slow first tokens under load. Groq is allowed as a fallback, with an OpenAI-compatible API, about 30 RPM and 1K RPD per model, and a low TPM.

The SSE contract shows tokens as they arrive. Once a token is on screen, switching to another model would mean discarding visible text. The UI must also say which model wrote the answer.

## Decision

1. **One `LLM` adapter, `app/adapters/llm_router.py`, owns the provider chain.** The service sees only the `LLM` port. Each call to `stream()` yields `LLMFailover*`, then `LLMDelta*`, then exactly one `LLMDone`. If no provider produced a first token, it raises `LLMUnavailableError`.

2. **The chain is ordered, OpenRouter first.** It is configured from the `LLM_CHAIN` env JSON (provider, base_url, model), with one `AsyncOpenAI` client per base URL. The initial order:
   1. OpenRouter `google/gemma-4-31b-it:free`
   2. OpenRouter `nvidia/nemotron-3.5-super-vl:free`
   3. OpenRouter `openrouter/free`
   4. Groq `openai/gpt-oss-120b`
   5. Groq `qwen/qwen3.8-27b`

   The best-scoring OpenRouter model on the provider subset eval is always primary. Reordering is a config change, not a code change.

3. **Failover happens only before the first token.** The trigger is a 429, a 5xx or transport error, or no first token within the TTFT deadline. The router yields `LLMFailover` and tries the next entry. The service turns each failover into a soft SSE `error` with `degraded_to = fallback_llm`. After the first token there is no switch. A broken stream ends with `LLMDone(finish_reason = error)`, which the service reports as `llm_interrupted` and `flags.truncated = true`.

4. **Circuit breaker per provider entry.** A failure puts the entry in cooldown. A 429 is never retried on the same entry within the request. Entries in cooldown are skipped (`provider_skipped`) without spending a request.

5. **Quota ledger for OpenRouter.** A background task polls `GET /api/v1/key` and keeps the last `free_model_daily_requests` value in memory. When the remaining count is at or below the reserve threshold, every OpenRouter entry is skipped and the chain starts at Groq. `LLM.status()` exposes the remaining count and the per-entry cooldowns to `/ready` and `/stats`.

6. **Initial values** come from PLAN section 5. dg-ml-lead owns their final values in `docs/specs/llm-router.md`:

   | Setting | Initial value |
   |---|---|
   | TTFT deadline | 8 s |
   | Cooldown after 429, 5xx or timeout | 60 s |
   | Quota poll interval | 5 min |
   | Skip OpenRouter when remaining is at or below | 3 |
   | `temperature` | 0 |
   | `max_tokens` | 500 (English), 700 (Arabic, `ARABIC_MAX_TOKENS`) |

7. **Groq answers are labelled.** `end.provider` and `end.model` always name the provider and model that wrote the text, including for cache hits. The UI shows them in the latency strip.

8. **When every provider fails, the service degrades, not the router.** On `LLMUnavailableError`, the service emits a soft `error{all_providers_unavailable, degraded_to: retrieval_only}`, then a templated summary over the source cards with no LLM (`answer_mode = retrieval_only`). If retrieval also failed, it sends the static busy message instead. Each of the four levels of the degradation ladder has a test with fake adapters.

9. **Keys live only in env:** `OPENROUTER_API_KEY` and `GROQ_API_KEY`, from HF Space secrets or a local `.env`. They are never in the image, in CI, or in logs. `LLMFailover.detail` and SSE error messages are log-safe and user-safe.

Verify at build: whether OpenRouter BYOK with the Groq key would let one client cover both providers. If it works, it is a config option, not a change to this decision.

## Alternatives rejected

- **OpenRouter only.** 50 requests a day cannot survive Friday's reviewers plus the warm cache. Without a fallback, the demo stops answering mid-review.
- **Groq first.** It breaks the assignment's free-OpenRouter requirement in spirit, and the user chose OpenRouter as primary.
- **OpenRouter's server-side `models` fallback array.** It only covers OpenRouter models, every attempt still counts against the same 50, and it hides which attempt failed, so we could not emit precise failover events.
- **LiteLLM Router.** It does chains and cooldowns, but adds a large dependency and its own retry semantics. We need exact control of "no failover after the first token" and of the event stream.
- **Retrying the same provider on 429.** It burns quota (failed calls count) and adds latency, and the next provider is usually faster.

## Consequences

- The primary path stays on the free OpenRouter model while quota lasts. Overflow goes to Groq automatically, and the answer is labelled.
- The UI never retracts text, because failover is limited to the window before the first token.
- In-memory cooldowns and the quota value reset on restart (ADR-0008). The first poll after startup restores the quota view, and cooldowns are short anyway.
- Answers from different models differ in style. The provider-subset eval gates each model (5/5 format and citations, at least 4/5 grounded) before it may stay in the chain.
- Groq's low TPM makes long prompts a risk. The context budget (1,200 tokens of cards, 300 of history) keeps each request small.

## Trade-off accepted

When the free OpenRouter cap is hit, Groq answers, labelled with its model. The quota reserve sacrifices a few OpenRouter requests to keep the chain from spending failed calls at the edge of the cap.
