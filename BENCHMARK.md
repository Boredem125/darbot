# Benchmark Results

Git sha: 067fa2c
Index version: 77284224a04cf674
Date: 2026-11-20
Hardware (Part A): Windows 11, AMD Ryzen (dev box, shared, not 2-vCPU Docker)
Hardware (Part B): HuggingFace ZeroGPU Space (boredem125-dar-bot)

---

## Part A: System latency with a stubbed LLM (TTFT 800 ms)

LLM: fake_llm_server.py (TTFT 800 ms, 40 tok/s)
Provider label: fake (reported as "openrouter" in SSE due to chain config; the LLM is local)
Per-IP rate limits: disabled (set to 100000/min, 100000/day)
Server restarted between concurrency 1 and concurrency 5 runs (no carried L1/session state).

### Concurrency 1

Requests: 200, Errors: 0
Mix: L0=50, L1_prime=25, L1_repeat=25, lookup=60, aggregate=40
Answer modes: cache=105, llm=66, refusal=29
L3/L4 cached: 70, retrieval ran: 130

#### S1 Cache Hit (server total_ms, cache_tier L0/L1)

p50: 29.4 ms, p95: 46.1 ms, n=105 -- **PASS** (target < 50 ms)

#### S2 Retrieval Path (server, excl L3/L4 cached)

p50: 0.0 ms, p95: 21.5 ms, n=97 -- **PASS** (target < 150 ms)
L3/L4-cached retrieval: p50=0.2 ms, n=40 (not counted in S2)

Note: S2 p50 of 0.0 ms includes L0-cached lookup queries that skipped retrieval. The p95 reflects actual retrieval runs.

#### S3 SQL (indicative, dev box)

p50: 12.3 ms, p95: 18.1 ms, n=38 -- **PASS** (target < 20 ms)

S3 first-aggregate cold vs warm: first aggregate query sql_ms was in the 16-18 ms range (DuckDB JIT cost); subsequent queries settled to 10-12 ms. No single cold outlier above the 20 ms target.

#### S4 TTFT (server llm_ttft_ms, fake provider)

fake: p50=807.3 ms, p95=810.3 ms, n=66 (expected: configured TTFT 800 ms + network)

#### S5 Client-side (reported, not gated)

| Tier | n | TTFB p50 | TTFB p95 | First Token p50 | First Token p95 | End p50 | End p95 |
|------|---|----------|----------|-----------------|-----------------|---------|---------|
| L0 (cache) | 105 | 47 ms | 63 ms | 47 ms | 63 ms | 47 ms | 63 ms |
| miss | 95 | 47 ms | 78 ms | 1422 ms | 1610 ms | 1422 ms | 1610 ms |

#### S6 start-before-token

200/200 (100%) -- **PASS**

### Concurrency 5 (fresh server)

Requests: 200, Errors: 0
Mix: L0=50, L1_prime=25, L1_repeat=25, lookup=60, aggregate=40
Answer modes: cache=105, llm=66, refusal=29
L3/L4 cached: 70, retrieval ran: 130

#### S1 Cache Hit (server total_ms, cache_tier L0/L1)

p50: 30.7 ms, p95: 44.7 ms, n=105 -- **PASS** (target < 50 ms)

#### S2 Retrieval Path (server, excl L3/L4 cached)

p50: 0.0 ms, p95: 20.9 ms, n=97 -- **PASS** (target < 150 ms)
L3/L4-cached retrieval: p50=0.1 ms, n=40 (not counted in S2)

#### S3 SQL (indicative, dev box)

p50: 12.4 ms, p95: 20.8 ms, n=38 -- **FAIL** (target < 20 ms; 20.8 ms, indicative on shared dev box)

#### S4 TTFT (server llm_ttft_ms, fake provider)

fake: p50=806.9 ms, p95=810.8 ms, n=66

#### S5 Client-side (reported, not gated)

| Tier | n | TTFB p50 | TTFB p95 | First Token p50 | First Token p95 | End p50 | End p95 |
|------|---|----------|----------|-----------------|-----------------|---------|---------|
| L0 (cache) | 105 | 47 ms | 78 ms | 47 ms | 78 ms | 47 ms | 78 ms |
| miss | 95 | 47 ms | 63 ms | 1422 ms | 1641 ms | 1422 ms | 1641 ms |

#### S6 start-before-token

200/200 (100%) -- **PASS**

---

## Part B: Public URL (https://boredem125-dar-bot.hf.space)

Space was already awake (/ready in 1 second, no cold start measured).
Query mix: 6 PLAN section 12 probes + 4 L0 warm-cache queries = 10 per concurrency level.
Rate limit: 6 s between requests (staying within 10/min from one IP).
Budget: 20 requests total, 10 OpenRouter max. Actual: 1 Groq LLM call, 0 OpenRouter.

### Concurrency 1

Requests: 10, Errors: 0
Mix: probe=4, followup=1, injection=1, L0=4
Answer modes: cache=7, llm=1, refusal=2

#### S1 Cache Hit (server total_ms, cache_tier L0)

max(n=7): 23.2 ms -- **PASS** (target < 50 ms)

#### S2 Retrieval Path (server, excl L3/L4 cached)

max(n=6): 13.7 ms -- **PASS** (target < 150 ms)

#### S4 TTFT (server llm_ttft_ms, real provider)

groq: 209.3 ms, n=1

#### S5 Client-side (reported, not gated)

| Tier | n | TTFB max | First Token max | End max |
|------|---|----------|-----------------|---------|
| L0 | 7 | 1750 ms | 1766 ms | 1766 ms |
| miss | 3 | 922 ms | 1625 ms | 1625 ms |

Environment gap (section 4 rule): Part B L0 client TTFB max 1750 ms vs Part A L0 client TTFB p50 47 ms. The gap is network latency to the HF Space (~1700 ms round-trip), not server processing. Server-side S1 is consistent: 23 ms (B) vs 29 ms (A).

#### S6 start-before-token

10/10 (100%) -- **PASS**

### Concurrency 5

Requests: 10, Errors: 1 (HTTP 429 from Space rate limiter)
Mix: probe=4, followup=1, injection=1, L0=4 (1 got 429'd)
Answer modes: cache=7, refusal=2, llm=0

#### S1 Cache Hit (server total_ms, cache_tier L0/L1)

max(n=7): 22.3 ms (L0), 19.4 ms (L1 n=1) -- **PASS**

#### S5 Client-side (reported, not gated)

| Tier | n | TTFB max | End max |
|------|---|----------|---------|
| L0 | 6 | 1360 ms | 1360 ms |
| L1 | 1 | 703 ms | 703 ms |
| miss | 2 | 781 ms | 781 ms |

#### S6 start-before-token

9/9 (100%, excluding the 429) -- **PASS**

Note: 1 request got HTTP 429 from the Space's per-IP rate limiter (10/min). This is expected at conc 5 with burst traffic. The 429 is not a test failure (per the brief).

---

## SLO Summary

| SLO | Conc 1 (A) | Conc 5 (A) | Conc 1 (B) | Conc 5 (B) |
|-----|-----------|-----------|-----------|-----------|
| S1 cache hit < 50 ms p95 | PASS (46.1 ms, n=105) | PASS (44.7 ms, n=105) | PASS (23.2 ms max, n=7) | PASS (22.3 ms max, n=7) |
| S2 retrieval < 150 ms p95 | PASS (21.5 ms, n=97; spec asks n >= 100) | PASS (20.9 ms, n=97; spec asks n >= 100) | PASS (13.7 ms max, n=6) | n/a (no retrieval ran; all cached or refused) |
| S3 SQL < 20 ms p95 | PASS (18.1 ms, n=38; spec asks n >= 100) | **FAIL** (20.8 ms, n=38, dev box) | n/a | n/a |
| S4 TTFT | 807 ms (fake) | 807 ms (fake) | 209 ms (groq, n=1) | n/a |
| S5 e2e | reported | reported | reported | reported |
| S6 start-before-token | 200/200 | 200/200 | 10/10 | 9/9 |

S3 at concurrency 5 is a failure: p95 20.8 ms against a 20 ms target, on a shared dev box (indicative), with n=38, below the n >= 100 the spec asks for. S3 was not measured on the Space: the Part B aggregate queries were served from L0, so no SQL ran. The S2 n of 97 is also just under the stated 100. These are reported as measured, not re-run.
