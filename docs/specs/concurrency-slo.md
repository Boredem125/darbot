# Concurrency and SLO spec (ADR-0008, PLAN §5 and §8)

Status: Wave 1b, 18 Nov 2026. Owner: dg-ml-lead, who also signs off the SLOs from `BENCHMARK.md` in Wave 4. Implementers: dg-backend-engineer (runtime), dg-qa-engineer (`scripts/bench.py`, the fake LLM server), dg-devops-engineer (Dockerfile env).

## 1. Process and threads

| Item | Decision | Evidence or reason |
|---|---|---|
| Workers | 1 uvicorn worker (ADR-0008) | In-memory caches, sessions, ledger |
| Event loop | Never runs blocking work over 5 ms | One blocked loop stalls every open SSE stream |
| ONNX executor | `ThreadPoolExecutor(max_workers=2, thread_name_prefix="onnx")`, owned by the embedder adapter (and the reranker if enabled). Models are built with `threads=1` | 2 × 1 thread on 2 vCPU, no oversubscription (PLAN §5). Dense query embed p95 29.3 ms at threads=1 (retrieval.md F11) |
| Dense + bm25 query embed | One executor job computes both; bm25 is pure Python at 0.02 ms (F12) | Saves an executor hop. `sparse_ms` is reported inside the job (the port stop-gap below) |
| Hybrid search (numpy) | Runs **on the event loop** | p95 1.54 ms at 6k points (retrieval.md F5), under the 5 ms rule. If the HF bench shows `retrieve_ms` p95 > 5 ms, move it to the ONNX executor (one config switch, `RETRIEVE_IN_EXECUTOR=1`) |
| Qdrant | **No runtime role** (retrieval.md §1) | Local mode runs synchronously on the loop: 653 ms p95 filtered at 6k points (F1). It would stall all streams |
| BLAS threads | Dockerfile env `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` | numpy matmul must not contend with the two ONNX threads. A 6k×384 matmul is under 1 ms single-threaded (F5) |
| DuckDB | One `duckdb.connect()` at startup (read-only, `SET threads=1`). Each query uses `con.cursor()` and runs in `asyncio.to_thread` | DuckDB releases the GIL, and a cursor per call is its documented thread-safe pattern. One thread keeps 2 vCPU free for ONNX |
| LLM | Async network I/O on the loop; `LLM_MAX_CONCURRENCY=4` (llm-router.md §7) | |
| Caches and sessions | Touched only on the loop thread (caching.md §1) | `cachetools` is not thread-safe |
| fastembed models | Loaded from `FASTEMBED_CACHE_PATH` baked into the image; `HF_HUB_OFFLINE=1` at runtime; warm-up of one embed call before `/ready` turns green | The first call pays the ONNX session init. C: is full locally, so local builds point `FASTEMBED_CACHE_PATH` at E: |

**Sessions**: `TTLCache(maxsize=1000, ttl=1800)`, holding the last 2 turns plus slots, keyed by an HttpOnly cookie (PLAN §5).

**Admission control**:
- At most `MAX_INFLIGHT_CHATS = 12` concurrent `/api/chat` streams. Beyond that, the request gets degradation level 4: a normal SSE stream with the static busy text and `answer_mode=static_busy`, as the SSE contract allows. 12 is above 5 users × 2 tabs and caps memory and CPU on 2 vCPU.
- Per-IP limits are 10/min and 40/day (PLAN §5).

**Embed and sparse timing (CC-2).** The earlier `last_timings` adapter attribute is superseded by CC-2 (`.orchestration/CONTRACT_CHANGES.md`); an attribute on a shared adapter is racy under concurrent requests. The embedder returns `QueryEmbedding.sparse_ms` (the bm25 part, timed inside its executor job). The service times the whole `await embed_query(...)` as `W` and sets `sparse_ms = QueryEmbedding.sparse_ms` and `embed_ms = max(W - (sparse_ms or 0), 0)`, so `embed_ms` includes the executor hop and queue wait. A cached L3 hit sets both to null.

## 2. SLOs (PLAN §8): definitions

All latencies are **server-side stage timings** from `end.timings`, plus client-side timings from the bench. They are reported as p50/p95 with n. "p95" over n < 20 is reported as "max (n<20)", never as p95.

| SLO | Metric (exact definition) | Target | Sample |
|---|---|---|---|
| S1 cache hit | `total_ms` of requests with `cache_tier ∈ {L0, L1}` (exact). L2 is reported separately, because its time includes an embed | p95 < 50 ms | n ≥ 100 |
| S1b L2 hit (reported) | `total_ms` with `cache_tier = L2` | p95 < 50 ms is the goal, not gated (embed p95 is 29 ms per F11) | n ≥ 30 |
| S2 retrieval path | `cache_lookup_ms + embed_ms + sparse_ms + retrieve_ms + rerank_ms` (nulls as 0) on lookup and project_info routes, i.e. the work before the prompt is built | p95 < 150 ms | n ≥ 100 |
| S3 SQL | `sql_ms` on the aggregate route | p95 < 20 ms | n ≥ 100 |
| S4 TTFT | `llm_ttft_ms` per provider/model | reported only | all real calls |
| S5 end-to-end | Client-side: request sent → `start` received (TTFB), → first `token`, → `end`; at 1 and 5 concurrent users | reported only (p50/p95) | see §3 |
| S6 start-before-token | Every non-cached stream has `start` before its first `token` | 100% | all |

**Projected budget for S2** (dev box, threads=1; F5, F11, F12): embed 29.3 + bm25 0.02 + numpy retrieve 1.5 + cache lookup about 0.1 is about **31 ms p95**. That leaves roughly 4.8× headroom for slower HF cores and contention. With rerank enabled, the 5-pair rerank alone measured 164–183 ms p50 (F13), so the S2 target cannot hold, which is why rerank is off by default (retrieval.md §3.7).

## 3. How the bench measures (dg-qa-engineer, `scripts/bench.py`)

The free OpenRouter quota (50/day) and Groq's low TPM make it impossible to measure the system's own latency under concurrency with real LLM calls. The bench therefore has two parts.

**Part A, system bench (no quota spent).**
- Run the production image locally with `docker run --cpus=2 --memory=16g`, which approximates HF.
- `LLM_CHAIN` points at `scripts/fake_llm_server.py`, an OpenAI-compatible SSE stub with configurable TTFT (default 800 ms) and token rate (default 40 tok/s), emitting 150-word answers with valid `[n]` citations taken from the prompt.
- Query mix: 200 requests, stratified 25% exact repeats (L1), 25% L0 warm queries, 30% lookup/project_info misses, 20% aggregate.
- Run at concurrency 1 and 5.
- Yields S1, S2, S3, S5-system and S6 with n ≥ 100 where required.
- Reported as "system latency with a stubbed LLM (TTFT 800 ms)".

**Part B, public URL (real providers, quota-limited).**
- Concurrency 1: the six PLAN §12 probes plus four L0 queries, n = 10.
- Concurrency 5: 5 users × 2 requests, n = 10, staying within 10/min from one IP.
- At most 10 OpenRouter calls (PLAN §9 Friday "probes and bench ≤ 10"); Groq overflow is automatic.
- Cold vs warm: the first request after a Space wake versus subsequent requests.
- Yields S4 and S5-real with honest small n (reported as "max (n<20)").
- Also records the cold start: time from container start to `/ready` green.

`BENCHMARK.md` gets one table per part, the exact query mix, the git sha, `index_version`, the date and the hardware.

## 4. Sign-off rule (dg-ml-lead, Wave 4)

- S1, S2 and S3 **pass** if Part A meets the target with the stated n.
- If Part B, the public URL, shows a p50 more than 2× worse than Part A on the same tier, I report it as an environment gap with both numbers, and I do not mark the SLO failed silently.
- A miss is reported as a failure with the number. It is never hidden or re-run until it passes.
- The fix order for an S2 miss:
  1. Move retrieve to the executor (if `retrieve_ms` is the cause).
  2. Check that `OMP_NUM_THREADS=1` is effective.
  3. Lower `MAX_INFLIGHT_CHATS`.
- An S1 miss on L0/L1 means a bug, because the lookup is a dict get. Investigate; do not tune.

## 5. Parameters

| Name | Value | Rule |
|---|---|---|
| uvicorn workers | 1 | ADR-0008 |
| ONNX executor workers / threads per model | 2 / 1 | PLAN §5; matches 2 vCPU |
| `MAX_INFLIGHT_CHATS` | 12 | Lower it to 8 if the Part A 5-user run shows S2 p95 > 150 ms only under load |
| `RETRIEVE_IN_EXECUTOR` | 0 | Set it to 1 if HF `retrieve_ms` p95 > 5 ms |
| DuckDB threads | 1 | Raise it to 2 only if S3 misses and CPU is idle |
| Session cache | 1,000 / 30 min | PLAN §5 |
| Per-IP limits | 10/min, 40/day | PLAN §5 |

## 6. Open item for security review
`sse-v1.md` keys the rate limiter on the **first** `X-Forwarded-For` hop. If HF's proxy appends to a client-supplied XFF header rather than overwriting it, the first hop is client-controlled and the per-IP limit (and so the free quota) can be bypassed. The correct key is then the right-most hop added by the trusted proxy. This needs one header-echo check on a deployed Space (Wave 4, dg-devops-engineer), and a decision by dg-lead-architect plus the security auditor. It is not mine to change.
