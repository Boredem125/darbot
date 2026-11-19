# ADR-0008: Single-process stateful runtime (one uvicorn worker)

- Status: Accepted
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0001, ADR-0003, ADR-0005, ADR-0006, `docs/specs/concurrency-slo.md` (dg-ml-lead)

## Context

The Space has 2 vCPU and 16 GB of RAM, one container, an ephemeral disk, and sleeps after about 48 hours idle. The runtime holds several kinds of state:
- the in-memory Qdrant index and the DuckDB view of `listings.parquet` (ADR-0003);
- the L1 and L2 answer caches and the L3 and L4 inner caches (ADR-0006);
- sessions with the last two turns and merged slots, for follow-ups (ADR-0004);
- the per-IP rate limiter;
- the OpenRouter quota value and the provider cooldowns (ADR-0005);
- the telemetry ring buffer behind `/stats`.

With several worker processes, each would load its own copy of the index and the ONNX models, keep its own caches (lowering hit rates), keep its own rate-limit counters (so the limit would be per worker, not per user), and keep its own view of the quota. Fixing that means a shared store such as Redis, which is another process to run and secure in a single-container Space.

The heavy work is CPU-bound ONNX inference (dense embed, bm25 sparse, rerank). The LLM call is I/O-bound and long, so async concurrency handles it well.

## Decision

1. **One uvicorn worker.** The image starts `uvicorn app.api:app --workers 1 --proxy-headers --forwarded-allow-ips="*" --port 7860`. All state lives in that process's memory.

2. **CPU work runs on a dedicated `ThreadPoolExecutor(max_workers=2)`,** not the default executor (which allows about 40 threads). Each ONNX session runs with `threads=1`, so 2 workers × 1 thread match the 2 vCPU with no oversubscription. The event loop never runs ONNX directly.

3. **DuckDB uses its own connection per request** (for example `cursor()` on one shared database that holds a view over `listings.parquet`), so concurrent aggregates never share a connection. DuckDB calls also run on the dedicated executor, because they block.

4. **The in-memory state and its limits:**

   | State | Store | Limit |
   |---|---|---|
   | Sessions | `TTLCache`, keyed by an HttpOnly cookie `sid` | 30 min TTL, 1,000 sessions, last 2 turns plus slots |
   | L1 answers | `TTLCache` | 24 h TTL, 2,000 entries |
   | L2 answers | Qdrant `qa_cache` collection, in memory | Size limit set in `docs/specs/caching.md` |
   | L3, L4 | LRU | Size limits set in `docs/specs/caching.md` |
   | Rate limits | slowapi, memory storage | 10/min and 40/day per client IP |
   | Quota and cooldowns | `LLMRouter` fields | Refreshed by the poll every 5 min |
   | Telemetry | Ring buffer | Size set by the backend worker |

5. **Client identity comes from the proxy header.** HF Spaces sits behind a proxy, so the socket peer is the proxy. uvicorn runs with `--proxy-headers`, and the rate limiter keys on the **first** `X-Forwarded-For` hop. Without that, every user would share one bucket.

6. **Health is split.** `/health` is liveness and does nothing expensive. `/ready` turns green only after the manifest checks pass, the index is loaded, and the models are warm. A keep-alive workflow requests `/ready` every 6 hours so the Space does not sleep.

7. **The process exits on unrecoverable startup errors** (ADR-0003 fail-fast) instead of serving degraded answers from a bad index.

## Alternatives rejected

- **Several uvicorn or gunicorn workers.** Each would load the index and the ONNX models again, caches would be split, and per-IP limits would be multiplied by the worker count. With 2 vCPU and CPU work already on 2 threads, more processes would not add throughput.
- **Redis for sessions, caches, rate limits and quota.** It is the right move for scale-out (ADR-0001), but here it adds a process, a network hop on the cache-hit path, and more to secure, with no benefit at one replica.
- **Stateless runtime with client-held sessions (state in a signed cookie or in the request).** That works for sessions, but not for caches, rate limits or the quota. It also widens the request contract.
- **Persisting state to disk.** The Space disk is ephemeral and is wiped on restart, so persistence would give a false sense of durability.
- **The default executor for ONNX.** Up to about 40 threads competing for 2 cores. Tail latency gets worse, not better.

## Consequences

- Hit rates and rate limits are exact for the single replica, and there is no cross-process coordination.
- A restart or a sleep loses sessions, L1, L2, L3 and L4, counters, and the telemetry buffer. The L0 warm cache is baked into the image, so the probe questions are still fast and quota-free right after a cold start. The quota value comes back on the first poll.
- `/stats` describes the current process lifetime only. The benchmark (`BENCHMARK.md`) records cold and warm runs separately.
- Throughput is bounded by one event loop plus two CPU threads. The bench at 1 and 5 concurrent users measures where that bound sits.
- Horizontal scaling is not possible without the Redis and Qdrant-server migration. The ports (ADR-0001) keep that migration to new adapters.

## Trade-off accepted

State is lost on restart. The L0 warm cache baked into the image mitigates the user-visible part (cold-start latency and quota on common questions). Scale-out is deferred until it is needed.
