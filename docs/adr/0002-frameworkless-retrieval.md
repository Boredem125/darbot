# ADR-0002: Retrieval without a framework (numpy hybrid search + fastembed + openai SDK)

- Status: Accepted; amended 18 Nov 2026 (amendment 1: numpy runtime retriever and L2 cache, see the end)
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0001, ADR-0003, ADR-0005, `docs/specs/retrieval.md` (dg-ml-lead)

## Context

The answer path is short: embed the query, run a hybrid search with payload filters, optionally rerank, build a prompt, stream one completion. The user's priorities are architecture first and measured latency. Every stage needs its own timer (`embed_ms`, `sparse_ms`, `retrieve_ms`, `rerank_ms`, `llm_ttft_ms`, `llm_total_ms`), and the SLOs (cache hit p95 < 50 ms, retrieval p95 < 150 ms) leave little room for framework overhead.

Three facts constrain the choice:
- The runtime is async (FastAPI, SSE). Some RAG frameworks wrap sync clients or pin one client type, which blocks the event loop or forces a thread hop per call.
- OpenRouter and Groq both speak the OpenAI chat API, so one SDK covers both.
- The image ships to HF Spaces, where image size affects build and cold-start time.

## Decision

The runtime uses three libraries directly, each behind a port (ADR-0001):

| Need | Library | Port |
|---|---|---|
| Dense + sparse query embedding, rerank | `fastembed` (ONNX): `paraphrase-multilingual-MiniLM-L12-v2` (384-d), `Qdrant/bm25`, `Xenova/ms-marco-MiniLM-L-6-v2` | `Embedder`, `Reranker` |
| Hybrid search | In-process numpy engine (`NumpyHybridRetriever`) implementing the Qdrant Query API semantics: dense top-20 prefetch plus bm25 top-20 prefetch, weighted RRF with explicit k, payload filters on both prefetches. `qdrant-client` local mode is the test oracle only (amendment 1) | `Retriever` |
| L2 semantic answer cache | numpy, one matrix per cache scope (amendment 1) | `Cache` |
| Aggregates | `duckdb` over `listings.parquet`, whitelisted SQL templates only (`AggregateOp`) | `StructuredQuery` |
| Generation | `openai` SDK, one `AsyncOpenAI` client per base URL | `LLM` (ADR-0005) |

`llama-index-core` is used **only at build time**, in `organize/`, for its sentence splitter. It sits in the `build` extra of `pyproject.toml` and never enters the runtime image. The glue (filter translation, RRF call, prompt assembly, streaming loop) is hand-written in adapters and the service. The core call is in the PLAN appendix and must be checked against the pinned qdrant-client at build time.

Retrieval tuning (RRF limits, score floor, rerank keep or drop, chunk size) belongs to dg-ml-lead in `docs/specs/retrieval.md`. This ADR fixes only the library choice and the seams.

## Alternatives rejected

- **LlamaIndex or LangChain query engines at runtime.** They add layers between us and the stage timers, they bring large dependency trees into the image, and their sync/async client handling is a known source of event-loop blocking. Their value (many connectors, agents) is not needed for one store and one completion call.
- **sentence-transformers + PyTorch for embeddings.** Hundreds of MB of torch in the image and slower CPU inference than ONNX on 2 vCPU.
- **A hosted embedding API.** Adds a network hop and a second quota to every query, and needs a key in CI.
- **LiteLLM as the LLM layer.** A useful router, but the chain here is five entries over two OpenAI-compatible endpoints. Our own router (ADR-0005) is about 200 lines and exposes exactly the failover events the SSE contract needs.
- **Dense-only search.** Exact strings matter here (district names, project names, "3-bed"), and bm25 handles them better. The hybrid costs one extra prefetch.

## Consequences

- Every stage is timed at the port boundary by the service. The `end` event and `/stats` report real per-stage numbers, not framework totals.
- We own the glue, so we own its bugs. The adapters need their own tests: the numpy retriever against Qdrant local mode as an oracle (amendment 1), and the SQL adapter against a small DuckDB file.
- The runtime image carries only the runtime pins in `pyproject.toml`. Build-time libraries stay in the `build` extra.
- The vendor-containment rule (R9 in the dependency spec) keeps `qdrant_client`, `fastembed`, `openai`, `duckdb` and `onnxruntime` out of `domain`, `service` and `api`.
- qdrant-client API details (Prefetch, `RrfQuery(rrf=Rrf(k, weights))`, sparse IDF modifier in local mode) were verified against 1.20.0 in Wave 1b; the parity test pins them.

## Trade-off accepted

We hand-write the glue that a framework would give us. In return we get fewer layers, exact per-stage timing, no sync/async client lock-in, and a smaller image.

## Amendment 1 (18 Nov 2026, WP-2-contracts): numpy at runtime, Qdrant as the test oracle

### Context

Wave 1b measured Qdrant local mode (`AsyncQdrantClient(":memory:")`, qdrant-client 1.20) on the dev box (`docs/specs/retrieval.md` §0, `.orchestration/WP-1b-aispec.report.md` P1):
- A filtered hybrid query at 6,000 points took p95 653 ms, and 1,002 ms at 12,000 points. The retrieval-path SLO is p95 < 150 ms for embed plus search plus rerank.
- Local mode evaluates payload conditions per point in Python and ignores payload indexes (it warns that they have no effect).
- `AsyncQdrantLocal.query_points` is async in signature only. The search runs synchronously on the event loop, so one query stalls every open SSE stream.
- An in-process numpy engine with the same semantics (dense matmul, boolean filter mask, bm25 posting lists, RRF) took p95 1.54 ms on the same data and filters.
- Local mode's RRF uses k = 2 by default. The fused score depends only on rank, so it cannot drive a score floor.

### Decision

1. The `Retriever` port is implemented by an exact, in-process numpy engine (`NumpyHybridRetriever`) over the unchanged `points_*.parquet` files. It reproduces Qdrant's semantics exactly: filter rules (a null target field fails its condition), bm25 IDF `ln((N - df + 0.5)/(df + 0.5) + 1)` over the whole collection, and weighted RRF `w / ((p+1) + w(k-1))` with k set explicitly (initial 60). Parameters live in `docs/specs/retrieval.md`.
2. The L2 semantic answer cache is numpy too: one float32 matrix per cache scope (`docs/specs/caching.md` §4). There is no `qa_cache` collection at runtime.
3. Qdrant has no runtime role. `qdrant-client` stays a pinned dependency for:
   - the parity test `tests/test_retriever_parity.py`, which asserts identical top-20 ids and fused scores within 1e-6 against `RrfQuery(rrf=Rrf(k=60, weights=[1, 1]))`;
   - an optional `RETRIEVER_BACKEND=qdrant_local` adapter. It is a should-have and is the first thing cut.
4. The port gains `RetrievalResult.top_dense_score`, the raw cosine of the filtered dense top-1. The score floor uses it, never the RRF score (CONTRACT_CHANGES CC-1).
5. The numpy search runs on the event loop (p95 well under the 5 ms blocking rule). If the HF bench shows `retrieve_ms` p95 > 5 ms, it moves to the ONNX executor behind a config switch (`docs/specs/concurrency-slo.md` §1).

### Alternatives rejected

- **Keep Qdrant local mode and relax the SLO.** Latency is the user's second priority, and the loop would still block.
- **Local mode on a dedicated thread** (sync client in an executor). It unblocks the loop, but each query still costs about 200 ms or more.
- **A Qdrant server sidecar in the container.** It is fast and truly async, but it adds a second process to supervise, startup upserts and image size, all for fewer than 10k points.
- **Keep in-memory Qdrant for L2 only** (AI-SPEC option a). Scope-filtered local search scales to about 55 ms at the 2,000-entry L2 cap. That alone exceeds the 50 ms cache-hit SLO, and it blocks the loop.

### Consequences

- About 1.5 ms per filtered hybrid query instead of hundreds, with exact (brute-force) recall.
- We own a small search engine whose semantics must not drift from Qdrant's. The parity test is the guard, and it needs no models and no network.
- The upgrade path from ADR-0001/0008 still holds. Scaling out means a Qdrant server adapter behind the same port; the file formats already match Qdrant's model.
- R9 vendor containment is unchanged: inside `app`, `qdrant_client` may appear only in `adapters`. Tests are outside the rule's walk.
- Brute force is linear in points. At about 100k points (matmul in the tens of ms on 2 vCPU) this needs revisiting, the same bound ADR-0003 already states.

### Trade-off accepted

We maintain a small search engine instead of calling a library. In return the retrieval path fits its SLO by two orders of magnitude, the event loop never blocks on search, and a test proves the semantics match Qdrant's.
