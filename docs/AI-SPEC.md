# AI-SPEC: DarGlobal/Wasalt RAG chatbot

Status: draft for Wave 1b, 18 Nov 2026. Author: gsd-ai-researcher (T1). Scope of this file: sections 3, 4 and 4b.

| Input | Value |
|---|---|
| Framework | **Option B**: no orchestration framework. `qdrant-client==1.20.0` (Query API, local `:memory:` mode), `fastembed==0.9.1`, `openai==3.55.0` (SDK pointed at OpenRouter and Groq). `llama-index-core==0.14.26` is used for build-time chunking only. |
| System type | RAG (hybrid dense + bm25 retrieval with payload filters), plus a structured-query route (DuckDB templates) and offline LLM extraction |
| Model provider | Model-agnostic over OpenAI-compatible endpoints: OpenRouter free models first, then Groq (ADR-0005) |
| Phase | Thu AM step 3 (PLAN section 11). The spec feeds Wave 2 to 4 workers (`dg-backend-engineer`, `dg-data-engineer`, `dg-qa-engineer`) |
| Canonical context | `.orchestration/PLAN.md`; frozen contracts in `common/schemas.py`, `app/domain/ports.py` and `docs/contracts/` |

Sections 1 and 2 (system classification and framework decision) are covered by PLAN sections 1 to 2 and ADR-0002. They are not repeated here.

**Verification legend.** Each fact or snippet carries one tag:
- **[I]** The signature was introspected from the installed package in `.venv` (`inspect.signature`, `model_fields`, source reading).
- **[R]** The snippet was executed offline in `.venv`, against an in-memory Qdrant, the real `Bm25` class, or an `httpx2.MockTransport` that fakes OpenRouter and Groq. No model downloads and no LLM calls.
- **[D]** Taken from official provider docs (URLs in 3.7), because the package cannot answer it.
- **[U]** Unverified. Must be verified in the build (Wave 2 to 4); the check to run is stated.

All `[snippet:...]` blocks are extracted and executed by `.orchestration/tmp/aispec/verify_spec.py` (see `.orchestration/WP-1b-aispec.report.md`).

---

## 3. Framework Quick Reference

### 3.1 Installation

The pins already in `pyproject.toml` are correct for the installed environment [I]:

```bash
# runtime image
pip install "qdrant-client[fastembed]==1.20.0" "fastembed==0.9.1" "openai==3.55.0"
# build-time only (organize/, chunking)
pip install "llama-index-core==0.14.26"
# or, from the repo
pip install -e ".[build,dev]"
```

Three things the pins pull in transitively that matter:
- **`openai==3.55.0` runs on `httpx2` (2.16.2), not `httpx`** [I]. `httpx==0.28.1` is still installed for our own calls (quota poller, tests). `openai.Timeout` and `openai.DefaultAsyncHttpxClient` are the httpx2 types. An `httpx.AsyncClient` passed as `http_client` was also accepted at request time [R], but use the openai-exported types so timeouts and transports stay one library.
- `fastembed` brings `onnxruntime` 1.32.0, `tokenizers`, `mmh3` and `py-rust-stemmers` (bm25 stemming, no download).
- `llama-index-core` ships its own tiktoken and nltk caches under `llama_index/core/_static/`. `SentenceSplitter` worked with the network blocked [R].

Model files are not pip packages. FastEmbed downloads them on first construction, into `FASTEMBED_CACHE_PATH` or `cache_dir`. **The default is `%TEMP%\fastembed_cache`, which is on the nearly full C: drive locally** [I]. Sizes from the fastembed registry [I]: dense 0.22 GB, bm25 0.01 GB, reranker 0.08 GB.

### 3.2 Key imports (all resolved in `.venv`) [I][R]

```python
# [snippet:imports]
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion  # client-side RRF (score-floor variant)
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
import openai
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from llama_index.core.node_parser import SentenceSplitter  # build-time only, never in app/
```

### 3.3 Minimal entry point for this system type (hybrid RAG turn)

This is a self-contained script: index three documents, run the PLAN appendix hybrid query with a payload filter, and stream a grounded answer. On first run it downloads about 0.23 GB of models and needs `OPENROUTER_API_KEY` (one free-tier call). It was executed offline with a fake dense model, the real `Bm25` class and a mocked OpenRouter endpoint [R].

```python
# [snippet:quickstart]
# quickstart.py: one hybrid-RAG turn with Option B. Run: python quickstart.py
import asyncio
import os

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models

DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384-d
SPARSE_MODEL = "Qdrant/bm25"
DOCS = [
    "3-bed villa for sale in Al Narjis, Riyadh. Price SAR 1,950,000. Area 350 sqm.",
    "2-bed apartment for rent in Al Rawdah, Jeddah. Rent SAR 60,000 yearly.",
    "Trump Tower Jeddah by Dar Global: branded residences on the Jeddah Corniche.",
]


def unit(v: np.ndarray) -> list[float]:
    v = v.astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()  # fastembed does NOT L2-normalise this model


def to_qdrant(s) -> models.SparseVector:  # fastembed SparseEmbedding -> Qdrant SparseVector
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.astype(float).tolist())


async def main() -> None:
    dense = TextEmbedding(DENSE_MODEL, threads=1)
    sparse = SparseTextEmbedding(SPARSE_MODEL)
    client = AsyncQdrantClient(":memory:")
    await client.create_collection(
        collection_name="kb",
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    await client.upsert(
        "kb",
        points=[
            models.PointStruct(
                id=i,
                vector={"dense": unit(d), "bm25": to_qdrant(s)},  # doc side: embed()
                payload={"text": text, "city": "Riyadh" if "Riyadh" in text else "Jeddah"},
            )
            for i, (text, d, s) in enumerate(zip(DOCS, dense.embed(DOCS), sparse.embed(DOCS)))
        ],
        wait=True,
    )

    question = "3 bedroom villa in Riyadh under 2M"
    q_dense = unit(next(iter(dense.query_embed(question))))
    q_sparse = to_qdrant(next(iter(sparse.query_embed(question))))  # query side: values all 1
    flt = models.Filter(
        must=[models.FieldCondition(key="city", match=models.MatchValue(value="Riyadh"))]
    )
    res = await client.query_points(
        collection_name="kb",
        prefetch=[
            models.Prefetch(query=q_dense, using="dense", limit=20, filter=flt),
            models.Prefetch(query=q_sparse, using="bm25", limit=20, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5,
        with_payload=True,
    )
    context = "\n".join(
        f'<source n="{n}">{p.payload["text"]}</source>' for n, p in enumerate(res.points, 1)
    )

    llm = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=0,  # every retry is another request against the 50/day free cap
        timeout=30.0,
    )
    stream = await llm.chat.completions.create(
        model="google/gemma-4-31b-it:free",
        messages=[
            {"role": "system", "content": "Answer only from <source>. Cite [n] after each fact."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=500,
        temperature=0,
    )
    try:
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:  # usage chunk may have no choices
                print(chunk.choices[0].delta.content, end="", flush=True)
    finally:
        await stream.close()  # releases the HTTP stream even on Ctrl-C / cancellation
        await llm.close()
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### 3.4 Core abstractions in this system

| Abstraction | Where it lives | What it does here |
|---|---|---|
| `AsyncQdrantClient(":memory:")` + `query_points` [I][R] | `app/adapters/qdrant_retriever.py`, `caches.py` (L2 `qa_cache`) | Holds collection `kb` (named vectors `dense` 384-d cosine and `bm25` sparse with `Modifier.IDF`), loaded from `points_*.parquet` at startup. `query_points` returns a `QueryResponse`; hits are in `.points` (`ScoredPoint`: `id`, `score`, `payload`). In local mode it is an exact brute-force search **that runs synchronously on the event loop** (pitfall P1). |
| `models.Prefetch` + `models.FusionQuery(fusion=models.Fusion.RRF)` or `models.RrfQuery(rrf=models.Rrf(k=...))` [I][R] | retriever | Dense top-20 and bm25 top-20, each with the payload `Filter`, fused by RRF. The default RRF constant is **k = 2** (`qdrant_client.hybrid.fusion.DEFAULT_RANKING_CONSTANT_K`), so fused scores are rank-only (pitfall P2). `RrfQuery` exposes `k` and per-prefetch `weights`. |
| `TextEmbedding`, `SparseTextEmbedding("Qdrant/bm25")`, `TextCrossEncoder` [I] | `app/adapters/fastembed_embedder.py`, reranker; `organize/build_index.py` | ONNX models. Dense `query_embed`/`embed` yield numpy vectors (mean-pooled, **not normalised**). Sparse doc-side `embed` yields BM25 term weights; `query_embed` yields hashed token ids with value 1. `TextCrossEncoder.rerank(query, documents, batch_size=64)` yields one float per document. |
| `AsyncOpenAI` + `AsyncStream` [I][R] | `app/adapters/llm_router.py`, `collect/extract_projects.py` | One client per `base_url` (OpenRouter, Groq), `max_retries=0`, explicit `openai.Timeout`. Streaming uses `chat.completions.create(stream=True, stream_options={"include_usage": True})`. `AsyncStream` supports `async for`, `async with` and `await .close()`. Use `.with_raw_response.create(...)` to read rate-limit headers; `.parse()` on that result is **synchronous** and returns the `AsyncStream`. |
| `SentenceSplitter` [I][R] | `organize/` only | Splits project and article text into chunks. `chunk_size` and `chunk_overlap` are counted in **tiktoken cl100k tokens**, not MiniLM word pieces (pitfall P9). |

### 3.5 Pitfalls (specific to these versions; each was checked as tagged)

**P1. The local "async" Qdrant client blocks the event loop, and filtered hybrid search is slow.** [R]
- `AsyncQdrantLocal.query_points` is an `async def` that calls the synchronous `LocalCollection.query_points` directly. There is no thread offload, so the brute-force search and the per-point Python filter evaluation run on the event loop.
- Measured in `.venv` on this laptop, with 4,000 points (384-d dense, about 40 sparse terms each) and the filter `city`, `bedrooms` range and `price_sar` range, p50 over 9 runs:

  | Query | p50 |
  |---|---|
  | dense, no filter | 11.8 ms |
  | dense, filter | 50.8 ms |
  | sparse, no filter | 45.1 ms |
  | sparse, filter | 179.9 ms |
  | **hybrid (appendix call), filter** | **199.9 ms** |
  | hybrid, filter pre-resolved to `HasIdCondition` (59 ids) | 118.6 ms |
  | numpy reference (same semantics: cosine, bm25 term match, boolean mask, RRF k=2) | 0.53 ms |

- `create_payload_index` has no effect in local mode. It emits `UserWarning: Payload indexes have no effect in the local Qdrant` [R]. The manifest contract's "create payload indexes" step is harmless but buys nothing.
- **Consequence:** the PLAN SLO "retrieval path p95 < 150 ms" is at risk at our corpus size (~2k sale + 1k rent listings plus project and article chunks). While a query runs, every other request, including SSE token forwarding, is stalled. HF's 2 vCPU are unlikely to be faster than this laptop.
- **Decision needed (dg-ml-lead with dg-lead-architect; ADR-0002 and ADR-0003 are affected).** Options, cheapest first:
  - (a) Keep `AsyncQdrantClient(":memory:")` for the L2 `qa_cache` (small), and implement the `Retriever` port for `kb` as a numpy matrix plus an inverted index plus boolean payload masks. This uses the same semantics as the Qdrant call, about 1 ms, and costs no extra dependency. The port makes this a single-adapter swap.
  - (b) Run Qdrant server in the container (real HNSW and payload indexes, truly async HTTP). The same client code works with `AsyncQdrantClient(url=...)`, but it adds a binary and a process.
  - (c) Keep local mode but move the call onto a dedicated single-thread executor using a sync `QdrantClient(":memory:")`. This unblocks the loop, but per-query latency stays around 200 ms.
- Whatever is chosen, measure `retrieve_ms` at the real corpus size in Wave 2 before tuning.

**P2. RRF scores are rank-only, so a score floor on them is meaningless.** [R]
- With the default k = 2, a point ranked first by one prefetch scores 1/2, and a point ranked first by both scores 1/2 + 1/3 = 0.833 (observed 0.8333 [R]).
- The top fused hit therefore always scores at least 0.5, whatever the query. A "score below the floor, refuse" rule (PLAN section 5, step 4) cannot be applied to `RetrievedHit.score` after fusion.
- Use the **raw dense cosine of the best dense hit** as the floor input. Get it from the dense-only query (`query_batch_points` with two `QueryRequest`s, then `reciprocal_rank_fusion(...)` client-side; see 4.4), or use the rerank score for English.
- This needs a small, backward-compatible contract change (see the report: `RetrievalResult.top_dense_score`).

**P3. Sparse vector format.** [I][R]
- *Doc side vs query side.* `SparseTextEmbedding("Qdrant/bm25").embed(...)` produces BM25 term-frequency weights (float64). `.query_embed(...)` produces unique token ids with value `1` (**int64**).
  - Using `embed` for queries double-weights TF.
  - Using `query_embed` for documents loses TF.
  - The IDF part is applied by Qdrant only when the collection declares `SparseVectorParams(modifier=models.Modifier.IDF)`. Without it, bm25 degrades to term overlap.
- *Types.* Indices are `abs(mmh3.hash(token))`, an int64 below 2^31, so they fit the manifest's `list<uint32>`. Both `qdrant_client.models.SparseVector` and `common.schemas.SparseVector` accept numpy arrays (pydantic coerces them) [R]. Still call `.tolist()` and cast values to float, so the int64 query values become `1.0` and parquet stays `float32`.
- *Uniqueness.* Duplicate indices are rejected on both upsert and query: `ValueError: Indices must be unique` [R]. fastembed already deduplicates; any hand-built sparse vector (tests, fakes) must too. An empty sparse query (all stopwords) is accepted and simply returns no bm25 hits [R].
- *Same parameters both sides.* `Bm25(..., k=1.2, b=0.75, avg_len=256.0, language="english")` are constructor defaults [I]. `avg_len` should be set at build time to the corpus mean token count (listing cards are far shorter than 256). `language` must be identical at build and query time, and is recorded only implicitly through `Manifest.sparse_model`.
- *Arabic and numbers.* With `language="english"`, Arabic tokens are kept but neither stemmed nor stopword-filtered ("الرياض" and "رياض" stay different tokens) [R]. `"SAR 1,950,000"` tokenises to `sar`, `1`, `950`, `000` [R]. Normalise numbers (`common/text.py`) before building embed-mode text, and rely on payload filters, not bm25, for prices.

**P4. The dense vectors are not normalised.** [I]
- fastembed registers `paraphrase-multilingual-MiniLM-L12-v2` as a `PooledEmbedding` (mean pooling only), not a `PooledNormalizedEmbedding`.
- Qdrant normalises stored vectors for cosine itself (stored norm 1.0 [R]). Our own consumers do not, and the contracts require normalised vectors: `QueryEmbedding.dense` ("L2-normalised"), the manifest's float16 `dense` column, the L2 semantic-cache cosine and any numpy retriever.
- Normalise in float32, **then** cast to float16 at build time.

**P5. openai SDK defaults are wrong for a free-tier quota.** [I][R]
- `DEFAULT_MAX_RETRIES = 2`. With a mocked 429, `max_retries=2` produced **3 HTTP requests** for one call [R]. Against OpenRouter's 50 free requests per day, where failed calls count (P8), SDK retries triple the burn and also hide which provider failed.
- `DEFAULT_TIMEOUT` is `Timeout(connect=5.0, read=600, write=600, pool=600)`.
- Set `max_retries=0` and an explicit `openai.Timeout(...)` on every client. LLMRouter owns failover; a 429 is never retried on the same provider (ADR-0005).

**P6. Streaming cancellation (client disconnect, TTFT deadline).** [R][D]
- When the browser disconnects, sse-starlette cancels the response task. `CancelledError` surfaces inside whatever `await` the generator chain is in.
- If the router's generator does not `await stream.close()` in a `finally`, the upstream HTTP stream stays open until GC.
- A test with a slow mock body confirmed that cancelling the consumer mid-stream and closing in `finally` closes the upstream body [R].
- The TTFT deadline works as `async with asyncio.timeout(8): create(...) + first content chunk`, raising `TimeoutError` after 0.50 s in the test [R]. **Never `yield` inside that `asyncio.timeout` block**: the deadline would keep running while the consumer holds the generator.
- Aborting does not refund work everywhere. OpenRouter documents that for providers without abort support, Groq and Google among them, "the model will continue processing and you will be billed for the complete response" [D]. Assume the same for Groq direct: a cancelled Groq stream still consumes its TPM. A TTFT-timeout failover from a slow OpenRouter call still counted as a request (P8).
- After a disconnect: emit no `end` event and cache nothing. Still record telemetry with `finish_reason=error`.

**P7. Streaming error and usage shapes differ between OpenAI, OpenRouter and Groq.** [R][D]
- *Before the first byte:* a normal HTTP status. A 429 maps to `openai.RateLimitError` (`.status_code`, `.response.headers["retry-after"]`) [R].
- *After the stream starts* (OpenRouter): HTTP stays 200, and the last chunk has a top-level `error` plus `finish_reason: "error"` [D]. openai 3.55 **raises `openai.APIError` while iterating** on that chunk [R]. A dropped connection mid-stream raises `openai.APIConnectionError`, also an `APIError` subclass [R]. So after the first token, one `except openai.APIError` maps to `LLMDone(finish_reason=error)`. Before it, the same exception means fail over.
- *Keep-alive comments* (`: OPENROUTER PROCESSING`) are skipped by the SDK's SSE parser [R].
- *Usage chunk:* OpenAI sends `choices: []`. OpenRouter sends one choice with an empty delta that **repeats `finish_reason`** [D]. Read `chunk.usage` before touching `chunk.choices[0]`, and treat `finish_reason` as idempotent.
- A 200 stream that ends with no content is a failure, not an empty answer [D]. Fail over if no content delta arrived.

**P8. OpenRouter free-tier quota counting.** [D]
- The limits are **50 requests per UTC day, shared across all `:free` models (including the `openrouter/free` router), plus 20 RPM**, while lifetime credit purchases are under 10 credits [D].
- The official limits page states that failed requests count only for key-management endpoints; it is silent for free models. PLAN section 3 records failed calls as counting, so **treat every request that reached OpenRouter as consumed**: 429s, 5xx, TTFT timeouts and mid-stream errors.
- Implementation:
  - Increment the local ledger **before** sending, not after success.
  - Reconcile every 5 min with `GET https://openrouter.ai/api/v1/key` → `data.free_model_daily_requests.{used, limit, remaining}` [D].
  - Successful inference responses carry **no** `X-RateLimit-*` headers; only 429s do [D]. Never try to infer the quota from response headers.
  - Skip OpenRouter when `remaining <= 3` (ADR-0005).
  - Never call OpenRouter in CI or in the fake-adapter tests.

**P9. Build-time chunking and the 128-token dense model.** [I][R][U]
- `SentenceSplitter(chunk_size=1024, chunk_overlap=200)` are the defaults [I], measured in cl100k tokens.
- `paraphrase-multilingual-MiniLM-L12-v2` was trained with a 128 word-piece window. fastembed truncates at the tokenizer's `model_max_length` (from `tokenizer_config.json`, not readable without the download) [I][U].
- Keep project and article chunks small; 128 to 200 cl100k tokens is a starting point and the final value is dg-ml-lead's call. Verify in Wave 2 by checking `TextEmbedding(...).model.tokenizer.truncation["max_length"]` after the model is cached [U].
- The default `secondary_chunking_regex` has no Arabic punctuation (`،` `؟`), so Arabic text splits on spaces.

**P10. Reasoning models eat `max_tokens` and TTFT.** [D][U]
- On OpenRouter, reasoning tokens are output tokens and count against `max_tokens`. If reasoning uses the whole budget, the response has no visible output [D].
- Groq exposes `reasoning_effort` (`low`, `medium`, `high` for gpt-oss; `none` for qwen3.8-27b, which "does not return reasoning tokens at this setting") and `include_reasoning` [D]. Its docs do not say whether reasoning counts toward `max_completion_tokens` [U].
- Set the per-provider reasoning control in `LLM_CHAIN` (4.1) and measure `usage.completion_tokens` against the visible answer length on the 5-query provider subset.
- The TTFT deadline counts until the first **content** delta, so reasoning time also burns it.

### 3.6 Folder structure (where each API is used)

```
common/schemas.py              SparseVector, ChunkPayload, SearchFilters, ChatEvent* (frozen)
collect/extract_projects.py    AsyncOpenAI -> Groq, json_schema/json_object, ProjectExtraction (4b.1)
organize/build_index.py        SentenceSplitter; TextEmbedding.embed + SparseTextEmbedding.embed (doc side)
                               -> points_*.parquet (float16 dense, uint32/float32 sparse) + manifest.json
app/adapters/
  fastembed_embedder.py        TextEmbedding.query_embed + SparseTextEmbedding.query_embed on a 2-thread executor
  qdrant_retriever.py          AsyncQdrantClient(":memory:"), create_collection, upsert, query_points (or numpy, P1)
  reranker.py (or inside the retriever module)  TextCrossEncoder.rerank
  llm_router.py                AsyncOpenAI per base_url, streaming, failover, quota ledger, key poller
  caches.py                    L2 qa_cache collection on the same AsyncQdrantClient
app/service.py                 consumes ports only; contextlib.aclosing(llm.stream(...))
app/api.py                     EventSourceResponse(ping=...), maps ChatEvent -> SSE
```

### 3.7 Sources

Installed packages (primary source; introspected 18 Nov 2026): `qdrant-client` 1.20.0, `fastembed` 0.9.1, `openai` 3.55.0 (with `httpx2` 2.16.2), `llama-index-core` 0.14.26, `pydantic` 2.14.3, `sse-starlette` 3.7.0.

Official docs used where the package cannot answer:
- OpenRouter limits and the key endpoint: https://openrouter.ai/docs/api-reference/limits
- OpenRouter streaming (comments, mid-stream errors, cancellation, usage chunk): https://openrouter.ai/docs/api-reference/streaming
- OpenRouter reasoning tokens: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
- Groq rate limits and headers: https://console.groq.com/docs/rate-limits
- Groq structured outputs: https://console.groq.com/docs/structured-outputs
- Groq reasoning: https://console.groq.com/docs/reasoning

---

## 4. Implementation Guidance

### 4.1 Models and parameters

| Role | Model id | Parameters | Source |
|---|---|---|---|
| Dense embed | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | dim 384 (`TextEmbedding.get_embedding_size` → 384 [R]); `threads=1`; L2-normalise in float32 | [I] |
| Sparse | `Qdrant/bm25` | `language="english"`, `avg_len=<corpus mean>` at build; `query_embed` at runtime; collection `Modifier.IDF` | [I] |
| Rerank (flag `RERANK_ENABLED`, English only) | `Xenova/ms-marco-MiniLM-L-6-v2` | `threads=1`; rerank top-20, keep 5; scores are raw model outputs [U: check range before using as a floor] | [I] |
| LLM 1 | OpenRouter `google/gemma-4-31b-it:free` | `temperature=0`, `max_tokens=500` (700 Arabic), `stream=True`, `stream_options={"include_usage": True}` | PLAN, [U] availability |
| LLM 2 | OpenRouter `nvidia/nemotron-3.5-super-vl:free` | as above, plus `extra_body={"reasoning": {"effort": "none"}}` if `/api/v1/models` does not mark reasoning mandatory | PLAN, [D] reasoning param, [U] model |
| LLM 3 | OpenRouter `openrouter/free` | as above; the routed model varies, so log `chunk.model` | PLAN |
| LLM 4 | Groq `openai/gpt-oss-120b` | `reasoning_effort="low"`, `extra_body={"include_reasoning": False}`, `max_tokens` cap with reasoning headroom (start at 1,000, measure) | [D], [U] headroom |
| LLM 5 | Groq `qwen/qwen3.8-27b` | `reasoning_effort="none"` | [D] |
| Extraction (offline) | Groq `openai/gpt-oss-120b` | non-streamed, `response_format` json_schema `strict: true`, `temperature=0`, `max_tokens` about 1,200 | [D] |

Client settings [I][R]:

| Client | base_url | Settings |
|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `max_retries=0`, `timeout=openai.Timeout(30.0, connect=5.0)` |
| Groq | `https://api.groq.com/openai/v1` | `max_retries=0`, `timeout=openai.Timeout(30.0, connect=5.0)` |

The TTFT deadline (8 s) is enforced with `asyncio.timeout`, not with the HTTP read timeout. An HTTP read timeout is per-read, and keep-alive comments would reset it.

`LLM_CHAIN` (env JSON) carries per-entry `provider`, `model`, `base_url`, `api_key_env`, `max_tokens_cap`, `reasoning_effort` and `extra_body`, so these knobs are config, not code (PLAN: "cut list is flag flips").

### 4.2 Index load at startup (local mode) [I][R]

```python
# [snippet:index_load]
import json

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import AsyncQdrantClient, models

from common.schemas import DENSE_VECTOR_NAME, KB_COLLECTION, SPARSE_VECTOR_NAME, ChunkPayload


async def load_kb(client: AsyncQdrantClient, shard_paths: list[str], dim: int) -> int:
    await client.create_collection(
        collection_name=KB_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    n = 0
    for path in shard_paths:
        table = pq.read_table(path).to_pylist()
        points = []
        for row in table:
            dense = np.frombuffer(row["dense"], dtype="<f2").astype(np.float32)  # float16 -> float32
            payload = ChunkPayload.model_validate_json(row["payload"])  # contract check at load
            points.append(
                models.PointStruct(
                    id=row["point_id"],  # uuid5 string
                    vector={
                        DENSE_VECTOR_NAME: dense.tolist(),
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=row["sparse_indices"], values=row["sparse_values"]
                        ),
                    },
                    payload=json.loads(payload.model_dump_json()),  # JSON-safe (datetimes as str)
                )
            )
        for i in range(0, len(points), 512):
            await client.upsert(KB_COLLECTION, points=points[i : i + 512], wait=True)
        n += len(points)
    return n
```

Notes:
- Payload values must be JSON-safe. `model_dump(mode="json")` or the round-trip above stores datetimes as ISO strings. Range filters on `scraped_at` are not used.
- Skip `create_payload_index` in local mode (P1), or call it and filter the `UserWarning`. Keep the call if option (b) of P1 is chosen.
- Loading 4,000 synthetic points took 0.54 s [R].

### 4.3 Embedder adapter (implements `ports.Embedder`) [I][R]

```python
# [snippet:embedder]
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding

from app.domain.ports import EmbeddingError, QueryEmbedding
from common.schemas import SparseVector


class FastEmbedEmbedder:
    def __init__(
        self,
        dense_model: str,
        sparse_model: str,
        executor: ThreadPoolExecutor,  # the shared ONNX pool, max_workers=2 (PLAN section 5)
        *,
        cache_dir: str | None = None,  # = FASTEMBED_CACHE_PATH baked into the image
    ) -> None:
        # local_files_only: the runtime must never reach the HF hub (models are baked in)
        self._dense = TextEmbedding(
            dense_model, cache_dir=cache_dir, threads=1, local_files_only=True
        )
        self._sparse = SparseTextEmbedding(sparse_model, cache_dir=cache_dir, local_files_only=True)
        self._names = (dense_model, sparse_model)
        self._dim = TextEmbedding.get_embedding_size(dense_model)
        self._executor = executor

    @property
    def dense_model(self) -> str:
        return self._names[0]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def sparse_model(self) -> str:
        return self._names[1]

    def _embed_sync(self, text: str) -> QueryEmbedding:
        d = next(iter(self._dense.query_embed(text))).astype(np.float32)
        d /= np.linalg.norm(d)  # P4: contract says L2-normalised
        s = next(iter(self._sparse.query_embed(text)))  # P3: query side, int64 values of 1
        return QueryEmbedding(
            dense=tuple(d.tolist()),
            sparse=SparseVector(indices=s.indices.tolist(), values=s.values.astype(float).tolist()),
        )

    async def embed_query(self, text: str) -> QueryEmbedding:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._executor, self._embed_sync, text)
        except Exception as exc:  # translated to the port error (chained)
            raise EmbeddingError(str(exc)) from exc
```

Notes:
- Warm it once at startup, before `/ready` turns green; the first ONNX run is slow. Wrap it with the L3 LRU keyed on the normalised query.
- `HF_HUB_OFFLINE=1` in the runtime image has the same effect as `local_files_only=True` [I].
- `sparse_ms` is reported on the per-call result, not through the adapter: time the bm25 call inside `_embed_sync` and return it as `QueryEmbedding.sparse_ms` (CC-2, `.orchestration/CONTRACT_CHANGES.md`). The service derives `embed_ms = max(W - (sparse_ms or 0), 0)` from the awaited wall time `W`. The earlier `last_timings` attribute idea is superseded, because a shared adapter attribute races under concurrent requests.

### 4.4 Retriever adapter (implements `ports.Retriever`) [I][R]

Filter mapping, exactly as documented on `SearchFilters`:

```python
# [snippet:filters]
from qdrant_client import models

from common.schemas import Purpose, SearchFilters


def build_filter(f: SearchFilters) -> models.Filter | None:
    must: list[models.Condition] = []

    def eq(key: str, value: str) -> None:
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

    def rng(key: str, lo: float | None, hi: float | None) -> None:
        if lo is not None or hi is not None:  # inclusive range (gte/lte)
            must.append(models.FieldCondition(key=key, range=models.Range(gte=lo, lte=hi)))

    if f.kinds:
        must.append(
            models.FieldCondition(key="kind", match=models.MatchAny(any=[str(k) for k in f.kinds]))
        )
    for key in ("source", "purpose", "property_type", "city", "district", "country"):
        value = getattr(f, key)
        if value is not None:
            eq(key, str(value))  # StrEnum -> its value; places are canonical English names
    if f.project_id is not None:
        eq("doc_id", f.project_id)
    rng("bedrooms", f.bedrooms_min, f.bedrooms_max)
    rng("area_sqm", f.area_min_sqm, f.area_max_sqm)
    price_key = "annual_rent_sar" if f.purpose is Purpose.rent else "price_sar"
    rng(price_key, f.price_min_sar, f.price_max_sar)
    return models.Filter(must=must) if must else None
```

Hybrid search, as in the PLAN appendix (verified unchanged against 1.20.0 [R]):

```python
# [snippet:retriever]
from qdrant_client import AsyncQdrantClient, models

from app.domain.ports import QueryEmbedding, RetrievalError, RetrievalResult, RetrievedHit
from common.schemas import (
    DENSE_VECTOR_NAME,
    KB_COLLECTION,
    SPARSE_VECTOR_NAME,
    ChunkPayload,
    SearchFilters,
)


class QdrantRetriever:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client

    async def search(
        self, embedding: QueryEmbedding, filters: SearchFilters, *, limit: int = 20
    ) -> RetrievalResult:
        flt = build_filter(filters)  # from [snippet:filters]
        sparse = models.SparseVector(
            indices=embedding.sparse.indices, values=embedding.sparse.values
        )
        try:
            res = await self._client.query_points(
                collection_name=KB_COLLECTION,
                prefetch=[
                    models.Prefetch(
                        query=list(embedding.dense), using=DENSE_VECTOR_NAME, limit=limit, filter=flt
                    ),
                    models.Prefetch(query=sparse, using=SPARSE_VECTOR_NAME, limit=limit, filter=flt),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),  # k = 2 (P2)
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:  # translated to the port error (chained)
            raise RetrievalError(str(exc)) from exc
        return RetrievalResult(
            hits=tuple(
                RetrievedHit(payload=ChunkPayload.model_validate(p.payload), score=p.score)
                for p in res.points
            )
        )
```

Notes:
- Putting the filter on each `Prefetch` (as above) and putting it once as top-level `query_filter` both returned only matching points in local mode [R]. Keep it per-prefetch: it is explicit and does not depend on propagation semantics.

**Score-floor variant (P2).** One batch call returns both raw lists. RRF is then applied client-side with the same function and default k that local mode uses, and the raw dense cosine stays available for the refusal floor [I][R]:

```python
# [snippet:retriever_floor]
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion


async def hybrid_with_floor(
    client: AsyncQdrantClient,
    dense: list[float],
    sparse: models.SparseVector,
    flt: models.Filter | None,
    limit: int = 20,
) -> tuple[list[models.ScoredPoint], float]:
    dense_res, sparse_res = await client.query_batch_points(
        collection_name="kb",
        requests=[
            models.QueryRequest(query=dense, using="dense", filter=flt, limit=limit, with_payload=True),
            models.QueryRequest(query=sparse, using="bm25", filter=flt, limit=limit, with_payload=True),
        ],
    )
    top_cosine = dense_res.points[0].score if dense_res.points else 0.0  # floor input
    fused = reciprocal_rank_fusion([dense_res.points, sparse_res.points], limit=limit)
    return fused, top_cosine
```

- `reciprocal_rank_fusion(responses, limit=10, ranking_constant_k=None, weights=None)` is importable from `qdrant_client.hybrid.fusion` but is not re-exported at package top level. Pin-protected, but re-check it on any upgrade.
- If P1 option (a) is chosen, the numpy retriever computes the same three numbers (dense ranks, bm25 ranks, top cosine) directly.

### 4.5 Reranker adapter (implements `ports.Reranker`) [I]

```python
# [snippet:reranker]
import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.domain.ports import RetrievalError, RetrievedHit


class FastEmbedReranker:
    def __init__(self, model: str, executor: ThreadPoolExecutor, *, cache_dir: str | None = None):
        self._model = TextCrossEncoder(model, cache_dir=cache_dir, threads=1, local_files_only=True)
        self._executor = executor

    async def rerank(
        self, query: str, hits: Sequence[RetrievedHit], *, top_k: int
    ) -> list[RetrievedHit]:
        docs = [h.payload.text for h in hits]
        loop = asyncio.get_running_loop()
        try:
            scores = await loop.run_in_executor(
                self._executor, lambda: list(self._model.rerank(query, docs, batch_size=32))
            )
        except Exception as exc:  # translated to the port error (chained)
            raise RetrievalError(str(exc)) from exc
        rescored = [
            RetrievedHit(payload=h.payload, score=float(s))
            for h, s in zip(hits, scores, strict=True)
        ]
        return sorted(rescored, key=lambda h: h.score, reverse=True)[:top_k]
```

The service skips it for Arabic (`ms-marco` is English-only) and when `RERANK_ENABLED=0`.

### 4.6 LLMRouter core (implements `ports.LLM.stream`) [I][R]

This is the part workers most often get wrong. The snippet shows the per-attempt logic: ledger-before-send, TTFT deadline, failover before the first token, no failover after it, and cleanup on cancellation. Cooldown bookkeeping, `status()` and the key poller are left to the worker.

```python
# [snippet:llm_router]
import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import openai
from openai import AsyncOpenAI

from app.domain.ports import (
    ChatMessage,
    LLMDelta,
    LLMDone,
    LLMFailover,
    LLMStreamItem,
    LLMUnavailableError,
)
from common.schemas import ErrorCode, FinishReason, Provider, TokenUsage

_FINISH = {
    "stop": FinishReason.stop,
    "length": FinishReason.length,
    "content_filter": FinishReason.content_filter,
}


@dataclass(frozen=True)
class ChainEntry:  # one item of the LLM_CHAIN env JSON
    provider: Provider
    model: str
    base_url: str
    max_tokens_cap: int | None = None  # reasoning headroom (P10)
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


class _NoContent(Exception):
    """The stream ended without a single content delta (P7)."""


def _classify(exc: BaseException) -> ErrorCode:
    if isinstance(exc, openai.RateLimitError):
        return ErrorCode.provider_rate_limited
    if isinstance(exc, TimeoutError | openai.APITimeoutError):
        return ErrorCode.provider_timeout
    return ErrorCode.provider_error  # 5xx, 4xx, transport, mid-stream error chunk, no content


class LLMRouter:
    def __init__(
        self,
        chain: Sequence[ChainEntry],
        clients: dict[str, AsyncOpenAI],  # one per base_url, max_retries=0
        *,
        ttft_s: float = 8.0,
        cooldown_s: float = 60.0,
    ) -> None:
        self._chain, self._clients = tuple(chain), clients
        self._ttft_s, self._cooldown_s = ttft_s, cooldown_s
        self._cool_until: dict[tuple[str, str], float] = {}
        self.openrouter_attempts_today = 0  # reconciled with GET /api/v1/key (P8)
        self.openrouter_remaining: int | None = None

    def _skip_reason(self, e: ChainEntry) -> str | None:
        if time.monotonic() < self._cool_until.get((e.base_url, e.model), 0.0):
            return "cooldown"
        if e.provider is Provider.openrouter and (
            self.openrouter_remaining is not None and self.openrouter_remaining <= 3
        ):
            return "openrouter quota guard"
        return None

    async def stream(
        self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float = 0.0
    ) -> AsyncIterator[LLMStreamItem]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        attempts: list[LLMFailover] = []
        for e in self._chain:
            if reason := self._skip_reason(e):
                attempts.append(LLMFailover(e.provider, e.model, ErrorCode.provider_skipped, reason))
                yield attempts[-1]
                continue
            if e.provider is Provider.openrouter:
                self.openrouter_attempts_today += 1  # count BEFORE sending (P8)
                if self.openrouter_remaining is not None:
                    self.openrouter_remaining -= 1
            kwargs: dict[str, Any] = {}
            if e.reasoning_effort:
                kwargs["reasoning_effort"] = e.reasoning_effort
            if e.extra_body:
                kwargs["extra_body"] = e.extra_body
            t0 = time.perf_counter()
            stream = None
            try:
                async with asyncio.timeout(self._ttft_s):  # P6: no yield inside this block
                    stream = await self._clients[e.base_url].chat.completions.create(
                        model=e.model,
                        messages=payload,
                        stream=True,
                        stream_options={"include_usage": True},
                        max_tokens=min(max_tokens, e.max_tokens_cap or max_tokens),
                        temperature=temperature,
                        **kwargs,
                    )
                    chunks = aiter(stream)
                    first = ""
                    while not first:  # skip role-only / reasoning-only chunks
                        chunk = await anext(chunks, None)
                        if chunk is None:
                            raise _NoContent()
                        if chunk.choices and chunk.choices[0].delta.content:
                            first = chunk.choices[0].delta.content
            except (TimeoutError, openai.APIError, _NoContent) as exc:
                if stream is not None:
                    await stream.close()
                self._cool_until[(e.base_url, e.model)] = time.monotonic() + self._cooldown_s
                attempts.append(LLMFailover(e.provider, e.model, _classify(exc), type(exc).__name__))
                yield attempts[-1]
                continue
            except BaseException:  # CancelledError (client gone) while waiting for token 1
                if stream is not None:
                    await stream.close()
                raise

            ttft_ms = (time.perf_counter() - t0) * 1000
            finish, usage = FinishReason.error, TokenUsage()
            try:
                yield LLMDelta(first)
                async for chunk in chunks:
                    if chunk.usage:  # P7: read usage before choices
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                        )
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.delta.content:
                        yield LLMDelta(choice.delta.content)
                    if choice.finish_reason:  # may repeat on OpenRouter's usage chunk
                        finish = _FINISH.get(choice.finish_reason, FinishReason.error)
            except openai.APIError:  # mid-stream error chunk or dropped connection: no failover
                finish = FinishReason.error
            finally:
                await stream.close()  # P6: also runs on CancelledError / aclose()
            yield LLMDone(
                provider=e.provider,
                model=e.model,
                finish_reason=finish,
                usage=usage,
                ttft_ms=ttft_ms,
                total_ms=(time.perf_counter() - t0) * 1000,
            )
            return
        raise LLMUnavailableError(attempts)
```

Notes:
- `aiter`/`anext` are builtins (3.10+). Iterating `chunks` after the first `anext` continues the same underlying stream.
- For Groq TPM throttling, swap `create(...)` for `with_raw_response.create(...)`, read `raw.headers["x-ratelimit-remaining-tokens"]`, then call `stream = raw.parse()`. `parse()` is synchronous [R]. Groq's `x-ratelimit-*-requests` headers are per day (RPD) and `*-tokens` headers are per minute (TPM) [D].
- The service consumes it with `async with contextlib.aclosing(llm.stream(...)) as items:`. Then an early `break` or a client disconnect closes the generator, and its `finally` closes the HTTP stream [R].
- 429s are classified `provider_rate_limited` and never retried on the same provider, because the cooldown starts immediately.

### 4.7 OpenRouter quota poller [D][R]

```python
# [snippet:quota]
import httpx


def parse_free_remaining(body: dict) -> int | None:
    """`GET /api/v1/key` -> data.free_model_daily_requests.remaining (UTC day)."""
    fmr = (body.get("data") or {}).get("free_model_daily_requests") or {}
    remaining = fmr.get("remaining")
    return int(remaining) if remaining is not None else None


async def poll_openrouter_quota(http: httpx.AsyncClient, api_key: str) -> int | None:
    r = await http.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=5.0,
    )
    r.raise_for_status()
    return parse_free_remaining(r.json())
```

- Run it every 5 min from a lifespan task, and set `router.openrouter_remaining` from the result. Log `used`, `limit` and `remaining` to `eval/quota_log.csv` for the PLAN section 9 ledger.
- The key endpoint is a key-management request. It does not consume the free-model inference quota [D].

### 4.8 Tool use, state, context window

- **Tool use: none.** There is one LLM call per turn (ADR-0004). Routing, filters and slots are rule-based in `app/domain/router.py`. Do not pass `tools=`: Groq structured outputs and tool use cannot be streamed [D], and every extra call spends quota.
- **State:**
  - Sessions live in a `cachetools.TTLCache(maxsize=1000, ttl=1800)` keyed by the HttpOnly cookie, holding the last 2 turns plus slots (ADR-0008).
  - There is no server-side conversation memory beyond that, and no framework memory object.
  - LLM clients, the Qdrant client, the ONNX models and the executor are process singletons created in the FastAPI lifespan and closed on shutdown (`await client.close()` for both `AsyncOpenAI` and `AsyncQdrantClient` [I]).
- **Context window:** the budget per turn is fixed (PLAN section 6). The system prompt is at most 350 tokens, context at most 1,200 (5 cards), history at most 300, and `max_tokens` is 500, or 700 for Arabic. That is about 2,400 tokens per call, well inside every chain model's window, so no compaction is needed. 4b.4 covers how the budget is enforced.

### 4.9 Offline project extraction (Groq, non-streamed)

The extraction call itself is in 4b.1. Two operational points:
- Throttle on Groq's TPM. Use about 8K TPM for gpt-oss-120b [D], and read `x-ratelimit-remaining-tokens` via `with_raw_response`.
- Run it in `collect/` with `asyncio.run(main())` at the top of the script, never inside the app.

---

## 4b. AI Systems Best Practices

### 4b.1 Structured outputs with Pydantic

**Where structure applies in this system:**
1. **Offline project extraction** (the only LLM call whose output is parsed as JSON).
2. **Every runtime event**: `ChatEvent*` in `common/schemas.py` are pydantic models with cross-field validators, and the API emits only `event.to_sse()`.

The chat answer itself is free text by design (streamed). Its "schema" is enforced by the deterministic grounding check (ADR-0007), not by JSON mode.

**LLM-facing extraction model.** Keep it deliberately looser than `common.schemas.Project`:
- Types and nullability only, every field required, `extra="forbid"`. Groq strict mode requires every property in `required` and `additionalProperties: false`; optional values are expressed as a union with null [D]. `model_json_schema()` produces exactly that for this model [R].
- No `pattern`, `ge` or `le` in this schema. Groq's docs do not list `pattern` support [D]. Those constraints are enforced afterwards, when the result is mapped into `Project`, whose validators run.

```python
# [snippet:extraction]
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError


class ProjectExtraction(BaseModel):
    """What the LLM may return for one DarGlobal project page. All nullable; never guessed."""

    model_config = ConfigDict(extra="forbid")
    name: str | None
    brand: str | None
    country: str | None
    city: str | None
    location: str | None
    status: str | None
    handover: str | None
    unit_types: list[str]
    bedrooms_min: int | None
    bedrooms_max: int | None
    area_sqm_min: float | None
    area_sqm_max: float | None
    price_from_amount: int | None
    price_from_currency: str | None
    payment_plan: str | None
    amenities: list[str]


SCHEMA = ProjectExtraction.model_json_schema()  # all required, additionalProperties false


async def extract_project(
    client: AsyncOpenAI,  # Groq client, max_retries=0
    model: str,
    system_prompt: str,  # app/prompts/extract_project_v1.txt
    page_text: str,
    *,
    json_schema: bool = True,  # False -> json_object + schema in the prompt
) -> tuple[ProjectExtraction | None, list[str]]:
    response_format = (
        {"type": "json_schema",
         "json_schema": {"name": "ProjectExtraction", "strict": True, "schema": SCHEMA}}
        if json_schema
        else {"type": "json_object"}
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<page>\n{page_text}\n</page>"},  # page is data, not instructions
    ]
    errors: list[str] = []
    for _attempt in range(2):  # 1 try + 1 retry carrying the validation error
        resp = await client.chat.completions.create(
            model=model, messages=messages, response_format=response_format,
            temperature=0, max_tokens=1200,
        )
        choice = resp.choices[0]
        raw = choice.message.content or ""
        if choice.finish_reason == "length":
            errors.append("finish_reason=length")  # truncated JSON: do not retry blindly
            break
        try:
            return ProjectExtraction.model_validate_json(raw), errors
        except ValidationError as exc:
            errors.append(f"{exc.error_count()} errors: {exc.errors()[0]['loc']}")
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid JSON for the schema:\n{exc}\nReturn only corrected JSON."},
            ]
    return None, errors
```

**Retry and surfacing policy:**
- At most 2 attempts per page (temperature 0, so a third try rarely changes anything).
- Log `slug`, the attempt number, `errors`, `finish_reason` and `usage` to the extraction log.
- On final failure, write the page with all extracted fields null and count it against the gate "schema-valid 40/40". Do not drop the page.
- After validation, run the **verbatim check**: every non-null string or number must appear on its page after whitespace and number normalisation, or it is set to null and logged. Only then map into `common.schemas.Project`, which enforces the currency pattern and the min ≤ max ranges.
- `client.chat.completions.parse(response_format=ProjectExtraction)` also works [R]. It sends `json_schema` with `strict: true` and returns `.choices[0].message.parsed`. It is not used because the `json_object` fallback path needs `model_validate_json` anyway; one code path is easier to test.

### 4b.2 Async-first design

How async works in this stack:
- `AsyncOpenAI` and `httpx` are truly non-blocking.
- `AsyncQdrantClient(":memory:")` is async in signature only (P1).
- fastembed is synchronous CPU work. It goes on the dedicated `ThreadPoolExecutor(max_workers=2)` via `loop.run_in_executor`, with ONNX `threads=1` per model (PLAN section 5; ONNX sets intra- and inter-op threads from `threads` [I]).
- DuckDB, also synchronous, uses `asyncio.to_thread` with one connection per request.

```python
# [snippet:lifespan]
import contextlib
from concurrent.futures import ThreadPoolExecutor

import openai
from fastapi import FastAPI
from openai import AsyncOpenAI


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    onnx_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="onnx")
    timeout = openai.Timeout(30.0, connect=5.0)
    app.state.llm_clients = {
        "https://openrouter.ai/api/v1": AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key="set-from-env",
            max_retries=0, timeout=timeout),
        "https://api.groq.com/openai/v1": AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1", api_key="set-from-env",
            max_retries=0, timeout=timeout),
    }
    app.state.onnx_pool = onnx_pool
    try:
        yield
    finally:
        for c in app.state.llm_clients.values():
            await c.close()
        onnx_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)
```

In the real app the keys come from `pydantic-settings`, never literals. The placeholders keep the snippet runnable without secrets.

**The common mistakes:**
- **Calling `asyncio.run()` (or a sync helper that calls it) from inside a FastAPI handler or the lifespan.** It raises `RuntimeError: asyncio.run() cannot be called from a running event loop`. Scripts (`collect/`, `organize/`, `scripts/`) call `asyncio.run(main())` once at the top level; library code is `async def` all the way down.
- **Calling `TextEmbedding.embed`/`query_embed` directly in an `async def`.** It blocks the loop for the full ONNX run (tens of ms). Always go through the executor.
- **Using the default executor (`None`).** It sizes itself to `min(32, cpu + 4)` threads and oversubscribes the 2 vCPUs.
- **Yielding inside `asyncio.timeout(...)` in an async generator** (P6).

**Stream vs await:**
- Stream the chat answer, since the SSE `start` event goes before the first token and TTFT is an SLO.
- Await the full response for anything that must be validated as a whole: project extraction, the LLM judge, and the grounding check (which runs after the stream ends; the `end` event carries its result).
- Groq cannot stream structured outputs at all [D].

### 4b.3 Prompt engineering discipline

- **System vs user separation:**
  - The system message is `app/prompts/system_v1.txt` only (at most 350 tokens, zero-shot, model-agnostic). It is static per deploy, so it is identical across requests and its version `compute_prompt_version("system_v1", bytes)` goes into every cache key and every `end` event.
  - Everything dynamic goes in the user message, in this order:
    1. `<source n="k">` cards (context-mode render with `[n]`, URL and `scraped_at`), or `<sql_result>`;
    2. the trimmed history;
    3. the question.
  - Source text is sanitised and wrapped, and the system prompt says to treat it as data. Never interpolate user text into the system message.
- **Few-shot:** none inline. The prompt is zero-shot by decision (PLAN section 6), because examples cost tokens on every call against a TPM-limited fallback and invite copying of example numbers, which the grounding check would then flag. If a provider fails the 5-query subset on format, prefer one tightened instruction over examples. Dynamic few-shot retrieval is out of scope.
- **Always set `max_tokens`:**
  - 500, or 700 for Arabic (`ARABIC_MAX_TOKENS`), capped per chain entry for reasoning models (P10).
  - For extraction, 1,200.
  - Never leave it unset. Groq's default `max_completion_tokens` is 1,024 [D], and other providers default to the model maximum.
  - `finish_reason == "length"` sets `flags.truncated`, and the answer is never cached (the contract enforces this).
- **Temperature 0 everywhere.** Do not send `seed`: it is not honoured uniformly across OpenRouter's upstreams, and caching already gives repeatability.
- **Change control:** follow PLAN section 6. Change one thing at a time, run the no-regression check on dev, then rebuild the warm cache. The manifest fail-fast rejects a stale warm cache automatically.

### 4b.4 Context window management (RAG)

The window is never the constraint (about 2.4K tokens per call). The **token budget** is: it drives Groq TPM, latency and cost. Enforce it before the call:

```python
# [snippet:budget]
import math


def est_tokens(text: str) -> int:
    """Conservative estimate without a tokenizer in the runtime image (tiktoken is build-only).
    Arabic script tokenises much denser than Latin. Calibrate both ratios against
    usage.prompt_tokens from the Groq dev eval, and log the ratio in telemetry."""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    return math.ceil((len(text) - arabic) / 3.5 + arabic / 1.5)


def fit_cards(rendered_cards: list[str], budget_tokens: int = 1200) -> list[str]:
    """Keep whole cards in rank order until the budget is reached; never cut a card mid-way,
    so every [n] the model sees maps to a complete source."""
    kept: list[str] = []
    used = 0
    for card in rendered_cards:
        cost = est_tokens(card)
        if kept and used + cost > budget_tokens:
            break
        kept.append(card)
        used += cost
    return kept
```

- **Ordering:**
  1. Retrieve 20 (hybrid).
  2. Optionally rerank to 5 (English).
  3. Deduplicate by `doc_id`, keeping the best chunk per document, so 5 cards are 5 documents.
  4. Run `fit_cards` with 1,200 tokens.
  5. Renumber `[n]` = 1..k only after fitting, so the `start` event's `sources[].n` matches what the model saw.
- Descriptions are already capped at 400 characters at organize time, so one card can never blow the budget by itself.
- **History:** the last 2 turns only, user text plus assistant text, trimmed from the oldest with the same estimator to at most 300 tokens. There is no LLM summarisation: it would be a second call per turn (ADR-0004) and burn quota. Slot carry-over ("cheaper ones?") is handled by the router's structured slots, not by history text, so trimming history never loses filters.
- **SQL route:** `<sql_result>` is a few lines plus up to 3 example cards, always under budget.

### 4b.5 Cost and latency budget

**Money: $0 by construction.** Only free tiers are used. The real budget is **quota and rate**:

| Resource | Limit | Per-turn usage | Capacity |
|---|---|---|---|
| OpenRouter `:free` | 50 req/day, 20 RPM, shared, failed calls counted [D] | 1 request (more on failover) | about 45 reviewer turns/day after the PLAN section 9 reserve; L0/L1/L2 hits cost 0 |
| Groq gpt-oss-120b | 30 RPM, 1K RPD, **8K TPM** [D] | about 1,900 prompt tokens + up to 500 output + reasoning ≈ 2.4 to 3K tokens | **2 to 3 turns per minute** before 429; failover then goes to qwen3.8-27b (another 8K TPM) |
| Groq qwen3.8-27b | 30 RPM, 1K RPD, 8K TPM [D] | same | 2 to 3 turns per minute |

At the expected reviewer volume (5 concurrent users, at most a few dozen turns), the chain plus the caches cover it. A burst of more than 6 uncached turns per minute reaches degradation level 3 (retrieval-only), which is the intended behaviour and is tested.

**Caching** (ADR-0006), cheapest first:
- **L0 warm cache:** about 25 grounded answers to the probe and demo queries, baked into the image.
- **L1 exact match:** a `TTLCache` keyed on `index_version + prompt_version + normalised query + filters.canonical_json() + numbers`.
- **L2 semantic:** Qdrant `qa_cache`, cosine ≥ τ, same `scope` (filters and numbers), standalone turns only. Tune τ on `cache_pairs.jsonl` to the lowest value with 0 false hits.
- **L3 query-embedding LRU** and **L4 retrieval-result LRU** skip ONNX and Qdrant work on repeats.
- Only grounded, untruncated answers are stored, enforced by `CachedAnswer`'s validator.

**Cheaper paths for sub-tasks:**
- Routing, intent, filters and slots: regex and dictionaries (0 LLM calls).
- Out-of-scope and below-floor queries: fixed refusal (0 calls).
- Aggregates: DuckDB templates, where the LLM only phrases the result.
- The LLM judge (reported, not gated) and project extraction run on Groq, never on the OpenRouter quota.
- The judge uses a different Groq model from the generator.

**Latency budget (2 vCPU, targets from PLAN section 8; P1 must be resolved for the retrieval line):**

| Stage | Budget | Notes |
|---|---|---|
| cache lookup (L0/L1) | < 5 ms | dict lookup + key build; SLO for a cache hit is p95 < 50 ms end-to-end |
| embed + sparse | 15 to 40 ms | one ONNX query; L3 hit = 0 |
| retrieve | < 20 ms target | **local-mode Qdrant measured about 200 ms filtered (P1)**; numpy ≈ 1 ms |
| rerank (20 pairs) | 30 to 80 ms | English only, flag-gated; drop it if it gains fewer than 3 points on dev |
| sql | < 20 ms | DuckDB template on `listings.parquet` |
| prompt + `start` event | < 5 ms | |
| LLM TTFT | provider-bound; 8 s deadline | reported per provider |
| grounding | < 5 ms | regex + dict lookups after the stream |
| retrieval-path total before TTFT | **p95 < 150 ms** | the SLO |

Every stage is timed by the service around the port call and reported in `StageTimings`, `Server-Timing` and `/stats`.
