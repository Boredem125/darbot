# Retrieval spec (lookup and project_info routes)

Status: Wave 1b, 18 Nov 2026. Owner: dg-ml-lead. Implementers: dg-data-engineer (index side, WP-2-index), dg-backend-engineer (domain side in WP-2-domain; adapters in Wave 4).
Sources: PLAN §5, §7 and the appendix, ADR-0002, `common/schemas.py`, `app/domain/ports.py`, `docs/contracts/manifest.md`.

Every number below has an **initial value** and a **tuning rule**. Tuning happens on the dev split only, in Wave 3, and every change is recorded in `docs/tuning.md` with the measured before/after numbers. The test split is never used to choose a value.

## 0. Measured facts this spec rests on

All measurements were taken on the dev box (16 logical cores, shared with three other agents while running, so the numbers are noisy and probably pessimistic). Probe scripts and raw outputs are in `.orchestration/tmp/ml/`. Every SLO is re-measured on HF (2 vCPU) in Wave 4.

| # | Fact | Value | How measured |
|---|---|---|---|
| F1 | Qdrant **local mode** (`AsyncQdrantClient(":memory:")`), hybrid RRF query with 3 payload conditions, 6,000 points | p50 331 ms, **p95 653 ms** | `probe_qdrant.py`, 200 queries, synthetic 384-d vectors + 40-term sparse docs |
| F2 | Same at 12,000 points | p50 523 ms, **p95 1,002 ms** | same |
| F3 | Same, no filter, 6,000 points | p50 75 ms, p95 156 ms | same |
| F4 | Local mode stage split at 6,000 points: dense+filter p95 164 ms; sparse+filter p95 440 ms; hybrid with a `HasIdCondition` prefilter p95 384 ms | | `probe_stages.py`, 100 queries |
| F5 | In-process numpy hybrid (dense matmul + boolean mask + posting-list bm25 + RRF), same 6,000 points and filters | p50 1.17 ms, **p95 1.54 ms** | `probe_stages.py`, 100 queries |
| F6 | qdrant-client 1.20 local mode warns "Payload indexes have no effect in the local Qdrant" | | `create_payload_index` UserWarning |
| F7 | `FusionQuery(fusion=RRF)` in local mode uses **k = 2** (`DEFAULT_RANKING_CONSTANT_K = 2` in `qdrant_client/hybrid/fusion.py`); the RRF score of rank position p (0-based) is `1 / ((p+1)/w + k - 1)`. `RrfQuery(rrf=Rrf(k=..., weights=[...]))` exists in 1.20 | | source read + probe (single-list top score 0.5000 = 1/2) |
| F8 | Qdrant local IDF: `ln((N - df + 0.5) / (df + 0.5) + 1)`, computed over the whole collection | | `qdrant_client/local/local_collection.py:343` |
| F9 | Dense model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (fastembed 0.9.1): dim 384, **truncates input at 128 tokens** (`tokenizer.truncation.max_length == 128`; registry text "128 input tokens truncation"); adds 2 special tokens | | fastembed registry + tokenizer introspection |
| F10 | One 43-word English listing card (255 chars) = **71 tokens** with that tokenizer (1.65 tokens/word) | | `probe_models.py` |
| F11 | Dense `query_embed`, threads=1 | p50 19.7 ms, **p95 29.3 ms** | 55 queries, EN + AR |
| F12 | bm25 `query_embed` (pure Python) | p95 0.02 ms | same |
| F13 | Cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`, threads=1, ~83-token pairs, dynamic padding already on | 5 pairs p50 164–183 ms; 10 pairs p50 328–986 ms; 20 pairs p50 605–878 ms | `probe_models.py`, `probe_rerank.py` (two runs) |
| F14 | Rerank scores are raw logits (relevant 5.13, irrelevant −11.27), not probabilities | | `probe_models.py` |
| F15 | `Qdrant/bm25`: `avg_len` defaults to 256 and is a doc-side parameter (verified passthrough `SparseTextEmbedding("Qdrant/bm25", avg_len=70.0)`); query-side values are all 1.0; `token_count()` exists; one language per instance, default `english` | | introspection |
| F16 | Cosine (dense) top-1 over 6 probe cards: 8 in-scope queries min **0.52**; 7 out-of-scope queries max **0.15** | | `probe_cos.py` (hand-written probe set, not an eval split; vectors normalised explicitly) |
| F17 | fastembed 0.9.1 does **not** L2-normalise this model's output: doc norms 5.47 and 5.59, query norm 6.04 | | one-off check, confirms the AI-SPEC finding |

The AI-SPEC researcher measured Qdrant local mode independently and got the same picture: about 200 ms p50 at 4,000 filtered points, with the event loop blocked, against about 0.5 ms for numpy (`.orchestration/WP-1b-aispec.report.md`).

## 1. Decision R1: the runtime hybrid search is an in-process numpy engine, not Qdrant local mode

**Decision.** `app/adapters/` implements the `Retriever` port with an exact (brute-force) in-process engine over numpy arrays: `NumpyHybridRetriever`. The port, the manifest and the on-disk files are unchanged. Qdrant local mode is kept only as a **parity oracle in tests** and as an optional second adapter (`RETRIEVER_BACKEND=qdrant_local`, a should-have that is cut first).

**Why (measured).** The retrieval-path SLO is p95 < 150 ms for the whole path (PLAN §8). With filters, which most real-estate queries have, Qdrant local mode alone costs p95 653 ms at 6k points (F1) and 1,002 ms at 12k (F2). That is 4–7× the whole budget before embedding (F11) is added. Local mode evaluates payload conditions in Python per point and ignores payload indexes (F4, F6). The numpy engine does the same work at p95 1.54 ms (F5), which is about 420× faster, and is exact, so it has no recall loss.

**Rejected alternatives.**
- Keeping local mode and relaxing the SLO. The user ranked latency second only to architecture.
- A Qdrant server binary as a sidecar process in the container. It is fast, but it adds process supervision, startup upserts and image size on deploy day for a corpus of under 10k points.

**What this changes elsewhere** (raised in the report as change requests; nothing frozen is edited here):
- ADR-0002 needs an amendment: "qdrant-client Query API" becomes "the Query API semantics (prefetch, RRF, payload filters) implemented in numpy behind the Retriever port; qdrant-client is the test oracle". dg-lead-architect owns that ADR.
- `docs/contracts/manifest.md`, "Load procedure (app)": collection and payload-index creation become a detail of the optional Qdrant adapter only. The file formats are untouched.
- PLAN §5 L2 "Qdrant `qa_cache`" becomes in-process numpy too (see `caching.md` §4). The AI-SPEC researcher proposed keeping in-memory Qdrant for L2 only. I decide against it on the numbers. L2 sits on the cache-hit path, whose SLO is p95 < 50 ms. Scope-filtered dense search in local mode measured p95 164 ms at 6k points (F4); scaled linearly to the 2k-entry L2 cap that is about 55 ms, which alone exceeds the hit budget, and it runs synchronously on the event loop. A per-scope numpy matrix costs microseconds. Qdrant then has no runtime role at all, which keeps the image and ADR-0002 simpler.
- `KB_COLLECTION`, `QA_CACHE_COLLECTION`, `DENSE_VECTOR_NAME` and `SPARSE_VECTOR_NAME` stay in `common/schemas.py`, used by the oracle test and the optional adapter.

## 2. Index-side parameters (dg-data-engineer, `organize/build_index.py`)

### 2.1 Dense
- Model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, dim 384, cosine. fastembed returns **unnormalised** vectors (F17). The build L2-normalises in **float32** (`v / max(norm, 1e-12)`) and only then casts to float16 for storage. It asserts `|norm − 1| < 1e-3` on a sample after the float16 round trip. The query-side Embedder adapter normalises the same way. The `QueryEmbedding.dense` port contract says L2-normalised.
- Build: `TextEmbedding(model_name, threads=None)` (all cores at build time), `embed(texts, batch_size=64)`. **No prefixes** (registry: "Prefixes for queries/documents: not necessary").
- **Hard rule: every embedded text is ≤ 126 model tokens.** That is 128 (F9) minus 2 special tokens. The build counts tokens with the model's own tokenizer (§2.3) and **fails** if any chunk exceeds 126. Silent truncation would hide content from the embedder.

### 2.2 Sparse (bm25)
- Model `Qdrant/bm25`, `language="english"`, `k=1.2`, `b=0.75` (fastembed defaults).
- **`avg_len` = the measured corpus mean**, not the default 256 (F15). Compute `SparseTextEmbedding("Qdrant/bm25").token_count(all_chunk_texts) / n_chunks`, round to one decimal, pass it as `avg_len=` for the document embedding run, and log it in the build log. Listing chunks are about 71 tokens (F10), so the default of 256 would over-reward long chunks.
- Query side (runtime) uses the same model id and language. `avg_len` does not affect query vectors (all values 1.0, F15).
- Arabic text: the corpus is English (Wasalt `_en_sa` sitemaps, DarGlobal EN site). Arabic query tokens mostly match nothing in bm25, so the sparse list is empty and RRF reduces to dense alone, which is multilingual. This is intended. Entity aliases in Arabic are handled by the router's filters, not by bm25.

### 2.3 Chunking (decides WP-1a open issue 1b)

Tokens are counted with the dense model's tokenizer, loaded as `tokenizers.Tokenizer.from_file(<fastembed model dir>/tokenizer.json)`, followed by **`no_truncation()` and `no_padding()`**. The file ships with truncation at 128, and a counter that truncates silently caps every count at 128 (verified). Count with `encode(text, add_special_tokens=False)`.

| Doc kind | Chunk 0 | Chunks 1..n | Splitter | Cap |
|---|---|---|---|---|
| listing | Embed-mode card + description snippet, **one chunk only**. If it is over 126 tokens, trim the **description** at a word boundary until it fits. Never trim card fields. | none | none | 1 |
| project | Embed-mode card: name, brand, city, country, status, handover, unit types, bedrooms range, price-from, payment plan, amenities | `header + text_chunk` over the sanitised main page text | `SentenceSplitter(chunk_size=96, chunk_overlap=16, tokenizer=<counter above>)` | 12 chunks per project |
| article | `header + text_chunk` from the start of the body (no separate card chunk) | same | same | **8 chunks per article** (`MAX_ARTICLE_CHUNKS`) |

- **header** = `"{project name} ({city}, {country}). "` for projects, or `"{article title}. "` for articles. If the header is over 28 tokens, truncate it at a word boundary to 28. The budget is 2 special + ≤ 28 header + ≤ 96 chunk = ≤ 126 (F9).
- **chunk_overlap 16** is about 10 English words (F10: 1.65 tokens/word). A fact that straddles a boundary still appears whole in one chunk.
- **Article cap 8**: 363 articles × ≤ 8 is ≤ 2,904 points. With ~3,000 listings and ~40 projects × ≤ 12 chunks, the index is ≤ ~6,400 points. That is the size benchmarked in F5, and it keeps `points_*.parquet` to about 2 shards.
- `SentenceSplitter` was verified offline with an injected tokenizer: 1,050 tokens of text gave 15 chunks of ≤ 70 tokens. It splits at sentence granularity, so chunks are at most 96 tokens.
- **Project page text.** The main page text (trafilatura output, sanitised with the `Article.body` rules: no URLs, phones, emails, `<` or `>`) is the `Project.body` field (CC-3, `.orchestration/CONTRACT_CHANGES.md`). The earlier build-internal `data/processed/project_text.jsonl` proposal is superseded. Organize re-applies its sanitiser before chunking `body` into project chunks 1..n; when `body` is null the project has chunk 0 only.

**Tuning rules (Wave 3, dev, no LLM).**
- chunk_size/overlap: compare {96/16, 64/12} on the dev part of `projects.jsonl` plus the dev project questions in `handwritten.jsonl`. Metrics are entity hit@5 and support hit@5 (the gold answer string appears in one of the top-5 retrieved chunk texts). Switch to 64/12 only if support hit@5 improves by ≥ 1 question with no question lost. The sets are too small for a percentage rule; one question is 10 points.
- `MAX_ARTICLE_CHUNKS`: raise 8 to 16 only if ≥ 2 dev questions have their gold support only beyond chunk 8 (the build log lists `chunk_index` per doc).
- `avg_len`: not tuned. It is a measured statistic of the corpus.

## 3. Query-side algorithm (Retriever adapter, Wave 4; pure parts in domain, Wave 2)

### 3.1 Inputs
`QueryEmbedding` (dense normalised + bm25 query sparse vector), `SearchFilters`, `limit=20`.

### 3.2 Filter semantics (identical to Qdrant `must` conditions, `SearchFilters` mapping)
- `kinds` uses match-any on `kind`. Every other scalar filter is an exact match.
- Ranges are inclusive.
- `price_min/max_sar` target `annual_rent_sar` when `purpose == rent`, else `price_sar`.
- **A point whose target field is null fails that condition.** This matches Qdrant: a range on a missing field does not match. For example, a bedrooms filter excludes projects and articles, and a price filter excludes listings without a price.
- numpy implementation: one column array per filter field (int64 with a sentinel plus a null mask, or category codes for keywords). The mask is an AND of the per-condition masks. It is computed once per query and shared by both lists.

### 3.3 Candidate lists ("prefetch")
- Dense: `scores = M @ q` over float32 `M` (upcast from float16 once at load), masked points set to −inf, then top `PREFETCH_LIMIT = 20` by `argpartition`+`argsort`. Points with score −inf are never returned.
- Sparse: score(d) = Σ over query terms t of `idf(t) · v_d(t)`. Here `v_d` are the stored doc-side bm25 values and idf uses the formula of F8, computed once at load over the **whole** collection (not the filtered subset), as Qdrant does. Use posting lists `term -> (doc_idx int32[], value float32[])`, masked docs score 0, and keep the top 20 with score > 0.
- Ties on both lists break by ascending point index, so results are deterministic.

### 3.4 Fusion
- Weighted RRF with **Qdrant's exact formula** (F7): `rrf(d) = Σ_lists w_list / ((p+1) + w_list·(k−1))`, where p is the 0-based position in that list. This is the same function as `1/((p+1)/w + k − 1)`, written to avoid dividing by w=0. A list with w = 0 contributes 0.
- Initial **`RRF_K = 60`**, **`RRF_WEIGHTS = (dense 1.0, bm25 1.0)`**. k is always set explicitly, because Qdrant local mode would default to 2 (F7).
- Output: top `limit` points by fused score (ties broken by best dense rank), as `RetrievedHit(payload, score)`.
- **`top_dense_score`** = the cosine of the top-1 of the *filtered* dense list (§3.3), or `None` when that list is empty. It feeds the score floor (§3.6). The port's `RetrievedHit.score` is the fused score. **Dependency:** the queued contract change adds `RetrievalResult.top_dense_score: float | None = None` (raised by the AI-SPEC researcher; I endorse it as CR-ML-1 with exactly this definition). Wave 2 domain code takes `top_dense_score` as a plain function argument, so it does not wait on the change. The Wave 4 adapter fills the field once it is approved. If it is rejected, the fallback is an adapter-specific method `last_top_dense_score()` on `NumpyHybridRetriever`, called by the service right after `search()`.

**Tuning rule (Wave 3, dev, no LLM).**
- Grid `RRF_K ∈ {10, 60}` × `RRF_WEIGHTS ∈ {(1,1), (1,0.5), (0.5,1)}`, which is 6 configs.
- Primary metric: the mean of constraint precision@5 (filter queries) and entity hit@5 on `templated_dev` + dev handwritten. Tie-break on MRR.
- Move off (60, (1,1)) only if the primary metric gains ≥ 3 points. That is the PLAN §7 rule; one dev query is about 1.7 points at n ≈ 60, so 3 points is about two queries. A small grid with a gain threshold limits overfitting the dev set.

### 3.5 From hits to cards (domain, pure, `app/domain/retrieval_post.py`)
- **Doc collapse.** Group hits by `doc_id`, keep the best-scoring chunk per doc, and order docs by that score. Take the top `TOP_K_DOCS = 5`. That is the PLAN §6 context budget of 1,200 tokens for 5 cards; not tuned.
- **project_info with `project_id` set** (single doc): one card for the project. Its context is chunk 0 plus up to 4 more chunks of that project in score order, all cited as `[1]`.
- Cards are numbered `n = 1..k` in final order (the SSE `start` contract).

### 3.6 Score floor (refusal without an LLM call)
- Signal: **`top_dense_score`** (raw dense cosine of the filtered dense top-1, §3.4), not the RRF score. RRF scores depend only on rank: the top hit always scores about 2/(k+1) whatever the query, so a floor on RRF scores cannot separate in-scope from out-of-scope queries.
- Rule: refuse with the fixed refusal line, `answer_mode=refusal` and no LLM call, when
  (a) there are zero hits, or
  (b) `filters.is_empty()` and `top_dense_score < SCORE_FLOOR`.
  When the router extracted any filter and hits exist, the floor does not apply. A matched entity, type or price constraint is in-domain evidence, and this keeps false refusals down.
- Initial **`SCORE_FLOOR = 0.30`**. On the probe set (F16), out-of-scope top-1 had a max of 0.15 and in-scope a min of 0.52, so 0.30 sits in the gap, nearer the out-of-scope side to favour fewer false refusals.
- **Tuning rule (Wave 3, dev, real index, no LLM).**
  1. Compute top-1 dense cosine for every dev in-scope query that has empty filters, and every dev out-of-scope query that reaches the lookup route.
  2. If the two sets are separable, set the floor to the midpoint of (max OOS, min in-scope), rounded down to 0.01.
  3. Otherwise pick the highest floor with dev false refusals ≤ 2% of in-scope queries (stricter than the 5% test gate, leaving margin), and report the OOS leak count it leaves.
  4. Re-run after any index rebuild that changes `embed_model`.
  The real index has thousands of cards, so the max over cards rises for OOS queries too. That is why the probe value is only a prior.

### 3.7 Rerank (decides PLAN §7 "rerank keep/drop")
- **Initial: `RERANK_ENABLED=0`.**
- Measured cost (F13): 5 pairs p50 164–183 ms at threads=1. That alone exceeds the 150 ms retrieval-path budget. With F11 (29 ms) added, even the smallest useful rerank breaks the SLO.
- Rerank only reorders candidates that already satisfy the hard filters, so its likely gain is in MRR, not in constraint precision.
- **Tuning rule (Wave 3, dev, English only).** Measure rerank of the fused top 10, keeping the top 5, against no rerank on the §3.4 primary metric and MRR. Record the gain in `docs/tuning.md` either way. Enable it only if **both** hold:
  1. Gain ≥ 3 points (PLAN §7).
  2. The Wave 4 HF bench shows retrieval-path p95 < 150 ms with rerank on.

  If (1) holds and (2) fails, the gain is reported as "quality foregone for latency". Arabic is always skipped (English-only model).
- If enabled: `TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2", threads=1)`, rerank candidates `RERANK_CANDIDATES = 10`, documents = chunk `text`, and the logit score replaces `score` (F14; never compare it with RRF scores or with the floor).

### 3.8 Sorting and superlatives (router contract, see WP-2-domain)
The retriever ignores `filters.sort` (port docstring). Queries whose meaning is an order go to the SQL route, where the order is exact:
- "cheapest" goes to `aggregate/min_price` and "most expensive" to `max_price`, with `example_limit=3`.
- "newest/latest" goes to `aggregate/count` with `sort=newest` and `example_limit=5`.

## 4. Parity test (dg-qa-engineer, Wave 4; the test generates its own synthetic fixture)
`tests/test_retriever_parity.py` uses a fixture of 300 synthetic points (random normalised dense vectors + random sparse vectors + payloads covering every filter field, including nulls) and 30 queries with mixed filters. For every query it asserts:
- The top-20 ids of `NumpyHybridRetriever` equal those of Qdrant local mode with `RrfQuery(rrf=Rrf(k=60, weights=[1,1]))`.
- The fused scores agree within 1e-6.

This proves the numpy engine implements the Qdrant semantics the ADR names. It needs no models and no network.

## 5. Parameter table (initial values; Wave 3 records changes in `docs/tuning.md`)

| Name | Initial | Tuning rule | Evidence |
|---|---|---|---|
| `PREFETCH_LIMIT` | 20 per list | fixed (PLAN §5) | |
| `RRF_K` | 60 (Wave 3: kept, best grid gain 0.18 pts < 3) | §3.4 grid, ≥ 3-pt gain | F7; `docs/tuning.md` §1 |
| `RRF_WEIGHTS` | (1.0, 1.0) (Wave 3: kept) | §3.4 grid | `docs/tuning.md` §1 |
| `TOP_K_DOCS` | 5 | fixed (PLAN §6 budget) | |
| `SCORE_FLOOR` (`top_dense_score`; empty filters only) | **0.52** (Wave 3, was 0.30) | §3.6 | F16; `docs/tuning.md` §2 |
| `RERANK_ENABLED` | 0 (Wave 3: kept off, primary gain 0 pts) | §3.7 (gain ≥ 3 pts **and** HF p95 < 150 ms) | F13; `docs/tuning.md` §3 |
| `RERANK_CANDIDATES` | 10 | only if rerank is enabled | F13 |
| chunk_size / overlap (projects, articles) | 96 / 16 model tokens | §2.3 | F9, F10 |
| header cap | 28 tokens | fixed by the 126-token budget | F9 |
| max embedded tokens | 126 (build fails above) | fixed | F9 |
| `MAX_ARTICLE_CHUNKS` | 8 | §2.3 | F5 sizing |
| project chunk cap | 12 | fixed | |
| bm25 `avg_len` | measured corpus mean | not tuned | F15 |
| bm25 k, b | 1.2, 0.75 | not tuned (no budget) | |

## 6. Unverified items (marked so nobody builds on them blindly)
- U1. Latencies on HF 2 vCPU. All numbers above come from the shared dev box. Wave 4 bench re-measures them.
- U2. That `fastembed` multi-core build embedding (`parallel=`) works on Windows. The build may use `parallel=None`. It is only slower.
