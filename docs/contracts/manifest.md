# Index artifact contract (`data/index/`)

Status: frozen in Wave 1a (18 Nov 2026); amended and re-frozen in WP-2-contracts (18 Nov 2026,
see `.orchestration/CONTRACT_CHANGES.md` CC-3, CC-5, CC-6). Owner: dg-lead-architect. Source of truth: the
`Manifest`, `ManifestFile`, `ManifestCounts`, `ChunkPayload`, `Listing`, `Entities` and
`WarmCacheRecord` models in `common/schemas.py`. Changes need a note in
`.orchestration/CONTRACT_CHANGES.md` plus an orchestrator broadcast.

The offline half (`organize/build_index.py`) and the online half (`app/`) share **files
only** (ADR-0003). Neither half imports the other. This document lists everything one half
writes and the other reads.

## Directory layout

```
data/index/
├─ manifest.json          written last by organize/build_index.py; prompt_version later set by scripts/warm_cache.py
├─ points_000.parquet     Qdrant points, sharded, each file < 10,000,000 bytes
├─ points_001.parquet     ...
├─ listings.parquet       one row per Listing, read by DuckDB
├─ entities.json          Entities dictionary (router + grounding)
└─ warm_cache.jsonl       L0 answers (optional; WarmCacheRecord per line)
```

All file names are flat, with no subdirectories. Every file except `manifest.json` and
`warm_cache.jsonl` must be listed in `manifest.files`. `warm_cache.jsonl` is deliberately
not listed: it is written after the index is built, and listing it would change
`index_version` and every cache key. It gets no size or sha check; it is guarded by
rule 3 and by the per-record checks in rule 5.

## `manifest.json` (`Manifest`, `schema_version` 1)

Every key is **required**. `prompt_version` must be present, but its value may be null.

| Field | Type | Rule |
|---|---|---|
| `schema_version` | `1` | Bump on any breaking change to this contract. |
| `index_version` | 16 lowercase hex chars | `compute_index_version(files)`: the first 16 hex chars of sha256 over the lines `"{path}\t{sha256}\n"`, sorted by `path`. The model rejects a mismatch. |
| `built_at` | ISO 8601 datetime with an offset | UTC. |
| `embed_model` | string | Dense model id, the exact fastembed id. Default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`EMBED_MODEL_DEFAULT`); the builder takes it from env `EMBED_MODEL`. |
| `dim` | int > 0 | 384 for the model above (`EMBED_DIM_DEFAULT`). |
| `sparse_model` | string | Default `Qdrant/bm25` (`SPARSE_MODEL_DEFAULT`); the builder takes it from env `SPARSE_MODEL`. |
| `distance` | `"cosine"` | |
| `vector_dtype` | `"float16"` | How dense vectors are stored on disk. They are upcast to float32 at load time. |
| `counts.listings_sale` | int ≥ 0 | Rows with `purpose == sale` in `listings.parquet`. |
| `counts.listings_rent` | int ≥ 0 | Rows with `purpose == rent`. |
| `counts.projects` | int ≥ 0 | Distinct project `doc_id`s in the points. |
| `counts.articles` | int ≥ 0 | Distinct article `doc_id`s in the points. |
| `counts.points` | int ≥ the sum of the four counts above | Total rows across `points_*.parquet`. |
| `scraped_at_min` / `scraped_at_max` | datetime, min ≤ max | Range over all documents. The UI and README say "data as of `scraped_at_max`". |
| `git_sha` | 7–40 hex chars, optionally ending in `-dirty`, or the literal `unknown` | Commit of the code that built the index. |
| `prompt_version` | `"<stem>@<8 hex>"` or null | The prompt version the warm cache was generated with. It is null until `scripts/warm_cache.py` runs. That script rewrites only this key, which does not change `index_version`. |
| `files` | `ManifestFile[]`, at least 1 | See below. Must include `listings.parquet`, `entities.json` and at least one `points_*.parquet`. Paths must be unique. |

`ManifestFile`: `path` (flat file name), `sha256` (64 lowercase hex chars), `size_bytes`
(int, `0 ≤ size_bytes < 10,000,000`), and `rows` (parquet row count, or null for JSON).

`prompt_version` is computed with `compute_prompt_version(stem, file_bytes)`. For example,
`system_v1@1a2b3c4d` is the first 8 hex chars of the sha256 of `app/prompts/system_v1.txt`.
Any edit to the prompt changes the version.

## Fail-fast rules (app startup, before `/ready` turns green)

The app loads `manifest.json`, validates it with `Manifest.model_validate_json`, and then
calls:

```python
manifest.assert_compatible(
    embed_model=settings.embed_model, dim=embedder.dim, sparse_model=settings.sparse_model,
    prompt_version=current_prompt_version, warm_cache_present=warm_cache_path.exists(),
)
```

### Model-id env vars (one name per model, used by every producer and consumer)

| Env var | Default (`common.schemas`) | Read by | Written to / checked against |
|---|---|---|---|
| `EMBED_MODEL` | `EMBED_MODEL_DEFAULT` = `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `organize/build_index.py`, `app/config.py` (`settings.embed_model`), Dockerfile `ARG`/`ENV` | `manifest.embed_model`; rule 2 below |
| `SPARSE_MODEL` | `SPARSE_MODEL_DEFAULT` = `Qdrant/bm25` | `organize/build_index.py`, `app/config.py` (`settings.sparse_model`), Dockerfile `ARG`/`ENV` | `manifest.sparse_model`; rule 2 below |
| `RERANK_MODEL` | `RERANK_MODEL_DEFAULT` = `Xenova/ms-marco-MiniLM-L-6-v2` | `app/config.py` (`settings.rerank_model`), Dockerfile `ARG`/`ENV` | Not in the manifest: rerank is query-side only and does not change the index |

`MODEL_ENV_DEFAULTS` in `common/schemas.py` maps each env name to its default. Code reads
`os.environ.get(name, MODEL_ENV_DEFAULTS[name])`; `common` itself does no env reads. The
Dockerfile `ARG` defaults must equal these values. `dim` is not configured: the app takes it
from the loaded dense model (`embedder.dim`).

Startup **aborts** (the process exits non-zero, and `/ready` never turns green) when:
1. `manifest.json` is missing, is not valid JSON, or fails `Manifest` validation. That covers a missing key, a wrong type, a bad hash format, an `index_version` that does not match `files`, a file ≥ 10 MB, or a required file missing from the list.
2. The configured `embed_model`, `dim` or `sparse_model` differs from the manifest. `ManifestMismatchError` lists every reason.
3. `warm_cache.jsonl` exists and `manifest.prompt_version` differs from the current prompt version. This includes the case where `manifest.prompt_version` is null.
4. A file listed in `files` is missing on disk, or its size differs. The loader (`app/adapters/index_loader.py`) checks existence and size only; sha256 is not recomputed at startup (as built, 20 Nov 2026). The model has no I/O.

Not an abort (as built, 20 Nov 2026; `app/adapters/caches.py`):

5. A `WarmCacheRecord` that fails validation, or whose `entry.index_version` or `entry.prompt_version` differs from the running versions, is **skipped** with a log line and counted in the L0 skipped total. The rest of L0 loads. Earlier text made this an abort; the code's skip behaviour is now the contract.

If `warm_cache.jsonl` is absent, startup proceeds with an empty L0. That is not an error.

Tests (dg-qa-engineer, `tests/test_manifest_failfast.py`) cover each numbered rule with a
temporary `data/index` built in `tmp_path`.

## `points_*.parquet` (Qdrant points)

There is one row per chunk. The organize step writes shards of at most `N` rows, choosing
`N` so that each file stays under 10,000,000 bytes. Shards are numbered from `000`.

| Column | Arrow type | Notes |
|---|---|---|
| `point_id` | string | `point_id(doc_id, chunk_index)`: uuid5 with the fixed `POINT_ID_NAMESPACE`. |
| `doc_id` | string | `wasalt:<id>`, `darglobal:project:<slug>`, or `darglobal:<press\|insight\|blog>:<slug>` |
| `chunk_index` | int32 | 0-based. Listings always have exactly one chunk, index 0. |
| `dense` | fixed_size_binary(`dim * 2`) | Little-endian float16 bytes of the L2-normalised dense vector (`np.float16(...).tobytes()`). |
| `sparse_indices` | list<uint32> | bm25 document-side indices. |
| `sparse_values` | list<float32> | Same length as `sparse_indices`. |
| `payload` | string | `ChunkPayload.model_dump_json()`. Validated with `model_validate_json` at load; feeds the filter columns and the cards. |

Load procedure (app, runtime numpy engine; ADR-0002 amendment 1): read every
`points_*.parquet` listed in `manifest.files`, in path order (row order within a shard is
kept, so point indexes are deterministic), validate each `payload` with
`ChunkPayload.model_validate_json`, and build, once at startup:
- a float32 dense matrix `[points, dim]`, upcast from the float16 bytes (already
  L2-normalised; the loader asserts `|norm - 1| < 1e-3` on a sample);
- bm25 posting lists `term -> (point_idx int32[], value float32[])` from the sparse
  columns, plus per-term IDF `ln((N - df + 0.5) / (df + 0.5) + 1)` over the **whole**
  collection (Qdrant's formula, `docs/specs/retrieval.md` §3.3);
- one column array per `SearchFilters` target field (`kind`, `source`, `purpose`,
  `property_type`, `city`, `district`, `country`, `doc_id`, `bedrooms`, `price_sar`,
  `annual_rent_sar`, `area_sqm`) with a null mask, for the boolean filter mask.

Runtime search is numpy only. **No Qdrant collection is created and there are no payload
indexes at runtime.** Payload indexes do nothing in Qdrant local mode anyway
(qdrant-client 1.20 warns "Payload indexes have no effect in the local Qdrant").

The Qdrant load (collection `kb` = `KB_COLLECTION`, named dense vector `dense` with size
`dim` and cosine, named sparse vector `bm25` with the IDF modifier, upsert of every row) is
used only by the parity test oracle (`tests/test_retriever_parity.py`) and by the optional
`RETRIEVER_BACKEND=qdrant_local` adapter. The file formats are the same for both.

How chunks are built (`ChunkPayload.text`, embed mode, with no `[n]`, URL or id):
- **listing**: the card, rendered in embed mode, plus the description snippet. One chunk.
- **project**: chunk 0 is the card; chunks 1..n are `Project.body` (sanitised page text from the processed `Project` record), split by the llama-index sentence splitter. Each of those chunks starts with a header that repeats the project name. A project with `body == null` has chunk 0 only. Chunk size and caps are in `docs/specs/retrieval.md` §2.3.
- **article**: the title plus the body, split the same way.

## `listings.parquet` (DuckDB)

There is one row per `Listing`, and the column names are exactly the `Listing` field names:

| Columns | Arrow type |
|---|---|
| `id`, `source`, `source_id`, `url`, `title`, `description`, `purpose`, `property_type`, `property_type_raw`, `city`, `district`, `rent_period` | string (enums as their string values) |
| `price_sar`, `annual_rent_sar` | int64 |
| `bedrooms`, `bathrooms` | int16 |
| `area_sqm`, `lat`, `lon` | float64 |
| `amenities` | list<string> |
| `published_at`, `scraped_at` | timestamp[us, tz=UTC] |

Nulls are allowed wherever the `Listing` field is optional. Rows are deduplicated (no two
rows share an `id`). Near-duplicates are removed by `organize/dedupe.py` before the write.

## `entities.json` (`Entities`)

```json
{"schema_version": 1, "entities": [
  {"kind": "city", "canonical": "Riyadh", "aliases": ["riyadh", "الرياض", "ar riyadh"], "parent": "Saudi Arabia", "doc_id": null},
  {"kind": "district", "canonical": "Al Narjis", "aliases": ["narjis", "النرجس"], "parent": "Riyadh", "doc_id": null},
  {"kind": "project", "canonical": "Trump Tower Jeddah", "aliases": ["برج ترامب جدة"], "parent": "Jeddah", "doc_id": "darglobal:project:trump-tower-jeddah"},
  {"kind": "brand", "canonical": "Trump", "aliases": [], "parent": null, "doc_id": null}
]}
```

- `canonical` is the exact string stored in payloads and in `listings.parquet`, and used in `SearchFilters`.
- `(kind, canonical, parent)` is unique. The same district name can appear under two cities.
- Aliases are matched case-insensitively after normalisation (`common/text.py`). The canonical name is an implicit alias.

## `warm_cache.jsonl` (`WarmCacheRecord`, L0)

There is one JSON object per line: `{"key": str, "query": str, "entry": CachedAnswer}`.
`key` must come from the same key function that L1 uses. `entry` must be grounded (the
model enforces this) and carry the current `prompt_version` and `index_version`. The writer
must skip any answer whose LLM finish_reason was not `stop`, because truncated answers are
never cached. `scripts/warm_cache.py` writes this file and then sets
`manifest.prompt_version`.

## Size budget

Every listed file is under 10,000,000 bytes, which is the HF limit for files without LFS.
At 384-d float16, a dense vector takes 768 bytes. One shard of about 8,000 points, payload
included, fits comfortably. `organize` asserts the size of each file after writing it, and
the `Manifest` model rejects any `size_bytes` ≥ 10,000,000.
