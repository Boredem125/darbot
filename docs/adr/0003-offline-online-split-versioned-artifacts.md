# ADR-0003: Offline/online split through versioned file artifacts

- Status: Accepted; amended 18 Nov 2026 (amendment 1: local-mode note, see the end)
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0001, ADR-0006, ADR-0008, `docs/contracts/manifest.md`, `common/schemas.py` (`Manifest`, `ChunkPayload`, `Listing`, `Entities`, `WarmCacheRecord`)

## Context

Scraping is slow (Wasalt at 1 request/s, DarGlobal behind a JS challenge) and needs a 2 GB Playwright image. Index building needs `llama-index-core` and, for DarGlobal projects, Groq calls. None of that belongs in a request path or in the deployed image.

The deployed Space has ephemeral disk, sleeps after about 48 hours idle, and restarts cold. It must come up with the same data every time, without network calls to build anything. The data is a snapshot anyway: listings change daily, and the demo answers "as of" the scrape date.

Two hosting facts shape the files:
- HF rejects git-pushed files of 10 MB or more without LFS/Xet. Deploy uses `huggingface_hub.upload_folder` (PLAN section 10), but small files keep both paths open and keep the repo browsable.
- At 384 dimensions, a float32 vector is 1,536 bytes. A few thousand listings plus project and article chunks fit easily, but only if the files are sharded.

## Decision

1. **The two halves share files only.** `collect` writes `data/raw/`, `organize` reads it and writes `data/processed/` and `data/index/`, and `app` reads `data/index/` at startup. `app` never imports `collect` or `organize` (R10), and they never import `app` (R11). The file formats are frozen in `docs/contracts/manifest.md`.

2. **The index is built offline and shipped in the image.** At startup the app loads `points_*.parquet` into in-process numpy arrays (dense matrix, bm25 posting lists, filter columns; ADR-0002 amendment 1 and the load procedure in `manifest.md`) and opens `listings.parquet` with DuckDB. Nothing is fetched at runtime. Qdrant is not loaded at runtime; the same files feed the Qdrant parity oracle in tests.

3. **Every artifact is versioned by content.** `manifest.json` lists each file with its sha256 and size. `index_version` is the first 16 hex characters of a sha256 over the sorted `path<TAB>sha256` lines (`compute_index_version`), and the `Manifest` model rejects a mismatch. The same `index_version` appears in every `end` event and every cache key (ADR-0006).

4. **Storage format.**
   - Dense vectors are L2-normalised, stored as little-endian **float16** bytes in a `fixed_size_binary(dim * 2)` column, and upcast to float32 at load. This halves the size. Raw bytes avoid depending on parquet's float16 logical type support across pyarrow versions.
   - Sparse vectors are stored as `list<uint32>` indices and `list<float32>` values.
   - The payload is a JSON string column holding `ChunkPayload.model_dump_json()`.
   - Points are sharded into `points_000.parquet`, `points_001.parquet`, and so on. **Every listed file is under 10,000,000 bytes.** `organize` asserts this after each write, and `ManifestFile` rejects anything larger.

5. **Fail fast at startup.** The app validates the manifest and calls `Manifest.assert_compatible(...)`, and the loader checks hashes and sizes. Startup aborts, and `/ready` never turns green, when the manifest is invalid, when the configured embed model, dim or sparse model differ from the manifest, when a listed file is missing or altered, or when the warm cache does not match (point 6). The five numbered rules are in `manifest.md`.

6. **`prompt_version` lifecycle.**
   - Format: `"<stem>@<first 8 hex of sha256(file bytes)>"`, computed by `compute_prompt_version`, for example `system_v1@1a2b3c4d`. Any byte change to the prompt file changes the version, including whitespace.
   - `organize/build_index.py` writes `manifest.prompt_version = null`. The index is built in Wave 2 and the prompt is frozen in Wave 3, so the build cannot know it.
   - `scripts/warm_cache.py` (Wave 4) generates grounded answers with the frozen prompt, writes `warm_cache.jsonl`, and then rewrites **only** the `prompt_version` key in `manifest.json`. `warm_cache.jsonl` and `manifest.json` are excluded from `files`, so `index_version` does not change.
   - At startup, if `warm_cache.jsonl` exists, `manifest.prompt_version` must equal the version of the prompt the app loads, and every record's `entry.index_version` must equal `manifest.index_version`. Otherwise startup aborts. If `warm_cache.jsonl` is absent, L0 is empty and startup proceeds.
   - So a prompt edit has two legal outcomes: rebuild the warm cache, or delete `warm_cache.jsonl`. A stale warm cache can never be served.
   - L1 and L2 entries carry `prompt_version` and `index_version` in both the key and the `CachedAnswer`, so a redeploy with a new prompt or index cannot hit an old entry.

7. **Freshness is shown, not hidden.** Every card carries `scraped_at`, and the UI shows "as of `scraped_at`" on listing cards. The README states the data range from `manifest.scraped_at_min` and `scraped_at_max`.

## Alternatives rejected

- **Scraping or indexing at startup or on a schedule inside the Space.** Startup would take minutes and depend on third-party sites being up. The Playwright browser would bloat the runtime image. The Space sleeps, so a schedule inside it is unreliable anyway.
- **A hosted vector DB (Qdrant Cloud) or a Qdrant server container.** A network hop on every query, a second service to keep alive, and a credential to manage. For a few thousand points, in-process search is faster.
- **On-disk local Qdrant storage (`path=...`) committed to the repo.** An opaque binary format tied to the client version, harder to shard under 10 MB, and not inspectable in review.
- **float32 vectors.** Twice the size for no measurable retrieval gain at this scale. Rank changes from float16 rounding are far below the RRF noise.
- **A separate data repo or dataset download at startup.** Adds a network dependency to cold start, and one more token to manage.

## Consequences

- The deployed image is self-contained and reproducible. The same image always answers from the same `index_version`.
- `index_version` and `prompt_version` give cache invalidation for free (ADR-0006).
- Startup cost grows with the point count, because every point is decoded into memory. At the planned scale (about 3k listings plus project and article chunks) this takes seconds, and it is measured as part of `/ready`.
- The in-process numpy engine does exact search, linear in the point count. That is fine at this scale. The retrieval p95 < 150 ms SLO is the guard. Above roughly 100k points this decision would need revisiting.
- Any data refresh is a rebuild plus a redeploy, run from the laptop or CI. There is no hot reload.

## Trade-off accepted

Fresher data needs a rebuild and a redeploy. Listings are shown "as of `scraped_at`", and the README states the scrape window.

## Amendment 1 (18 Nov 2026, WP-2-contracts): local mode and payload indexes

- **Context.** Wave 1b measured Qdrant local mode at p95 653 ms for a filtered hybrid query at 6,000 points. Local mode evaluates payload conditions in Python and ignores payload indexes (qdrant-client 1.20 warns "Payload indexes have no effect in the local Qdrant"). ADR-0002 amendment 1 therefore moves runtime search and the L2 cache to numpy.
- **Decision.** Point 2 above now reads "numpy arrays" instead of "in-memory Qdrant". The artifact contract itself does not change: the file names, Arrow types, float16 dense bytes, sparse columns, `ChunkPayload` JSON, manifest and fail-fast rules are all as before. The "create the collection and payload indexes" step in `manifest.md` applies only to the Qdrant parity oracle and the optional `RETRIEVER_BACKEND=qdrant_local` adapter. Payload indexes are a no-op there in local mode, and they are kept only so that the same code works against a Qdrant server.
- **Alternatives rejected.** Changing the on-disk format to a numpy-native one (`.npy` matrices or a pickled index). That would break the Qdrant-shaped files that the oracle and a future server migration rely on. It would also make the artifacts less inspectable, and the load-time saving at this scale is negligible.
- **Consequences.** `index_version` and every cache key are unaffected. A future move to a Qdrant server needs only an adapter that upserts the same files.
- **Trade-off accepted.** The runtime decodes a Qdrant-shaped format into numpy structures at startup instead of loading a native format. That costs a little startup time and keeps one artifact format for runtime, tests and a later migration.
