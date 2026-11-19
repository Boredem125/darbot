"""Build the search index from processed data.

Usage:
    python -m organize.build_index
    python -m organize.build_index --out-dir .orchestration/tmp/index_next

Reads data/raw/wasalt/*.json.gz and data/processed/*.jsonl,
produces data/index/ (or the directory given by --out-dir) with
points_*.parquet, listings.parquet, entities.json and manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common.schemas import (
    EMBED_DIM_DEFAULT,
    EMBED_MODEL_DEFAULT,
    MANIFEST_SCHEMA_VERSION,
    MAX_SHARD_BYTES,
    SPARSE_MODEL_DEFAULT,
    Article,
    ChunkPayload,
    Listing,
    Manifest,
    ManifestCounts,
    ManifestFile,
    Project,
    Purpose,
    compute_index_version,
)
from organize.chunking import (
    build_token_counter,
    chunk_article,
    chunk_listing,
    chunk_project,
)
from organize.dedupe import dedupe
from organize.entities import build_entities
from organize.normalise import load_raw_listings

logger = logging.getLogger(__name__)

RAW_WASALT_DIR = Path("data/raw/wasalt")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")

MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB (T0 lowered from 3 GB)


def _check_disk_space() -> None:
    usage = shutil.disk_usage("E:/")
    free_gb = usage.free / (1024**3)
    logger.info("E: free space: %.1f GB", free_gb)
    if usage.free < MIN_FREE_BYTES:
        raise RuntimeError(f"Disk space below 2 GB: {free_gb:.1f} GB free")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_articles(processed_dir: Path) -> list[Article]:
    """Load Article records from processed JSONL."""
    articles: list[Article] = []
    path = processed_dir / "darglobal_articles.jsonl"
    if not path.exists():
        logger.info("No articles file at %s", path)
        return articles
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(Article.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Invalid article: %s", exc)
    logger.info("Loaded %d articles", len(articles))
    return articles


def _load_projects(processed_dir: Path) -> list[Project]:
    """Load Project records from processed JSONL."""
    projects: list[Project] = []
    path = processed_dir / "darglobal_projects.jsonl"
    if not path.exists():
        logger.info("No projects file at %s", path)
        return projects
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                projects.append(Project.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Invalid project: %s", exc)
    logger.info("Loaded %d projects", len(projects))
    return projects


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _write_listings_parquet(listings: list[Listing], out_dir: Path) -> ManifestFile:
    """Write listings.parquet with exact column types per manifest.md."""
    rows: list[dict] = []
    for li in listings:
        rows.append(li.model_dump(mode="json"))

    if not rows:
        # Write an empty parquet with the right schema
        schema = pa.schema([
            ("id", pa.string()),
            ("source", pa.string()),
            ("source_id", pa.string()),
            ("url", pa.string()),
            ("title", pa.string()),
            ("description", pa.string()),
            ("purpose", pa.string()),
            ("property_type", pa.string()),
            ("property_type_raw", pa.string()),
            ("city", pa.string()),
            ("district", pa.string()),
            ("price_sar", pa.int64()),
            ("annual_rent_sar", pa.int64()),
            ("bedrooms", pa.int16()),
            ("bathrooms", pa.int16()),
            ("area_sqm", pa.float64()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("amenities", pa.list_(pa.string())),
            ("rent_period", pa.string()),
            ("published_at", pa.timestamp("us", tz="UTC")),
            ("scraped_at", pa.timestamp("us", tz="UTC")),
        ])
        table = pa.table({}, schema=schema)
    else:
        # Build columns
        ids = [r["id"] for r in rows]
        sources = [r["source"] for r in rows]
        source_ids = [r["source_id"] for r in rows]
        urls = [r["url"] for r in rows]
        titles = [r["title"] for r in rows]
        descriptions = [r.get("description") for r in rows]
        purposes = [r["purpose"] for r in rows]
        property_types = [r["property_type"] for r in rows]
        property_type_raws = [r.get("property_type_raw") for r in rows]
        cities = [r["city"] for r in rows]
        districts = [r.get("district") for r in rows]
        price_sars = [r.get("price_sar") for r in rows]
        annual_rent_sars = [r.get("annual_rent_sar") for r in rows]
        bedrooms_list = [r.get("bedrooms") for r in rows]
        bathrooms_list = [r.get("bathrooms") for r in rows]
        area_sqms = [r.get("area_sqm") for r in rows]
        lats = [r.get("lat") for r in rows]
        lons = [r.get("lon") for r in rows]
        amenities_list = [r.get("amenities", []) for r in rows]
        rent_periods = [r.get("rent_period") for r in rows]
        published_ats = [r.get("published_at") for r in rows]
        scraped_ats = [r["scraped_at"] for r in rows]

        table = pa.table({
            "id": pa.array(ids, type=pa.string()),
            "source": pa.array(sources, type=pa.string()),
            "source_id": pa.array(source_ids, type=pa.string()),
            "url": pa.array(urls, type=pa.string()),
            "title": pa.array(titles, type=pa.string()),
            "description": pa.array(descriptions, type=pa.string()),
            "purpose": pa.array(purposes, type=pa.string()),
            "property_type": pa.array(property_types, type=pa.string()),
            "property_type_raw": pa.array(property_type_raws, type=pa.string()),
            "city": pa.array(cities, type=pa.string()),
            "district": pa.array(districts, type=pa.string()),
            "price_sar": pa.array(price_sars, type=pa.int64()),
            "annual_rent_sar": pa.array(annual_rent_sars, type=pa.int64()),
            "bedrooms": pa.array(bedrooms_list, type=pa.int16()),
            "bathrooms": pa.array(bathrooms_list, type=pa.int16()),
            "area_sqm": pa.array(area_sqms, type=pa.float64()),
            "lat": pa.array(lats, type=pa.float64()),
            "lon": pa.array(lons, type=pa.float64()),
            "amenities": pa.array(amenities_list, type=pa.list_(pa.string())),
            "rent_period": pa.array(rent_periods, type=pa.string()),
            "published_at": pa.array(
                [datetime.fromisoformat(v) if v else None for v in published_ats],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "scraped_at": pa.array(
                [datetime.fromisoformat(v) if v else None for v in scraped_ats],
                type=pa.timestamp("us", tz="UTC"),
            ),
        })

    path = out_dir / "listings.parquet"
    pq.write_table(
        table, path,
        write_statistics=False,
        store_schema=True,
    )
    size = path.stat().st_size
    assert size < MAX_SHARD_BYTES, f"listings.parquet too large: {size}"
    return ManifestFile(
        path="listings.parquet",
        sha256=_sha256_file(path),
        size_bytes=size,
        rows=len(rows),
    )


def _embed_texts(
    texts: list[str],
    embed_model: str,
    cache_dir: str,
) -> np.ndarray:
    """Embed texts using fastembed, L2-normalise, return float16."""
    from fastembed import TextEmbedding

    model = TextEmbedding(embed_model, cache_dir=cache_dir)
    embeddings = list(model.embed(texts, batch_size=64))
    arr = np.array(embeddings, dtype=np.float32)

    # L2-normalise
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    arr = arr / norms

    # Sample check before converting
    if len(arr) >= 100:
        sample_idx = np.random.choice(len(arr), 100, replace=False)
    else:
        sample_idx = np.arange(len(arr))

    # Cast to float16
    arr16 = arr.astype(np.float16)

    # Verify norm after float16 round trip
    for idx in sample_idx:
        v32 = arr16[idx].astype(np.float32)
        norm = float(np.linalg.norm(v32))
        assert abs(norm - 1.0) < 1e-3, f"Norm check failed: idx={idx}, norm={norm}"

    return arr16


def _sparse_embed_texts(
    texts: list[str],
    sparse_model: str,
    cache_dir: str,
) -> tuple[list[list[int]], list[list[float]], float]:
    """Compute bm25 sparse embeddings. Returns (indices, values, avg_len)."""
    from fastembed import SparseTextEmbedding

    # First pass: compute avg_len
    model_count = SparseTextEmbedding(sparse_model, cache_dir=cache_dir)
    total_tokens = model_count.token_count(texts)  # returns int (total)
    avg_len = round(total_tokens / max(len(texts), 1), 1)
    logger.info("BM25 avg_len: %.1f (total tokens %d, %d texts)", avg_len, total_tokens, len(texts))

    # Second pass: embed with correct avg_len
    model = SparseTextEmbedding(sparse_model, avg_len=avg_len, cache_dir=cache_dir)
    all_indices: list[list[int]] = []
    all_values: list[list[float]] = []

    for sparse_vec in model.embed(texts, batch_size=64):
        indices = sparse_vec.indices.tolist()
        values = sparse_vec.values.tolist()
        # Sort by index and deduplicate
        pairs = sorted(set(zip(indices, values)))
        if pairs:
            indices, values = zip(*pairs)
            all_indices.append(list(indices))
            all_values.append(list(values))
        else:
            all_indices.append([])
            all_values.append([])

    return all_indices, all_values, avg_len


def _write_points_parquet(
    payloads: list[ChunkPayload],
    dense: np.ndarray,
    sparse_indices: list[list[int]],
    sparse_values: list[list[float]],
    out_dir: Path,
    dim: int,
) -> list[ManifestFile]:
    """Write points_*.parquet shards, each under MAX_SHARD_BYTES."""
    n = len(payloads)
    if n == 0:
        raise ValueError("No points to write")

    # Estimate rows per shard
    # Dense: dim * 2 bytes, sparse: ~100 bytes avg, payload: ~500 bytes avg
    est_row_bytes = dim * 2 + 200 + 600
    rows_per_shard = max(1, (MAX_SHARD_BYTES - 10000) // est_row_bytes)

    files: list[ManifestFile] = []
    shard_idx = 0
    offset = 0

    while offset < n:
        end = min(offset + rows_per_shard, n)

        # Build arrays for this shard
        point_ids = [f"{payloads[i].doc_id}#{payloads[i].chunk_index}"
                     for i in range(offset, end)]
        doc_ids = [payloads[i].doc_id for i in range(offset, end)]
        chunk_indices = [payloads[i].chunk_index for i in range(offset, end)]

        # Dense vectors as fixed_size_binary
        dense_bytes = [dense[i].tobytes() for i in range(offset, end)]

        # Sparse
        s_indices = [sparse_indices[i] for i in range(offset, end)]
        s_values = [sparse_values[i] for i in range(offset, end)]

        # Payloads as JSON strings
        payload_jsons = [payloads[i].model_dump_json() for i in range(offset, end)]

        table = pa.table({
            "point_id": pa.array(point_ids, type=pa.string()),
            "doc_id": pa.array(doc_ids, type=pa.string()),
            "chunk_index": pa.array(chunk_indices, type=pa.int32()),
            "dense": pa.array(dense_bytes, type=pa.binary(dim * 2)),
            "sparse_indices": pa.array(
                [pa.array(idx, type=pa.uint32()) for idx in s_indices],
                type=pa.list_(pa.uint32()),
            ),
            "sparse_values": pa.array(
                [pa.array(val, type=pa.float32()) for val in s_values],
                type=pa.list_(pa.float32()),
            ),
            "payload": pa.array(payload_jsons, type=pa.string()),
        })

        filename = f"points_{shard_idx:03d}.parquet"
        path = out_dir / filename
        pq.write_table(
            table, path,
            write_statistics=False,
            store_schema=True,
        )

        size = path.stat().st_size
        if size >= MAX_SHARD_BYTES:
            # Reduce rows per shard and retry
            rows_per_shard = max(1, rows_per_shard // 2)
            path.unlink()
            continue

        files.append(ManifestFile(
            path=filename,
            sha256=_sha256_file(path),
            size_bytes=size,
            rows=end - offset,
        ))

        logger.info("  Shard %s: %d rows, %d bytes", filename, end - offset, size)
        shard_idx += 1
        offset = end

    return files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for build_index."""
    parser = argparse.ArgumentParser(description="Build the search index")
    parser.add_argument(
        "--out-dir", type=str, default="data/index",
        help="Output directory for the index (default: data/index)",
    )
    parser.add_argument(
        "--extra-listings", type=str, default=None,
        help="Path to a JSONL file of extra Listing records to append (e.g. canary listings)",
    )
    return parser.parse_args(argv)


def _load_extra_listings(path: Path) -> list[Listing]:
    """Load extra Listing records from a JSONL file."""
    extra: list[Listing] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            extra.append(Listing.model_validate_json(line))
    logger.info("Loaded %d extra listings from %s", len(extra), path)
    return extra


def build_index(out_dir: str | None = None, extra_listings_path: str | None = None) -> None:
    """Main build entry point.

    Parameters:
        out_dir: output directory for the index; defaults to data/index.
        extra_listings_path: optional JSONL file of extra Listing records to append.
    """
    t0 = time.monotonic()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Resolve output directory
    index_dir = Path(out_dir) if out_dir else INDEX_DIR
    logger.info("Output directory: %s", index_dir)

    _check_disk_space()

    # Environment
    cache_dir = os.environ.get(
        "FASTEMBED_CACHE_PATH",
        str(Path.home() / ".cache" / "fastembed"),
    )
    embed_model = os.environ.get("EMBED_MODEL", EMBED_MODEL_DEFAULT)
    sparse_model = os.environ.get("SPARSE_MODEL", SPARSE_MODEL_DEFAULT)
    dim = EMBED_DIM_DEFAULT

    logger.info("Embed model: %s (dim=%d)", embed_model, dim)
    logger.info("Sparse model: %s", sparse_model)
    logger.info("Cache dir: %s", cache_dir)

    # 1. Load and normalise Wasalt listings
    logger.info("--- Step 1: Loading raw listings ---")
    listings, _rejection_histogram = load_raw_listings(RAW_WASALT_DIR)

    # 2. Deduplicate
    logger.info("--- Step 2: Deduplication ---")
    listings, _merged_pairs = dedupe(listings)

    # 2b. Append extra listings if provided
    if extra_listings_path:
        extra = _load_extra_listings(Path(extra_listings_path))
        listings.extend(extra)
        logger.info("Total listings after extra: %d", len(listings))

    # 3. Load articles and projects
    logger.info("--- Step 3: Loading articles and projects ---")
    articles = _load_articles(PROCESSED_DIR)
    projects = _load_projects(PROCESSED_DIR)

    # 4. Build entities
    logger.info("--- Step 4: Building entities ---")
    entities = build_entities(listings, projects, articles)

    # 5. Chunk everything
    logger.info("--- Step 5: Chunking ---")
    token_count = build_token_counter()

    all_chunks: list[ChunkPayload] = []
    trimmed_count = 0
    total_listings = len(listings)

    for listing in listings:
        chunks = chunk_listing(listing, token_count)
        # Check if description was trimmed
        card_only = chunks[0].text
        if listing.description and listing.description not in card_only:
            trimmed_count += 1
        all_chunks.extend(chunks)

    for project in projects:
        chunks = chunk_project(project, token_count)
        all_chunks.extend(chunks)

    for article in articles:
        chunks = chunk_article(article, token_count)
        all_chunks.extend(chunks)

    # Sort by (doc_id, chunk_index) for determinism
    all_chunks.sort(key=lambda c: (c.doc_id, c.chunk_index))

    logger.info(
        "Total chunks: %d (listings=%d, projects=%d, articles=%d)",
        len(all_chunks),
        sum(1 for c in all_chunks if c.kind.value == "listing"),
        sum(1 for c in all_chunks if c.kind.value == "project"),
        sum(1 for c in all_chunks if c.kind.value == "article"),
    )
    logger.info(
        "Listings trimmed: %d / %d (%.1f%%)",
        trimmed_count, total_listings,
        trimmed_count / max(total_listings, 1) * 100,
    )

    # Verify 126-token gate
    texts = [c.text for c in all_chunks]
    over_limit = []
    token_lengths: list[int] = []
    for i, text in enumerate(texts):
        tc = token_count(text)
        token_lengths.append(tc)
        if tc > 126:
            over_limit.append((i, all_chunks[i].doc_id, all_chunks[i].chunk_index, tc))

    if over_limit:
        for idx, doc_id, ci, tc in over_limit[:10]:
            logger.error("OVER LIMIT: %s chunk %d = %d tokens", doc_id, ci, tc)
        raise RuntimeError(f"{len(over_limit)} chunks exceed 126 tokens")

    # Token length histogram
    if token_lengths:
        hist_bins = [0, 20, 40, 60, 80, 100, 126]
        import collections
        hist = collections.Counter()
        for tl in token_lengths:
            for b in hist_bins:
                if tl <= b:
                    hist[f"<={b}"] += 1
                    break
            else:
                hist[f">{hist_bins[-1]}"] += 1
        logger.info("Token length distribution: %s", dict(sorted(hist.items())))

    # 6. Embed
    logger.info("--- Step 6: Embedding ---")
    _check_disk_space()

    dense = _embed_texts(texts, embed_model, cache_dir)
    sparse_idx, sparse_val, avg_len = _sparse_embed_texts(texts, sparse_model, cache_dir)

    # 7. Write output
    logger.info("--- Step 7: Writing index ---")
    index_dir.mkdir(parents=True, exist_ok=True)

    # listings.parquet
    listings_file = _write_listings_parquet(listings, index_dir)

    # entities.json
    entities_path = index_dir / "entities.json"
    entities_json = entities.model_dump_json(indent=2)
    entities_path.write_text(entities_json, encoding="utf-8")
    entities_file = ManifestFile(
        path="entities.json",
        sha256=_sha256_file(entities_path),
        size_bytes=entities_path.stat().st_size,
    )

    # points_*.parquet
    points_files = _write_points_parquet(
        all_chunks, dense, sparse_idx, sparse_val, index_dir, dim,
    )

    # Manifest
    all_files = [listings_file, entities_file] + points_files
    index_version = compute_index_version(all_files)

    # Counts
    sale_count = sum(1 for li in listings if li.purpose == Purpose.sale)
    rent_count = sum(1 for li in listings if li.purpose == Purpose.rent)
    project_doc_ids = {c.doc_id for c in all_chunks if c.kind.value == "project"}
    article_doc_ids = {c.doc_id for c in all_chunks if c.kind.value == "article"}

    # scraped_at range
    all_scraped = [li.scraped_at for li in listings]
    for a in articles:
        all_scraped.append(a.scraped_at)
    for p in projects:
        all_scraped.append(p.scraped_at)

    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        index_version=index_version,
        built_at=datetime.now(UTC),
        embed_model=embed_model,
        dim=dim,
        sparse_model=sparse_model,
        distance="cosine",
        vector_dtype="float16",
        counts=ManifestCounts(
            listings_sale=sale_count,
            listings_rent=rent_count,
            projects=len(project_doc_ids),
            articles=len(article_doc_ids),
            points=len(all_chunks),
        ),
        scraped_at_min=min(all_scraped),
        scraped_at_max=max(all_scraped),
        git_sha=_git_sha(),
        prompt_version=None,
        files=all_files,
    )

    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    elapsed = time.monotonic() - t0
    logger.info("--- Build complete in %.1f s ---", elapsed)
    logger.info("Index version: %s", index_version)
    logger.info("Counts: sale=%d rent=%d projects=%d articles=%d points=%d",
                sale_count, rent_count, len(project_doc_ids),
                len(article_doc_ids), len(all_chunks))
    logger.info("avg_len: %.1f", avg_len)
    logger.info("Files: %s", [f.path for f in all_files])

    # Price distributions
    sale_prices = sorted(li.price_sar for li in listings
                         if li.purpose == Purpose.sale and li.price_sar)
    rent_prices = sorted(li.annual_rent_sar for li in listings
                         if li.purpose == Purpose.rent and li.annual_rent_sar)

    if sale_prices:
        p5 = sale_prices[int(len(sale_prices) * 0.05)]
        p50 = sale_prices[int(len(sale_prices) * 0.5)]
        p95 = sale_prices[int(len(sale_prices) * 0.95)]
        logger.info("Sale price_sar: p5=%d p50=%d p95=%d", p5, p50, p95)

    if rent_prices:
        p5 = rent_prices[int(len(rent_prices) * 0.05)]
        p50 = rent_prices[int(len(rent_prices) * 0.5)]
        p95 = rent_prices[int(len(rent_prices) * 0.95)]
        logger.info("Rent annual_rent_sar: p5=%d p50=%d p95=%d", p5, p50, p95)


if __name__ == "__main__":
    args = _parse_args()
    build_index(out_dir=args.out_dir, extra_listings_path=args.extra_listings)
