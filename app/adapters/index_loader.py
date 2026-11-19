"""Load the index from data/index/ into an immutable LoadedIndex.

Reads manifest.json, points_*.parquet, listings.parquet, entities.json.
Fail-fast on model mismatch (CC-5, manifest.md). Float16->float32 upcast,
bm25 postings with whole-collection IDF, filter columns with null masks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common.schemas import (
    ChunkPayload,
    Entities,
    Manifest,
)

logger = logging.getLogger(__name__)

# Sentinel for null numeric values in filter arrays
_NULL_INT = np.iinfo(np.int64).min
_NULL_FLOAT = np.finfo(np.float64).min


@dataclass(frozen=True, slots=True)
class LoadedIndex:
    """Immutable in-memory index loaded from data/index/."""

    manifest: Manifest
    n_points: int
    dim: int
    dense: np.ndarray  # shape (n_points, dim), float32, L2-normalised rows
    payloads: tuple[ChunkPayload, ...]
    # bm25 posting lists: term_id -> (point_indices int32[], values float32[])
    postings: dict[int, tuple[np.ndarray, np.ndarray]]
    idf: dict[int, float]  # term_id -> idf value
    # Filter columns: field_name -> array (ints or category codes)
    filter_cols: dict[str, np.ndarray]
    null_masks: dict[str, np.ndarray]  # field_name -> bool array (True = null)
    # Category maps for string fields: field_name -> {value: code}
    category_maps: dict[str, dict[str, int]]
    entities: Entities
    listings_path: Path  # for DuckDB


def load_index(
    index_dir: Path,
    *,
    embed_model: str,
    dim: int,
    sparse_model: str,
    prompt_version: str | None = None,
    warm_cache_present: bool = False,
) -> LoadedIndex:
    """Load and validate the index. Raises on fail-fast violations."""
    # 1. Load manifest
    manifest_path = index_dir / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = Manifest.model_validate_json(manifest_text)

    # 2. Fail-fast: model compatibility
    manifest.assert_compatible(
        embed_model=embed_model,
        dim=dim,
        sparse_model=sparse_model,
        prompt_version=prompt_version or "unused@00000000",
        warm_cache_present=warm_cache_present,
    )

    # 3. Check all files exist and have correct sizes
    for mf in manifest.files:
        fp = index_dir / mf.path
        if not fp.exists():
            raise FileNotFoundError(f"Index file missing: {fp}")
        actual_size = fp.stat().st_size
        if actual_size != mf.size_bytes:
            raise ValueError(
                f"{mf.path}: size {actual_size} != manifest {mf.size_bytes}"
            )

    # 4. Load points parquets in path order
    point_files = sorted(
        mf.path for mf in manifest.files
        if mf.path.startswith("points_") and mf.path.endswith(".parquet")
    )

    all_payloads: list[ChunkPayload] = []
    all_dense_rows: list[np.ndarray] = []
    all_sparse_indices: list[list[int]] = []
    all_sparse_values: list[list[float]] = []

    for pf in point_files:
        table = pq.read_table(index_dir / pf)
        for i in range(table.num_rows):
            # Parse payload
            payload_json = table.column("payload")[i].as_py()
            payload = ChunkPayload.model_validate_json(payload_json)
            all_payloads.append(payload)

            # Dense: fixed_size_binary -> float16 bytes -> float32
            dense_bytes = table.column("dense")[i].as_py()
            vec16 = np.frombuffer(dense_bytes, dtype=np.float16)
            vec32 = vec16.astype(np.float32)
            all_dense_rows.append(vec32)

            # Sparse
            sp_idx = table.column("sparse_indices")[i].as_py()
            sp_val = table.column("sparse_values")[i].as_py()
            all_sparse_indices.append(sp_idx)
            all_sparse_values.append(sp_val)

    n_points = len(all_payloads)
    logger.info("Loaded %d points from %d shard(s)", n_points, len(point_files))

    # 5. Build dense matrix and check norms
    dense = np.stack(all_dense_rows, axis=0)  # (n_points, dim)
    assert dense.shape == (n_points, dim), f"Dense shape {dense.shape} != ({n_points}, {dim})"

    # Sampled norm assert (check 10% or at least 10 points)
    sample_size = max(min(n_points, 10), n_points // 10)
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(n_points, size=sample_size, replace=False)
    sample_norms = np.linalg.norm(dense[sample_indices], axis=1)
    bad = np.abs(sample_norms - 1.0) >= 1e-2
    if np.any(bad):
        worst = float(np.max(np.abs(sample_norms - 1.0)))
        raise ValueError(
            f"Dense vectors not L2-normalised: worst deviation {worst:.4f} "
            f"({int(bad.sum())}/{sample_size} samples)"
        )

    # 6. Build bm25 posting lists and IDF
    # document frequency per term
    df: dict[int, int] = {}
    for sp_idx in all_sparse_indices:
        for term_id in sp_idx:
            df[term_id] = df.get(term_id, 0) + 1

    # IDF: ln((N - df + 0.5) / (df + 0.5) + 1), Qdrant formula
    idf: dict[int, float] = {}
    for term_id, dfreq in df.items():
        idf[term_id] = math.log((n_points - dfreq + 0.5) / (dfreq + 0.5) + 1.0)

    # Posting lists: term -> (doc_indices[], values[])
    postings_raw: dict[int, tuple[list[int], list[float]]] = {}
    for point_idx, (sp_idx, sp_val) in enumerate(
        zip(all_sparse_indices, all_sparse_values)
    ):
        for tid, val in zip(sp_idx, sp_val):
            if tid not in postings_raw:
                postings_raw[tid] = ([], [])
            postings_raw[tid][0].append(point_idx)
            postings_raw[tid][1].append(val)

    postings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tid, (idxs, vals) in postings_raw.items():
        postings[tid] = (
            np.array(idxs, dtype=np.int32),
            np.array(vals, dtype=np.float32),
        )

    # 7. Build filter columns and null masks
    filter_cols, null_masks, category_maps = _build_filter_arrays(all_payloads, n_points)

    # 8. Load entities
    entities_text = (index_dir / "entities.json").read_text(encoding="utf-8")
    entities = Entities.model_validate_json(entities_text)

    listings_path = index_dir / "listings.parquet"

    return LoadedIndex(
        manifest=manifest,
        n_points=n_points,
        dim=dim,
        dense=dense,
        payloads=tuple(all_payloads),
        postings=postings,
        idf=idf,
        filter_cols=filter_cols,
        null_masks=null_masks,
        category_maps=category_maps,
        entities=entities,
        listings_path=listings_path,
    )


def _build_filter_arrays(
    payloads: list[ChunkPayload],
    n: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Build per-field filter arrays and null masks from payloads."""
    filter_cols: dict[str, np.ndarray] = {}
    null_masks: dict[str, np.ndarray] = {}
    category_maps: dict[str, dict[str, int]] = {}

    # String/enum fields that use category codes
    string_fields = [
        "kind", "source", "purpose", "property_type",
        "city", "district", "country", "doc_id",
    ]
    for field_name in string_fields:
        values = [getattr(p, field_name) for p in payloads]
        # Build category map
        unique_vals = sorted({v for v in values if v is not None})
        cat_map = {v: i for i, v in enumerate(unique_vals)}
        category_maps[field_name] = cat_map

        codes = np.full(n, -1, dtype=np.int32)
        is_null = np.zeros(n, dtype=bool)
        for i, v in enumerate(values):
            if v is None:
                is_null[i] = True
            else:
                sv = str(v)  # StrEnum -> str
                codes[i] = cat_map.get(sv, -1)
                if codes[i] == -1:
                    is_null[i] = True
        filter_cols[field_name] = codes
        null_masks[field_name] = is_null

    # Numeric fields
    numeric_fields = [
        ("bedrooms", "int"),
        ("price_sar", "int"),
        ("annual_rent_sar", "int"),
        ("area_sqm", "float"),
    ]
    for field_name, dtype in numeric_fields:
        values = [getattr(p, field_name) for p in payloads]
        is_null = np.array([v is None for v in values], dtype=bool)
        if dtype == "int":
            arr = np.array([v if v is not None else _NULL_INT for v in values], dtype=np.int64)
        else:
            arr = np.array(
                [v if v is not None else _NULL_FLOAT for v in values], dtype=np.float64
            )
        filter_cols[field_name] = arr
        null_masks[field_name] = is_null

    return filter_cols, null_masks, category_maps
