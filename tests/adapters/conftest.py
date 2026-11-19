"""Shared fixtures for adapter tests: a synthetic mini-index on disk."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from common.schemas import (
    ChunkPayload,
    Manifest,
    ManifestFile,
    compute_index_version,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 11, 18, 12, 0, 0, tzinfo=UTC)
_TS_ISO = _TS.isoformat()


def _make_payload(
    doc_id: str,
    chunk_index: int,
    *,
    kind: str = "listing",
    source: str = "wasalt",
    title: str = "Test listing",
    url: str = "https://example.com/1",
    purpose: str | None = "sale",
    property_type: str | None = "villa",
    city: str | None = "Riyadh",
    district: str | None = None,
    price_sar: int | None = 1000000,
    annual_rent_sar: int | None = None,
    rent_period: str | None = None,
    bedrooms: int | None = 3,
    bathrooms: int | None = 2,
    area_sqm: float | None = 200.0,
    country: str | None = None,
    brand: str | None = None,
    text: str = "Test text for embedding",
) -> ChunkPayload:
    return ChunkPayload(
        doc_id=doc_id,
        chunk_index=chunk_index,
        kind=kind,
        source=source,
        title=title,
        url=url,
        scraped_at=_TS,
        purpose=purpose,
        property_type=property_type,
        city=city,
        district=district,
        price_sar=price_sar,
        annual_rent_sar=annual_rent_sar,
        rent_period=rent_period,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        area_sqm=area_sqm,
        country=country,
        brand=brand,
        text=text,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > 1e-12:
        return v / n
    return v


# 10 synthetic points with known dense vectors (dim=4 for speed)
DIM = 4
N_POINTS = 10

# Data arrays used by both points and listings fixtures.
# Rent listings use: annual_rent_sar = price_sar * factor.
#   yearly: factor 1
#   monthly: factor 12
#
# Index: 0    1    2       3    4        5    6    7    8       9
_PURPOSES =       ["sale", "sale", "rent",  "sale", "rent",   None,  None,  "sale", "rent",  "sale"]
_CITIES =         ["Riyadh","Jeddah","Riyadh","Dammam","Riyadh",None, None,"Jeddah","Riyadh","Riyadh"]
_KINDS =          ["listing","listing","listing","listing","listing","project","article","listing","listing","listing"]
_SOURCES =        ["wasalt","wasalt","wasalt","wasalt","wasalt","darglobal","darglobal","wasalt","wasalt","wasalt"]
_PROP_TYPES =     ["villa","apartment","villa","land","apartment",None, None,"villa","townhouse","duplex"]
_BEDROOMS =       [3, 2, 4, None, 1, None, None, 5, 2, 3]
# Rent rows: 2 (yearly, 5000/yr -> annual 5000), 4 (monthly, 2000/mo -> annual 24000),
#             8 (yearly, 3000/yr -> annual 3000)
_PRICES =         [1000000, 500000, 5000, 300000, 2000, None, None, 2000000, 3000, 800000]
_ANNUAL_RENTS =   [None, None, 5000, None, 24000, None, None, None, 3000, None]
_RENT_PERIODS =   [None, None, "yearly", None, "monthly", None, None, None, "yearly", None]
_AREAS =          [200.0, 100.0, 150.0, 500.0, 80.0, None, None, 300.0, 120.0, 250.0]
_DISTRICTS =      ["Al Narjis", None, "Al Narjis", None, None, None, None, "Al Hamra", None, "Al Narjis"]
_COUNTRIES =      [None, None, None, None, None, "Saudi Arabia", None, None, None, None]


def _build_points_table(rng: np.random.Generator) -> tuple[pa.Table, list[ChunkPayload]]:
    """Build a points parquet table with 10 rows."""
    payloads: list[ChunkPayload] = []
    point_ids: list[str] = []
    doc_ids: list[str] = []
    chunk_indices: list[int] = []
    dense_bytes_list: list[bytes] = []
    sparse_indices_list: list[list[int]] = []
    sparse_values_list: list[list[float]] = []

    for i in range(N_POINTS):
        doc_id = f"wasalt:test-{i}" if _SOURCES[i] == "wasalt" else (
            f"darglobal:project:test-{i}" if _KINDS[i] == "project"
            else f"darglobal:blog:test-{i}"
        )
        payload = _make_payload(
            doc_id=doc_id,
            chunk_index=0,
            kind=_KINDS[i],
            source=_SOURCES[i],
            title=f"Test {i}",
            url=f"https://example.com/{i}",
            purpose=_PURPOSES[i],
            property_type=_PROP_TYPES[i],
            city=_CITIES[i],
            district=_DISTRICTS[i],
            price_sar=_PRICES[i],
            annual_rent_sar=_ANNUAL_RENTS[i],
            rent_period=_RENT_PERIODS[i],
            bedrooms=_BEDROOMS[i],
            bathrooms=2 if _BEDROOMS[i] else None,
            area_sqm=_AREAS[i],
            country=_COUNTRIES[i],
            text=f"Test text for point {i}",
        )
        payloads.append(payload)

        from common.schemas import point_id
        pid = point_id(doc_id, 0)
        point_ids.append(pid)
        doc_ids.append(doc_id)
        chunk_indices.append(0)

        # Dense: normalised random float32 -> stored as float16 bytes
        v = rng.standard_normal(DIM).astype(np.float32)
        v = _normalise(v)
        dense_bytes_list.append(v.astype(np.float16).tobytes())

        # Sparse: each point gets terms {i, i+1} with values
        sparse_indices_list.append([i, (i + 1) % 20])
        sparse_values_list.append([1.5, 0.8])

    table = pa.table({
        "point_id": pa.array(point_ids, type=pa.string()),
        "doc_id": pa.array(doc_ids, type=pa.string()),
        "chunk_index": pa.array(chunk_indices, type=pa.int32()),
        "dense": pa.array(dense_bytes_list, type=pa.binary(DIM * 2)),
        "sparse_indices": pa.array(sparse_indices_list, type=pa.list_(pa.uint32())),
        "sparse_values": pa.array(sparse_values_list, type=pa.list_(pa.float32())),
        "payload": pa.array(
            [p.model_dump_json() for p in payloads], type=pa.string()
        ),
    })
    return table, payloads


def _build_listings_table() -> pa.Table:
    """Build a listings parquet with 10 rows using same data as points (listing rows only).

    All 10 rows are listings for DuckDB testing. Rent rows have consistent
    annual_rent_sar = price_sar * ANNUAL_RENT_FACTOR[rent_period].
    """
    # For listings.parquet all 10 rows are listings (not project/article).
    # Reuse the same data arrays; project/article rows become sale listings.
    purposes = ["sale", "sale", "rent", "sale", "rent", "sale", "sale", "sale", "rent", "sale"]
    cities = [
        "Riyadh", "Jeddah", "Riyadh", "Dammam", "Riyadh",
        "Riyadh", "Riyadh", "Jeddah", "Riyadh", "Riyadh",
    ]
    prop_types = [
        "villa", "apartment", "villa", "land", "apartment",
        "villa", "apartment", "villa", "townhouse", "duplex",
    ]
    # Sale: price_sar set, annual_rent_sar=None, rent_period=None
    # Rent: price_sar=periodic amount, annual_rent_sar=price*factor, rent_period set
    # Row 2: rent, yearly, price=5000, annual=5000 (factor=1)
    # Row 4: rent, monthly, price=2000, annual=24000 (factor=12)
    # Row 8: rent, yearly, price=3000, annual=3000 (factor=1)
    prices = [1000000, 500000, 5000, 300000, 2000, 400000, 600000, 2000000, 3000, 800000]
    annual_rents = [None, None, 5000, None, 24000, None, None, None, 3000, None]
    rent_periods = [None, None, "yearly", None, "monthly", None, None, None, "yearly", None]
    bedrooms_vals = [3, 2, 4, None, 1, 2, 3, 5, 2, 3]
    areas = [200.0, 100.0, 150.0, 500.0, 80.0, 180.0, 90.0, 300.0, 120.0, 250.0]
    districts = [
        "Al Narjis", None, "Al Narjis", None, None,
        None, None, "Al Hamra", None, "Al Narjis",
    ]

    rows: list[dict] = []
    for i in range(10):
        rows.append({
            "id": f"wasalt:test-{i}",
            "source": "wasalt",
            "source_id": f"test-{i}",
            "url": f"https://example.com/{i}",
            "title": f"Test listing {i}",
            "description": f"Description for listing {i}" if i % 2 == 0 else None,
            "purpose": purposes[i],
            "property_type": prop_types[i],
            "property_type_raw": None,
            "city": cities[i],
            "district": districts[i],
            "price_sar": prices[i],
            "annual_rent_sar": annual_rents[i],
            "bedrooms": bedrooms_vals[i],
            "bathrooms": 2 if bedrooms_vals[i] else None,
            "area_sqm": areas[i],
            "lat": None,
            "lon": None,
            "amenities": [],
            "rent_period": rent_periods[i],
            "published_at": None,
            "scraped_at": _TS,
        })

    table = pa.table({
        "id": pa.array([r["id"] for r in rows], type=pa.string()),
        "source": pa.array([r["source"] for r in rows], type=pa.string()),
        "source_id": pa.array([r["source_id"] for r in rows], type=pa.string()),
        "url": pa.array([r["url"] for r in rows], type=pa.string()),
        "title": pa.array([r["title"] for r in rows], type=pa.string()),
        "description": pa.array([r["description"] for r in rows], type=pa.string()),
        "purpose": pa.array([r["purpose"] for r in rows], type=pa.string()),
        "property_type": pa.array([r["property_type"] for r in rows], type=pa.string()),
        "property_type_raw": pa.array([r["property_type_raw"] for r in rows], type=pa.string()),
        "city": pa.array([r["city"] for r in rows], type=pa.string()),
        "district": pa.array([r["district"] for r in rows], type=pa.string()),
        "price_sar": pa.array([r["price_sar"] for r in rows], type=pa.int64()),
        "annual_rent_sar": pa.array([r["annual_rent_sar"] for r in rows], type=pa.int64()),
        "bedrooms": pa.array([r["bedrooms"] for r in rows], type=pa.int16()),
        "bathrooms": pa.array([r["bathrooms"] for r in rows], type=pa.int16()),
        "area_sqm": pa.array([r["area_sqm"] for r in rows], type=pa.float64()),
        "lat": pa.array([r["lat"] for r in rows], type=pa.float64()),
        "lon": pa.array([r["lon"] for r in rows], type=pa.float64()),
        "amenities": pa.array([r["amenities"] for r in rows], type=pa.list_(pa.string())),
        "rent_period": pa.array([r["rent_period"] for r in rows], type=pa.string()),
        "published_at": pa.array(
            [r["published_at"] for r in rows], type=pa.timestamp("us", tz="UTC")
        ),
        "scraped_at": pa.array(
            [r["scraped_at"] for r in rows], type=pa.timestamp("us", tz="UTC")
        ),
    })
    return table


def _build_entities() -> dict:
    return {
        "schema_version": 1,
        "entities": [
            {
                "kind": "city",
                "canonical": "Riyadh",
                "aliases": ["riyadh"],
                "parent": "Saudi Arabia",
                "doc_id": None,
            },
            {
                "kind": "city",
                "canonical": "Jeddah",
                "aliases": ["jeddah"],
                "parent": "Saudi Arabia",
                "doc_id": None,
            },
            {
                "kind": "district",
                "canonical": "Al Narjis",
                "aliases": ["narjis"],
                "parent": "Riyadh",
                "doc_id": None,
            },
        ],
    }


@pytest.fixture()
def synthetic_index(tmp_path: Path) -> Path:
    """Build a synthetic index directory and return its path."""
    rng = np.random.default_rng(42)
    idx_dir = tmp_path / "index"
    idx_dir.mkdir()

    # Points
    points_table, _payloads = _build_points_table(rng)
    points_path = idx_dir / "points_000.parquet"
    pq.write_table(points_table, points_path)

    # Listings
    listings_table = _build_listings_table()
    listings_path = idx_dir / "listings.parquet"
    pq.write_table(listings_table, listings_path)

    # Entities
    entities = _build_entities()
    entities_path = idx_dir / "entities.json"
    entities_path.write_text(json.dumps(entities), encoding="utf-8")

    # Build manifest
    files_info: list[ManifestFile] = []
    for name in ["points_000.parquet", "listings.parquet", "entities.json"]:
        p = idx_dir / name
        data = p.read_bytes()
        rows = None
        if name.endswith(".parquet"):
            rows = pq.read_metadata(p).num_rows
        files_info.append(ManifestFile(
            path=name,
            sha256=_sha256(data),
            size_bytes=len(data),
            rows=rows,
        ))

    index_version = compute_index_version(files_info)
    manifest = Manifest(
        schema_version=1,
        index_version=index_version,
        built_at=_TS,
        embed_model="test-model",
        dim=DIM,
        sparse_model="test-sparse",
        distance="cosine",
        vector_dtype="float16",
        counts={
            "listings_sale": 7,
            "listings_rent": 3,
            "projects": 0,
            "articles": 0,
            "points": N_POINTS,
        },
        scraped_at_min=_TS,
        scraped_at_max=_TS,
        git_sha="unknown",
        prompt_version=None,
        files=files_info,
    )
    (idx_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    return idx_dir


@pytest.fixture()
def payloads_and_dense(tmp_path: Path) -> tuple[list[ChunkPayload], np.ndarray]:
    """Return the payloads list and the dense matrix for the synthetic index."""
    rng = np.random.default_rng(42)
    _table, payloads = _build_points_table(rng)

    # Reconstruct dense from same rng
    rng2 = np.random.default_rng(42)
    dense = np.zeros((N_POINTS, DIM), dtype=np.float32)
    for i in range(N_POINTS):
        v = rng2.standard_normal(DIM).astype(np.float32)
        v = _normalise(v)
        # float16 round-trip
        v16 = v.astype(np.float16)
        dense[i] = v16.astype(np.float32)

    return payloads, dense
