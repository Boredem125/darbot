"""Tests for index_loader: manifest validation, dense matrix, bm25 postings, filters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.adapters.conftest import DIM, N_POINTS


class TestIndexLoaderBasic:
    """Load from the synthetic index and check shapes and types."""

    def test_load_success(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(
            synthetic_index,
            embed_model="test-model",
            dim=DIM,
            sparse_model="test-sparse",
        )
        assert idx.n_points == N_POINTS
        assert idx.dim == DIM
        assert idx.dense.shape == (N_POINTS, DIM)
        assert idx.dense.dtype == np.float32

    def test_dense_norms_near_one(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        norms = np.linalg.norm(idx.dense, axis=1)
        assert np.all(np.abs(norms - 1.0) < 1e-2)  # float16 round-trip

    def test_payloads_count(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert len(idx.payloads) == N_POINTS

    def test_posting_lists_populated(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert len(idx.postings) > 0
        assert len(idx.idf) > 0

    def test_manifest_stored(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert idx.manifest.index_version is not None
        assert idx.manifest.dim == DIM


class TestIndexLoaderFailFast:
    """Manifest fail-fast rules: wrong model, wrong dim, missing file."""

    def test_wrong_embed_model(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index
        from common.schemas import ManifestMismatchError

        with pytest.raises(ManifestMismatchError, match="embed_model"):
            load_index(
                synthetic_index,
                embed_model="wrong-model",
                dim=DIM,
                sparse_model="test-sparse",
            )

    def test_wrong_dim(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index
        from common.schemas import ManifestMismatchError

        with pytest.raises(ManifestMismatchError, match="dim"):
            load_index(
                synthetic_index,
                embed_model="test-model",
                dim=999,
                sparse_model="test-sparse",
            )

    def test_wrong_sparse_model(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index
        from common.schemas import ManifestMismatchError

        with pytest.raises(ManifestMismatchError, match="sparse_model"):
            load_index(
                synthetic_index,
                embed_model="test-model",
                dim=DIM,
                sparse_model="wrong-sparse",
            )

    def test_missing_file(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        (synthetic_index / "points_000.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            load_index(
                synthetic_index,
                embed_model="test-model",
                dim=DIM,
                sparse_model="test-sparse",
            )

    def test_corrupted_manifest(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        (synthetic_index / "manifest.json").write_text("{bad json", encoding="utf-8")
        with pytest.raises((ValueError, KeyError)):
            load_index(
                synthetic_index,
                embed_model="test-model",
                dim=DIM,
                sparse_model="test-sparse",
            )


class TestFilterColumns:
    """Filter column arrays and null masks."""

    def test_filter_kind(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert "kind" in idx.filter_cols
        assert idx.filter_cols["kind"].shape == (N_POINTS,)

    def test_filter_purpose_with_nulls(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert "purpose" in idx.filter_cols
        # Points 5, 6 have purpose=None in payloads
        assert idx.null_masks["purpose"].shape == (N_POINTS,)

    def test_filter_city_with_nulls(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert "city" in idx.filter_cols
        mask = idx.null_masks["city"]
        # Points 5, 6 have city=None
        assert mask[5] and mask[6]  # True means null

    def test_filter_bedrooms(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert "bedrooms" in idx.filter_cols
        # Point 3 has bedrooms=None
        assert idx.null_masks["bedrooms"][3]

    def test_filter_price_sar(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert "price_sar" in idx.filter_cols

    def test_filter_area(self, synthetic_index: Path) -> None:
        from app.adapters.index_loader import load_index

        idx = load_index(synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse")
        assert "area_sqm" in idx.filter_cols
