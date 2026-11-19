"""Tests for NumpyHybridRetriever: filter semantics, RRF order, top_dense_score."""

from __future__ import annotations

import numpy as np
import pytest

from app.domain.ports import QueryEmbedding
from common.schemas import (
    DocKind,
    PropertyType,
    Purpose,
    SearchFilters,
    SparseVector,
)
from tests.adapters.conftest import DIM


class TestRetrieverFilters:
    """Filter semantics: each field, null handling, combined filters."""

    @pytest.fixture(autouse=True)
    def _load_index(self, synthetic_index):
        from app.adapters.index_loader import load_index
        from app.adapters.retriever_numpy import NumpyHybridRetriever

        self.idx = load_index(
            synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse"
        )
        self.retriever = NumpyHybridRetriever(self.idx, rrf_k=60)

    def _make_embedding(self, dense: np.ndarray | None = None) -> QueryEmbedding:
        if dense is None:
            dense = np.ones(DIM, dtype=np.float32)
            dense = dense / np.linalg.norm(dense)
        return QueryEmbedding(
            dense=tuple(dense.tolist()),
            sparse=SparseVector(indices=[0, 1], values=[1.0, 1.0]),
            cached=False,
            sparse_ms=0.1,
        )

    @pytest.mark.asyncio
    async def test_no_filters_returns_results(self):
        result = await self.retriever.search(
            self._make_embedding(), SearchFilters(), limit=5
        )
        assert len(result.hits) > 0
        assert len(result.hits) <= 5

    @pytest.mark.asyncio
    async def test_filter_purpose_sale(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(purpose="sale"),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.purpose == Purpose.sale

    @pytest.mark.asyncio
    async def test_filter_purpose_rent(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(purpose="rent"),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.purpose == Purpose.rent

    @pytest.mark.asyncio
    async def test_filter_city(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(city="Riyadh"),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.city == "Riyadh"

    @pytest.mark.asyncio
    async def test_filter_property_type(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(property_type="villa"),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.property_type == PropertyType.villa

    @pytest.mark.asyncio
    async def test_filter_bedrooms_exact(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(bedrooms_min=3, bedrooms_max=3),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.bedrooms == 3

    @pytest.mark.asyncio
    async def test_filter_bedrooms_range(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(bedrooms_min=2, bedrooms_max=4),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.bedrooms is not None
            assert 2 <= hit.payload.bedrooms <= 4

    @pytest.mark.asyncio
    async def test_null_bedrooms_excluded_by_filter(self):
        """Points with null bedrooms must not appear when bedrooms filter is set."""
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(bedrooms_min=1, bedrooms_max=10),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.bedrooms is not None

    @pytest.mark.asyncio
    async def test_filter_price_sale(self):
        """Sale price filter uses price_sar."""
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(purpose="sale", price_max_sar=600000),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.purpose == Purpose.sale
            assert hit.payload.price_sar is not None
            assert hit.payload.price_sar <= 600000

    @pytest.mark.asyncio
    async def test_filter_price_rent(self):
        """Rent price filter uses annual_rent_sar."""
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(purpose="rent", price_max_sar=50000),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.purpose == Purpose.rent
            assert hit.payload.annual_rent_sar is not None
            assert hit.payload.annual_rent_sar <= 50000

    @pytest.mark.asyncio
    async def test_filter_area_range(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(area_min_sqm=100.0, area_max_sqm=200.0),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.area_sqm is not None
            assert 100.0 <= hit.payload.area_sqm <= 200.0

    @pytest.mark.asyncio
    async def test_filter_kinds(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(kinds=(DocKind.listing,)),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.kind == DocKind.listing

    @pytest.mark.asyncio
    async def test_filter_district(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(district="Al Narjis"),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.district == "Al Narjis"

    @pytest.mark.asyncio
    async def test_impossible_filter_returns_empty(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(city="Nonexistent City"),
            limit=20,
        )
        assert len(result.hits) == 0
        assert result.top_dense_score is None

    @pytest.mark.asyncio
    async def test_combined_filters(self):
        result = await self.retriever.search(
            self._make_embedding(),
            SearchFilters(purpose="sale", city="Riyadh", bedrooms_min=3, bedrooms_max=3),
            limit=20,
        )
        for hit in result.hits:
            assert hit.payload.purpose == Purpose.sale
            assert hit.payload.city == "Riyadh"
            assert hit.payload.bedrooms == 3


class TestRRFOrder:
    """RRF fusion order on a hand-checked example."""

    @pytest.fixture(autouse=True)
    def _load_index(self, synthetic_index):
        from app.adapters.index_loader import load_index
        from app.adapters.retriever_numpy import NumpyHybridRetriever

        self.idx = load_index(
            synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse"
        )
        self.retriever = NumpyHybridRetriever(self.idx, rrf_k=60)

    @pytest.mark.asyncio
    async def test_results_sorted_desc(self):
        """Results must be sorted by fused score descending."""
        dense = np.ones(DIM, dtype=np.float32)
        dense = dense / np.linalg.norm(dense)
        emb = QueryEmbedding(
            dense=tuple(dense.tolist()),
            sparse=SparseVector(indices=[0, 1], values=[1.0, 1.0]),
            cached=False,
            sparse_ms=0.1,
        )
        result = await self.retriever.search(emb, SearchFilters(), limit=10)
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rrf_scores_positive(self):
        dense = np.ones(DIM, dtype=np.float32)
        dense = dense / np.linalg.norm(dense)
        emb = QueryEmbedding(
            dense=tuple(dense.tolist()),
            sparse=SparseVector(indices=[0], values=[1.0]),
            cached=False,
            sparse_ms=0.1,
        )
        result = await self.retriever.search(emb, SearchFilters(), limit=5)
        for hit in result.hits:
            assert hit.score > 0


class TestTopDenseScore:
    """top_dense_score must be the raw cosine of the filtered top-1, or None."""

    @pytest.fixture(autouse=True)
    def _load_index(self, synthetic_index):
        from app.adapters.index_loader import load_index
        from app.adapters.retriever_numpy import NumpyHybridRetriever

        self.idx = load_index(
            synthetic_index, embed_model="test-model", dim=DIM, sparse_model="test-sparse"
        )
        self.retriever = NumpyHybridRetriever(self.idx, rrf_k=60)

    @pytest.mark.asyncio
    async def test_top_dense_score_matches_manual(self):
        """top_dense_score must match the cosine of the dense query vs the filtered top-1 point."""
        # Use the first point's vector as query so we know what the top-1 should be
        query_vec = self.idx.dense[0].copy()
        query_vec /= max(np.linalg.norm(query_vec), 1e-12)
        emb = QueryEmbedding(
            dense=tuple(query_vec.tolist()),
            sparse=SparseVector(indices=[], values=[]),
            cached=False,
            sparse_ms=0.0,
        )
        result = await self.retriever.search(emb, SearchFilters(), limit=5)
        assert result.top_dense_score is not None
        # Compute expected
        scores = self.idx.dense @ query_vec
        expected = float(np.max(scores))
        assert abs(result.top_dense_score - expected) < 1e-5

    @pytest.mark.asyncio
    async def test_top_dense_score_none_on_no_candidates(self):
        """top_dense_score is None when no point passes the filters."""
        emb = QueryEmbedding(
            dense=tuple(np.ones(DIM).tolist()),
            sparse=SparseVector(indices=[], values=[]),
            cached=False,
            sparse_ms=0.0,
        )
        result = await self.retriever.search(
            emb, SearchFilters(city="DOES NOT EXIST"), limit=5
        )
        assert result.top_dense_score is None
        assert len(result.hits) == 0

    @pytest.mark.asyncio
    async def test_top_dense_score_with_filter(self):
        """top_dense_score is computed over filtered points only."""
        query_vec = self.idx.dense[0].copy()
        query_vec /= max(np.linalg.norm(query_vec), 1e-12)
        emb = QueryEmbedding(
            dense=tuple(query_vec.tolist()),
            sparse=SparseVector(indices=[], values=[]),
            cached=False,
            sparse_ms=0.0,
        )
        result = await self.retriever.search(emb, SearchFilters(city="Jeddah"), limit=5)
        if result.top_dense_score is not None:
            # Verify it's the cosine only over Jeddah points
            jeddah_indices = [
                i for i, p in enumerate(self.idx.payloads)
                if p.city == "Jeddah"
            ]
            if jeddah_indices:
                expected = float(max(self.idx.dense[j] @ query_vec for j in jeddah_indices))
                assert abs(result.top_dense_score - expected) < 1e-5
