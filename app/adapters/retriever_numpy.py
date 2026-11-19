"""Numpy hybrid retriever: dense matmul + bm25 posting lists + RRF fusion.

Implements the Retriever port with Qdrant Query API semantics (retrieval.md section 3).
Runs on the event loop (p95 ~1.5 ms at 6k points, well under the 5 ms threshold).
"""

from __future__ import annotations

import numpy as np

from app.adapters.index_loader import LoadedIndex
from app.domain.ports import (
    QueryEmbedding,
    RetrievalError,
    RetrievalResult,
    RetrievedHit,
)
from common.schemas import Purpose, SearchFilters

# Default parameters; tuning changes them via constructor
DEFAULT_RRF_K = 60
DEFAULT_PREFETCH_LIMIT = 20


class NumpyHybridRetriever:
    """In-process hybrid search: dense + bm25, RRF fusion, payload filters."""

    def __init__(
        self,
        index: LoadedIndex,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    ) -> None:
        self._idx = index
        self._rrf_k = rrf_k
        self._dense_w = dense_weight
        self._sparse_w = sparse_weight
        self._prefetch = prefetch_limit

    async def search(
        self,
        embedding: QueryEmbedding,
        filters: SearchFilters,
        *,
        limit: int = 20,
    ) -> RetrievalResult:
        try:
            return self._search_sync(embedding, filters, limit)
        except Exception as exc:
            raise RetrievalError(str(exc)) from exc

    def _search_sync(
        self,
        embedding: QueryEmbedding,
        filters: SearchFilters,
        limit: int,
    ) -> RetrievalResult:
        idx = self._idx
        n = idx.n_points

        # 1. Build filter mask (True = passes)
        mask = self._build_mask(filters)
        n_pass = int(mask.sum())
        if n_pass == 0:
            return RetrievalResult(hits=(), top_dense_score=None)

        # 2. Dense candidate list
        query_dense = np.array(embedding.dense, dtype=np.float32)
        scores_dense = idx.dense @ query_dense  # (n_points,)
        scores_dense[~mask] = -np.inf
        dense_top_indices, dense_top_scores = self._topk(
            scores_dense, self._prefetch, mask
        )

        # top_dense_score: raw cosine of the filtered top-1
        top_dense_score: float | None = None
        if len(dense_top_indices) > 0:
            top_dense_score = float(dense_top_scores[0])

        # 3. Sparse (bm25) candidate list
        sparse_scores = np.zeros(n, dtype=np.float64)
        for tid, qval in zip(embedding.sparse.indices, embedding.sparse.values):
            if tid in idx.postings:
                doc_indices, doc_values = idx.postings[tid]
                idf_val = idx.idf.get(tid, 0.0)
                sparse_scores[doc_indices] += idf_val * doc_values * qval
        sparse_scores[~mask] = 0.0
        sparse_top_indices, _sparse_top_scores = self._topk_sparse(
            sparse_scores, self._prefetch
        )

        # 4. RRF fusion
        fused = self._rrf_fuse(
            dense_top_indices, sparse_top_indices, dense_top_scores
        )

        # 5. Build results
        hits: list[RetrievedHit] = []
        for point_idx, fused_score in fused[:limit]:
            hits.append(
                RetrievedHit(
                    payload=idx.payloads[point_idx],
                    score=fused_score,
                )
            )

        return RetrievalResult(
            hits=tuple(hits),
            top_dense_score=top_dense_score,
        )

    def _build_mask(self, filters: SearchFilters) -> np.ndarray:
        """Build boolean mask: True = point passes all filter conditions."""
        idx = self._idx
        n = idx.n_points
        mask = np.ones(n, dtype=bool)

        # kinds: match-any
        if filters.kinds is not None:
            codes = [
                idx.category_maps["kind"].get(str(k), -1) for k in filters.kinds
            ]
            valid_codes = [c for c in codes if c >= 0]
            if valid_codes:
                match = np.zeros(n, dtype=bool)
                for c in valid_codes:
                    match |= idx.filter_cols["kind"] == c
                mask &= match
            else:
                mask[:] = False

        # Scalar string filters
        for attr, field_name in [
            ("source", "source"),
            ("purpose", "purpose"),
            ("property_type", "property_type"),
            ("city", "city"),
            ("district", "district"),
            ("country", "country"),
            ("project_id", "doc_id"),
        ]:
            val = getattr(filters, attr)
            if val is not None:
                code = idx.category_maps[field_name].get(str(val), -1)
                if code >= 0:
                    mask &= (idx.filter_cols[field_name] == code)
                else:
                    mask[:] = False

        # Bedrooms range (null fails the condition)
        if filters.bedrooms_min is not None:
            mask &= ~idx.null_masks["bedrooms"]
            mask &= idx.filter_cols["bedrooms"] >= filters.bedrooms_min
        if filters.bedrooms_max is not None:
            mask &= ~idx.null_masks["bedrooms"]
            mask &= idx.filter_cols["bedrooms"] <= filters.bedrooms_max

        # Price range: annual_rent_sar when purpose==rent, else price_sar
        price_field = (
            "annual_rent_sar"
            if filters.purpose == Purpose.rent
            else "price_sar"
        )
        if filters.price_min_sar is not None:
            mask &= ~idx.null_masks[price_field]
            mask &= idx.filter_cols[price_field] >= filters.price_min_sar
        if filters.price_max_sar is not None:
            mask &= ~idx.null_masks[price_field]
            mask &= idx.filter_cols[price_field] <= filters.price_max_sar

        # Area range (null fails)
        if filters.area_min_sqm is not None:
            mask &= ~idx.null_masks["area_sqm"]
            mask &= idx.filter_cols["area_sqm"] >= filters.area_min_sqm
        if filters.area_max_sqm is not None:
            mask &= ~idx.null_masks["area_sqm"]
            mask &= idx.filter_cols["area_sqm"] <= filters.area_max_sqm

        return mask

    def _topk(
        self,
        scores: np.ndarray,
        k: int,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k from dense scores (descending). Ties broken by ascending index."""
        valid = np.where((scores > -np.inf) & mask)[0]
        if len(valid) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        k_actual = min(k, len(valid))
        # argpartition + argsort for deterministic tie-break
        if k_actual < len(valid):
            part_idx = np.argpartition(-scores[valid], k_actual)[:k_actual]
            top_valid = valid[part_idx]
        else:
            top_valid = valid

        # Sort by score desc, then by index asc
        order = np.lexsort((top_valid, -scores[top_valid]))
        top_valid = top_valid[order]
        return top_valid, scores[top_valid]

    def _topk_sparse(
        self,
        scores: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k from sparse scores (> 0 only), ties broken by ascending index."""
        valid = np.where(scores > 0)[0]
        if len(valid) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        k_actual = min(k, len(valid))
        if k_actual < len(valid):
            part_idx = np.argpartition(-scores[valid], k_actual)[:k_actual]
            top_valid = valid[part_idx]
        else:
            top_valid = valid

        order = np.lexsort((top_valid, -scores[top_valid]))
        top_valid = top_valid[order]
        return top_valid, scores[top_valid]

    def _rrf_fuse(
        self,
        dense_indices: np.ndarray,
        sparse_indices: np.ndarray,
        dense_scores: np.ndarray,
    ) -> list[tuple[int, float]]:
        """Weighted RRF fusion with Qdrant's formula.

        rrf(d) = sum over lists: w / ((p+1) + w*(k-1))
        where p is 0-based position, w is list weight, k is rrf_k.
        Ties broken by best dense rank (ascending index in dense list).
        """
        k = self._rrf_k
        fused: dict[int, float] = {}
        dense_rank: dict[int, int] = {}

        # Dense list contribution
        w_d = self._dense_w
        for pos, idx in enumerate(dense_indices):
            idx_int = int(idx)
            if w_d > 0:
                score = w_d / ((pos + 1) + w_d * (k - 1))
                fused[idx_int] = fused.get(idx_int, 0.0) + score
            dense_rank[idx_int] = pos

        # Sparse list contribution
        w_s = self._sparse_w
        for pos, idx in enumerate(sparse_indices):
            idx_int = int(idx)
            if w_s > 0:
                score = w_s / ((pos + 1) + w_s * (k - 1))
                fused[idx_int] = fused.get(idx_int, 0.0) + score

        # Sort by fused score desc, then by dense rank asc (inf for not in dense list)
        items = list(fused.items())
        items.sort(key=lambda x: (-x[1], dense_rank.get(x[0], float("inf"))))
        return items
