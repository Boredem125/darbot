"""FastEmbed-based Embedder adapter: dense + bm25 query embedding in one executor job.

Model ids from env with common.schemas.MODEL_ENV_DEFAULTS. Reports sparse_ms (CC-2).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.domain.ports import EmbeddingError, QueryEmbedding
from common.schemas import (
    MODEL_ENV_DEFAULTS,
    SparseVector,
)

logger = logging.getLogger(__name__)


class FastEmbedEmbedder:
    """Embedder port implementation using fastembed dense + bm25 models."""

    def __init__(
        self,
        *,
        executor: ThreadPoolExecutor | None = None,
        embed_model: str | None = None,
        sparse_model: str | None = None,
    ) -> None:
        self._embed_model_id = embed_model or os.environ.get(
            "EMBED_MODEL", MODEL_ENV_DEFAULTS["EMBED_MODEL"]
        )
        self._sparse_model_id = sparse_model or os.environ.get(
            "SPARSE_MODEL", MODEL_ENV_DEFAULTS["SPARSE_MODEL"]
        )
        self._executor = executor or ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="onnx"
        )

        # Lazy init: models loaded on first call
        self._dense_model_obj = None
        self._sparse_model_obj = None
        self._dim: int | None = None

    def _ensure_models(self) -> None:
        """Load models if not already loaded."""
        if self._dense_model_obj is None:
            from fastembed import SparseTextEmbedding, TextEmbedding

            self._dense_model_obj = TextEmbedding(
                self._embed_model_id, threads=1
            )
            self._sparse_model_obj = SparseTextEmbedding(
                self._sparse_model_id
            )
            # Determine dim from a test embed
            test_vecs = list(self._dense_model_obj.embed(["test"]))
            self._dim = test_vecs[0].shape[0]

    @property
    def dense_model(self) -> str:
        return self._embed_model_id

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._ensure_models()
        return self._dim  # type: ignore[return-value]

    @property
    def sparse_model(self) -> str:
        return self._sparse_model_id

    async def embed_query(self, text: str) -> QueryEmbedding:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor, self._embed_sync, text
            )
            return result
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc

    def _embed_sync(self, text: str) -> QueryEmbedding:
        """Run dense + bm25 embedding in one executor job."""
        self._ensure_models()

        # Dense
        dense_vecs = list(self._dense_model_obj.embed([text]))
        vec = dense_vecs[0].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec = vec / norm
        dense = tuple(vec.tolist())

        # Sparse (bm25) - timed for sparse_ms
        t0 = time.perf_counter()
        sparse_vecs = list(self._sparse_model_obj.query_embed(text))
        sparse_ms = (time.perf_counter() - t0) * 1000.0

        if sparse_vecs:
            sv = sparse_vecs[0]
            indices = sv.indices.tolist()
            values = sv.values.tolist()
        else:
            indices = []
            values = []

        return QueryEmbedding(
            dense=dense,
            sparse=SparseVector(indices=indices, values=values),
            cached=False,
            sparse_ms=sparse_ms,
        )
