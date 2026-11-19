"""FastEmbed-based Reranker adapter (retrieval.md section 3.7).

Constructed only when asked; rerank is off by default (RERANK_ENABLED=0).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from app.domain.ports import RetrievalError, RetrievedHit
from common.schemas import MODEL_ENV_DEFAULTS

logger = logging.getLogger(__name__)


class FastEmbedReranker:
    """Reranker port implementation using fastembed cross-encoder."""

    def __init__(
        self,
        *,
        executor: ThreadPoolExecutor | None = None,
        rerank_model: str | None = None,
    ) -> None:
        self._model_id = rerank_model or os.environ.get(
            "RERANK_MODEL", MODEL_ENV_DEFAULTS["RERANK_MODEL"]
        )
        self._executor = executor or ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="onnx"
        )
        self._model_obj = None

    def _ensure_model(self) -> None:
        if self._model_obj is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model_obj = TextCrossEncoder(self._model_id, threads=1)

    async def rerank(
        self, query: str, hits: Sequence[RetrievedHit], *, top_k: int
    ) -> list[RetrievedHit]:
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._rerank_sync, query, hits, top_k
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(str(exc)) from exc

    def _rerank_sync(
        self, query: str, hits: Sequence[RetrievedHit], top_k: int
    ) -> list[RetrievedHit]:
        self._ensure_model()
        if not hits:
            return []

        pairs = [(query, hit.payload.text) for hit in hits]
        scores = list(self._model_obj.rerank(query, [p[1] for p in pairs]))

        # Build (hit, score) pairs and sort
        scored = []
        for i, hit in enumerate(hits):
            scored.append((hit, float(scores[i])))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievedHit(payload=hit.payload, score=sc)
            for hit, sc in scored[:top_k]
        ]
