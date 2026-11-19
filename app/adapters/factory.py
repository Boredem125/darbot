"""Factory: build_read_side assembles the four read-side adapters + LoadedIndex.

Parameters that tuning will vary are overridable here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.adapters.index_loader import LoadedIndex, load_index
from app.adapters.retriever_numpy import NumpyHybridRetriever
from app.adapters.sql_duckdb import DuckDBStructuredQuery


@dataclass(frozen=True, slots=True)
class ReadSide:
    """Container for the four read-side adapters and the loaded index."""

    index: LoadedIndex
    embedder: object  # FastEmbedEmbedder or any Embedder
    retriever: NumpyHybridRetriever
    reranker: object | None  # FastEmbedReranker or None
    sql: DuckDBStructuredQuery


def build_read_side(
    index_dir: Path,
    *,
    rerank: bool = False,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    prefetch_limit: int = 20,
    embed_model: str | None = None,
    sparse_model: str | None = None,
    rerank_model: str | None = None,
    prompt_version: str | None = None,
    warm_cache_present: bool = False,
    executor: ThreadPoolExecutor | None = None,
) -> ReadSide:
    """Build all read-side adapters from an index directory.

    Parameters that tuning varies:
        rrf_k: RRF fusion constant (default 60)
        dense_weight: RRF weight for dense list (default 1.0)
        sparse_weight: RRF weight for sparse list (default 1.0)
        prefetch_limit: top-k per candidate list (default 20)
        rerank: whether to enable the reranker (default False)
    """
    import os

    from common.schemas import MODEL_ENV_DEFAULTS

    _embed_model = embed_model or os.environ.get(
        "EMBED_MODEL", MODEL_ENV_DEFAULTS["EMBED_MODEL"]
    )
    _sparse_model = sparse_model or os.environ.get(
        "SPARSE_MODEL", MODEL_ENV_DEFAULTS["SPARSE_MODEL"]
    )

    # Read manifest to get dim (needed for compatibility check)
    import json

    manifest_data = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_dim = manifest_data["dim"]

    # Load index
    idx = load_index(
        index_dir,
        embed_model=_embed_model,
        dim=manifest_dim,
        sparse_model=_sparse_model,
        prompt_version=prompt_version,
        warm_cache_present=warm_cache_present,
    )

    # Executor shared by embedder and reranker
    _executor = executor or ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="onnx"
    )

    # Embedder
    from app.adapters.embedder_fastembed import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(
        executor=_executor,
        embed_model=_embed_model,
        sparse_model=_sparse_model,
    )

    # Retriever
    retriever = NumpyHybridRetriever(
        idx,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        prefetch_limit=prefetch_limit,
    )

    # Reranker (optional)
    reranker = None
    if rerank:
        from app.adapters.reranker_fastembed import FastEmbedReranker

        reranker = FastEmbedReranker(
            executor=_executor,
            rerank_model=rerank_model,
        )

    # SQL
    sql = DuckDBStructuredQuery(idx.listings_path)

    return ReadSide(
        index=idx,
        embedder=embedder,
        retriever=retriever,
        reranker=reranker,
        sql=sql,
    )


def build_llm(
    chain_ids: list[str] | None = None,
) -> object:
    """Build the LLMRouter from LLM_CHAIN env or the default chain.

    Args:
        chain_ids: if given, filter the chain to these entry ids only.

    Returns:
        An LLMRouter implementing the LLM port.
    """
    import os

    from app.adapters.llm_router import LLMRouter

    chain_json = os.environ.get("LLM_CHAIN", "")
    if chain_json:
        chain = __import__("json").loads(chain_json)
    else:
        # Default chain: groq-oss, or-gemma, or-free, groq-qwen
        # Order per WP-3-geneval probe 19 Nov: or-gemma upstream 429,
        # or-nemotron timeout, or-free wrong language; user decision: Groq first.
        # or-nemotron dropped from default; kept available via LLM_CHAIN JSON.
        chain = [
            {
                "id": "groq-oss",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key_env": "GROQ_API_KEY",
                "quota": None,
                "tpm": 8000,
                "extra_body": {"reasoning_effort": "low", "include_reasoning": False},
            },
            {
                "id": "or-gemma",
                "provider": "openrouter",
                "model": "google/gemma-4-31b-it:free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "quota": "openrouter_free",
                "extra_body": {},
            },
            {
                "id": "or-free",
                "provider": "openrouter",
                "model": "openrouter/free",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "quota": "openrouter_free",
                "extra_body": {},
            },
            {
                "id": "groq-qwen",
                "provider": "groq",
                "model": "qwen/qwen3.8-27b",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key_env": "GROQ_API_KEY",
                "quota": None,
                "tpm": 6000,
                "extra_body": {},
            },
        ]

    if chain_ids:
        chain = [e for e in chain if e["id"] in chain_ids]

    return LLMRouter(chain)
