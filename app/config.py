"""App configuration via pydantic-settings. Keys from env only.

All defaults equal the tuned values from docs/tuning.md.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings

from common.schemas import EMBED_DIM_DEFAULT, MODEL_ENV_DEFAULTS


class Settings(BaseSettings):
    # --- Index and models ---
    INDEX_DIR: str = "data/index"
    EMBED_MODEL: str = MODEL_ENV_DEFAULTS["EMBED_MODEL"]
    SPARSE_MODEL: str = MODEL_ENV_DEFAULTS["SPARSE_MODEL"]
    RERANK_MODEL: str = MODEL_ENV_DEFAULTS["RERANK_MODEL"]
    EMBED_DIM: int = EMBED_DIM_DEFAULT

    # --- Feature flags ---
    RERANK_ENABLED: int = 0
    SEMANTIC_CACHE_ENABLED: int = 1
    METRICS_ENABLED: int = 1
    JUDGE_ENABLED: int = 0

    # --- LLM ---
    ARABIC_MAX_TOKENS: int = 700
    LLM_CHAIN: str = ""  # JSON string; empty => default chain
    LLM_MAX_CONCURRENCY: int = 4

    # --- Retrieval tuning ---
    TAU: float = 0.93
    SCORE_FLOOR: float = 0.52
    RRF_K: int = 60
    DENSE_WEIGHT: float = 1.0
    SPARSE_WEIGHT: float = 1.0
    PREFETCH_LIMIT: int = 20

    # --- Concurrency ---
    MAX_INFLIGHT_CHATS: int = 12
    RETRIEVE_IN_EXECUTOR: int = 0

    # --- Rate limiting ---
    RATE_LIMIT_PER_MIN: int = 10
    RATE_LIMIT_PER_DAY: int = 40
    RATE_LIMIT_XFF_HOP: str = "first"  # "first" or "last"

    # --- API key env names (not the keys themselves) ---
    OPENROUTER_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    model_config = {"env_prefix": "", "case_sensitive": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
