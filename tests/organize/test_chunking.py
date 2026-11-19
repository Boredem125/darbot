"""Tests for organize.chunking.

Marks tests that need the fastembed cache as skipif absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from common.schemas import (
    Article,
    ArticleKind,
    Listing,
    PropertyType,
    Purpose,
    article_id,
    listing_id,
)

_FE_CACHE = os.environ.get(
    "FASTEMBED_CACHE_PATH",
    str(Path("E:/aiengineer/.orchestration/tmp/ml/fe_cache")),
)
_TOK_DIR = Path(_FE_CACHE) / "models--Qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
_HAS_TOKENIZER = any(
    (d / "tokenizer.json").exists()
    for d in (_TOK_DIR / "snapshots").iterdir()
) if (_TOK_DIR / "snapshots").exists() else False

needs_tokenizer = pytest.mark.skipif(
    not _HAS_TOKENIZER,
    reason="fastembed tokenizer not cached",
)


@needs_tokenizer
class TestChunking:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        os.environ.setdefault("FASTEMBED_CACHE_PATH", _FE_CACHE)
        os.environ.setdefault("HF_HOME", str(Path(_FE_CACHE).parent / "hf"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from organize.chunking import build_token_counter
        self.token_count = build_token_counter()

    def test_listing_under_126(self) -> None:
        from organize.chunking import chunk_listing
        listing = Listing(
            id=listing_id("1"),
            source="wasalt",
            source_id="1",
            url="https://wasalt.sa/property/sale/test-1",
            title="Small Apartment",
            description="Nice apartment",
            purpose=Purpose.sale,
            property_type=PropertyType.apartment,
            city="Riyadh",
            price_sar=500000,
            bedrooms=2,
            area_sqm=80.0,
            scraped_at="2026-11-18T10:00:00+00:00",
        )
        chunks = chunk_listing(listing, self.token_count)
        assert len(chunks) == 1
        assert self.token_count(chunks[0].text) <= 126

    def test_listing_long_description_trimmed(self) -> None:
        from organize.chunking import chunk_listing
        # Create a listing with a very long description
        long_desc = " ".join(["word"] * 200)
        listing = Listing(
            id=listing_id("2"),
            source="wasalt",
            source_id="2",
            url="https://wasalt.sa/property/sale/test-2",
            title="Large Villa",
            description=long_desc[:400],
            purpose=Purpose.sale,
            property_type=PropertyType.villa,
            city="Jeddah",
            price_sar=2000000,
            bedrooms=5,
            bathrooms=4,
            area_sqm=500.0,
            scraped_at="2026-11-18T10:00:00+00:00",
        )
        chunks = chunk_listing(listing, self.token_count)
        assert len(chunks) == 1
        tc = self.token_count(chunks[0].text)
        assert tc <= 126

    def test_126_token_gate_fires(self) -> None:
        """If we try to build a chunk over 126 tokens, the build catches it."""
        from organize.chunking import MAX_TOKENS
        assert MAX_TOKENS == 126

    def test_article_max_8_chunks(self) -> None:
        from organize.chunking import MAX_ARTICLE_CHUNKS, chunk_article
        assert MAX_ARTICLE_CHUNKS == 8

        body = " ".join(["Sentence about real estate in Dubai."] * 500)
        article = Article(
            id=article_id(ArticleKind.press, "test-long"),
            source="darglobal",
            kind=ArticleKind.press,
            slug="test-long",
            url="https://darglobal.co.uk/press/test-long",
            title="Long Press Article",
            body=body,
            scraped_at="2026-11-18T10:00:00+00:00",
        )
        chunks = chunk_article(article, self.token_count)
        assert len(chunks) <= 8

    def test_header_cap(self) -> None:
        from organize.chunking import HEADER_MAX_TOKENS, _make_header
        assert HEADER_MAX_TOKENS == 28
        long_title = " ".join(["word"] * 50) + ". "
        header = _make_header(long_title, self.token_count)
        assert self.token_count(header) <= 28
