"""Tests for near-duplicate dedup in collapse_to_docs and HTML entity cleanup in render_card."""

from __future__ import annotations

import datetime

from app.domain.retrieval_post import collapse_to_docs, dedup_docs
from common.cards import render_card
from common.schemas import (
    ChunkPayload,
    DocFields,
    DocKind,
    PropertyType,
    Purpose,
    Source,
)

_NOW = datetime.datetime(2026, 11, 19, tzinfo=datetime.UTC)


def _chunk(
    doc_id: str = "wasalt:1",
    title: str = "Villa",
    price: int = 2000000,
    area: float = 478.0,
    bedrooms: int = 3,
    district: str = "Al Mahdiyah",
) -> ChunkPayload:
    return ChunkPayload(
        doc_id=doc_id,
        kind=DocKind.listing,
        source=Source.wasalt,
        title=title,
        url=f"https://wasalt.sa/en/property/{doc_id}",
        scraped_at=_NOW,
        purpose=Purpose.sale,
        property_type=PropertyType.villa,
        city="Riyadh",
        district=district,
        price_sar=price,
        bedrooms=bedrooms,
        area_sqm=area,
        chunk_index=0,
        text=f"{title} in {district}",
    )


class TestDedupDocs:
    """Near-duplicate listings with same (title, price, area, bedrooms, district) are dropped."""

    def test_duplicate_pair_collapsed(self):
        """Two hits with same key fields but different doc_ids: only the higher-scored one kept."""
        hits = [
            (_chunk("wasalt:1", "Villa 478 SQM", 2000000, 478.0, 3, "Al Mahdiyah"), 0.9),
            (_chunk("wasalt:99", "Different villa", 1500000, 300.0, 4, "Al Narjis"), 0.8),
            (_chunk("wasalt:2", "Villa 478 SQM", 2000000, 478.0, 3, "Al Mahdiyah"), 0.7),
        ]
        docs = collapse_to_docs(hits, top_k=5)
        deduped = dedup_docs(docs)
        # wasalt:2 is a near-dup of wasalt:1, should be removed
        ids = [d.doc_id for d in deduped]
        assert "wasalt:1" in ids
        assert "wasalt:99" in ids
        assert "wasalt:2" not in ids

    def test_five_cards_after_backfill(self):
        """With a dup in top 5, backfill from remaining candidates to keep 5 cards."""
        hits = [
            (_chunk("wasalt:1", "Villa A", 2000000, 478.0, 3, "Al Mahdiyah"), 0.95),
            (_chunk("wasalt:2", "Villa B", 1500000, 300.0, 4, "Al Narjis"), 0.90),
            (_chunk("wasalt:3", "Villa C", 1800000, 400.0, 3, "Dirab"), 0.85),
            (_chunk("wasalt:4", "Villa D", 1200000, 250.0, 2, "An Nahdah"), 0.80),
            (_chunk("wasalt:5", "Villa A", 2000000, 478.0, 3, "Al Mahdiyah"), 0.75),  # dup of 1
            (_chunk("wasalt:6", "Villa E", 900000, 200.0, 5, "Al Rawdah"), 0.70),
        ]
        docs = collapse_to_docs(hits, top_k=6)
        deduped = dedup_docs(docs, top_k=5)
        assert len(deduped) == 5
        ids = [d.doc_id for d in deduped]
        assert "wasalt:5" not in ids  # dup removed
        assert "wasalt:6" in ids  # backfilled

    def test_no_dups_unchanged(self):
        """When there are no duplicates, the list is unchanged."""
        hits = [
            (_chunk("wasalt:1", "Villa A", 2000000, 478.0, 3, "Al Mahdiyah"), 0.9),
            (_chunk("wasalt:2", "Villa B", 1500000, 300.0, 4, "Al Narjis"), 0.8),
        ]
        docs = collapse_to_docs(hits, top_k=5)
        deduped = dedup_docs(docs)
        assert len(deduped) == 2


class TestHtmlEntityCleanup:
    """HTML entities like &nbsp; should be decoded in render_card output."""

    def test_nbsp_decoded_in_snippet(self):
        doc = DocFields(
            doc_id="wasalt:1",
            kind=DocKind.listing,
            source=Source.wasalt,
            title="Villa",
            url="https://wasalt.sa/property/1",
            scraped_at=_NOW,
            purpose=Purpose.sale,
            property_type=PropertyType.villa,
            city="Riyadh",
            price_sar=1500000,
            bedrooms=3,
        )
        snippet = "Park&nbsp;-&nbsp;School&nbsp;-&nbsp;Mosque"
        result = render_card(doc, mode="context", n=1, snippet=snippet)
        assert "&nbsp;" not in result
        assert "Park" in result
        assert "School" in result

    def test_entities_decoded_in_embed_mode(self):
        doc = DocFields(
            doc_id="wasalt:1",
            kind=DocKind.listing,
            source=Source.wasalt,
            title="Villa &amp; Garden",
            url="https://wasalt.sa/property/1",
            scraped_at=_NOW,
            purpose=Purpose.sale,
            property_type=PropertyType.villa,
            city="Riyadh",
            price_sar=1500000,
        )
        result = render_card(doc, mode="embed")
        assert "&amp;" not in result

    def test_whitespace_collapsed(self):
        doc = DocFields(
            doc_id="wasalt:1",
            kind=DocKind.listing,
            source=Source.wasalt,
            title="Villa",
            url="https://wasalt.sa/property/1",
            scraped_at=_NOW,
            purpose=Purpose.sale,
            property_type=PropertyType.villa,
            city="Riyadh",
            price_sar=1500000,
        )
        snippet = "Park  -  School  -  Mosque"
        result = render_card(doc, mode="context", n=1, snippet=snippet)
        assert "  " not in result  # no double spaces
