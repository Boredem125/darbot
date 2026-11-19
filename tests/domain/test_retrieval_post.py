"""Tests for retrieval post-processing."""

from datetime import UTC, datetime

from app.domain.retrieval_post import (
    below_floor,
    collapse_to_docs,
    to_cards,
)
from common.schemas import ChunkPayload, DocKind, Source


def _make_chunk(
    doc_id: str = "wasalt:1",
    chunk_index: int = 0,
    kind: DocKind = DocKind.listing,
) -> ChunkPayload:
    return ChunkPayload(
        doc_id=doc_id,
        kind=kind,
        source=Source.wasalt,
        title=f"Listing {doc_id}",
        url=f"https://wasalt.sa/en/property/{doc_id}",
        scraped_at=datetime(2026, 11, 18, tzinfo=UTC),
        chunk_index=chunk_index,
        text=f"Chunk {chunk_index} of {doc_id}",
        purpose="sale",
        property_type="villa",
        city="Riyadh",
    )


class TestCollapseToDoc:
    def test_basic_collapse(self):
        hits = [
            (_make_chunk("wasalt:1", 0), 0.9),
            (_make_chunk("wasalt:1", 1), 0.8),
            (_make_chunk("wasalt:2", 0), 0.85),
        ]
        docs = collapse_to_docs(hits, top_k=5)
        assert len(docs) == 2
        assert docs[0].doc_id == "wasalt:1"
        assert docs[0].best_score == 0.9
        assert docs[1].doc_id == "wasalt:2"

    def test_top_k_limit(self):
        hits = [(_make_chunk(f"wasalt:{i}", 0), 1.0 - i * 0.1) for i in range(10)]
        docs = collapse_to_docs(hits, top_k=3)
        assert len(docs) == 3

    def test_project_mode(self):
        project_id = "darglobal:project:test"
        hits = [
            (_make_chunk(project_id, 0, DocKind.project), 0.9),
            (_make_chunk(project_id, 1, DocKind.project), 0.7),
            (_make_chunk(project_id, 2, DocKind.project), 0.6),
            (_make_chunk("wasalt:other", 0), 0.95),  # different doc, ignored
        ]
        docs = collapse_to_docs(hits, project_id=project_id)
        assert len(docs) == 1
        assert docs[0].doc_id == project_id
        assert len(docs[0].extra_chunks) == 2


class TestToCards:
    def test_numbering(self):
        hits = [
            (_make_chunk("wasalt:1", 0), 0.9),
            (_make_chunk("wasalt:2", 0), 0.8),
        ]
        docs = collapse_to_docs(hits)
        cards = to_cards(docs)
        assert cards[0].n == 1
        assert cards[1].n == 2

    def test_snippet_truncated(self):
        chunk = _make_chunk("wasalt:1", 0)
        # Override with long text
        long_chunk = ChunkPayload(
            doc_id=chunk.doc_id, kind=chunk.kind, source=chunk.source,
            title=chunk.title, url=chunk.url, scraped_at=chunk.scraped_at,
            chunk_index=0, text="x" * 500,
            purpose=chunk.purpose, property_type=chunk.property_type, city=chunk.city,
        )
        from app.domain.retrieval_post import DocHit
        docs = [DocHit("wasalt:1", 0.9, long_chunk)]
        cards = to_cards(docs)
        assert len(cards[0].snippet) <= 400


class TestBelowFloor:
    def test_zero_hits(self):
        assert below_floor(n_hits=0, filters_is_empty=True, top_dense_score=None) is True

    def test_empty_filters_low_score(self):
        assert below_floor(n_hits=5, filters_is_empty=True, top_dense_score=0.1) is True

    def test_empty_filters_high_score(self):
        # Score above SCORE_FLOOR (0.52) -> not refused
        assert below_floor(n_hits=5, filters_is_empty=True, top_dense_score=0.55) is False

    def test_has_filters_low_score(self):
        """With filters present, the floor does not apply."""
        assert below_floor(n_hits=5, filters_is_empty=False, top_dense_score=0.1) is False

    def test_empty_filters_no_score(self):
        assert below_floor(n_hits=5, filters_is_empty=True, top_dense_score=None) is True
