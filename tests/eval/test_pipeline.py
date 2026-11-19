"""Tests for the eval pipeline against fakes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.domain.ports import (
    AggregateOp,
    RetrievedHit,
    SqlResult,
)
from app.domain.router import EntityIndex
from common.schemas import (
    ChunkPayload,
    DocKind,
    Entities,
    Entity,
    EntityKind,
    Purpose,
    Source,
)
from scripts.evallib.pipeline import PipelineConfig, run_case
from tests.fakes.fake_adapters import (
    FakeEmbedder,
    FakeRetriever,
    FakeStructuredQuery,
)


def _make_entities() -> EntityIndex:
    return EntityIndex(Entities(entities=[
        Entity(kind=EntityKind.city, canonical="Riyadh", aliases=["riyadh"]),
        Entity(kind=EntityKind.city, canonical="Jeddah", aliases=["jeddah"]),
    ]))


def _ts() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def _make_chunk(doc_id: str = "wasalt:1", city: str = "Riyadh") -> ChunkPayload:
    return ChunkPayload(
        doc_id=doc_id,
        kind=DocKind.listing,
        source=Source.wasalt,
        title="Test Villa",
        url="https://example.com/1",
        scraped_at=_ts(),
        purpose=Purpose.sale,
        property_type="villa",
        city=city,
        chunk_index=0,
        text="A nice villa in " + city,
        price_sar=1500000,
        bedrooms=3,
    )


class TestPipelineLookup:
    def test_lookup_route(self):
        entities = _make_entities()
        embedder = FakeEmbedder()
        retriever = FakeRetriever()
        chunk = _make_chunk()
        retriever.set_hits(
            [RetrievedHit(payload=chunk, score=0.8)],
            top_dense_score=0.75,
        )
        config = PipelineConfig()
        case = {
            "id": "test-001",
            "turns": ["villas in Riyadh"],
            "expect": {"route": "lookup"},
        }
        result = asyncio.run(
            run_case(
                case,
                entities=entities,
                embedder=embedder,
                retriever=retriever,
                config=config,
            )
        )
        assert result["route"] == "lookup"
        assert len(result["cards"]) == 1
        assert result["refused"] is False

    def test_out_of_scope(self):
        entities = _make_entities()
        config = PipelineConfig()
        case = {
            "id": "test-002",
            "turns": ["What is the weather today?"],
            "expect": {"route": "out_of_scope"},
        }
        result = asyncio.run(
            run_case(
                case,
                entities=entities,
                config=config,
            )
        )
        assert result["route"] == "out_of_scope"
        assert result["refused"] is True

    def test_score_floor_refusal(self):
        entities = _make_entities()
        embedder = FakeEmbedder()
        retriever = FakeRetriever()
        # Set low dense score and empty filters (no entity or filter cues in query)
        retriever.set_hits(
            [RetrievedHit(payload=_make_chunk(), score=0.1)],
            top_dense_score=0.15,
        )
        config = PipelineConfig(score_floor=0.30)
        # Use a query that has a domain cue to avoid out_of_scope,
        # but no filters to trigger score floor
        case = {
            "id": "test-003",
            "turns": ["property listings"],
            "expect": {"route": "lookup"},
        }
        result = asyncio.run(
            run_case(
                case,
                entities=entities,
                embedder=embedder,
                retriever=retriever,
                config=config,
            )
        )
        # With empty filters and low score, should be refused
        assert result["refused"] is True


class TestPipelineAggregate:
    def test_aggregate_route(self):
        entities = _make_entities()
        sq = FakeStructuredQuery()
        sq.set_result(
            AggregateOp.count,
            SqlResult(op=AggregateOp.count, value=42, unit="count", n_matched=42),
        )
        config = PipelineConfig()
        case = {
            "id": "test-agg-001",
            "turns": ["how many villas in Riyadh"],
            "expect": {"route": "aggregate", "op": "count"},
        }
        result = asyncio.run(
            run_case(
                case,
                entities=entities,
                structured_query=sq,
                config=config,
            )
        )
        assert result["route"] == "aggregate"
        assert result["sql_value"] == 42


class TestPipelineMultiturn:
    def test_session_carry_over(self):
        entities = _make_entities()
        embedder = FakeEmbedder()
        retriever = FakeRetriever()
        chunk = _make_chunk()
        retriever.set_hits(
            [RetrievedHit(payload=chunk, score=0.8)],
            top_dense_score=0.75,
        )
        config = PipelineConfig()
        # Two turns: first sets city, second is a follow-up
        case = {
            "id": "test-mt-001",
            "turns": ["villas in Riyadh", "cheaper ones?"],
            "expect": {"route": "lookup"},
        }
        result = asyncio.run(
            run_case(
                case,
                entities=entities,
                embedder=embedder,
                retriever=retriever,
                config=config,
            )
        )
        # Should route to lookup (the follow-up should carry over)
        assert result["route"] == "lookup"
