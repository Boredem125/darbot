"""Fake adapters for testing. Each implements its Protocol from app.domain.ports."""

from tests.fakes.fake_adapters import (
    FakeCache,
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    FakeRetriever,
    FakeStructuredQuery,
)

__all__ = [
    "FakeCache",
    "FakeEmbedder",
    "FakeLLM",
    "FakeReranker",
    "FakeRetriever",
    "FakeStructuredQuery",
]
