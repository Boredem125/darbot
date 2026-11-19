"""Tests that each fake implements its Protocol from ports.py."""


from app.domain.ports import (
    LLM,
    Cache,
    Embedder,
    Reranker,
    Retriever,
    StructuredQuery,
)
from tests.fakes import (
    FakeCache,
    FakeEmbedder,
    FakeLLM,
    FakeReranker,
    FakeRetriever,
    FakeStructuredQuery,
)


class TestFakeProtocolConformance:
    """Assert isinstance(fake, Protocol) for each fake adapter."""

    def test_embedder(self):
        assert isinstance(FakeEmbedder(), Embedder)

    def test_retriever(self):
        assert isinstance(FakeRetriever(), Retriever)

    def test_reranker(self):
        assert isinstance(FakeReranker(), Reranker)

    def test_structured_query(self):
        assert isinstance(FakeStructuredQuery(), StructuredQuery)

    def test_llm(self):
        assert isinstance(FakeLLM(), LLM)

    def test_cache(self):
        assert isinstance(FakeCache(), Cache)
