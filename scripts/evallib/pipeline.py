"""Eval pipeline: run each case through the domain functions.

Composes: router -> (embedder + retriever + optional reranker) | structured_query,
retrieval_post, score-floor check, card rendering. For LLM mode: prompting, LLM stream,
grounding.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from app.domain.messages import REFUSAL_AR, REFUSAL_EN
from app.domain.retrieval_post import (
    TOP_K_DOCS,
    below_floor,
    collapse_to_docs,
    to_cards,
)
from app.domain.router import EntityIndex, RouteDecision, SessionSlots, route
from common.schemas import (
    Card,
    Language,
    Route,
    SearchFilters,
)


@runtime_checkable
class EvalEmbedder(Protocol):
    async def embed_query(self, text: str) -> Any: ...


@runtime_checkable
class EvalRetriever(Protocol):
    async def search(self, embedding: Any, filters: SearchFilters, *, limit: int = 20) -> Any: ...


@runtime_checkable
class EvalReranker(Protocol):
    async def rerank(self, query: str, hits: Any, *, top_k: int) -> list: ...


@runtime_checkable
class EvalStructuredQuery(Protocol):
    async def aggregate(self, query: Any) -> Any: ...


@runtime_checkable
class EvalLLM(Protocol):
    def stream(self, messages: Any, *, max_tokens: int, temperature: float = 0.0) -> Any: ...


@dataclass
class PipelineConfig:
    """Tuning knobs exposed via CLI flags."""
    rrf_k: int = 60
    list_size: int = 20
    top_k_docs: int = TOP_K_DOCS
    score_floor: float = 0.52
    rerank_enabled: bool = False
    tau: float = 0.92
    llm_mode: bool = False
    system_prompt: str = ""
    budget: int = 0


@dataclass
class PipelineResult:
    """Result of running one eval case through the pipeline."""
    case_id: str = ""
    route: str = ""
    filters: dict = field(default_factory=dict)
    cards: list[dict] = field(default_factory=list)
    refused: bool = False
    sql_value: Any = None
    answer: str = ""
    grounding: dict | None = None
    error: str | None = None


def run_case_sync(
    case: dict,
    *,
    entities: EntityIndex,
    embedder: EvalEmbedder | None = None,
    retriever: EvalRetriever | None = None,
    reranker: EvalReranker | None = None,
    structured_query: EvalStructuredQuery | None = None,
    llm: EvalLLM | None = None,
    config: PipelineConfig,
) -> dict:
    """Run a single case synchronously (wraps the async pipeline)."""
    return asyncio.get_event_loop().run_until_complete(
        run_case(
            case,
            entities=entities,
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            structured_query=structured_query,
            llm=llm,
            config=config,
        )
    )


async def run_case(
    case: dict,
    *,
    entities: EntityIndex,
    embedder: EvalEmbedder | None = None,
    retriever: EvalRetriever | None = None,
    reranker: EvalReranker | None = None,
    structured_query: EvalStructuredQuery | None = None,
    llm: EvalLLM | None = None,
    config: PipelineConfig,
) -> dict:
    """Run one eval case through the pipeline. Returns a dict with predicted fields."""
    turns = case.get("turns", [])
    if not turns:
        return {"error": "no turns"}

    session: SessionSlots | None = None
    decision: RouteDecision | None = None

    # Process each turn, carrying over session state
    for i, turn_text in enumerate(turns):
        decision = route(
            turn_text,
            session=session,
            entities=entities,
            lang_hint=None,
        )
        # Build session for next turn
        session = SessionSlots(
            filters=decision.filters,
            last_prices=(),
            last_areas=(),
            history=(),
        )

    if decision is None:
        return {"error": "no decision"}

    result: dict[str, Any] = {
        "case_id": case.get("id", ""),
        "route": decision.route.value,
        "filters": decision.filters.model_dump(exclude_none=True, exclude={"sort"}),
        "cards": [],
        "refused": False,
        "sql_value": None,
        "answer": "",
        "grounding": None,
    }

    # Out-of-scope -> refusal
    if decision.route == Route.out_of_scope:
        result["refused"] = True
        refusal = REFUSAL_AR if decision.language == Language.ar else REFUSAL_EN
        result["answer"] = refusal
        return result

    # Aggregate route
    if decision.route == Route.aggregate and structured_query is not None:
        from app.domain.ports import AggregateQuery

        agg_query = AggregateQuery(
            op=decision.op,
            filters=decision.filters,
            example_limit=decision.example_limit,
        )
        try:
            sql_result = await structured_query.aggregate(agg_query)
            result["sql_value"] = sql_result.value
            result["_sql_result"] = sql_result  # H-3: keep full SqlResult for LLM step
            # Convert examples to cards
            if sql_result.examples:
                from app.domain.retrieval_post import DocHit
                doc_hits = []
                for ex in sql_result.examples:
                    from common.schemas import ChunkPayload, DocKind, Source
                    payload = ChunkPayload(
                        doc_id=ex.id,
                        kind=DocKind.listing,
                        source=Source.wasalt,
                        title=ex.title,
                        url=ex.url,
                        scraped_at=ex.scraped_at,
                        purpose=ex.purpose,
                        property_type=ex.property_type,
                        city=ex.city,
                        district=ex.district,
                        price_sar=ex.price_sar,
                        rent_period=ex.rent_period,
                        annual_rent_sar=ex.annual_rent_sar,
                        bedrooms=ex.bedrooms,
                        bathrooms=ex.bathrooms,
                        area_sqm=ex.area_sqm,
                        chunk_index=0,
                        text=ex.title,
                    )
                    doc_hits.append(DocHit(ex.id, 1.0, payload))
                cards = to_cards(doc_hits)
                result["cards"] = [_card_to_dict(c) for c in cards]
        except (ValueError, TypeError, RuntimeError) as e:
            result["error"] = str(e)
        return result

    # Lookup / project_info route: embed + retrieve
    if embedder is not None and retriever is not None:
        query_text = turns[-1]
        try:
            embedding = await embedder.embed_query(query_text)
            retrieval = await retriever.search(
                embedding, decision.filters, limit=config.list_size
            )

            # Score floor check
            if below_floor(
                n_hits=len(retrieval.hits),
                filters_is_empty=decision.filters.is_empty(),
                top_dense_score=retrieval.top_dense_score,
                floor=config.score_floor,
            ):
                result["refused"] = True
                refusal = REFUSAL_AR if decision.language == Language.ar else REFUSAL_EN
                result["answer"] = refusal
                return result

            # Collapse to docs and cards
            hits_for_collapse = [
                (hit.payload, hit.score) for hit in retrieval.hits
            ]
            project_id = (
                decision.filters.project_id
                if decision.route == Route.project_info
                else None
            )
            # H2: collapse to 10 when reranking (retrieval.md section 3.7),
            # then rerank keeps top_k_docs.  Without rerank, collapse
            # directly to top_k_docs.
            collapse_k = (
                10
                if config.rerank_enabled and reranker is not None
                else config.top_k_docs
            )
            doc_hits = collapse_to_docs(
                hits_for_collapse,
                top_k=collapse_k,
                project_id=project_id,
            )

            # Optional rerank
            if config.rerank_enabled and reranker is not None and doc_hits:
                from app.domain.ports import RetrievedHit
                rh_list = []
                for dh in doc_hits:
                    rh_list.append(RetrievedHit(payload=dh.best_chunk, score=dh.best_score))
                reranked = await reranker.rerank(
                    query_text, rh_list, top_k=config.top_k_docs
                )
                from app.domain.retrieval_post import DocHit
                doc_hits = [
                    DocHit(rh.payload.doc_id, rh.score, rh.payload)
                    for rh in reranked
                ]

            cards = to_cards(doc_hits)
            result["cards"] = [_card_to_dict(c) for c in cards]

            # Populate last_prices/areas in session for downstream
            prices = tuple(c.price_sar for c in cards if c.price_sar is not None)
            areas = tuple(c.area_sqm for c in cards if c.area_sqm is not None)
            result["_prices"] = prices
            result["_areas"] = areas

        except (ValueError, TypeError, RuntimeError) as e:
            result["error"] = str(e)

    return result


def _card_to_dict(card: Card) -> dict:
    """Convert a Card to a plain dict for metric evaluation."""
    return card.model_dump(exclude_none=True)
