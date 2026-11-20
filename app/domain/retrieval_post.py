"""Post-retrieval processing: doc collapse, cards, score floor.

Implements retrieval.md sections 3.5-3.6. Imports only common.* and stdlib (R4).
"""

from __future__ import annotations

import html
import re
import unicodedata

from common.cards import is_area_unreliable
from common.schemas import Card, ChunkPayload
from common.text import strip_instructions

# Pattern to strip area measurements like "3183 sqm", "1,666 sqm", "120.5 sqm"
_AREA_SQM_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*sqm\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K_DOCS = 5
SCORE_FLOOR = 0.52  # B10: score-floor refusal threshold (tuning.md decision 2)
TAU_DEFAULT = 0.93  # L2 semantic-cache tau (tuning.md decision 4; kept after B9, see report)


# ---------------------------------------------------------------------------
# DocHit: a collapsed document with its best chunk and optional extras
# ---------------------------------------------------------------------------


class DocHit:
    """A document after collapse: best chunk + optional extra chunks."""

    __slots__ = ("best_chunk", "best_score", "doc_id", "extra_chunks")

    def __init__(
        self,
        doc_id: str,
        best_score: float,
        best_chunk: ChunkPayload,
        extra_chunks: list[ChunkPayload] | None = None,
    ) -> None:
        self.doc_id = doc_id
        self.best_score = best_score
        self.best_chunk = best_chunk
        self.extra_chunks = extra_chunks or []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collapse_to_docs(
    hits: list[tuple[ChunkPayload, float]],
    top_k: int = TOP_K_DOCS,
    *,
    project_id: str | None = None,
) -> list[DocHit]:
    """Group hits by doc_id, keep the best chunk per doc, order by score.

    For project_info with a project_id: return one doc with up to 5 chunks.
    """
    if project_id:
        # Single-doc mode: collect all chunks for the project
        project_chunks: list[tuple[ChunkPayload, float]] = [
            (payload, score) for payload, score in hits if payload.doc_id == project_id
        ]
        if not project_chunks:
            return []
        project_chunks.sort(key=lambda x: x[1], reverse=True)
        best_payload, best_score = project_chunks[0]
        extras = [p for p, _s in project_chunks[1:5]]
        return [DocHit(project_id, best_score, best_payload, extras)]

    # General case: group by doc_id, keep best chunk per doc
    docs: dict[str, DocHit] = {}
    for payload, score in hits:
        did = payload.doc_id
        if did not in docs or score > docs[did].best_score:
            docs[did] = DocHit(did, score, payload)

    # Sort by score descending, take top_k
    sorted_docs = sorted(docs.values(), key=lambda d: d.best_score, reverse=True)
    return sorted_docs[:top_k]


def _dedup_key(dh: DocHit) -> tuple:
    """Normalised (title, price, area, bedrooms, district) for near-duplicate detection."""
    p = dh.best_chunk
    title = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", p.title)).strip().lower()
    return (title, p.price_sar, p.area_sqm, p.bedrooms, (p.district or "").lower())


def dedup_docs(docs: list[DocHit], top_k: int = TOP_K_DOCS) -> list[DocHit]:
    """Drop near-duplicate hits and backfill to top_k from the tail.

    A hit is a near-duplicate if its (normalised title, price, area, bedrooms,
    district) equals a higher-ranked hit's.  The input list is in score-descending
    order from collapse_to_docs.
    """
    seen: set[tuple] = set()
    kept: list[DocHit] = []
    deferred: list[DocHit] = []
    for dh in docs:
        key = _dedup_key(dh)
        if key in seen:
            continue
        seen.add(key)
        if len(kept) < top_k:
            kept.append(dh)
        else:
            deferred.append(dh)
    # Backfill if we dropped dups from the first top_k
    for dh in deferred:
        if len(kept) >= top_k:
            break
        kept.append(dh)
    return kept


def _clean_snippet(text: str | None) -> str | None:
    """Clean a card snippet: decode HTML entities, strip angle brackets, collapse whitespace, strip injections."""
    if not text:
        return text
    s = html.unescape(text)
    # S3: strip < and > after unescape to prevent </source> injection
    s = s.replace("<", "").replace(">", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = strip_instructions(s)
    return s[:400] if s else None


def to_cards(doc_hits: list[DocHit]) -> list[Card]:
    """Convert DocHits to numbered Cards (1..k).

    Applies display-time transformations:
    - MJ-5: nulls area_sqm on apartments when unreliable, strips from snippet
    - MN-4: replaces pipe with space in district for display
    """
    cards: list[Card] = []
    for i, dh in enumerate(doc_hits, start=1):
        payload = dh.best_chunk
        snippet = _clean_snippet(payload.text[:400] if payload.text else None)

        # MJ-5: null unreliable area on card and strip from snippet
        area = payload.area_sqm
        area_hide = is_area_unreliable(
            property_type=payload.property_type,
            area_sqm=payload.area_sqm,
            purpose=payload.purpose,
            price_sar=payload.price_sar,
        )
        if area_hide:
            area = None
            if snippet:
                snippet = _AREA_SQM_RE.sub("", snippet)
                snippet = re.sub(r",\s*,", ",", snippet)  # collapse double commas
                snippet = re.sub(r"\s+", " ", snippet).strip()
                snippet = snippet or None

        # MN-4: display-clean district (pipe -> space)
        district = payload.district
        if district and "|" in district:
            district = district.replace("|", " ")
            if snippet:
                snippet = snippet.replace("|", " ")

        card = Card(
            n=i,
            score=dh.best_score,
            snippet=snippet,
            doc_id=payload.doc_id,
            kind=payload.kind,
            source=payload.source,
            title=payload.title,
            url=payload.url,
            scraped_at=payload.scraped_at,
            purpose=payload.purpose,
            property_type=payload.property_type,
            city=payload.city,
            district=district,
            price_sar=payload.price_sar,
            rent_period=payload.rent_period,
            annual_rent_sar=payload.annual_rent_sar,
            bedrooms=payload.bedrooms,
            bathrooms=payload.bathrooms,
            area_sqm=area,
            country=payload.country,
            brand=payload.brand,
            handover=payload.handover,
            price_from_amount=payload.price_from_amount,
            price_from_currency=payload.price_from_currency,
            article_kind=payload.article_kind,
            published_at=payload.published_at,
        )
        cards.append(card)
    return cards


def below_floor(
    *,
    n_hits: int,
    filters_is_empty: bool,
    top_dense_score: float | None,
    floor: float = SCORE_FLOOR,
) -> bool:
    """Score-floor refusal check (retrieval.md section 3.6).

    Returns True when the query should be refused:
    - Zero hits
    - Empty filters and (no dense score or dense score below floor)
    """
    if n_hits == 0:
        return True
    return filters_is_empty and (top_dense_score is None or top_dense_score < floor)
