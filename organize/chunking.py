"""Chunking and embedding for the search index.

Follows retrieval.md section 2 exactly:
  - Token counter uses the dense model's tokenizer.json with no_truncation/no_padding
  - Listings: one chunk = embed-mode card + description, capped at 126 tokens
  - Projects: chunk 0 = card + amenities/unit types/payment plan, chunks 1..n = page text
  - Articles: header + body chunks, max 8 chunks
  - Hard rule: every embedded text <= 126 tokens
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from common.cards import render_card
from common.schemas import (
    Article,
    ChunkPayload,
    DocFields,
    DocKind,
    Listing,
    Project,
    Source,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 126
HEADER_MAX_TOKENS = 28
MAX_ARTICLE_CHUNKS = 8
MAX_PROJECT_CHUNKS = 12


def _find_tokenizer_path() -> Path:
    """Find the tokenizer.json for the dense model."""
    cache_dir = os.environ.get(
        "FASTEMBED_CACHE_PATH",
        str(Path.home() / ".cache" / "fastembed"),
    )
    base = Path(cache_dir)
    model_dir = base / "models--Qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
    # Find the snapshot
    snapshots = model_dir / "snapshots"
    if snapshots.exists():
        for d in snapshots.iterdir():
            tok = d / "tokenizer.json"
            if tok.exists():
                return tok
    raise FileNotFoundError(
        f"tokenizer.json not found under {model_dir}. "
        "Set FASTEMBED_CACHE_PATH to the fastembed cache directory."
    )


def build_token_counter(tokenizer_path: Path | None = None) -> Callable[[str], int]:
    """Build a token counter using the dense model's tokenizer.

    Returns a function text -> token_count.
    CRITICAL: Calls no_truncation() and no_padding() so counts are accurate.
    Without no_truncation(), every count silently caps at 128.
    """
    from tokenizers import Tokenizer

    path = tokenizer_path or _find_tokenizer_path()
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def _trim_at_word_boundary(text: str, token_count: Callable[[str], int],
                           max_tokens: int) -> str:
    """Trim text at a word boundary to fit within max_tokens."""
    if token_count(text) <= max_tokens:
        return text
    words = text.split()
    # Binary search for the right cutoff
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = " ".join(words[:mid])
        if token_count(candidate) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo]) if lo > 0 else ""


def _make_header(text: str, token_count: Callable[[str], int]) -> str:
    """Create a header, capped at HEADER_MAX_TOKENS."""
    return _trim_at_word_boundary(text, token_count, HEADER_MAX_TOKENS)


def _doc_fields_from_listing(listing: Listing) -> DocFields:
    """Create DocFields from a Listing."""
    return DocFields(
        doc_id=listing.id,
        kind=DocKind.listing,
        source=Source.wasalt,
        title=listing.title,
        url=listing.url,
        scraped_at=listing.scraped_at,
        purpose=listing.purpose,
        property_type=listing.property_type,
        city=listing.city,
        district=listing.district,
        price_sar=listing.price_sar,
        rent_period=listing.rent_period,
        annual_rent_sar=listing.annual_rent_sar,
        bedrooms=listing.bedrooms,
        bathrooms=listing.bathrooms,
        area_sqm=listing.area_sqm,
    )


def _doc_fields_from_project(project: Project) -> DocFields:
    """Create DocFields from a Project."""
    return DocFields(
        doc_id=project.id,
        kind=DocKind.project,
        source=Source.darglobal,
        title=project.name,
        url=project.url,
        scraped_at=project.scraped_at,
        city=project.city,
        country=project.country,
        brand=project.brand,
        handover=project.handover,
        price_from_amount=project.price_from_amount,
        price_from_currency=project.price_from_currency,
    )


def _doc_fields_from_article(article: Article) -> DocFields:
    """Create DocFields from an Article."""
    return DocFields(
        doc_id=article.id,
        kind=DocKind.article,
        source=Source.darglobal,
        title=article.title,
        url=article.url,
        scraped_at=article.scraped_at,
        article_kind=article.kind,
        published_at=article.published_at,
    )


def chunk_listing(
    listing: Listing,
    token_count: Callable[[str], int],
) -> list[ChunkPayload]:
    """Chunk a listing into exactly one chunk.

    The chunk is the embed-mode card + description. If over 126 tokens,
    the description is trimmed at a word boundary.
    """
    doc_fields = _doc_fields_from_listing(listing)
    card_text = render_card(doc_fields, mode="embed")

    # Add description
    full_text = card_text
    if listing.description:
        full_text = card_text + " " + listing.description

    # Check token count and trim description if needed
    if token_count(full_text) > MAX_TOKENS and listing.description:
        # Calculate tokens available for description
        card_tokens = token_count(card_text + " ")
        desc_budget = MAX_TOKENS - card_tokens
        if desc_budget > 0:
            trimmed_desc = _trim_at_word_boundary(
                listing.description, token_count, desc_budget
            )
            if trimmed_desc:
                full_text = card_text + " " + trimmed_desc
            else:
                full_text = card_text
        else:
            full_text = card_text

    return [
        ChunkPayload(
            **doc_fields.model_dump(),
            chunk_index=0,
            text=full_text,
        )
    ]


def chunk_project(
    project: Project,
    token_count: Callable[[str], int],
) -> list[ChunkPayload]:
    """Chunk a project. chunk 0 = card + details, chunks 1..n = body text."""
    doc_fields = _doc_fields_from_project(project)
    card_text = render_card(doc_fields, mode="embed")

    # Chunk 0: card plus amenities, unit types and payment plan
    extras: list[str] = []
    if project.unit_types:
        extras.append("Unit types: " + ", ".join(project.unit_types) + ".")
    if project.amenities:
        extras.append("Amenities: " + ", ".join(project.amenities[:10]) + ".")
    if project.payment_plan:
        extras.append("Payment plan: " + project.payment_plan + ".")

    chunk0_text = card_text
    if extras:
        extra_text = " ".join(extras)
        candidate = card_text + " " + extra_text
        if token_count(candidate) <= MAX_TOKENS:
            chunk0_text = candidate
        else:
            # Trim extras to fit
            chunk0_text = _trim_at_word_boundary(candidate, token_count, MAX_TOKENS)

    chunks: list[ChunkPayload] = [
        ChunkPayload(
            **doc_fields.model_dump(),
            chunk_index=0,
            text=chunk0_text,
        )
    ]

    # Chunks 1..n from body text
    if project.body:
        from organize.sanitise import sanitise_text

        body = sanitise_text(project.body)
        header_raw = f"{project.name} ({project.city or ''}, {project.country or ''}). "
        header = _make_header(header_raw, token_count)

        body_chunks = _split_text(body, token_count, header)
        for i, chunk_text in enumerate(body_chunks[: MAX_PROJECT_CHUNKS - 1]):
            full = header + chunk_text if header else chunk_text
            # Ensure under limit
            if token_count(full) > MAX_TOKENS:
                full = _trim_at_word_boundary(full, token_count, MAX_TOKENS)
            chunks.append(
                ChunkPayload(
                    **doc_fields.model_dump(),
                    chunk_index=i + 1,
                    text=full,
                )
            )

    return chunks


def chunk_article(
    article: Article,
    token_count: Callable[[str], int],
) -> list[ChunkPayload]:
    """Chunk an article. header + body chunks, max 8."""
    doc_fields = _doc_fields_from_article(article)

    from organize.sanitise import sanitise_text

    body = sanitise_text(article.body)
    header_raw = f"{article.title}. "
    header = _make_header(header_raw, token_count)

    body_chunks = _split_text(body, token_count, header)

    chunks: list[ChunkPayload] = []
    for i, chunk_text in enumerate(body_chunks[:MAX_ARTICLE_CHUNKS]):
        full = header + chunk_text if header else chunk_text
        if token_count(full) > MAX_TOKENS:
            full = _trim_at_word_boundary(full, token_count, MAX_TOKENS)
        chunks.append(
            ChunkPayload(
                **doc_fields.model_dump(),
                chunk_index=i,
                text=full,
            )
        )

    return chunks


def _split_text(
    text: str,
    token_count: Callable[[str], int],
    header: str,
) -> list[str]:
    """Split text into chunks using SentenceSplitter.

    Each chunk will have the header prepended, so the chunk_size accounts
    for the header token budget.
    """
    from llama_index.core.node_parser import SentenceSplitter

    chunk_size = 96  # max tokens per chunk body
    chunk_overlap = 16

    # Create a tokenizer function for SentenceSplitter
    def tokenizer_fn(text: str) -> list[str]:
        """Return a list of token strings (SentenceSplitter uses len())."""
        n = token_count(text)
        return ["t"] * n

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tokenizer_fn,
    )

    nodes = splitter.split_text(text)
    return nodes
