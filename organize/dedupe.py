"""Near-duplicate listing deduplication.

Blocking key: (purpose, property_type, city, district, price_sar, bedrooms, round(area_sqm / 5)).
Within a block, two listings are duplicates if both descriptions are null, or
the token-set Jaccard of their normalised descriptions is >= JACCARD_THRESHOLD.
Keep the one with the latest published_at (tie: smallest source_id).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from common.schemas import Listing

logger = logging.getLogger(__name__)

JACCARD_THRESHOLD = 0.8


def _blocking_key(listing: Listing) -> tuple:
    """Compute the blocking key for a listing."""
    area_bucket = round(listing.area_sqm / 5) if listing.area_sqm else None
    return (
        listing.purpose,
        listing.property_type,
        listing.city,
        listing.district,
        listing.price_sar,
        listing.bedrooms,
        area_bucket,
    )


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase word tokens."""
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Token-set Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _pick_keeper(a: Listing, b: Listing) -> Listing:
    """Given two duplicate listings, pick the one to keep.

    Prefer latest published_at, then smallest source_id.
    """
    a_pub = a.published_at
    b_pub = b.published_at
    if a_pub is not None and b_pub is not None:
        if a_pub > b_pub:
            return a
        if b_pub > a_pub:
            return b
    elif a_pub is not None:
        return a
    elif b_pub is not None:
        return b
    # Tie: smallest source_id
    return a if a.source_id <= b.source_id else b


def dedupe(
    listings: list[Listing],
    threshold: float = JACCARD_THRESHOLD,
) -> tuple[list[Listing], list[tuple[str, str, str, str]]]:
    """Deduplicate listings.

    Returns (deduplicated_listings, merged_pairs).
    merged_pairs is a list of (kept_id, kept_title, dropped_id, dropped_title).
    """
    # Group by blocking key
    blocks: dict[tuple, list[Listing]] = defaultdict(list)
    for listing in listings:
        key = _blocking_key(listing)
        blocks[key].append(listing)

    kept: list[Listing] = []
    merged_pairs: list[tuple[str, str, str, str]] = []

    for block in blocks.values():
        if len(block) == 1:
            kept.append(block[0])
            continue

        # Sort block by published_at desc, then source_id asc for determinism
        block.sort(
            key=lambda x: (
                x.published_at or "",
                x.source_id,
            ),
            reverse=True,
        )

        # Mark duplicates within the block
        is_dup = [False] * len(block)
        tokens = [
            _tokenize(b.description) if b.description else None
            for b in block
        ]

        for i in range(len(block)):
            if is_dup[i]:
                continue
            for j in range(i + 1, len(block)):
                if is_dup[j]:
                    continue

                # Both null descriptions: duplicate
                if tokens[i] is None and tokens[j] is None:
                    is_dup[j] = True
                    keeper = _pick_keeper(block[i], block[j])
                    dropped = block[j] if keeper is block[i] else block[i]
                    merged_pairs.append(
                        (keeper.id, keeper.title, dropped.id, dropped.title)
                    )
                    continue

                # One null, one not: not duplicate
                if tokens[i] is None or tokens[j] is None:
                    continue

                sim = _jaccard(tokens[i], tokens[j])
                if sim >= threshold:
                    is_dup[j] = True
                    keeper = _pick_keeper(block[i], block[j])
                    dropped = block[j] if keeper is block[i] else block[i]
                    merged_pairs.append(
                        (keeper.id, keeper.title, dropped.id, dropped.title)
                    )

        for i, listing in enumerate(block):
            if not is_dup[i]:
                kept.append(listing)

    input_count = len(listings)
    output_count = len(kept)
    dedupe_rate = 1 - output_count / input_count if input_count > 0 else 0

    logger.info(
        "Dedupe: %d -> %d (%.1f%% removed, %d pairs)",
        input_count, output_count, dedupe_rate * 100, len(merged_pairs),
    )

    # Sample 20 random pairs for audit
    audit_pairs = random.sample(merged_pairs, min(20, len(merged_pairs)))
    for kept_id, kept_title, dropped_id, dropped_title in audit_pairs:
        logger.info(
            "  Merge: kept %s (%s) dropped %s (%s)",
            kept_id, kept_title[:50], dropped_id, dropped_title[:50],
        )

    return kept, merged_pairs
