"""Cache key computation (caching.md section 2).

Pure, deterministic functions. Imports only common.* and stdlib (R4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from common.schemas import Language, Route

CACHE_KEY_SCHEMA = "ck2"


@dataclass(frozen=True, slots=True)
class CacheKeyParts:
    index_version: str
    prompt_version: str
    language: Language
    route: Route
    op: str | None  # AggregateOp value for aggregate, else None
    filters_json: str  # SearchFilters.canonical_json()
    numbers: tuple[str, ...]  # canonical numbers in order of appearance
    negated: bool
    query_key_text: str  # normalise_for_key(query)
    amenity_tokens: tuple[str, ...] = ()  # B9: sorted canonical amenity/view tokens


def scope_preimage(p: CacheKeyParts) -> str:
    """Build the preimage string for the scope hash."""
    return "\x1f".join([
        CACHE_KEY_SCHEMA,
        p.index_version,
        p.prompt_version,
        p.language,
        p.route,
        p.op or "-",
        p.filters_json,
        ",".join(p.numbers),
        "neg" if p.negated else "pos",
        ",".join(p.amenity_tokens),  # B9: amenity/view tokens in scope
    ])


def build_scope(p: CacheKeyParts) -> str:
    """Build the L2 scope string."""
    prefix = CACHE_KEY_SCHEMA + ":s:"
    return prefix + hashlib.sha256(scope_preimage(p).encode()).hexdigest()[:32]


def build_key(p: CacheKeyParts) -> str:
    """Build the exact cache key (L0, L1)."""
    full = scope_preimage(p) + "\x1e" + p.query_key_text
    prefix = CACHE_KEY_SCHEMA + ":k:"
    return prefix + hashlib.sha256(full.encode()).hexdigest()[:32]


def parts_from(
    decision: object,  # RouteDecision (avoid circular import by using duck typing)
    query: str,
    index_version: str,
    prompt_version: str,
) -> CacheKeyParts:
    """Build CacheKeyParts from a RouteDecision and the raw query."""
    from common.text import extract_amenity_tokens, normalise_for_key

    nk = normalise_for_key(query)
    return CacheKeyParts(
        index_version=index_version,
        prompt_version=prompt_version,
        language=decision.language,  # type: ignore[attr-defined]
        route=decision.route,  # type: ignore[attr-defined]
        op=decision.op.value if decision.op is not None else None,  # type: ignore[attr-defined]
        filters_json=decision.filters.canonical_json(),  # type: ignore[attr-defined]
        numbers=decision.numbers,  # type: ignore[attr-defined]
        negated=decision.negated,  # type: ignore[attr-defined]
        query_key_text=nk,
        amenity_tokens=extract_amenity_tokens(nk),
    )
