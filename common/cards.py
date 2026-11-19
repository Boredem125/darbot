"""Shared card renderer for search results.

Renders DocFields into text cards used in both the embedding index (embed mode)
and the LLM context window (context mode).

Imports only common.schemas and the stdlib.
"""

from __future__ import annotations

from typing import Literal

from common.schemas import DocFields, DocKind, Purpose
from common.text import strip_instructions


def _fmt_area(area: float) -> str:
    """Format area: int when whole, else one decimal."""
    if area == int(area):
        return str(int(area))
    return f"{area:.1f}"


def _render_listing(doc: DocFields) -> str:
    """Render a listing card."""
    parts: list[str] = []

    # Property type for sale/rent in district, city
    ptype = doc.property_type.value.capitalize() if doc.property_type else "Property"

    if doc.purpose == Purpose.sale:
        loc_parts = []
        if doc.district:
            loc_parts.append(doc.district)
        if doc.city:
            loc_parts.append(doc.city)
        location = ", ".join(loc_parts) if loc_parts else ""
        if location:
            parts.append(f"{ptype} for sale in {location}.")
        else:
            parts.append(f"{ptype} for sale.")
    elif doc.purpose == Purpose.rent:
        loc_parts = []
        if doc.district:
            loc_parts.append(doc.district)
        if doc.city:
            loc_parts.append(doc.city)
        location = ", ".join(loc_parts) if loc_parts else ""
        if location:
            parts.append(f"{ptype} for rent in {location}.")
        else:
            parts.append(f"{ptype} for rent.")

    # Bedrooms, bathrooms, area
    details: list[str] = []
    if doc.bedrooms is not None:
        details.append(f"{doc.bedrooms} bedrooms")
    if doc.bathrooms is not None:
        details.append(f"{doc.bathrooms} bathrooms")
    if doc.area_sqm is not None:
        details.append(f"{_fmt_area(doc.area_sqm)} sqm")
    if details:
        parts.append(", ".join(details) + ".")

    # Price
    if doc.purpose == Purpose.sale and doc.price_sar is not None:
        parts.append(f"Price SAR {doc.price_sar:,}.")
    elif doc.purpose == Purpose.rent and doc.price_sar is not None:
        if doc.rent_period and doc.annual_rent_sar:
            parts.append(
                f"Rent SAR {doc.price_sar:,} per {doc.rent_period.value}"
                f" (SAR {doc.annual_rent_sar:,} per year)."
            )
        elif doc.price_sar:
            parts.append(f"Rent SAR {doc.price_sar:,}.")

    return " ".join(parts)


def _render_project(doc: DocFields) -> str:
    """Render a project card."""
    parts: list[str] = []

    main = doc.title
    if doc.brand:
        main += f" by DarGlobal ({doc.brand} brand)"
    else:
        main += " by DarGlobal"

    loc_parts: list[str] = []
    if doc.city:
        loc_parts.append(doc.city)
    if doc.country:
        loc_parts.append(doc.country)
    if loc_parts:
        main += f" in {', '.join(loc_parts)}"

    parts.append(main + ".")

    if doc.handover:
        parts.append(f"Handover {doc.handover}.")

    if doc.price_from_amount is not None and doc.price_from_currency:
        parts.append(f"From {doc.price_from_currency} {doc.price_from_amount:,}.")

    return " ".join(parts)


def _render_article(doc: DocFields) -> str:
    """Render an article card."""
    parts: list[str] = [doc.title]
    meta: list[str] = []
    if doc.article_kind:
        meta.append(doc.article_kind.value)
    if doc.published_at:
        meta.append(doc.published_at.strftime("%Y-%m-%d"))
    if meta:
        parts[0] += f" ({', '.join(meta)})"
    return parts[0] + "."


def render_card(
    doc: DocFields,
    *,
    mode: Literal["embed", "context"],
    n: int | None = None,
    snippet: str | None = None,
) -> str:
    """Render a document card.

    embed mode: no [n], no URL, no id.
    context mode: [n] prefix, then the card, then URL and scraped_at.
    """
    if doc.kind == DocKind.listing:
        card = _render_listing(doc)
    elif doc.kind == DocKind.project:
        card = _render_project(doc)
    elif doc.kind == DocKind.article:
        card = _render_article(doc)
    else:
        card = doc.title

    # Append snippet (strip injected instructions in context mode)
    if snippet:
        if mode == "context":
            snippet = strip_instructions(snippet)
        card += " " + snippet

    if mode == "embed":
        return card

    # Context mode
    result = ""
    if n is not None:
        result = f"[{n}] "
    result += card
    result += f"\nURL: {doc.url}"
    result += f"\nAs of: {doc.scraped_at:%Y-%m-%d}"
    return result
