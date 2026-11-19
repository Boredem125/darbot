"""Tests for organize.dedupe."""

from __future__ import annotations

from common.schemas import Listing, PropertyType, Purpose, listing_id
from organize.dedupe import dedupe


def _make_listing(
    source_id: str,
    purpose: str = "sale",
    property_type: str = "apartment",
    city: str = "Riyadh",
    district: str | None = "Al Narjis",
    price_sar: int | None = 500000,
    bedrooms: int | None = 3,
    area_sqm: float | None = 100.0,
    description: str | None = "Nice apartment in Al Narjis area of Riyadh",
    published_at: str | None = "2026-11-10T08:00:00+00:00",
) -> Listing:
    return Listing(
        id=listing_id(source_id),
        source="wasalt",
        source_id=source_id,
        url=f"https://wasalt.sa/property/{purpose}/test-{source_id}",
        title=f"Test Listing {source_id}",
        description=description,
        purpose=Purpose(purpose),
        property_type=PropertyType(property_type),
        property_type_raw=None,
        city=city,
        district=district,
        price_sar=price_sar,
        rent_period=None,
        annual_rent_sar=None,
        bedrooms=bedrooms,
        bathrooms=None,
        area_sqm=area_sqm,
        lat=None,
        lon=None,
        amenities=[],
        published_at=published_at,
        scraped_at="2026-11-18T10:00:00+00:00",
    )


class TestDedupe:
    def test_duplicate_pair(self) -> None:
        """Two listings with same blocking key and similar descriptions are merged."""
        a = _make_listing("100", description="Nice apartment near park with great views")
        b = _make_listing("101", description="Nice apartment near park with great views")
        result, pairs = dedupe([a, b])
        assert len(result) == 1
        assert len(pairs) == 1

    def test_near_miss_different_price(self) -> None:
        """Different price means different blocking key, so not merged."""
        a = _make_listing("200", price_sar=500000)
        b = _make_listing("201", price_sar=600000)
        result, pairs = dedupe([a, b])
        assert len(result) == 2
        assert len(pairs) == 0

    def test_both_null_descriptions(self) -> None:
        """Two listings with null descriptions and same key are merged."""
        a = _make_listing("300", description=None)
        b = _make_listing("301", description=None)
        result, pairs = dedupe([a, b])
        assert len(result) == 1
        assert len(pairs) == 1

    def test_one_null_one_not(self) -> None:
        """One null description, one not: not merged."""
        a = _make_listing("400", description=None)
        b = _make_listing("401", description="Some text here about the apartment")
        result, pairs = dedupe([a, b])
        assert len(result) == 2
        assert len(pairs) == 0

    def test_different_descriptions_not_merged(self) -> None:
        """Very different descriptions should not be merged."""
        a = _make_listing("500", description="Luxury penthouse with sea views and pool access")
        b = _make_listing("501", description="Small studio near metro station basic furnishing")
        result, pairs = dedupe([a, b])
        assert len(result) == 2
        assert len(pairs) == 0

    def test_keeps_latest_published(self) -> None:
        """Keeps the listing with the latest published_at."""
        a = _make_listing("600", published_at="2026-11-01T00:00:00+00:00",
                          description="Same text for both listings here")
        b = _make_listing("601", published_at="2026-11-15T00:00:00+00:00",
                          description="Same text for both listings here")
        result, _pairs = dedupe([a, b])
        assert len(result) == 1
        assert result[0].source_id == "601"  # later published
