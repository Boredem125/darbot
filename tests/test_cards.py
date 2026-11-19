"""Tests for common.cards.render_card."""

from __future__ import annotations

from common.cards import render_card
from common.schemas import (
    ArticleKind,
    DocFields,
    DocKind,
    PropertyType,
    Purpose,
    RentPeriod,
    Source,
)


def _listing_sale() -> DocFields:
    return DocFields(
        doc_id="wasalt:111",
        kind=DocKind.listing,
        source=Source.wasalt,
        title="Villa for Sale",
        url="https://wasalt.sa/property/sale/villa-111",
        scraped_at="2026-11-18T10:00:00+00:00",
        purpose=Purpose.sale,
        property_type=PropertyType.villa,
        city="Riyadh",
        district="Al Narjis",
        price_sar=1950000,
        bedrooms=5,
        bathrooms=3,
        area_sqm=333.0,
    )


def _listing_rent() -> DocFields:
    return DocFields(
        doc_id="wasalt:222",
        kind=DocKind.listing,
        source=Source.wasalt,
        title="Apartment for Rent",
        url="https://wasalt.sa/property/rent/apt-222",
        scraped_at="2026-11-18T10:00:00+00:00",
        purpose=Purpose.rent,
        property_type=PropertyType.apartment,
        city="Jeddah",
        district="Al Hamra",
        price_sar=3000,
        rent_period=RentPeriod.monthly,
        annual_rent_sar=36000,
        bedrooms=2,
        area_sqm=100.5,
    )


def _project() -> DocFields:
    return DocFields(
        doc_id="darglobal:project:trump-tower-jeddah",
        kind=DocKind.project,
        source=Source.darglobal,
        title="Trump Tower Jeddah",
        url="https://darglobal.co.uk/trump-tower-jeddah",
        scraped_at="2026-11-18T10:00:00+00:00",
        city="Jeddah",
        country="Saudi Arabia",
        brand="Trump",
        handover="Q4 2027",
        price_from_amount=2000000,
        price_from_currency="SAR",
    )


def _article() -> DocFields:
    return DocFields(
        doc_id="darglobal:press:test-press",
        kind=DocKind.article,
        source=Source.darglobal,
        title="Dar Global Launches New Project",
        url="https://darglobal.co.uk/press/test-press",
        scraped_at="2026-11-18T10:00:00+00:00",
        article_kind=ArticleKind.press,
        published_at="2026-11-15T00:00:00+00:00",
    )


class TestListingSaleCard:
    def test_embed_mode(self) -> None:
        card = render_card(_listing_sale(), mode="embed")
        assert "Villa for sale in Al Narjis, Riyadh." in card
        assert "5 bedrooms" in card
        assert "3 bathrooms" in card
        assert "333 sqm" in card
        assert "Price SAR 1,950,000." in card

    def test_context_mode(self) -> None:
        card = render_card(_listing_sale(), mode="context", n=1)
        assert card.startswith("[1] ")
        assert "URL: https://wasalt.sa/property/sale/villa-111" in card
        assert "As of: 2026-11-18" in card


class TestListingRentCard:
    def test_embed_mode(self) -> None:
        card = render_card(_listing_rent(), mode="embed")
        assert "Apartment for rent in Al Hamra, Jeddah." in card
        assert "Rent SAR 3,000 per monthly (SAR 36,000 per year)." in card
        assert "100.5 sqm" in card

    def test_context_mode(self) -> None:
        card = render_card(_listing_rent(), mode="context", n=2)
        assert card.startswith("[2] ")


class TestProjectCard:
    def test_embed_mode(self) -> None:
        card = render_card(_project(), mode="embed")
        assert "Trump Tower Jeddah by DarGlobal (Trump brand)" in card
        assert "in Jeddah, Saudi Arabia." in card
        assert "Handover Q4 2027." in card
        assert "From SAR 2,000,000." in card

    def test_context_mode(self) -> None:
        card = render_card(_project(), mode="context", n=1)
        assert card.startswith("[1] ")
        assert "URL: https://darglobal.co.uk/trump-tower-jeddah" in card


class TestArticleCard:
    def test_embed_mode(self) -> None:
        card = render_card(_article(), mode="embed")
        assert "Dar Global Launches New Project" in card
        assert "press" in card
        assert "2026-11-15" in card

    def test_context_mode(self) -> None:
        card = render_card(_article(), mode="context", n=3)
        assert card.startswith("[3] ")


class TestNullFieldOmission:
    def test_no_district_no_bedrooms(self) -> None:
        doc = DocFields(
            doc_id="wasalt:333",
            kind=DocKind.listing,
            source=Source.wasalt,
            title="Land for Sale",
            url="https://wasalt.sa/property/sale/land-333",
            scraped_at="2026-11-18T10:00:00+00:00",
            purpose=Purpose.sale,
            property_type=PropertyType.land,
            city="Riyadh",
            price_sar=1000000,
        )
        card = render_card(doc, mode="embed")
        assert "bedrooms" not in card
        assert "bathrooms" not in card
        assert "sqm" not in card
        # District should not appear
        assert "None" not in card

    def test_project_no_handover_no_price(self) -> None:
        doc = DocFields(
            doc_id="darglobal:project:test",
            kind=DocKind.project,
            source=Source.darglobal,
            title="Test Project",
            url="https://darglobal.co.uk/test",
            scraped_at="2026-11-18T10:00:00+00:00",
        )
        card = render_card(doc, mode="embed")
        assert "Handover" not in card
        assert "From" not in card


class TestEmbedModeRestrictions:
    def test_no_citation_bracket(self) -> None:
        card = render_card(_listing_sale(), mode="embed")
        assert "[" not in card

    def test_no_http(self) -> None:
        card = render_card(_listing_sale(), mode="embed")
        assert "http" not in card

    def test_no_doc_id(self) -> None:
        card = render_card(_listing_sale(), mode="embed")
        assert "wasalt:111" not in card


class TestSnippet:
    def test_snippet_appended(self) -> None:
        card = render_card(_listing_sale(), mode="embed", snippet="Great location.")
        assert card.endswith("Great location.")

    def test_snippet_in_context_mode(self) -> None:
        card = render_card(_listing_sale(), mode="context", n=1, snippet="Spacious.")
        assert "Spacious." in card


class TestAreaFormatting:
    def test_whole_area_no_decimal(self) -> None:
        doc = DocFields(
            doc_id="wasalt:444",
            kind=DocKind.listing,
            source=Source.wasalt,
            title="Test",
            url="https://wasalt.sa/property/sale/test-444",
            scraped_at="2026-11-18T10:00:00+00:00",
            purpose=Purpose.sale,
            property_type=PropertyType.apartment,
            city="Riyadh",
            area_sqm=200.0,
        )
        card = render_card(doc, mode="embed")
        assert "200 sqm" in card
        assert "200.0" not in card

    def test_fractional_area_one_decimal(self) -> None:
        doc = DocFields(
            doc_id="wasalt:555",
            kind=DocKind.listing,
            source=Source.wasalt,
            title="Test",
            url="https://wasalt.sa/property/sale/test-555",
            scraped_at="2026-11-18T10:00:00+00:00",
            purpose=Purpose.sale,
            property_type=PropertyType.apartment,
            city="Riyadh",
            area_sqm=150.5,
        )
        card = render_card(doc, mode="embed")
        assert "150.5 sqm" in card
