"""Tests for the Wasalt listing parser.

Verifies parsing, PII removal, property type mapping, rent frequency mapping,
and conformance with common.schemas.Listing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from collect.wasalt import (
    extract_id_from_url,
    extract_next_data,
    map_property_type,
    map_rent_freq,
    parse_listing,
    parse_sitemap_urls,
    strip_pii,
)
from common.schemas import Listing, listing_id

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Property type mapping
# ---------------------------------------------------------------------------


class TestPropertyTypeMapping:
    def test_known_types(self) -> None:
        assert map_property_type("Villa") == ("villa", None)
        assert map_property_type("Apartment") == ("apartment", None)
        assert map_property_type("Townhouse") == ("townhouse", None)
        assert map_property_type("Duplex") == ("duplex", None)
        assert map_property_type("Floor") == ("floor", None)
        assert map_property_type("Land") == ("land", None)
        assert map_property_type("Building") == ("building", None)
        assert map_property_type("Office") == ("office", None)
        assert map_property_type("Shop") == ("shop", None)
        assert map_property_type("Warehouse") == ("warehouse", None)
        assert map_property_type("Chalet") == ("chalet", None)
        assert map_property_type("Farm") == ("farm", None)

    def test_rest_house_maps_to_chalet(self) -> None:
        mapped, raw = map_property_type("Rest House")
        assert mapped == "chalet"
        assert raw == "Rest House"

    def test_showroom_maps_to_shop(self) -> None:
        mapped, raw = map_property_type("Showroom")
        assert mapped == "shop"
        assert raw == "Showroom"

    def test_room_maps_to_other(self) -> None:
        mapped, raw = map_property_type("Room")
        assert mapped == "other"
        assert raw == "Room"

    def test_unknown_maps_to_other(self) -> None:
        mapped, raw = map_property_type("SomeNewType")
        assert mapped == "other"
        assert raw == "SomeNewType"

    def test_none_maps_to_other(self) -> None:
        mapped, raw = map_property_type(None)
        assert mapped == "other"
        assert raw is None

    def test_empty_maps_to_other(self) -> None:
        mapped, raw = map_property_type("")
        assert mapped == "other"
        assert raw is None


# ---------------------------------------------------------------------------
# Rent frequency mapping
# ---------------------------------------------------------------------------


class TestRentFreqMapping:
    def test_yearly(self) -> None:
        freq = {"yearly": {"amount": 35000, "freq": "yearly", "default_freq": True}}
        period, amount = map_rent_freq(freq)
        assert period == "yearly"
        assert amount == 35000

    def test_monthly_default(self) -> None:
        freq = {
            "monthly": {"amount": 5000, "freq": "monthly", "default_freq": True},
            "yearly": {"amount": 55000, "freq": "yearly", "default_freq": False},
        }
        period, amount = map_rent_freq(freq)
        assert period == "monthly"
        assert amount == 5000

    def test_no_default_prefers_yearly(self) -> None:
        freq = {
            "monthly": {"amount": 5000, "freq": "monthly"},
            "yearly": {"amount": 55000, "freq": "yearly"},
        }
        period, amount = map_rent_freq(freq)
        assert period == "yearly"
        assert amount == 55000

    def test_empty_string(self) -> None:
        period, amount = map_rent_freq("")
        assert period is None
        assert amount is None

    def test_empty_dict(self) -> None:
        period, amount = map_rent_freq({})
        assert period is None
        assert amount is None

    def test_none(self) -> None:
        period, amount = map_rent_freq(None)
        assert period is None
        assert amount is None

    def test_daily(self) -> None:
        freq = {"daily": {"amount": 200, "freq": "daily", "default_freq": True}}
        period, amount = map_rent_freq(freq)
        assert period == "daily"
        assert amount == 200


# ---------------------------------------------------------------------------
# PII stripping
# ---------------------------------------------------------------------------


class TestPIIStripping:
    def test_lead_contact_info_removed(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        details = fixture["props"]["pageProps"]["propertyDetailsV3"]
        assert "leadContactInfo" in details
        stripped = strip_pii(details)
        assert "leadContactInfo" not in stripped

    def test_owner_deny_fields_removed_company(self) -> None:
        """Company owner: deny-listed fields (managedBy, resEmployee, etc.) gone."""
        fixture = _load_fixture("wasalt_sale_listing.json")
        details = fixture["props"]["pageProps"]["propertyDetailsV3"]
        owner_before = details["propertyOwner"]
        assert owner_before.get("postedAs") == 49
        assert "resEmployee" in owner_before
        assert "ownerName" in owner_before
        assert "phoneNumber" in owner_before
        assert "email" in owner_before
        assert "managedBy" in owner_before
        assert "cmpLogoUrl" in owner_before

        stripped = strip_pii(details)
        owner = stripped["propertyOwner"]
        for banned in ("resEmployee", "ownerName", "phoneNumber", "email",
                       "managedBy", "cmpLogoUrl", "showUserCallWhatsapp",
                       "showInterestCta", "poaNumber"):
            assert banned not in owner, f"{banned} not stripped from company owner"

    def test_owner_company_allowed_fields_preserved(self) -> None:
        """Company owner: company names, licence numbers and metadata survive."""
        fixture = _load_fixture("wasalt_sale_listing.json")
        details = fixture["props"]["pageProps"]["propertyDetailsV3"]
        stripped = strip_pii(details)
        owner = stripped["propertyOwner"]
        assert owner["companyName"] == "Test Realty Co"
        assert owner["enCmpName"] == "Test Realty Co"
        assert owner["arCmpName"] == "Test Company AR"
        assert owner["regaAdvLicNo"] == "1234567890"
        assert owner["brand"] == "Test Realty Brand"
        assert owner["falLicenseNumber"] == "1100001111"
        assert owner["userCapacity"] == "company"
        assert owner["postedAs"] == 49
        assert owner["isOwnerProperty"] is False

    def test_owner_individual_name_fields_removed(self) -> None:
        """Individual owner (postedAs=48): name fields are dropped."""
        fixture = _load_fixture("wasalt_rent_listing.json")
        details = fixture["props"]["pageProps"]["propertyDetailsV3"]
        owner_before = details["propertyOwner"]
        assert owner_before.get("postedAs") == 48
        assert owner_before.get("userCapacity") == "individual"
        # Before stripping, name fields are present
        assert "enCmpName" in owner_before
        assert "arCmpName" in owner_before

        stripped = strip_pii(details)
        owner = stripped["propertyOwner"]
        # Individual name fields must be gone
        for banned in ("arCmpName", "enCmpName", "companyName", "brand",
                       "managedBy", "resEmployee", "ownerName",
                       "phoneNumber", "email"):
            assert banned not in owner, f"{banned} not stripped from individual owner"
        # Licence numbers still present
        assert owner["regaAdvLicNo"] == "9876543210"
        assert owner["falLicenseNumber"] == "1100009999"
        assert owner["userCapacity"] == "individual"
        assert owner["postedAs"] == 48

    def test_rega_verified_info_pii_removed(self) -> None:
        """responsibleEmployeeName, responsibleEmployeePhoneNumber, deedNumber
        must be stripped from regaVerifiedInfo[].fields[]."""
        fixture = _load_fixture("wasalt_sale_listing.json")
        details = fixture["props"]["pageProps"]["propertyDetailsV3"]
        # Verify these PII fields exist before stripping
        all_keys_before = set()
        for item in details["regaVerifiedInfo"]:
            for field in item.get("fields", []):
                all_keys_before.add(field["key"])
        assert "responsibleEmployeeName" in all_keys_before
        assert "responsibleEmployeePhoneNumber" in all_keys_before
        assert "deedNumber" in all_keys_before

        stripped = strip_pii(details)
        # After stripping they must be gone
        all_keys_after = set()
        for item in stripped["regaVerifiedInfo"]:
            for field in item.get("fields", []):
                all_keys_after.add(field["key"])
        assert "responsibleEmployeeName" not in all_keys_after
        assert "responsibleEmployeePhoneNumber" not in all_keys_after
        assert "deedNumber" not in all_keys_after
        # Non-PII fields still present
        assert "adLicenseNumber" in all_keys_after
        assert "status" in all_keys_after
        assert "location.city" in all_keys_after
        assert "propertyArea" in all_keys_after
        assert "isConstrained" in all_keys_after

    def test_rega_pii_removed_from_rent_fixture(self) -> None:
        """Rent fixture also has regaVerifiedInfo PII that must be stripped."""
        fixture = _load_fixture("wasalt_rent_listing.json")
        details = fixture["props"]["pageProps"]["propertyDetailsV3"]
        stripped = strip_pii(details)
        all_keys = set()
        for item in stripped.get("regaVerifiedInfo", []):
            for field in item.get("fields", []):
                all_keys.add(field["key"])
        assert "responsibleEmployeeName" not in all_keys
        assert "responsibleEmployeePhoneNumber" not in all_keys
        assert "deedNumber" not in all_keys

    def test_pii_stripped_from_all_fixtures(self) -> None:
        """Every fixture must have PII fields stripped cleanly."""
        for name in ("wasalt_sale_listing.json", "wasalt_rent_listing.json",
                     "wasalt_rent_multi_freq.json"):
            fixture = _load_fixture(name)
            details = fixture["props"]["pageProps"]["propertyDetailsV3"]
            stripped = strip_pii(details)
            assert "leadContactInfo" not in stripped
            owner = stripped.get("propertyOwner", {})
            for field in ("resEmployee", "ownerName", "phoneNumber", "email",
                         "phone", "mobile", "contactName", "managedBy",
                         "cmpLogoUrl", "showUserCallWhatsapp", "showInterestCta"):
                assert field not in owner, f"{field} still in {name}"
            # Check regaVerifiedInfo PII keys
            for item in stripped.get("regaVerifiedInfo", []):
                for field in item.get("fields", []):
                    assert field["key"] not in (
                        "responsibleEmployeeName",
                        "responsibleEmployeePhoneNumber",
                        "deedNumber",
                    ), f"PII key {field['key']} still in {name}"


# ---------------------------------------------------------------------------
# Parsing sale listing
# ---------------------------------------------------------------------------


class TestParseSaleListing:
    def test_basic_fields(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/en/property/sale/villa-5841671")
        assert result is not None
        assert result["source_id"] == "5841671"
        assert result["purpose"] == "sale"
        assert result["property_type"] == "villa"
        assert result["property_type_raw"] is None
        assert result["city"] == "Riyadh"
        assert result["district"] == "Al-Mahdiyah"
        assert result["price_sar"] == 1600000
        assert result["bedrooms"] == 5
        assert result["bathrooms"] == 3
        assert result["area_sqm"] == 333.0
        assert result["title"] == "Villa 333 SQM Facing East on 20m Width Street"
        assert result["rent_period"] is None
        assert result["annual_rent_sar"] is None

    def test_location(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert abs(result["lat"] - 24.6678) < 0.001
        assert abs(result["lon"] - 46.5358) < 0.001

    def test_description_html_stripped(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert "<p>" not in (result["description"] or "")
        assert "Villa for Sale" in (result["description"] or "")

    def test_published_at(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert result["published_at"] == "2026-11-10T08:30:00.000Z"

    def test_conforms_to_listing_schema(self) -> None:
        """The parsed data must be convertible to a valid Listing."""
        fixture = _load_fixture("wasalt_sale_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/en/property/sale/villa-5841671")
        assert result is not None
        result.pop("_raw_details")
        listing = Listing(
            id=listing_id(result["source_id"]),
            source_id=result["source_id"],
            url=result["url"],
            title=result["title"],
            description=result["description"],
            purpose=result["purpose"],
            property_type=result["property_type"],
            property_type_raw=result["property_type_raw"],
            city=result["city"],
            district=result["district"],
            price_sar=result["price_sar"],
            rent_period=result["rent_period"],
            annual_rent_sar=result["annual_rent_sar"],
            bedrooms=result["bedrooms"],
            bathrooms=result["bathrooms"],
            area_sqm=result["area_sqm"],
            lat=result["lat"],
            lon=result["lon"],
            amenities=result["amenities"],
            published_at=result["published_at"],
            scraped_at=datetime.now(UTC),
        )
        assert listing.id == "wasalt:5841671"
        assert listing.purpose.value == "sale"
        assert listing.property_type.value == "villa"


# ---------------------------------------------------------------------------
# Parsing rent listing
# ---------------------------------------------------------------------------


class TestParseRentListing:
    def test_basic_fields(self) -> None:
        fixture = _load_fixture("wasalt_rent_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/en/property/rent/apartment-5840301")
        assert result is not None
        assert result["source_id"] == "5840301"
        assert result["purpose"] == "rent"
        assert result["property_type"] == "apartment"
        assert result["price_sar"] == 35000
        assert result["rent_period"] == "yearly"
        assert result["annual_rent_sar"] == 35000  # yearly * 1
        assert result["bedrooms"] == 4
        assert result["area_sqm"] == 400.0

    def test_conforms_to_listing_schema(self) -> None:
        fixture = _load_fixture("wasalt_rent_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/en/property/rent/apartment-5840301")
        assert result is not None
        result.pop("_raw_details")
        listing = Listing(
            id=listing_id(result["source_id"]),
            source_id=result["source_id"],
            url=result["url"],
            title=result["title"],
            description=result["description"],
            purpose=result["purpose"],
            property_type=result["property_type"],
            property_type_raw=result["property_type_raw"],
            city=result["city"],
            district=result["district"],
            price_sar=result["price_sar"],
            rent_period=result["rent_period"],
            annual_rent_sar=result["annual_rent_sar"],
            bedrooms=result["bedrooms"],
            bathrooms=result["bathrooms"],
            area_sqm=result["area_sqm"],
            lat=result["lat"],
            lon=result["lon"],
            amenities=result["amenities"],
            published_at=result["published_at"],
            scraped_at=datetime.now(UTC),
        )
        assert listing.id == "wasalt:5840301"
        assert listing.purpose.value == "rent"
        assert listing.rent_period.value == "yearly"
        assert listing.annual_rent_sar == 35000


class TestParseRentMultiFreq:
    def test_monthly_default(self) -> None:
        fixture = _load_fixture("wasalt_rent_multi_freq.json")
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert result["rent_period"] == "monthly"
        assert result["price_sar"] == 5000
        assert result["annual_rent_sar"] == 60000  # 5000 * 12

    def test_empty_description_becomes_null(self) -> None:
        """WP-1a issue: empty descriptions must map to null."""
        fixture = _load_fixture("wasalt_rent_multi_freq.json")
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert result["description"] is None

    def test_conforms_to_listing_schema_monthly(self) -> None:
        fixture = _load_fixture("wasalt_rent_multi_freq.json")
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        result.pop("_raw_details")
        listing = Listing(
            id=listing_id(result["source_id"]),
            source_id=result["source_id"],
            url=result["url"],
            title=result["title"],
            description=result["description"],
            purpose=result["purpose"],
            property_type=result["property_type"],
            property_type_raw=result["property_type_raw"],
            city=result["city"],
            district=result["district"],
            price_sar=result["price_sar"],
            rent_period=result["rent_period"],
            annual_rent_sar=result["annual_rent_sar"],
            bedrooms=result["bedrooms"],
            bathrooms=result["bathrooms"],
            area_sqm=result["area_sqm"],
            lat=result["lat"],
            lon=result["lon"],
            amenities=result["amenities"],
            published_at=result["published_at"],
            scraped_at=datetime.now(UTC),
        )
        assert listing.rent_period.value == "monthly"
        assert listing.annual_rent_sar == 60000


# ---------------------------------------------------------------------------
# ID extraction from URLs
# ---------------------------------------------------------------------------


class TestExtractIdFromUrl:
    def test_sale_url(self) -> None:
        url = "https://wasalt.sa/en/property/sale/villa-333-sqm-facing-east-on-20m-width-street-5841671"
        assert extract_id_from_url(url) == "5841671"

    def test_rent_url(self) -> None:
        url = "https://wasalt.sa/en/property/rent/apartment-with-4-bedrooms-5840301"
        assert extract_id_from_url(url) == "5840301"

    def test_id_matches_schema_pattern(self) -> None:
        """WP-1a issue: Wasalt IDs must match ^[A-Za-z0-9][A-Za-z0-9._-]*$."""
        import re
        pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
        urls = [
            "https://wasalt.sa/en/property/sale/villa-5841671",
            "https://wasalt.sa/en/property/rent/apartment-5840301",
            "https://wasalt.sa/en/property/sale/land-5840190",
        ]
        for url in urls:
            wid = extract_id_from_url(url)
            assert wid is not None
            assert pattern.match(wid), f"ID {wid!r} from {url} does not match slug pattern"


# ---------------------------------------------------------------------------
# __NEXT_DATA__ extraction
# ---------------------------------------------------------------------------


class TestExtractNextData:
    def test_valid_html(self) -> None:
        html = '<html><script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{}}}</script></html>'
        data = extract_next_data(html)
        assert data is not None
        assert "props" in data

    def test_no_next_data(self) -> None:
        html = "<html><body>Hello</body></html>"
        assert extract_next_data(html) is None

    def test_invalid_json(self) -> None:
        html = '<html><script id="__NEXT_DATA__" type="application/json">{invalid}</script></html>'
        assert extract_next_data(html) is None


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------


class TestSitemapParsing:
    def test_parse_urls(self) -> None:
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <url><loc>https://wasalt.sa/en/property/sale/villa-123</loc></url>
          <url><loc>https://wasalt.sa/en/property/rent/apartment-456</loc></url>
        </urlset>'''
        urls = parse_sitemap_urls(xml)
        assert len(urls) == 2
        assert "villa-123" in urls[0]
        assert "apartment-456" in urls[1]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_property_details(self) -> None:
        data = {"props": {"pageProps": {}}}
        assert parse_listing(data, "https://wasalt.sa/test") is None

    def test_missing_title_uses_property_name(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        fixture["props"]["pageProps"]["propertyDetailsV3"]["propertyInfo"]["title"] = ""
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert result["title"] == "Villa for Sale"

    def test_no_id_returns_none(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        del fixture["props"]["pageProps"]["propertyDetailsV3"]["id"]
        assert parse_listing(fixture, "https://wasalt.sa/test") is None

    def test_sale_listing_no_rent_fields(self) -> None:
        """Sale listings must not set rent_period or annual_rent_sar."""
        fixture = _load_fixture("wasalt_sale_listing.json")
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert result["rent_period"] is None
        assert result["annual_rent_sar"] is None

    def test_long_description_truncated(self) -> None:
        fixture = _load_fixture("wasalt_sale_listing.json")
        long_desc = "A" * 500
        fixture["props"]["pageProps"]["propertyDetailsV3"]["propertyInfo"]["description"] = long_desc
        result = parse_listing(fixture, "https://wasalt.sa/test")
        assert result is not None
        assert len(result["description"]) <= 400
        assert result["description"].endswith("...")
