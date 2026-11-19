"""Tests for organize.normalise."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from organize.normalise import _canon_city, _canon_district, load_raw_listings, normalise_listing


class TestCityCanon:
    def test_known_city(self) -> None:
        assert _canon_city("Aldammam") == "Dammam"

    def test_makkah(self) -> None:
        assert _canon_city("Makkah Al Mukarramah") == "Makkah"

    def test_alttayif(self) -> None:
        assert _canon_city("Alttayif") == "Taif"

    def test_passthrough(self) -> None:
        assert _canon_city("Riyadh") == "Riyadh"

    def test_unknown_title_cased(self) -> None:
        assert _canon_city("some new city") == "Some New City"

    def test_none(self) -> None:
        assert _canon_city(None) is None

    def test_empty(self) -> None:
        assert _canon_city("") is None


class TestDistrictCanon:
    def test_hyphen_normalised(self) -> None:
        assert _canon_district("Al-Narjis") == "Al Narjis"

    def test_title_case(self) -> None:
        assert _canon_district("al mahdiyah") == "Al Mahdiyah"

    def test_none(self) -> None:
        assert _canon_district(None) is None

    def test_empty(self) -> None:
        assert _canon_district("") is None


class TestNormaliseListing:
    def test_sale_listing(self) -> None:
        parsed = {
            "source_id": "5841671",
            "url": "https://wasalt.sa/property/sale/villa-5841671",
            "title": "Villa for Sale",
            "description": "Nice villa in Riyadh",
            "purpose": "sale",
            "property_type": "villa",
            "property_type_raw": None,
            "city": "Riyadh",
            "district": "Al-Mahdiyah",
            "price_sar": 1600000,
            "rent_period": None,
            "annual_rent_sar": None,
            "bedrooms": 5,
            "bathrooms": 3,
            "area_sqm": 333.0,
            "lat": 24.667,
            "lon": 46.535,
            "amenities": ["Parking: 2"],
            "published_at": "2026-11-10T08:30:00.000Z",
        }
        listing = normalise_listing(parsed, "2026-11-18T10:00:00+00:00")
        assert listing is not None
        assert listing.id == "wasalt:5841671"
        assert listing.purpose.value == "sale"
        assert listing.price_sar == 1600000
        assert listing.city == "Riyadh"
        assert listing.district == "Al Mahdiyah"
        assert listing.rent_period is None
        assert listing.annual_rent_sar is None

    def test_rent_annualisation(self) -> None:
        parsed = {
            "source_id": "1234",
            "url": "https://wasalt.sa/property/rent/apt-1234",
            "title": "Apt for Rent",
            "description": None,
            "purpose": "rent",
            "property_type": "apartment",
            "property_type_raw": None,
            "city": "Jeddah",
            "district": None,
            "price_sar": 3000,
            "rent_period": "monthly",
            "annual_rent_sar": None,
            "bedrooms": 2,
            "bathrooms": 1,
            "area_sqm": 100.0,
            "lat": None,
            "lon": None,
            "amenities": [],
            "published_at": None,
        }
        listing = normalise_listing(parsed, "2026-11-18T10:00:00+00:00")
        assert listing is not None
        assert listing.rent_period.value == "monthly"
        assert listing.annual_rent_sar == 3000 * 12

    def test_empty_description_to_null(self) -> None:
        parsed = {
            "source_id": "9999",
            "url": "https://wasalt.sa/property/sale/test-9999",
            "title": "Test",
            "description": "",
            "purpose": "sale",
            "property_type": "apartment",
            "property_type_raw": None,
            "city": "Riyadh",
            "district": None,
            "price_sar": 500000,
            "rent_period": None,
            "annual_rent_sar": None,
            "bedrooms": None,
            "bathrooms": None,
            "area_sqm": None,
            "lat": None,
            "lon": None,
            "amenities": [],
            "published_at": None,
        }
        listing = normalise_listing(parsed, "2026-11-18T10:00:00+00:00")
        assert listing is not None
        assert listing.description is None

    def test_missing_city_rejected(self) -> None:
        parsed = {
            "source_id": "1111",
            "url": "https://wasalt.sa/property/sale/test-1111",
            "title": "Test",
            "description": None,
            "purpose": "sale",
            "property_type": "apartment",
            "property_type_raw": None,
            "city": None,
            "district": None,
            "price_sar": 100000,
            "rent_period": None,
            "annual_rent_sar": None,
            "bedrooms": None,
            "bathrooms": None,
            "area_sqm": None,
            "lat": None,
            "lon": None,
            "amenities": [],
            "published_at": None,
        }
        listing = normalise_listing(parsed, "2026-11-18T10:00:00+00:00")
        assert listing is None


class TestPlausibilityFilter:
    """Plausibility filter drops implausibly low prices."""

    @staticmethod
    def _make_raw(source_id: str, purpose: str, price: int, rent_period: str | None = None):
        return {
            "parsed": {
                "source_id": source_id,
                "url": f"https://wasalt.sa/property/{purpose}/test-{source_id}",
                "title": "Test Listing",
                "description": None,
                "purpose": purpose,
                "property_type": "apartment",
                "property_type_raw": None,
                "city": "Riyadh",
                "district": None,
                "price_sar": price,
                "rent_period": rent_period,
                "annual_rent_sar": None,
                "bedrooms": 2,
                "bathrooms": 1,
                "area_sqm": 100.0,
                "lat": None,
                "lon": None,
                "amenities": [],
                "published_at": None,
            },
            "scraped_at": "2026-11-18T10:00:00+00:00",
        }

    def test_low_rent_dropped(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # 60 SAR/year rent should be dropped (below 12,000 threshold)
        rec = self._make_raw("100", "rent", 60, "yearly")
        with gzip.open(raw_dir / "100.json.gz", "wt", encoding="utf-8") as f:
            json.dump(rec, f)
        # 5000 SAR/year rent should also be dropped (below 12,000 threshold)
        rec2 = self._make_raw("101", "rent", 5000, "yearly")
        with gzip.open(raw_dir / "101.json.gz", "wt", encoding="utf-8") as f:
            json.dump(rec2, f)
        # 15000 SAR/year rent should survive
        rec3 = self._make_raw("102", "rent", 15000, "yearly")
        with gzip.open(raw_dir / "102.json.gz", "wt", encoding="utf-8") as f:
            json.dump(rec3, f)

        listings, rejections = load_raw_listings(raw_dir)
        assert len(listings) == 1
        assert listings[0].source_id == "102"
        assert rejections["implausible_rent"] == 2

    def test_low_sale_dropped(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        # 750 SAR sale should be dropped
        rec = self._make_raw("200", "sale", 750)
        with gzip.open(raw_dir / "200.json.gz", "wt", encoding="utf-8") as f:
            json.dump(rec, f)
        # 500000 SAR sale should survive
        rec2 = self._make_raw("201", "sale", 500000)
        with gzip.open(raw_dir / "201.json.gz", "wt", encoding="utf-8") as f:
            json.dump(rec2, f)

        listings, rejections = load_raw_listings(raw_dir)
        assert len(listings) == 1
        assert listings[0].source_id == "201"
        assert rejections["implausible_sale"] == 1
