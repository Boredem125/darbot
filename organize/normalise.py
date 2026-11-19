"""Normalise raw Wasalt listings into validated Listing records.

Reads data/raw/wasalt/*.json.gz, applies place-name canonicalisation,
property type mapping, rent annualisation, and description cleanup.
Produces validated Listing records for downstream processing.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.schemas import (
    ANNUAL_RENT_FACTOR,
    Listing,
    PropertyType,
    Purpose,
    RentPeriod,
    listing_id,
)
from organize.sanitise import sanitise_description

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# City name canonicalisation
# ---------------------------------------------------------------------------

# Map raw Wasalt city names to canonical English names
CITY_CANON: dict[str, str] = {
    # Exact matches from Wasalt data (top cities from WP-1b report)
    "Riyadh": "Riyadh",
    "Jeddah": "Jeddah",
    "Khobar": "Khobar",
    "Makkah Al Mukarramah": "Makkah",
    "Aldammam": "Dammam",
    "Jazan": "Jazan",
    "Abha": "Abha",
    "Madinah": "Madinah",
    "Alttayif": "Taif",
    "Khamis Mushait": "Khamis Mushait",
    "Abu Arish": "Abu Arish",
    "Jubail": "Jubail",
    "Qatif": "Qatif",
    "Ahad Rafidah": "Ahad Rafidah",
    "Al Kharj": "Al Kharj",
    "Tabuk": "Tabuk",
    "Hafar Al Batin": "Hafar Al Batin",
    "Buraydah": "Buraydah",
    "Dhahran": "Dhahran",
    "Al Baha": "Al Baha",
    "Hail": "Hail",
    "Najran": "Najran",
    "Al Majmaah": "Al Majmaah",
    "Sakakah": "Sakakah",
    "Arar": "Arar",
    "Yanbu": "Yanbu",
    "Unaizah": "Unaizah",
    "Al Rass": "Al Rass",
    "Wadi Al Dawasir": "Wadi Al Dawasir",
    "Muhayil Asir": "Muhayil Asir",
    "Al Qurayyat": "Al Qurayyat",
    "Bishah": "Bishah",
    "Sabya": "Sabya",
    "Al Zulfi": "Al Zulfi",
    "Khulays": "Khulays",
    "Al Lith": "Al Lith",
    "Taif": "Taif",
    "Dammam": "Dammam",
    "Makkah": "Makkah",
    "Medina": "Madinah",
    "Al Madinah": "Madinah",
    "Al Khobar": "Khobar",
    "Mecca": "Makkah",
    "Jiddah": "Jeddah",
}

# District name normalisation: trim whitespace and title-case
def _canon_district(raw: str | None) -> str | None:
    if not raw:
        return None
    name = raw.strip()
    if not name:
        return None
    # Normalise hyphenation: "Al-Narjis" -> "Al Narjis"
    name = name.replace("-", " ").replace("  ", " ").strip()
    # Title-case
    name = name.title()
    # Fix common Arabic article: "Al " prefix
    if name.startswith("Al "):
        name = "Al " + name[3:]
    return name


def _canon_city(raw: str | None) -> str | None:
    if not raw:
        return None
    name = raw.strip()
    if not name:
        return None
    # Look up in canonicalisation table
    if name in CITY_CANON:
        return CITY_CANON[name]
    # Title-case as fallback
    return name.title()


def normalise_listing(parsed: dict[str, Any], scraped_at: str) -> Listing | None:
    """Normalise a single raw listing dict into a validated Listing.

    Returns None if the listing is invalid.
    """
    source_id = str(parsed.get("source_id", ""))
    if not source_id:
        return None

    purpose_raw = parsed.get("purpose", "")
    if purpose_raw not in ("sale", "rent"):
        return None

    purpose = Purpose(purpose_raw)

    title = parsed.get("title", "").strip()
    if not title:
        return None

    city = _canon_city(parsed.get("city"))
    if not city:
        return None

    district = _canon_district(parsed.get("district"))

    # Property type
    prop_type_raw = parsed.get("property_type_raw")
    prop_type_str = parsed.get("property_type", "other")
    try:
        prop_type = PropertyType(prop_type_str)
    except ValueError:
        prop_type = PropertyType.other
        prop_type_raw = prop_type_str

    # Description: sanitise
    description = sanitise_description(parsed.get("description"))

    # Price
    price_sar = parsed.get("price_sar")
    if price_sar is not None:
        try:
            price_sar = int(price_sar)
            if price_sar <= 0:
                price_sar = None
        except (ValueError, TypeError):
            price_sar = None

    # Rent period and annual rent
    rent_period: RentPeriod | None = None
    annual_rent_sar: int | None = None

    if purpose == Purpose.rent:
        rp_raw = parsed.get("rent_period")
        if rp_raw:
            try:
                rent_period = RentPeriod(rp_raw)
            except ValueError:
                rent_period = None

        if rent_period is not None and price_sar is not None:
            annual_rent_sar = price_sar * ANNUAL_RENT_FACTOR[rent_period]
        elif rp_raw and rent_period is None:
            # Unknown rent period: leave both null
            price_sar = None
            annual_rent_sar = None

    # Numeric fields
    bedrooms = parsed.get("bedrooms")
    if bedrooms is not None:
        try:
            bedrooms = int(bedrooms)
            if bedrooms < 0 or bedrooms > 50:
                bedrooms = None
        except (ValueError, TypeError):
            bedrooms = None

    bathrooms = parsed.get("bathrooms")
    if bathrooms is not None:
        try:
            bathrooms = int(bathrooms)
            if bathrooms < 0 or bathrooms > 50:
                bathrooms = None
        except (ValueError, TypeError):
            bathrooms = None

    area_sqm = parsed.get("area_sqm")
    if area_sqm is not None:
        try:
            area_sqm = float(area_sqm)
            if area_sqm <= 0:
                area_sqm = None
        except (ValueError, TypeError):
            area_sqm = None

    lat = parsed.get("lat")
    lon = parsed.get("lon")
    if lat is not None:
        try:
            lat = float(lat)
            if lat < -90 or lat > 90:
                lat = None
        except (ValueError, TypeError):
            lat = None
    if lon is not None:
        try:
            lon = float(lon)
            if lon < -180 or lon > 180:
                lon = None
        except (ValueError, TypeError):
            lon = None

    amenities = parsed.get("amenities", [])
    if not isinstance(amenities, list):
        amenities = []
    amenities = [str(a) for a in amenities if a]

    published_at = parsed.get("published_at")

    url = parsed.get("url", "")
    if not url:
        url = f"https://wasalt.sa/property/{purpose_raw}/{source_id}"

    try:
        listing = Listing(
            id=listing_id(source_id),
            source="wasalt",
            source_id=source_id,
            url=url,
            title=title,
            description=description,
            purpose=purpose,
            property_type=prop_type,
            property_type_raw=prop_type_raw,
            city=city,
            district=district,
            price_sar=price_sar,
            rent_period=rent_period,
            annual_rent_sar=annual_rent_sar,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            area_sqm=area_sqm,
            lat=lat,
            lon=lon,
            amenities=amenities,
            published_at=published_at,
            scraped_at=scraped_at,
        )
        return listing
    except Exception as exc:  # noqa: BLE001
        logger.debug("Validation failed for %s: %s", source_id, exc)
        return None


def load_raw_listings(raw_dir: Path) -> tuple[list[Listing], Counter[str]]:
    """Load and normalise all listings from raw dir.

    Returns (listings, rejection_reasons).
    """
    listings: list[Listing] = []
    rejections: Counter[str] = Counter()

    files = sorted(raw_dir.glob("*.json.gz"))
    logger.info("Loading %d raw files from %s", len(files), raw_dir)

    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            rejections["read_error"] += 1
            logger.debug("Failed to read %s: %s", path, exc)
            continue

        parsed = data.get("parsed")
        if not isinstance(parsed, dict):
            rejections["missing_parsed"] += 1
            continue

        scraped_at = data.get("scraped_at", datetime.now(UTC).isoformat())

        listing = normalise_listing(parsed, scraped_at)
        if listing is None:
            # Determine rejection reason
            if not parsed.get("source_id"):
                rejections["no_source_id"] += 1
            elif parsed.get("purpose") not in ("sale", "rent"):
                rejections["bad_purpose"] += 1
            elif not parsed.get("title", "").strip():
                rejections["no_title"] += 1
            elif not _canon_city(parsed.get("city")):
                rejections["no_city"] += 1
            else:
                rejections["validation_error"] += 1
            continue

        listings.append(listing)

    logger.info(
        "Loaded %d valid listings, %d rejected: %s",
        len(listings), sum(rejections.values()), dict(rejections),
    )

    # --- Plausibility filter ---
    # Drop listings with implausibly low prices that would poison aggregates.
    # Thresholds chosen from distribution analysis (see WP-2-index report):
    #   - Rent: 3,000 SAR/year. Below this, values are clearly data-entry errors
    #     (e.g. 60 SAR/year for a 217 sqm apartment). Even single rooms rent
    #     for more than 250 SAR/month in Saudi Arabia.
    #   - Sale: 50,000 SAR. Below this, values are per-sqm prices mistakenly
    #     entered as total price (e.g. land at 1,900 SAR for 321 sqm).
    MIN_ANNUAL_RENT_SAR = 12_000
    MIN_SALE_PRICE_SAR = 50_000

    before_plausibility = len(listings)
    plausible: list[Listing] = []
    dropped_rent = 0
    dropped_sale = 0

    for li in listings:
        if li.purpose == Purpose.rent and li.annual_rent_sar is not None and li.annual_rent_sar < MIN_ANNUAL_RENT_SAR:
            dropped_rent += 1
            rejections["implausible_rent"] += 1
            continue
        if li.purpose == Purpose.sale and li.price_sar is not None and li.price_sar < MIN_SALE_PRICE_SAR:
            dropped_sale += 1
            rejections["implausible_sale"] += 1
            continue
        plausible.append(li)

    listings = plausible

    if dropped_rent or dropped_sale:
        logger.info(
            "Plausibility filter: dropped %d rent (< %d SAR/yr) + %d sale (< %d SAR), "
            "%d -> %d listings",
            dropped_rent, MIN_ANNUAL_RENT_SAR,
            dropped_sale, MIN_SALE_PRICE_SAR,
            before_plausibility, len(listings),
        )

    return listings, rejections
