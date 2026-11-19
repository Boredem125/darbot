"""Wasalt listing collector.

Fetches Wasalt sale and rent listing pages, extracts __NEXT_DATA__
JSON, drops PII fields, and saves trimmed JSON (gzipped) to
data/raw/wasalt/{id}.json.gz.

Resumable: skips IDs whose .json.gz already exists.

Usage:
    python -m collect.wasalt [--max-sale 2000] [--max-rent 1000]

PII policy:
    - leadContactInfo is dropped entirely
    - propertyOwner person fields are dropped (keeps only company-level info)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from collect.politeness import (
    USER_AGENT,
    BackoffHandler,
    RateLimiter,
    is_allowed_wasalt,
)

logger = logging.getLogger(__name__)

# Paths
RAW_DIR = Path("data/raw/wasalt")

# Sitemap URLs
SALE_SITEMAP = "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz"
RENTAL_SITEMAP = "https://cdn.wasalt.sa/sitemap/rental_pdp_sitemap_en_sa.xml.gz"

# XML namespace for sitemaps
SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# __NEXT_DATA__ extraction pattern
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>'
)

# Allow-listed propertyOwner fields (everything else is dropped).
# For individuals (postedAs=48), name fields are also dropped.
_OWNER_ALLOW_COMPANY: set[str] = {
    # Company identity
    "arCmpName", "enCmpName", "companyName", "brand",
    # Licence / regulatory (public registry numbers, never PII)
    "regaAdvLicNo", "falLicenseNumber", "falLicenseUrl", "falExpiryDate",
    "regaAuthorizationNumber", "regaAdvertiserNumber",
    # Structural metadata
    "userCapacity", "postedAs", "enUserRole", "arUserRole",
    "isOwnerProperty", "licenseExpired",
}
# Individual advertisers: company-name fields actually hold a person's name.
_OWNER_ALLOW_INDIVIDUAL: set[str] = _OWNER_ALLOW_COMPANY - {
    "arCmpName", "enCmpName", "companyName", "brand",
}

# regaVerifiedInfo field keys that contain PII
_REGA_PII_KEYS: set[str] = {
    "responsibleEmployeeName",
    "responsibleEmployeePhoneNumber",
    "deedNumber",
}

# Minimum free space on E: before we stop (3 GB)
MIN_FREE_BYTES = 3 * 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# propertyMainType -> PropertyType mapping (WP-1a open issue #2)
# ---------------------------------------------------------------------------

# Wasalt uses propertyMainType for the broad category (Residential, Commercial)
# and propertySubType for the specific type. We map propertySubType to our enum.
SUBTYPE_MAP: dict[str, str] = {
    # Residential
    "Apartment": "apartment",
    "Villa": "villa",
    "Townhouse": "townhouse",
    "Duplex": "duplex",
    "Floor": "floor",
    "Land": "land",
    "Building": "building",
    "Chalet": "chalet",
    "Farm": "farm",
    "Rest House": "chalet",  # closest match
    "Room": "other",
    # Commercial
    "Office": "office",
    "Shop": "shop",
    "Warehouse": "warehouse",
    "Showroom": "shop",
    # Fallback
}

# rentFreq key -> RentPeriod enum value
RENT_FREQ_MAP: dict[str, str] = {
    "daily": "daily",
    "weekly": "weekly",
    "monthly": "monthly",
    "yearly": "yearly",
}


def map_property_type(sub_type: str | None) -> tuple[str, str | None]:
    """Map Wasalt propertySubType to schema PropertyType.

    Returns (mapped_type, raw_value_or_none).
    raw_value_or_none is non-None only when mapping differs from identity.
    """
    if not sub_type:
        return "other", None
    mapped = SUBTYPE_MAP.get(sub_type, "other")
    raw = sub_type if mapped == "other" or sub_type.lower() != mapped else None
    return mapped, raw


def map_rent_freq(rent_freq: Any) -> tuple[str | None, int | None]:
    """Extract rent period and amount from Wasalt rentFreq.

    rentFreq can be:
    - empty string (sale listings)
    - dict like {"yearly": {"amount": 35000, "freq": "yearly", ...}}
    - dict with multiple keys like {"monthly": ..., "yearly": ...}

    Returns (rent_period, rent_amount) for the default frequency.
    For multiple frequencies, prefer the one marked default_freq=True,
    then yearly, then the first available.
    """
    if not rent_freq or not isinstance(rent_freq, dict):
        return None, None

    # Find the default frequency
    default_entry = None
    for key, value in rent_freq.items():
        if isinstance(value, dict) and value.get("default_freq"):
            default_entry = (key, value)
            break

    # Fallback: prefer yearly, then first
    if default_entry is None:
        if "yearly" in rent_freq:
            default_entry = ("yearly", rent_freq["yearly"])
        else:
            key = next(iter(rent_freq))
            default_entry = (key, rent_freq[key])

    freq_key, freq_data = default_entry
    period = RENT_FREQ_MAP.get(freq_key)
    amount = None
    if isinstance(freq_data, dict):
        amount = freq_data.get("amount")
        if isinstance(amount, str):
            try:
                amount = int(amount)
            except (ValueError, TypeError):
                amount = None
        elif isinstance(amount, (int, float)):
            amount = int(amount)

    return period, amount


# ---------------------------------------------------------------------------
# PII removal
# ---------------------------------------------------------------------------

def strip_pii(details: dict[str, Any]) -> dict[str, Any]:
    """Remove PII fields from propertyDetailsV3.

    - Drops ``leadContactInfo`` entirely.
    - Applies an *allow-list* to ``propertyOwner``.  For companies
      (postedAs 49) the company name fields survive; for individuals
      (postedAs 48) they are dropped because they hold a person's name.
    - Strips ``responsibleEmployeeName``, ``responsibleEmployeePhoneNumber``
      and ``deedNumber`` from every ``regaVerifiedInfo[].fields[]`` entry.
    """
    details.pop("leadContactInfo", None)

    # --- propertyOwner: allow-list approach ---
    owner = details.get("propertyOwner")
    if isinstance(owner, dict):
        is_individual = (
            owner.get("postedAs") == 48
            or str(owner.get("userCapacity", "")).lower() == "individual"
        )
        allow = _OWNER_ALLOW_INDIVIDUAL if is_individual else _OWNER_ALLOW_COMPANY
        for key in list(owner.keys()):
            if key not in allow:
                del owner[key]

    # --- regaVerifiedInfo: drop PII field entries ---
    rega_info = details.get("regaVerifiedInfo")
    if isinstance(rega_info, list):
        for item in rega_info:
            if not isinstance(item, dict):
                continue
            fields = item.get("fields")
            if isinstance(fields, list):
                item["fields"] = [
                    f for f in fields
                    if f.get("key") not in _REGA_PII_KEYS
                ]

    return details


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_next_data(html: str) -> dict[str, Any] | None:
    """Extract __NEXT_DATA__ JSON from HTML."""
    m = NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_listing(next_data: dict[str, Any], url: str) -> dict[str, Any] | None:
    """Parse a Wasalt listing from __NEXT_DATA__ into a trimmed dict.

    Returns None if the page does not contain a valid propertyDetailsV3.
    """
    try:
        details = next_data["props"]["pageProps"]["propertyDetailsV3"]
    except (KeyError, TypeError):
        return None

    # Strip PII before anything else
    details = strip_pii(details)

    pi = details.get("propertyInfo", {})
    attrs = details.get("attributes", [])
    loc = details.get("location", {})

    source_id = str(details.get("id", ""))
    if not source_id:
        return None

    # Property type mapping
    sub_type = pi.get("propertySubType")
    prop_type, prop_type_raw = map_property_type(sub_type)

    # Purpose
    purpose = pi.get("propertyFor", "").lower()
    if purpose not in ("sale", "rent"):
        return None

    # Price
    price_sar: int | None = None
    if purpose == "sale":
        sp = pi.get("salePrice")
        if sp and sp != "":
            try:
                price_sar = int(sp)
            except (ValueError, TypeError):
                pass
    else:
        # Rent: use expectedRent or rentFreq amount
        er = pi.get("expectedRent")
        if er and er != "":
            try:
                price_sar = int(er)
            except (ValueError, TypeError):
                pass

    # Rent period
    rent_period: str | None = None
    annual_rent_sar: int | None = None
    if purpose == "rent":
        rent_freq_data = pi.get("rentFreq")
        rent_period, rent_amount = map_rent_freq(rent_freq_data)
        if rent_amount and price_sar is None:
            price_sar = rent_amount
        if rent_period and price_sar:
            factors = {"daily": 365, "weekly": 52, "monthly": 12, "yearly": 1}
            factor = factors.get(rent_period, 0)
            if factor:
                annual_rent_sar = price_sar * factor

    # Attributes
    bedrooms: int | None = None
    bathrooms: int | None = None
    area_sqm: float | None = None
    amenities: list[str] = []

    for attr in attrs:
        key = attr.get("key", "")
        val = attr.get("value")
        if key == "noOfBedrooms" and val is not None:
            try:
                bedrooms = int(val)
            except (ValueError, TypeError):
                pass
        elif key == "noOfBathrooms" and val is not None:
            try:
                bathrooms = int(val)
            except (ValueError, TypeError):
                pass
        elif key in ("carpetArea", "builtUpArea", "plotArea") and val is not None:
            try:
                candidate = float(val)
                # Keep the largest area value
                if area_sqm is None or candidate > area_sqm:
                    area_sqm = candidate
            except (ValueError, TypeError):
                pass
        elif key not in ("noOfBedrooms", "noOfBathrooms", "carpetArea",
                         "builtUpArea", "plotArea"):
            name = attr.get("name", "")
            if name and val:
                amenities.append(f"{name}: {val}")

    # Location
    lat: float | None = None
    lon: float | None = None
    if loc:
        try:
            lat = float(loc["lat"])
            lon = float(loc["lon"])
        except (KeyError, ValueError, TypeError):
            pass

    # Description: strip HTML tags, map empty to null
    description = pi.get("description", "") or ""
    if description:
        description = re.sub(r"<[^>]+>", " ", description).strip()
        description = re.sub(r"\s+", " ", description)
    if not description:
        description = None  # WP-1a: empty descriptions -> null
    elif len(description) > 400:
        description = description[:397] + "..."

    # Title
    title = pi.get("title", "").strip()
    if not title:
        title = pi.get("propertyName", "").strip()
    if not title:
        return None

    # Published at
    published_at = details.get("publishedAt")

    # City, district
    city = pi.get("city", "").strip() or None
    district = pi.get("district", "").strip() or None

    return {
        "source_id": source_id,
        "url": url,
        "title": title,
        "description": description,
        "purpose": purpose,
        "property_type": prop_type,
        "property_type_raw": prop_type_raw,
        "city": city,
        "district": district,
        "price_sar": price_sar if price_sar and price_sar > 0 else None,
        "rent_period": rent_period,
        "annual_rent_sar": annual_rent_sar if annual_rent_sar and annual_rent_sar > 0 else None,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area_sqm": area_sqm,
        "lat": lat,
        "lon": lon,
        "amenities": amenities,
        "published_at": published_at,
        "_raw_details": details,  # kept for saving the trimmed JSON
    }


# ---------------------------------------------------------------------------
# URL extraction from sitemaps
# ---------------------------------------------------------------------------

def parse_sitemap_urls(xml_text: str) -> list[str]:
    """Extract all <loc> URLs from a sitemap XML string."""
    root = ET.fromstring(xml_text)
    urls = []
    for url_elem in root.findall("s:url/s:loc", SITEMAP_NS):
        if url_elem.text:
            urls.append(url_elem.text.strip())
    return urls


def extract_id_from_url(url: str) -> str | None:
    """Extract the numeric Wasalt ID from a listing URL.

    URLs look like: .../property/sale/villa-333-sqm-...-5841671
    The ID is the last numeric segment after the last hyphen.
    """
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1]
    # ID is at the end after last hyphen
    parts = slug.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    # Try the full slug if it is all digits
    if slug.isdigit():
        return slug
    return None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def check_disk_space(path: str = "E:/") -> bool:
    """Return True if free space is above MIN_FREE_BYTES."""
    usage = shutil.disk_usage(path)
    if usage.free < MIN_FREE_BYTES:
        logger.error(
            "Disk space below 3 GB threshold: %.1f GB free. Stopping.",
            usage.free / (1024**3),
        )
        return False
    return True


def collect(
    max_sale: int = 2000,
    max_rent: int = 1000,
    output_dir: Path | None = None,
) -> dict[str, int]:
    """Run the Wasalt collection.

    Returns counts dict: {sale_saved, rent_saved, skipped, errors}.
    Uses curl for HTTP requests because Wasalt's Cloudflare setup blocks
    Python HTTP libraries (httpx, requests) via TLS fingerprinting while
    allowing standard curl.
    """
    if output_dir is None:
        output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rate_limiter = RateLimiter(interval=1.0)
    backoff = BackoffHandler()

    stats = {"sale_saved": 0, "rent_saved": 0, "skipped": 0, "errors": 0}

    # Fetch sitemaps (CDN does not block httpx/curl)
    logger.info("Fetching sale sitemap from CDN...")
    sale_urls = _fetch_sitemap(SALE_SITEMAP)
    logger.info("Found %d sale/rent URLs in product sitemap", len(sale_urls))

    logger.info("Fetching rental sitemap from CDN...")
    rental_urls = _fetch_sitemap(RENTAL_SITEMAP)
    logger.info("Found %d daily rental URLs", len(rental_urls))

    # Separate sale and rent from product sitemap
    sale_listing_urls = [u for u in sale_urls if "/property/sale/" in u]
    rent_listing_urls = [u for u in sale_urls if "/property/rent/" in u]

    logger.info(
        "Product sitemap: %d sale, %d rent",
        len(sale_listing_urls), len(rent_listing_urls),
    )

    # Process sale listings
    logger.info("Collecting up to %d sale listings...", max_sale)
    _collect_listings(
        sale_listing_urls, output_dir, max_sale,
        "sale", rate_limiter, backoff, stats,
    )

    # Process rent listings
    logger.info("Collecting up to %d rent listings...", max_rent)
    _collect_listings(
        rent_listing_urls, output_dir, max_rent,
        "rent", rate_limiter, backoff, stats,
    )

    logger.info(
        "Collection complete: %d sale, %d rent, %d skipped, %d errors",
        stats["sale_saved"], stats["rent_saved"], stats["skipped"], stats["errors"],
    )
    return stats


def _fetch_sitemap(url: str) -> list[str]:
    """Fetch and parse a sitemap URL using curl. Returns list of URLs."""
    try:
        body, status = _curl_fetch(url, timeout=60)
        if body is None:
            logger.error("Failed to fetch sitemap %s (HTTP %d)", url, status)
            return []
        return parse_sitemap_urls(body)
    except (OSError, ValueError):
        logger.exception("Failed to fetch sitemap %s", url)
        return []


def _collect_listings(
    urls: list[str],
    output_dir: Path,
    max_count: int,
    purpose: str,
    rate_limiter: RateLimiter,
    backoff: BackoffHandler,
    stats: dict[str, int],
) -> None:
    """Collect listings from a list of URLs."""
    saved_key = f"{purpose}_saved"
    count = 0

    for url in urls:
        if stats[saved_key] >= max_count:
            break

        # Check disk space every 100 listings
        if count % 100 == 0 and not check_disk_space():
            logger.error("Stopping due to low disk space")
            break

        # Check robots.txt
        parsed = urlparse(url)
        if not is_allowed_wasalt(parsed.path):
            logger.debug("Skipping disallowed path: %s", parsed.path)
            continue

        # Extract ID for resumability
        lid = extract_id_from_url(url)
        if not lid:
            stats["errors"] += 1
            continue

        # Skip if already saved (resumable)
        out_path = output_dir / f"{lid}.json.gz"
        if out_path.exists():
            stats["skipped"] += 1
            count += 1
            continue

        # Rate limit
        rate_limiter.wait()
        count += 1

        # Fetch with retry
        html = _fetch_with_retry(url, backoff)
        if html is None:
            stats["errors"] += 1
            continue

        # Extract and parse
        next_data = extract_next_data(html)
        if next_data is None:
            logger.warning("No __NEXT_DATA__ in %s", url)
            stats["errors"] += 1
            continue

        parsed_data = parse_listing(next_data, url)
        if parsed_data is None:
            logger.warning("Failed to parse listing from %s", url)
            stats["errors"] += 1
            continue

        # Verify purpose matches
        if parsed_data["purpose"] != purpose:
            # This can happen if URL categorisation doesn't match data
            logger.debug(
                "URL says %s but data says %s: %s",
                purpose, parsed_data["purpose"], url,
            )

        # Save trimmed JSON (without _raw_details internal key)
        raw_details = parsed_data.pop("_raw_details")
        save_data = {
            "parsed": parsed_data,
            "raw_property_details": raw_details,
            "scraped_at": datetime.now(UTC).isoformat(),
        }

        try:
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False)
            stats[saved_key] += 1

            if stats[saved_key] % 50 == 0:
                logger.info(
                    "%s: saved %d / %d target",
                    purpose, stats[saved_key], max_count,
                )
        except OSError:
            logger.exception("Failed to write %s", out_path)
            stats["errors"] += 1


def _curl_fetch(url: str, timeout: int = 30) -> tuple[str | None, int]:
    """Fetch a URL using curl subprocess.

    Returns (body, http_status). body is None on failure.
    """
    try:
        result = subprocess.run(
            [
                "curl", "-sS",
                "--max-time", str(timeout),
                "-H", f"User-Agent: {USER_AGENT}",
                "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "-H", "Accept-Language: en-US,en;q=0.5",
                "-w", "\n%{http_code}",
                "-L",  # follow redirects
                url,
            ],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout + 5,
        )
        if not result.stdout:
            logger.warning("curl returned empty output for %s", url)
            return None, 0
        # Last line is the HTTP status code
        lines = result.stdout.rsplit("\n", 1)
        if len(lines) == 2:
            body, status_str = lines
            try:
                status = int(status_str.strip())
            except ValueError:
                body = result.stdout
                status = 0
        else:
            body = result.stdout
            status = 0

        if status == 200:
            return body, 200
        if status > 0:
            logger.warning("curl HTTP %d for %s", status, url)
        else:
            logger.warning("curl failed for %s: %s", url, (result.stderr or "")[:200])
        return None, status

    except subprocess.TimeoutExpired:
        logger.warning("curl timeout for %s", url)
        return None, 0
    except FileNotFoundError:
        logger.error("curl not found in PATH")
        return None, 0


def _fetch_with_retry(
    url: str, backoff: BackoffHandler, max_attempts: int = 3,
) -> str | None:
    """Fetch a URL with retry and backoff on 429/5xx only.

    Does NOT retry on 4xx (except 429).
    """
    for attempt in range(max_attempts):
        body, status = _curl_fetch(url)
        if body is not None:
            return body
        # Only retry on 429, 5xx, or connection failure (status 0)
        retryable = status == 429 or status >= 500 or status == 0
        if retryable and attempt < max_attempts - 1:
            delay = backoff.should_retry(status or 503, attempt)
            if delay is not None:
                time.sleep(delay)
                continue
        # 4xx (except 429) or non-retryable: give up immediately
        break
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wasalt listing collector")
    parser.add_argument("--max-sale", type=int, default=2000)
    parser.add_argument("--max-rent", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    # Set up logging
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    out_dir = Path(args.output_dir) if args.output_dir else RAW_DIR
    stats = collect(max_sale=args.max_sale, max_rent=args.max_rent, output_dir=out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
