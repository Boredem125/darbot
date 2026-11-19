"""DarGlobal page collector.

Fetches DarGlobal project, press, insight, and blog pages using
Playwright (headless Chromium), extracts main-content text with
trafilatura, and saves gzipped JSON to data/raw/darglobal/{slug}.json.gz.

Resumable: skips slugs whose .json.gz already exists.

Usage (inside the Playwright Docker container):
    python -m collect.darglobal [--max-blog 80]

PII policy:
    - Contact names, phone numbers and emails of staff are stripped
      from extracted text at save time.

Politeness:
    - At most 1 page per 3 s (brief requirement).
    - Honest User-Agent (default Chromium UA, no faking).
    - No stealth plugins, no CAPTCHA solvers, no proxies.
    - Backoff on 429/5xx.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/darglobal")
MIN_FREE_BYTES = 3 * 1024 * 1024 * 1024  # 3 GB

# Sitemap namespace
SITEMAP_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Pages known NOT to be project/article content (structural, marketing, etc.)
_NON_CONTENT_SLUGS: set[str] = {
    "", "about", "become-a-broker", "become-an-agent", "blog", "bookYourUnit",
    "campaigns", "careers", "commercial", "development-management", "dg-circle",
    "exclusive-membership", "faq", "get-in-touch", "hospitality", "insights",
    "investor", "landing-page", "new-home-page", "one-of-one",
    "partners", "pay-online", "press", "privacy-policy",
    "success", "terms-conditions", "terms-of-uses", "thank-you",
    "tokenization", "why-invest", "win-a-trip-to-dubai", "sohar-islamic",
    "live-all-in", "luxury-golf-communities", "insignia", "internship-application",
    "internships",
}

# Patterns for multi-segment paths that are sub-projects (project children)
_SUB_PROJECT_PREFIXES = (
    "one-of-one/", "rayana/", "amaya/", "trump-plaza/", "landing-page/",
)

# Patterns for partner pages
_PARTNER_PREFIX = "partners/"

# Marketing / landing pages to skip
_MARKETING_SLUGS: set[str] = {
    "DarGlobal-exclusive-Off-Plan-Investment-Catalogue",
    "pre-launch-trump-jeddah-tower",
    "DarGlobal-Insignia-Card",
    "Invest-crypto",
    "exclusive-plots",
    "invest-in-aida",
    "broker-contest-terms-conditions",
}

# Keywords that indicate a blog post is in scope (mentions a project, Saudi/Gulf city, or market)
_BLOG_SCOPE_KEYWORDS = {
    # DarGlobal project keywords
    "trump", "missoni", "pagani", "aston martin", "astera", "lamborghini",
    "elie saab", "les vagues", "tierra viva", "w residences", "dg1",
    "davinci", "urban oasis", "marea", "mulliner", "neptune",
    "fendi", "mouawad", "aida", "amaya", "rayana", "fairway",
    "marriott", "nickelodeon", "paramount",
    # Saudi/Gulf cities and market terms
    "saudi", "jeddah", "riyadh", "dammam", "khobar", "makkah", "madinah",
    "dubai", "abu dhabi", "doha", "qatar", "oman", "muscat", "rak",
    "ras al khaimah", "gulf", "gcc", "mena",
    # Real estate terms relevant to the market
    "off-plan", "off plan", "real estate", "investment", "luxury living",
    "branded residences", "branded residence",
}

# Phone number patterns (Saudi and international)
_PHONE_RE = re.compile(
    r"""(?:"""
    r"""\+?(?:966|971|968|974)\s*[\d\s\-()]{7,15}"""  # Gulf country codes
    r"""|(?:\b0\d{1,2}[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b)"""  # local format
    r"""|(?:\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4})"""  # international
    r""")""",
    re.VERBOSE,
)

# Email pattern
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")


def parse_sitemap_urls(xml_text: str) -> list[str]:
    """Extract all <loc> URLs from a sitemap XML string.

    Handles both raw XML and XML rendered inside HTML (Chromium wraps XML
    in an HTML viewer, putting the real content inside
    <div id="webkit-xml-viewer-source-xml">).
    """
    # Try to extract the embedded XML from Chromium's HTML wrapper
    marker = '<div id="webkit-xml-viewer-source-xml">'
    idx = xml_text.find(marker)
    if idx != -1:
        start = idx + len(marker)
        end = xml_text.find("</div>", start)
        if end != -1:
            xml_text = xml_text[start:end]
        else:
            # Truncated file: take everything after the marker
            xml_text = xml_text[start:]

    # Ensure the XML is well-formed; if truncated, close the urlset
    if not xml_text.rstrip().endswith("</urlset>"):
        # Remove any partial <url> at the end
        last_complete = xml_text.rfind("</url>")
        if last_complete != -1:
            xml_text = xml_text[: last_complete + len("</url>")] + "\n</urlset>"

    root = ET.fromstring(xml_text)
    urls: list[str] = []
    for url_elem in root.findall("s:url/s:loc", SITEMAP_NS):
        if url_elem.text:
            urls.append(url_elem.text.strip())
    return urls


def classify_url(url: str) -> str | None:
    """Classify a DarGlobal URL into a page type.

    Returns one of: 'project', 'press', 'insight', 'blog', or None (skip).
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    if not path:
        return None

    # Press pages
    if path.startswith("press/") and path != "press":
        return "press"

    # Insights pages
    if path.startswith("insights/") and path != "insights":
        return "insight"

    # Blog pages
    if path.startswith("blog/") and path != "blog":
        return "blog"

    # Sub-project pages are still project pages (check before non-content filter)
    for prefix in _SUB_PROJECT_PREFIXES:
        if path.startswith(prefix) and path != prefix.rstrip("/"):
            return "project"

    # Skip known non-content pages
    slug = path.split("/")[0]
    if slug in _NON_CONTENT_SLUGS:
        return None

    # Skip marketing slugs
    if path in _MARKETING_SLUGS:
        return None

    # Skip partner pages
    if path.startswith(_PARTNER_PREFIX):
        return None

    # Skip pay-online sub-paths
    if path.startswith("pay-online/"):
        return None

    # Remaining root slugs and sub-project paths are project pages
    if "/" not in path or path.startswith("landingpage/"):
        # Root slugs like /trump-tower-jeddah, /the-astera etc.
        if path.startswith("landingpage/"):
            return None  # skip landing pages
        return "project"

    # Multi-segment paths not matching above patterns - skip
    return None


def slug_from_url(url: str) -> str:
    """Extract a filesystem-safe slug from a DarGlobal URL."""
    path = urlparse(url).path.strip("/")
    # Replace slashes with double underscores for sub-paths
    slug = path.replace("/", "__")
    # Sanitise for filesystem
    slug = re.sub(r"[^A-Za-z0-9._\-]", "_", slug)
    return slug or "index"


def blog_in_scope(slug: str) -> bool:
    """Check if a blog post slug mentions a project, city or market keyword."""
    lower = slug.lower().replace("-", " ").replace("_", " ")
    return any(kw in lower for kw in _BLOG_SCOPE_KEYWORDS)


def strip_pii_from_text(text: str) -> str:
    """Remove contact names, phone numbers and emails from extracted text."""
    text = _EMAIL_RE.sub("", text)
    text = _PHONE_RE.sub("", text)
    # Collapse multiple spaces left behind
    text = re.sub(r"  +", " ", text)
    return text.strip()


def extract_text(html: str) -> tuple[str | None, str | None]:
    """Extract main-content text and title from HTML using trafilatura.

    Returns (text, title). Both may be None on failure.
    """
    try:
        import trafilatura
    except ImportError:
        logger.error("trafilatura not installed; cannot extract text")
        return None, None

    result = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=False,
        output_format="txt",
    )
    title = None
    try:
        metadata = trafilatura.extract_metadata(html)
        if metadata and metadata.title:
            title = metadata.title
    except Exception:  # noqa: BLE001
        logger.debug("Could not extract metadata")

    return result, title


def check_disk_space(path: str | None = None) -> bool:
    """Return True if free space is above MIN_FREE_BYTES.

    Auto-detects E:/ on Windows, / on Linux/Docker.
    """
    if path is None:
        path = "E:/" if shutil.os.name == "nt" else "/"
    try:
        usage = shutil.disk_usage(path)
    except (FileNotFoundError, OSError):
        # Inside container, can't check host disk
        logger.warning("Cannot check disk space at %s, assuming OK", path)
        return True
    if usage.free < MIN_FREE_BYTES:
        logger.error(
            "Disk space below threshold: %.1f GB free. Stopping.",
            usage.free / (1024**3),
        )
        return False
    return True


def crawl(
    *,
    max_blog: int = 80,
    output_dir: Path | None = None,
    sitemap_html: str | None = None,
) -> dict[str, int | list[str]]:
    """Run the DarGlobal crawl.

    If sitemap_html is provided, it is used instead of fetching the sitemap.
    Returns stats dict with counts and skipped URLs.
    """
    from playwright.sync_api import sync_playwright

    if output_dir is None:
        output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int | list[str]] = {
        "project_saved": 0,
        "press_saved": 0,
        "insight_saved": 0,
        "blog_saved": 0,
        "skipped_existing": 0,
        "skipped_type": 0,
        "skipped_blog_scope": 0,
        "errors": 0,
        "blocked": [],
        "saved_pages": [],
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Fetch sitemap if not provided
        if sitemap_html is None:
            logger.info("Fetching sitemap...")
            time.sleep(3)
            try:
                resp = page.goto(
                    "https://www.darglobal.co.uk/sitemap.xml",
                    wait_until="networkidle",
                    timeout=30000,
                )
                page.wait_for_timeout(3000)
                sitemap_html = page.content()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to fetch sitemap: %s", exc)
                browser.close()
                return stats

        urls = parse_sitemap_urls(sitemap_html)
        logger.info("Sitemap contains %d URLs", len(urls))

        # Classify URLs
        classified: dict[str, list[str]] = {
            "project": [], "press": [], "insight": [], "blog": [],
        }
        for url in urls:
            page_type = classify_url(url)
            if page_type:
                classified[page_type].append(url)
            else:
                stats["skipped_type"] = int(stats["skipped_type"]) + 1

        logger.info(
            "Classified: %d project, %d press, %d insight, %d blog",
            len(classified["project"]),
            len(classified["press"]),
            len(classified["insight"]),
            len(classified["blog"]),
        )

        # Filter blog posts for scope
        in_scope_blogs: list[str] = []
        for url in classified["blog"]:
            slug = slug_from_url(url)
            if blog_in_scope(slug):
                in_scope_blogs.append(url)
            else:
                stats["skipped_blog_scope"] = int(stats["skipped_blog_scope"]) + 1
        in_scope_blogs = in_scope_blogs[:max_blog]

        # Fetch all pages in order: projects, press, insights, blogs
        all_pages: list[tuple[str, str]] = []
        for url in classified["project"]:
            all_pages.append((url, "project"))
        for url in classified["press"]:
            all_pages.append((url, "press"))
        for url in classified["insight"]:
            all_pages.append((url, "insight"))
        for url in in_scope_blogs:
            all_pages.append((url, "blog"))

        for url, page_type in all_pages:
            if not check_disk_space():
                logger.error("Stopping due to low disk space")
                break

            slug = slug_from_url(url)
            out_path = output_dir / f"{slug}.json.gz"

            # Resumable: skip if already saved
            if out_path.exists():
                stats["skipped_existing"] = int(stats["skipped_existing"]) + 1
                continue

            # Rate limit: at most 1 page per 3 s
            time.sleep(3)

            try:
                resp = page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(3000)
                status = resp.status if resp else None
                html = page.content()

                # Check for challenge/block
                title = page.title()
                body_text_len = len(page.inner_text("body"))
                challenge_markers = ["_Incapsula_Resource", "incapsula", "reese84"]
                is_blocked = (
                    body_text_len < 200
                    or any(m.lower() in html.lower() for m in challenge_markers)
                    and body_text_len < 500
                )

                if is_blocked:
                    logger.warning("Blocked on %s (status=%s, text=%d)", url, status, body_text_len)
                    blocked_list = stats["blocked"]
                    if isinstance(blocked_list, list):
                        blocked_list.append(url)
                    stats["errors"] = int(stats["errors"]) + 1
                    continue

                # Extract text
                text, extracted_title = extract_text(html)
                if not text or len(text) < 50:
                    logger.warning("No useful text from %s (len=%s)", url, len(text) if text else 0)
                    stats["errors"] = int(stats["errors"]) + 1
                    continue

                # Strip PII from text
                text = strip_pii_from_text(text)

                # Use page title if trafilatura didn't get one
                if not extracted_title:
                    extracted_title = title

                record = {
                    "url": url,
                    "title": extracted_title or slug,
                    "type": page_type,
                    "text": text,
                    "fetched_at": datetime.now(UTC).isoformat(),
                }

                with gzip.open(out_path, "wt", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False)

                key = f"{page_type}_saved"
                stats[key] = int(stats[key]) + 1

                saved_list = stats["saved_pages"]
                if isinstance(saved_list, list):
                    saved_list.append(f"{slug} ({page_type}, {len(text)} chars)")

                if (int(stats[key])) % 10 == 0:
                    logger.info("%s: saved %d", page_type, int(stats[key]))

            except Exception as exc:  # noqa: BLE001
                logger.warning("Error fetching %s: %s", url, exc)
                stats["errors"] = int(stats["errors"]) + 1

        browser.close()

    logger.info(
        "Crawl complete: %d project, %d press, %d insight, %d blog, %d errors",
        stats["project_saved"], stats["press_saved"],
        stats["insight_saved"], stats["blog_saved"], stats["errors"],
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="DarGlobal page collector")
    parser.add_argument("--max-blog", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    out_dir = Path(args.output_dir) if args.output_dir else RAW_DIR
    stats = crawl(max_blog=args.max_blog, output_dir=out_dir)
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, list)}, indent=2))
    print(f"\nSaved pages: {len(stats.get('saved_pages', []))}")
    if stats.get("blocked"):
        print(f"Blocked URLs: {stats['blocked']}")


if __name__ == "__main__":
    main()
