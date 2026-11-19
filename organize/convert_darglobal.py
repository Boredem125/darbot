"""Convert crawled DarGlobal json.gz files to processed Article JSONL.

Reads data/raw/darglobal/*.json.gz, writes data/processed/darglobal_articles.jsonl.
Only converts press, insight, and blog pages (not projects, which are
handled by the extraction WP with Groq).

Usage:
    python -m organize.convert_darglobal
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from common.schemas import Article, ArticleKind, article_id
from organize.sanitise import sanitise_text

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/darglobal")
PROCESSED_DIR = Path("data/processed")


def _slug_from_url(url: str) -> str:
    """Extract a schema-safe slug from a DarGlobal URL.

    Uses the URL path with slashes replaced by hyphens, since the
    Article slug pattern is [A-Za-z0-9][A-Za-z0-9._-]*.
    """
    path = urlparse(url).path.strip("/")
    # Replace slashes with hyphens for sub-paths
    slug = path.replace("/", "-")
    # Sanitise: keep only allowed chars
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    # Ensure starts with alnum
    slug = slug.lstrip("._-") or "unknown"
    return slug[:200]


def convert_articles(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, int]:
    """Convert crawled json.gz to Article JSONL.

    Returns counts dict.
    """
    if raw_dir is None:
        raw_dir = RAW_DIR
    if processed_dir is None:
        processed_dir = PROCESSED_DIR

    processed_dir.mkdir(parents=True, exist_ok=True)

    article_kinds = {"press", "insight", "blog"}
    counts: dict[str, int] = {
        "press": 0, "insight": 0, "blog": 0,
        "skipped_project": 0, "errors": 0,
    }

    articles: list[Article] = []

    for gz_path in sorted(raw_dir.glob("*.json.gz")):
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                record = json.load(f)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read %s", gz_path)
            counts["errors"] += 1
            continue

        page_type = record.get("type", "")
        if page_type not in article_kinds:
            counts["skipped_project"] += 1
            continue

        url = record.get("url", "").replace(" ", "%20")
        title = record.get("title", "")
        text = record.get("text", "")
        fetched_at = record.get("fetched_at", "")

        if not text or len(text) < 50:
            logger.warning("Skipping %s: text too short (%d chars)", gz_path.name, len(text))
            counts["errors"] += 1
            continue

        # Sanitise text (removes URLs, phones, emails, HTML)
        body = sanitise_text(text)
        if not body or len(body) < 20:
            logger.warning("Skipping %s: sanitised text too short", gz_path.name)
            counts["errors"] += 1
            continue

        slug = _slug_from_url(url)
        kind = ArticleKind(page_type)
        doc_id = article_id(kind, slug)

        try:
            scraped_at = datetime.fromisoformat(fetched_at) if fetched_at else datetime.now(UTC)
        except (ValueError, TypeError):
            scraped_at = datetime.now(UTC)

        try:
            article = Article(
                id=doc_id,
                source="darglobal",
                kind=kind,
                slug=slug,
                url=url,
                title=title[:300] if title else slug,
                body=body,
                published_at=None,
                scraped_at=scraped_at,
            )
            articles.append(article)
            counts[page_type] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid article from %s: %s", gz_path.name, exc)
            counts["errors"] += 1

    # Dedupe by id (keep first seen) and sort for determinism
    seen_ids: set[str] = set()
    unique_articles: list[Article] = []
    for article in articles:
        if article.id not in seen_ids:
            seen_ids.add(article.id)
            unique_articles.append(article)
    articles = unique_articles
    articles.sort(key=lambda a: a.id)

    # Write JSONL
    out_path = processed_dir / "darglobal_articles.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(article.model_dump_json() + "\n")

    logger.info(
        "Converted %d articles: %d press, %d insight, %d blog, %d errors",
        len(articles), counts["press"], counts["insight"], counts["blog"],
        counts["errors"],
    )
    return counts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    counts = convert_articles()
    print(json.dumps(counts, indent=2))
