"""Extract Project records from DarGlobal raw pages using Groq LLM.

Reads data/raw/darglobal/*.json.gz (type=project), calls Groq with
the extract_project_v1 prompt, applies the verbatim check, builds body
from page text, and writes data/processed/darglobal_projects.jsonl.

Placement rationale: this module is in `organize/` because it transforms
raw data into validated processed records, which is the organize stage's
responsibility. `collect/` only fetches raw data. R9 allows openai SDK
in organize. R12 forbids collect/organize cross-imports.

Usage:
    python -m organize.extract_projects
    python -m organize.extract_projects --slugs one-of-one__top-of-the-astera,rayana__trump-mansions --model qwen/qwen3.8-27b --response-format json_object
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from common.schemas import Project, project_id
from organize.sanitise import sanitise_text

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/darglobal")
PROCESSED_DIR = Path("data/processed")
PROMPT_PATH = Path("app/prompts/extract_project_v1.txt")
EVAL_DIR = Path("eval")

# Groq model chain
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "qwen/qwen3.8-27b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_CALLS = 110  # hard budget (original run)

# Minimum text length to consider a page extractable
MIN_TEXT_LEN = 50


# ---------------------------------------------------------------------------
# Slug extraction
# ---------------------------------------------------------------------------


def _slug_from_url(url: str) -> str:
    """Extract a schema-safe slug from a DarGlobal project URL.

    Uses __ for sub-paths (e.g. one-of-one/trump-penthouse -> one-of-one__trump-penthouse).
    """
    path = urlparse(url).path.strip("/")
    # Replace slashes with __ for sub-paths
    slug = path.replace("/", "__")
    # Sanitise: keep only allowed chars per schema pattern ^[A-Za-z0-9][A-Za-z0-9._-]*$
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-{2,}", "-", slug)
    # Ensure starts with alnum
    slug = slug.lstrip("._-") or "unknown"
    return slug[:120]


# ---------------------------------------------------------------------------
# Load raw project pages
# ---------------------------------------------------------------------------


def load_project_pages(raw_dir: Path | None = None) -> list[dict]:
    """Load raw project pages, deduplicating by URL casefold AND by text hash.

    Returns list of raw page dicts with an added 'slug' key.
    Keeps the first file seen per lowered URL or text hash (sorted order).
    The Trump Hotel Dubai URL-casing duplicate is caught by text hash
    since the URLs differ slightly (one has 'and-').
    """
    if raw_dir is None:
        raw_dir = RAW_DIR

    pages: list[dict] = []
    seen_urls: set[str] = set()
    seen_text_hashes: set[str] = set()

    for gz_path in sorted(raw_dir.glob("*.json.gz")):
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                record = json.load(f)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to read %s", gz_path)
            continue

        if record.get("type") != "project":
            continue

        text = record.get("text", "")
        if len(text) < MIN_TEXT_LEN:
            logger.info("Skipping %s: text too short (%d chars)", gz_path.name, len(text))
            continue

        url = record.get("url", "")
        url_key = url.lower()
        if url_key in seen_urls:
            logger.info("Dedup: skipping %s (URL casefold duplicate)", gz_path.name)
            continue
        seen_urls.add(url_key)

        # Also dedup by text content hash (catches the Trump Hotel Dubai case)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if text_hash in seen_text_hashes:
            logger.info("Dedup: skipping %s (identical text content)", gz_path.name)
            continue
        seen_text_hashes.add(text_hash)

        slug = _slug_from_url(url)
        record["slug"] = slug
        record["_filename"] = gz_path.name
        pages.append(record)

    logger.info("Loaded %d project pages (after dedup)", len(pages))
    return pages


# ---------------------------------------------------------------------------
# Verbatim check
# ---------------------------------------------------------------------------

def _normalise_text(s: str) -> str:
    """Light normalisation for verbatim matching: NFKC, casefold, collapse ws."""
    t = unicodedata.normalize("NFKC", s)
    t = t.casefold()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _verbatim_check(value, page_text: str) -> bool:
    """Check if a value appears verbatim on the page.

    - None always passes.
    - Strings: casefold + normalise, then substring match.
    - Numbers: check if the number appears (with or without commas).
    - Lists: each element must pass individually; returns True only if all pass.
    """
    if value is None:
        return True
    if isinstance(value, list):
        return all(_verbatim_check(v, page_text) for v in value)

    norm_page = _normalise_text(page_text)

    if isinstance(value, str):
        return _normalise_text(value) in norm_page

    if isinstance(value, (int, float)):
        # Try the number as-is
        num_str = str(int(value)) if isinstance(value, int) or value == int(value) else str(value)
        if num_str in norm_page:
            return True
        # Try with commas (e.g. 1200000 -> check if "1,200,000" is on the page)
        if isinstance(value, int) or value == int(value):
            formatted = f"{int(value):,}"
            if formatted.casefold() in norm_page:
                return True
        return False

    return True


def _enforce_verbatim(
    extracted: dict,
    page_text: str,
    slug: str,
    nulled_log: list[dict],
) -> dict:
    """Null out any extracted value not found verbatim on the page.

    For list fields (unit_types, amenities), filters to only the elements
    that appear on the page.
    """
    string_fields = [
        "name", "brand", "country", "city", "location",
        "status", "handover", "payment_plan",
    ]
    number_fields = [
        "bedrooms_min", "bedrooms_max",
        "area_sqm_min", "area_sqm_max",
        "price_from_amount",
    ]
    # price_from_currency is a 3-letter code, check verbatim too
    string_fields.append("price_from_currency")

    list_fields = ["unit_types", "amenities"]

    for field in string_fields + number_fields:
        val = extracted.get(field)
        if val is None:
            continue
        if not _verbatim_check(val, page_text):
            nulled_log.append({
                "page": slug,
                "field": field,
                "value": val if isinstance(val, str) else str(val),
            })
            extracted[field] = None

    # For list fields, filter to only verbatim elements
    for field in list_fields:
        vals = extracted.get(field, [])
        if not vals:
            continue
        kept = []
        for v in vals:
            if _verbatim_check(v, page_text):
                kept.append(v)
            else:
                nulled_log.append({
                    "page": slug,
                    "field": field,
                    "value": v,
                })
        extracted[field] = kept

    # Enforce price_from_amount and price_from_currency together
    if (extracted.get("price_from_amount") is None) != (extracted.get("price_from_currency") is None):
        if extracted.get("price_from_amount") is not None:
            nulled_log.append({
                "page": slug,
                "field": "price_from_amount",
                "value": str(extracted["price_from_amount"]),
            })
        if extracted.get("price_from_currency") is not None:
            nulled_log.append({
                "page": slug,
                "field": "price_from_currency",
                "value": str(extracted["price_from_currency"]),
            })
        extracted["price_from_amount"] = None
        extracted["price_from_currency"] = None

    return extracted


# ---------------------------------------------------------------------------
# Body extraction (not LLM)
# ---------------------------------------------------------------------------


def _build_body(text: str) -> str | None:
    """Build the body field from raw page text.

    Sanitised with Article.body rules: no URLs, phones, emails, < or >.
    Returns None if text is too short after sanitisation.
    """
    body = sanitise_text(text)
    if not body or len(body) < 20:
        return None
    return body


# ---------------------------------------------------------------------------
# Quota logging
# ---------------------------------------------------------------------------


def _init_quota_log(log_path: Path) -> None:
    """Create quota_log.csv with header if it does not exist."""
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "ts_utc,provider,model,purpose,wp,ok,prompt_tokens,completion_tokens\n",
            encoding="utf-8",
        )


def _log_call(
    log_path: Path,
    provider: str,
    model: str,
    purpose: str,
    wp: str,
    ok: bool,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Append one row to the quota log."""
    ts = datetime.now(UTC).isoformat()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, provider, model, purpose, wp, ok, prompt_tokens, completion_tokens])


# ---------------------------------------------------------------------------
# Rate limit parsing
# ---------------------------------------------------------------------------


def _parse_retry_after(error_msg: str) -> float:
    """Parse retry-after seconds from a Groq 429 error message.

    Example: "...Please try again in 4.8075s..."
    """
    match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(error_msg))
    if match:
        return float(match.group(1))
    return 10.0  # conservative default


def _parse_reset_duration(duration_str: str) -> float:
    """Parse a duration like '1m30s', '45s', '2m' into seconds."""
    total = 0.0
    m = re.search(r"(\d+)m", duration_str)
    if m:
        total += int(m.group(1)) * 60
    s = re.search(r"(\d+(?:\.\d+)?)s", duration_str)
    if s:
        total += float(s.group(1))
    return total if total > 0 else 10.0


# ---------------------------------------------------------------------------
# LLM extraction with proper rate limiting
# ---------------------------------------------------------------------------


async def _extract_one(
    client,
    model: str,
    prompt_text: str,
    page_text: str,
    tpm_state: dict,
    log_path: Path,
    call_counter: dict,
    response_format: str = "json_object",
    validation_error: str | None = None,
) -> dict | None:
    """Call Groq to extract fields from one page.

    Uses with_raw_response to read rate-limit headers.
    On 429, sleeps for the reset time and does NOT retry (the caller may retry).

    Parameters:
        response_format: "json_object" or "json_schema"
        validation_error: if set, appends a correction instruction to the prompt
    """
    from openai import RateLimitError

    # Build the actual prompt text, optionally with validation error feedback
    actual_prompt = prompt_text
    if validation_error:
        actual_prompt = (
            prompt_text
            + f"\n\nYour previous response had this issue: {validation_error}. "
            "Fix it and include all 16 keys."
        )

    messages = [
        {"role": "system", "content": actual_prompt},
        {"role": "user", "content": f"<page>\n{page_text}\n</page>"},
    ]

    # Estimate request tokens for the TPM guard
    msg_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
    est_prompt = -(-msg_bytes // 3)  # ceil division
    max_tokens = 1200
    est_total = est_prompt + max_tokens

    # TPM guard: sleep if the remaining bucket is below our estimate
    if tpm_state["remaining"] < est_total and tpm_state["reset_at"] > time.time():
        wait = tpm_state["reset_at"] - time.time() + 0.5  # small buffer
        if wait > 0:
            logger.info("TPM guard: waiting %.1fs for token reset", wait)
            tpm_state["waits"] += 1
            tpm_state["wait_total_s"] += wait
            await asyncio.sleep(wait)
            # After sleeping, reset the bucket estimate
            tpm_state["remaining"] = tpm_state["tpm_limit"]

    call_counter["total"] += 1

    # Build response_format parameter
    rf_param = {"type": response_format}

    # WP tag for quota log
    wp_tag = "WP-3-extract2"

    try:
        # Use with_raw_response to get headers
        raw = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format=rf_param,
            seed=0,
        )
        response = raw.parse()

        # Update TPM bucket from headers
        remaining_hdr = raw.headers.get("x-ratelimit-remaining-tokens")
        reset_hdr = raw.headers.get("x-ratelimit-reset-tokens")
        if remaining_hdr is not None:
            tpm_state["remaining"] = int(remaining_hdr)
        if reset_hdr is not None:
            tpm_state["reset_at"] = time.time() + _parse_reset_duration(reset_hdr)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else est_prompt
        completion_tokens = usage.completion_tokens if usage else 0

        _log_call(log_path, "groq", model, "extract", wp_tag, True,
                   prompt_tokens, completion_tokens)
        call_counter["ok"] += 1
        call_counter["prompt_tokens"] += prompt_tokens
        call_counter["completion_tokens"] += completion_tokens

        content = response.choices[0].message.content
        if not content:
            return None

        return json.loads(content)

    except RateLimitError as exc:
        call_counter["failed"] += 1
        _log_call(log_path, "groq", model, "extract", wp_tag, False, est_prompt, 0)
        wait = _parse_retry_after(str(exc))
        logger.info("429 rate limited, sleeping %.1fs", wait)
        tpm_state["waits"] += 1
        tpm_state["wait_total_s"] += wait
        tpm_state["remaining"] = 0
        tpm_state["reset_at"] = time.time() + wait
        await asyncio.sleep(wait)
        return None

    except Exception as exc:  # noqa: BLE001
        call_counter["failed"] += 1
        _log_call(log_path, "groq", model, "extract", wp_tag, False, est_prompt, 0)
        logger.warning("Groq call failed: %s", exc)

        # On 400 (bad JSON), wait briefly then return None (caller will skip)
        if "400" in str(exc):
            await asyncio.sleep(1.0)
        return None


async def _extract_with_retry(
    client,
    model: str,
    prompt_text: str,
    page: dict,
    tpm_state: dict,
    log_path: Path,
    call_counter: dict,
    max_calls: int = MAX_CALLS,
    response_format: str = "json_object",
) -> dict | None:
    """Extract with at most 2 calls per page (1 initial + 1 retry).

    The retry includes the validation error (missing keys) if applicable.
    A 429 that consumed a call slot counts as the page's attempt; the retry
    with the error message is the second and final call.
    """
    page_text = page["text"]

    # Call 1: initial attempt
    result = await _extract_one(
        client, model, prompt_text, page_text, tpm_state, log_path, call_counter,
        response_format=response_format,
    )

    if result is not None:
        # Validate: check the 16 expected keys
        expected_keys = {
            "name", "brand", "country", "city", "location", "status", "handover",
            "unit_types", "bedrooms_min", "bedrooms_max", "area_sqm_min", "area_sqm_max",
            "price_from_amount", "price_from_currency", "payment_plan", "amenities",
        }

        missing = expected_keys - result.keys()
        if missing:
            logger.warning("Missing keys %s for %s", sorted(missing), page.get("slug", "?"))
            # Call 2: retry with validation error
            if call_counter["total"] < max_calls:
                result2 = await _extract_one(
                    client, model, prompt_text, page_text, tpm_state, log_path, call_counter,
                    response_format=response_format,
                    validation_error=f"Missing keys: {sorted(missing)}",
                )
                if result2 is not None:
                    result = result2
            # Return whatever we have (possibly with missing keys)
        return result

    # result is None: initial call failed. Call 2: retry once.
    if call_counter["total"] < max_calls:
        result = await _extract_one(
            client, model, prompt_text, page_text, tpm_state, log_path, call_counter,
            response_format=response_format,
        )

    return result


# ---------------------------------------------------------------------------
# Project name cleanup (D2: fix bad project names)
# ---------------------------------------------------------------------------


_SEO_SUFFIX_RE = re.compile(
    r"\s*[|]\s*(?:DarGlobal|Dar\s*Global)\s*$",
    re.IGNORECASE,
)
_SEO_GENERAL_SUFFIX_RE = re.compile(
    r"\s*[|]\s*.*$",
)
_GENERIC_NAMES = frozenset({
    "darglobal",
    "dar global",
    "buy branded homes",
    "buy modern apartments",
    "ultra luxury apartments",
})


def _clean_project_name(name: str, slug: str, page_title: str) -> str:
    """Clean up a project name by stripping SEO title suffixes and
    replacing generic names with slug-derived names.

    D2: Strip suffixes like "| DarGlobal", handle page titles used as names.
    """
    cleaned = name.strip()

    # Strip "| DarGlobal" suffix
    cleaned = _SEO_SUFFIX_RE.sub("", cleaned).strip()

    # If still contains pipe characters or is very long, it is likely an SEO title.
    # Prefer the slug-derived name when the title is SEO-heavy.
    from_slug = False
    if "|" in cleaned or len(cleaned) > 80 or cleaned.lower().startswith(("ultra luxury", "buy ")):
        slug_name = _sub_project_from_slug(slug)

        # Try the first pipe segment first (often the real project name)
        first_seg = cleaned.split("|")[0].strip()
        if first_seg and len(first_seg) > 3 and _normalise_text(first_seg) not in _GENERIC_NAMES and "|" not in first_seg and len(first_seg) < 60:
            cleaned = first_seg
        else:
            cleaned = slug_name
            from_slug = True

    # If the name is still generic (DarGlobal, etc.), use slug
    if _normalise_text(cleaned) in _GENERIC_NAMES:
        cleaned = _sub_project_from_slug(slug)
        from_slug = True

    # Also check for the "projects" listing page name
    if _normalise_text(cleaned).startswith("buy modern apartments") or \
       _normalise_text(cleaned).startswith("buy branded homes"):
        cleaned = _sub_project_from_slug(slug)
        from_slug = True

    # Strip trailing location/master-project words from slug-derived names.
    # E.g. "The Great Escape 2 Aida Oman" -> "The Great Escape 2"
    if from_slug:
        _TAIL_WORDS = {"oman", "qatar", "dubai", "jeddah", "riyadh", "muscat",
                       "aida", "rayana", "amaya"}
        words = cleaned.split()
        while len(words) > 2 and words[-1].lower() in _TAIL_WORDS:
            words.pop()
        cleaned = " ".join(words)

    # Ensure not empty
    if not cleaned:
        cleaned = _sub_project_from_slug(slug)

    return cleaned[:200]


# ---------------------------------------------------------------------------
# Sub-project name disambiguation (step 3)
# ---------------------------------------------------------------------------


def _sub_project_from_title(base_name: str, page_title: str) -> str | None:
    """Extract the sub-project suffix from a page title.

    If the page title contains more information than the base project name,
    return the extra part. Otherwise return None.

    Skips generic site titles that do not relate to the project name.

    Example: base_name="Trump Plaza Jeddah", page_title="Trump Plaza Executive Residences"
    -> "Executive Residences"
    """
    if not page_title:
        return None

    # Strip common site suffixes like "| DarGlobal", "| Darglobal"
    cleaned_title = re.sub(r"\s*\|\s*(?:DarGlobal|Dar\s*Global)\s*$", "", page_title, flags=re.IGNORECASE).strip()

    norm_base = _normalise_text(base_name)
    norm_title = _normalise_text(cleaned_title)

    # If the cleaned title is the same as the base name, no extra info
    if norm_title == norm_base:
        return None

    # If the cleaned title is empty after stripping, no info
    if not norm_title:
        return None

    # Check if title contains any words from the base name (relevance check)
    base_words = set(norm_base.split())
    title_words = norm_title.split()
    title_word_set = set(title_words)
    overlap = base_words & title_word_set

    # If no overlap at all, the title is likely generic and not useful
    if not overlap:
        return None

    # Try to find overlapping prefix words (preserving order)
    base_word_list = norm_base.split()
    common = 0
    for bw, tw in zip(base_word_list, title_words):
        if bw == tw:
            common += 1
        else:
            break

    if common > 0 and common < len(title_words):
        # The suffix after the common prefix
        # Reconstruct from original cleaned title (preserve casing)
        orig_words = cleaned_title.split()
        suffix = " ".join(orig_words[common:])
        return suffix.strip() or None

    # If no common prefix but there is overlap, the title is different enough
    # to be a useful alternative name. But we prefer slug fallback for clarity.
    return None


def _sub_project_from_slug(slug: str) -> str:
    """Extract a human-readable sub-project name from the URL slug.

    Example: "trump-plaza__executive-residences" -> "Executive Residences"
    """
    if "__" in slug:
        sub_part = slug.split("__", 1)[1]
    else:
        sub_part = slug

    # Convert kebab-case to Title Case
    return sub_part.replace("-", " ").title()


def disambiguate_project_names(
    projects: list[dict],
) -> list[dict]:
    """Disambiguate duplicate project names by adding sub-project suffixes.

    For each group of 2+ projects sharing a name, append the sub-project
    to each name as "Base Name -- Sub Project". The sub-project comes from
    the page title if it carries extra info, else from the URL slug.

    Each project dict must have keys: slug, name, page_title.
    Returns the same list with 'name' values updated and a 'name_source' key
    indicating where the suffix came from ('title' or 'slug').
    """
    # Find duplicate names
    name_counts: Counter[str] = Counter(p["name"] for p in projects)
    duplicates = {name for name, count in name_counts.items() if count > 1}

    if not duplicates:
        for p in projects:
            p["name_source"] = "original"
        return projects

    for p in projects:
        if p["name"] not in duplicates:
            p["name_source"] = "original"
            continue

        # A slug without "__" is the parent page; it keeps the base name.
        # Only sub-pages (slugs with "__") get a disambiguating suffix.
        if "__" not in p["slug"]:
            p["name_source"] = "parent"
            continue

        # Try to get sub-project from page title
        suffix = _sub_project_from_title(p["name"], p.get("page_title", ""))
        if suffix:
            p["name"] = f'{p["name"]} – {suffix}'
            p["name_source"] = "title"
        else:
            # Fallback to slug sub-part
            suffix = _sub_project_from_slug(p["slug"])
            p["name"] = f'{p["name"]} – {suffix}'
            p["name_source"] = "slug"

    return projects


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


async def run_extraction(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    eval_dir: Path | None = None,
    slugs: list[str] | None = None,
    model: str | None = None,
    response_format: str = "json_object",
    max_calls: int = MAX_CALLS,
) -> dict:
    """Run the full extraction pipeline.

    Parameters:
        slugs: if set, only extract these slugs (targeted rerun)
        model: override the model (default: PRIMARY_MODEL)
        response_format: "json_object" or "json_schema"
        max_calls: hard budget for LLM calls

    Returns a summary dict with counts and stats.
    """
    from openai import AsyncOpenAI

    if raw_dir is None:
        raw_dir = RAW_DIR
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    if eval_dir is None:
        eval_dir = EVAL_DIR
    if model is None:
        model = PRIMARY_MODEL

    processed_dir.mkdir(parents=True, exist_ok=True)
    extract_raw_dir = processed_dir / "extract_raw"
    extract_raw_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")

    # Load pages
    all_pages = load_project_pages(raw_dir)

    # Filter to requested slugs if specified
    if slugs:
        slug_set = set(slugs)
        pages = [p for p in all_pages if p["slug"] in slug_set]
        logger.info("Filtered to %d pages matching slugs: %s", len(pages), slugs)
        missing_slugs = slug_set - {p["slug"] for p in pages}
        if missing_slugs:
            logger.warning("Slugs not found in raw pages: %s", sorted(missing_slugs))
    else:
        pages = all_pages

    # Groq client
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        max_retries=0,
    )

    log_path = eval_dir / "quota_log.csv"
    _init_quota_log(log_path)

    # State -- for targeted reruns, start counters at 0 (the budget is fresh)
    call_counter = {
        "total": 0, "ok": 0, "failed": 0,
        "prompt_tokens": 0, "completion_tokens": 0,
    }
    tpm_state = {
        "remaining": 8000,  # initial estimate from chain config
        "reset_at": 0.0,
        "tpm_limit": 8000,
        "waits": 0,
        "wait_total_s": 0.0,
    }

    new_projects: list[dict] = []  # newly extracted raw dicts
    nulled_log: list[dict] = []
    extract_errors: list[str] = []

    for i, page in enumerate(pages):
        slug = page["slug"]
        url = page.get("url", "").replace(" ", "%20")
        text = page["text"]
        fetched_at = page.get("fetched_at", "")

        # For targeted rerun: skip cache (re-extract)
        if slugs:
            extracted = None
        else:
            # Resume: skip pages that already have raw extraction
            raw_path = extract_raw_dir / f"{slug}.json"
            if raw_path.exists():
                try:
                    extracted = json.loads(raw_path.read_text(encoding="utf-8"))
                    logger.info("[%d/%d] Reusing cached extraction for %s", i + 1, len(pages), slug)
                except Exception:  # noqa: BLE001
                    extracted = None
                if extracted is None:
                    logger.info("[%d/%d] Re-extracting %s (corrupt cache)", i + 1, len(pages), slug)
            else:
                extracted = None

        if extracted is None:
            logger.info("[%d/%d] Extracting %s with model=%s format=%s",
                        i + 1, len(pages), slug, model, response_format)

            if call_counter["total"] >= max_calls:
                logger.warning("Budget exhausted at page %d/%d", i + 1, len(pages))
                extract_errors.append(f"Budget exhausted at {slug}")
                break

            # Extract from LLM
            extracted = await _extract_with_retry(
                client, model, prompt_text, page, tpm_state, log_path, call_counter,
                max_calls=max_calls,
                response_format=response_format,
            )

            if extracted is None:
                logger.warning("Extraction failed for %s", slug)
                extract_errors.append(slug)
                continue

            # Save raw extraction for audit
            raw_path = extract_raw_dir / f"{slug}.json"
            raw_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

        # Verbatim check
        extracted = _enforce_verbatim(extracted, text, slug, nulled_log)

        # Build body
        body = _build_body(text)

        # Parse scraped_at
        try:
            scraped_at = datetime.fromisoformat(fetched_at) if fetched_at else datetime.now(UTC)
        except (ValueError, TypeError):
            scraped_at = datetime.now(UTC)

        new_projects.append({
            "slug": slug,
            "url": url,
            "text": text,
            "page_title": page.get("title", ""),
            "extracted": extracted,
            "body": body,
            "scraped_at": scraped_at,
        })

    # For targeted rerun: merge new extractions into existing JSONL
    out_path = processed_dir / "darglobal_projects.jsonl"
    existing_projects: list[Project] = []

    # Build a lookup from slug to raw page (for page_title and original name)
    slug_to_raw_page = {p["slug"]: p for p in all_pages}

    if slugs and out_path.exists():
        # Load existing projects (keep those not being re-extracted)
        new_slugs = {p["slug"] for p in new_projects}
        with open(out_path, encoding="utf-8") as f:  # noqa: ASYNC230
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    proj = Project.model_validate_json(line)
                    if proj.slug not in new_slugs:
                        # Re-derive name from raw extraction if available
                        raw_page = slug_to_raw_page.get(proj.slug, {})
                        page_title = raw_page.get("title", "")
                        raw_path = extract_raw_dir / f"{proj.slug}.json"
                        if raw_path.exists():
                            try:
                                raw_ext = json.loads(raw_path.read_text(encoding="utf-8"))
                                raw_name = raw_ext.get("name") or page_title or proj.slug
                                clean_name = _clean_project_name(raw_name, proj.slug, page_title)
                                pd = proj.model_dump(mode="python")
                                pd["name"] = clean_name
                                proj = Project.model_validate(pd)
                            except Exception:  # noqa: BLE001
                                logger.debug("Could not re-derive name for %s", proj.slug)
                        existing_projects.append(proj)
                except Exception:  # noqa: BLE001
                    logger.debug("Skipping invalid project line during merge")
        logger.info("Loaded %d existing projects (kept, not re-extracted)", len(existing_projects))

    # Build Project objects from new extractions
    projects_to_write: list[Project] = list(existing_projects)
    # Collect info for disambiguation
    all_project_info: list[dict] = []

    for ep in existing_projects:
        page_title = slug_to_raw_page.get(ep.slug, {}).get("title", "")
        all_project_info.append({
            "slug": ep.slug,
            "name": ep.name,
            "page_title": page_title,
            "project": ep,
        })

    for np_data in new_projects:
        slug = np_data["slug"]
        url = np_data["url"]
        extracted = np_data["extracted"]
        body = np_data["body"]
        scraped_at = np_data["scraped_at"]

        doc_id = project_id(slug)
        try:
            project = Project(
                id=doc_id,
                source="darglobal",
                slug=slug,
                url=url,
                name=_clean_project_name(
                    extracted.get("name") or np_data["page_title"] or slug,
                    slug,
                    np_data["page_title"],
                ),
                brand=extracted.get("brand"),
                country=extracted.get("country"),
                city=extracted.get("city"),
                location=extracted.get("location"),
                status=extracted.get("status"),
                handover=extracted.get("handover"),
                unit_types=extracted.get("unit_types") or [],
                bedrooms_min=extracted.get("bedrooms_min"),
                bedrooms_max=extracted.get("bedrooms_max"),
                area_sqm_min=extracted.get("area_sqm_min"),
                area_sqm_max=extracted.get("area_sqm_max"),
                price_from_amount=extracted.get("price_from_amount"),
                price_from_currency=extracted.get("price_from_currency"),
                payment_plan=extracted.get("payment_plan"),
                amenities=extracted.get("amenities") or [],
                description=None,  # not extracted by LLM
                body=body,
                scraped_at=scraped_at,
            )
            projects_to_write.append(project)
            all_project_info.append({
                "slug": slug,
                "name": project.name,
                "page_title": np_data["page_title"],
                "project": project,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Project validation failed for %s: %s", slug, exc)
            extract_errors.append(f"{slug}: {exc}")

    # Disambiguate names (step 3)
    # Build a combined list for all projects (existing + new) with page titles
    all_info_for_disambig = [
        {
            "slug": proj.slug,
            "name": proj.name,
            "page_title": slug_to_raw_page.get(proj.slug, {}).get("title", ""),
        }
        for proj in projects_to_write
    ]

    disambiguated = disambiguate_project_names(all_info_for_disambig)

    # Apply disambiguated names back to projects
    name_map = {d["slug"]: d for d in disambiguated}
    for proj in projects_to_write:
        if proj.slug in name_map:
            new_name = name_map[proj.slug]["name"]
            if new_name != proj.name:
                logger.info("Disambiguated %s: %r -> %r", proj.slug, proj.name, new_name)
                # pydantic frozen model: rebuild with new name
                proj_dict = proj.model_dump(mode="python")
                proj_dict["name"] = new_name
                idx = projects_to_write.index(proj)
                projects_to_write[idx] = Project.model_validate(proj_dict)

    # Sort for determinism
    projects_to_write.sort(key=lambda p: p.id)

    # Write projects JSONL
    with open(out_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
        f.writelines(p.model_dump_json() + "\n" for p in projects_to_write)
    logger.info("Wrote %d projects to %s", len(projects_to_write), out_path)

    # Write nulled log (append for targeted rerun)
    nulled_path = processed_dir / "extract_nulled.jsonl"
    mode = "a" if slugs else "w"
    with open(nulled_path, mode, encoding="utf-8") as f:  # noqa: ASYNC230
        f.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in nulled_log)
    logger.info("Wrote %d nulled entries to %s", len(nulled_log), nulled_path)

    await client.close()

    summary = {
        "pages_total": len(pages),
        "projects_valid": len(projects_to_write),
        "extract_errors": len(extract_errors),
        "nulled_values": len(nulled_log),
        "calls_total": call_counter["total"],
        "calls_ok": call_counter["ok"],
        "calls_failed": call_counter["failed"],
        "prompt_tokens": call_counter["prompt_tokens"],
        "completion_tokens": call_counter["completion_tokens"],
        "tpm_waits": tpm_state["waits"],
        "tpm_wait_total_s": tpm_state["wait_total_s"],
        "errors": extract_errors,
        "disambiguation": {
            d["slug"]: {"name": d["name"], "source": d.get("name_source", "original")}
            for d in disambiguated if d.get("name_source") != "original"
        },
    }
    return summary


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Extract DarGlobal project pages")
    parser.add_argument("--slugs", type=str, default=None,
                        help="Comma-separated list of slugs to extract (targeted rerun)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the LLM model (e.g. qwen/qwen3.8-27b)")
    parser.add_argument("--response-format", type=str, default="json_object",
                        choices=["json_object", "json_schema"],
                        help="Groq response format (default: json_object)")
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS,
                        help=f"Max LLM calls (default: {MAX_CALLS})")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = _parse_cli_args()
    slugs = args.slugs.split(",") if args.slugs else None

    summary = asyncio.run(run_extraction(
        slugs=slugs,
        model=args.model,
        response_format=args.response_format,
        max_calls=args.max_calls,
    ))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
