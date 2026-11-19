"""Tests for organize.extract_projects.

Tests the extraction pipeline: loading raw pages, slug extraction,
deduplication, verbatim checking, body sanitisation, Project record
building, retry cap, and sub-project name disambiguation.
LLM calls are mocked; no network required.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers to create raw page fixtures
# ---------------------------------------------------------------------------

RAW_TEMPLATE = {
    "url": "https://darglobal.co.uk/test-project",
    "title": "Test Project | DarGlobal",
    "type": "project",
    "text": "Test Project is a luxury development in Dubai, UAE. It offers 1-3 bedroom apartments from 50 to 150 sqm. Amenities include pool, gym.",
    "fetched_at": "2026-11-18T10:00:00+00:00",
}

EXTRACT_RESPONSE = {
    "name": "Test Project",
    "brand": None,
    "country": "UAE",
    "city": "Dubai",
    "location": None,
    "status": None,
    "handover": None,
    "unit_types": ["apartments"],
    "bedrooms_min": 1,
    "bedrooms_max": 3,
    "area_sqm_min": 50.0,
    "area_sqm_max": 150.0,
    "price_from_amount": None,
    "price_from_currency": None,
    "payment_plan": None,
    "amenities": ["pool", "gym"],
}


def _write_raw(tmp_path: Path, filename: str, data: dict | None = None) -> Path:
    """Write a raw json.gz file to a temp directory."""
    d = data or dict(RAW_TEMPLATE)
    p = tmp_path / filename
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(d, f)
    return p


# ---------------------------------------------------------------------------
# Tests for slug_from_url
# ---------------------------------------------------------------------------


class TestSlugFromUrl:
    def test_simple(self):
        from organize.extract_projects import _slug_from_url
        assert _slug_from_url("https://darglobal.co.uk/the-astera") == "the-astera"

    def test_subpath(self):
        from organize.extract_projects import _slug_from_url
        s = _slug_from_url("https://darglobal.co.uk/one-of-one/trump-penthouse")
        assert s == "one-of-one__trump-penthouse"

    def test_casing_preserved(self):
        from organize.extract_projects import _slug_from_url
        s = _slug_from_url("https://darglobal.co.uk/Trump-International-Hotel-Tower-Dubai")
        assert "Trump" in s

    def test_max_length(self):
        from organize.extract_projects import _slug_from_url
        long_slug = "a" * 300
        s = _slug_from_url(f"https://darglobal.co.uk/{long_slug}")
        assert len(s) <= 120


# ---------------------------------------------------------------------------
# Tests for load_project_pages
# ---------------------------------------------------------------------------


class TestLoadProjectPages:
    def test_loads_project_type(self, tmp_path):
        from organize.extract_projects import load_project_pages
        _write_raw(tmp_path, "test.json.gz")
        pages = load_project_pages(tmp_path)
        assert len(pages) == 1
        assert pages[0]["type"] == "project"

    def test_skips_non_project(self, tmp_path):
        from organize.extract_projects import load_project_pages
        d = dict(RAW_TEMPLATE, type="blog")
        _write_raw(tmp_path, "blog.json.gz", d)
        pages = load_project_pages(tmp_path)
        assert len(pages) == 0

    def test_dedupes_by_url_casefold(self, tmp_path):
        from organize.extract_projects import load_project_pages
        d1 = dict(RAW_TEMPLATE, url="https://darglobal.co.uk/test-project")
        d2 = dict(RAW_TEMPLATE, url="https://darglobal.co.uk/Test-Project")
        _write_raw(tmp_path, "lower.json.gz", d1)
        _write_raw(tmp_path, "upper.json.gz", d2)
        pages = load_project_pages(tmp_path)
        assert len(pages) == 1

    def test_skips_too_short_text(self, tmp_path):
        from organize.extract_projects import load_project_pages
        d = dict(RAW_TEMPLATE, text="short")
        _write_raw(tmp_path, "short.json.gz", d)
        pages = load_project_pages(tmp_path)
        assert len(pages) == 0


# ---------------------------------------------------------------------------
# Tests for verbatim check
# ---------------------------------------------------------------------------


class TestVerbatimCheck:
    def test_string_present(self):
        from organize.extract_projects import _verbatim_check
        page_text = "This is a luxury villa in Dubai"
        assert _verbatim_check("Dubai", page_text) is True

    def test_string_absent(self):
        from organize.extract_projects import _verbatim_check
        page_text = "This is a luxury villa in Riyadh"
        assert _verbatim_check("Dubai", page_text) is False

    def test_number_present(self):
        from organize.extract_projects import _verbatim_check
        page_text = "Ranging from 50 to 150 sqm"
        assert _verbatim_check(50, page_text) is True
        assert _verbatim_check(150.0, page_text) is True

    def test_number_absent(self):
        from organize.extract_projects import _verbatim_check
        page_text = "Ranging from 50 to 150 sqm"
        assert _verbatim_check(200, page_text) is False

    def test_comma_number(self):
        from organize.extract_projects import _verbatim_check
        page_text = "Starting from 1,200,000 AED"
        assert _verbatim_check(1200000, page_text) is True

    def test_list_elements(self):
        from organize.extract_projects import _verbatim_check
        page_text = "Amenities include pool, gym, spa"
        assert _verbatim_check(["pool", "gym", "spa"], page_text) is True
        assert _verbatim_check(["pool", "cinema"], page_text) is False

    def test_none_always_true(self):
        from organize.extract_projects import _verbatim_check
        assert _verbatim_check(None, "any text") is True


# ---------------------------------------------------------------------------
# Tests for body extraction
# ---------------------------------------------------------------------------


class TestBuildBody:
    def test_sanitises_text(self):
        from organize.extract_projects import _build_body
        text = "Hello https://example.com world test@email.com end. This is a sufficiently long text for extraction purposes here."
        body = _build_body(text)
        assert body is not None
        assert "https://" not in body
        assert "test@email.com" not in body
        assert "Hello" in body

    def test_none_for_short_text(self):
        from organize.extract_projects import _build_body
        assert _build_body("hi") is None

    def test_strips_angles(self):
        from organize.extract_projects import _build_body
        body = _build_body("Hello <script>alert(1)</script> world. This is a long enough text for extraction.")
        assert "<" not in (body or "")
        assert ">" not in (body or "")


# ---------------------------------------------------------------------------
# Tests for null enforcement
# ---------------------------------------------------------------------------


class TestNullEnforcement:
    def test_nulls_invented_values(self):
        from organize.extract_projects import _enforce_verbatim
        page_text = "Test Project in Dubai, UAE. 1-3 bedrooms. 50 to 150 sqm. apartments. pool, gym."
        extracted = dict(EXTRACT_RESPONSE, brand="Invented Brand")
        nulled = []
        result = _enforce_verbatim(extracted, page_text, "test-slug", nulled)
        assert result["brand"] is None
        assert any(n["field"] == "brand" for n in nulled)

    def test_keeps_verbatim_values(self):
        from organize.extract_projects import _enforce_verbatim
        page_text = "Test Project is a luxury development in Dubai, UAE. 1 to 3 bedrooms. 50 to 150 sqm. apartments. pool, gym."
        extracted = dict(EXTRACT_RESPONSE)
        nulled = []
        result = _enforce_verbatim(extracted, page_text, "test-slug", nulled)
        assert result["name"] == "Test Project"
        assert result["city"] == "Dubai"
        assert len(nulled) == 0

    def test_list_partial_null(self):
        from organize.extract_projects import _enforce_verbatim
        page_text = "Amenities include pool but not anything else"
        extracted = dict(EXTRACT_RESPONSE, amenities=["pool", "invented_thing"])
        nulled = []
        result = _enforce_verbatim(extracted, page_text, "test-slug", nulled)
        assert "pool" in result["amenities"]
        assert "invented_thing" not in result["amenities"]


# ---------------------------------------------------------------------------
# Tests for quota log
# ---------------------------------------------------------------------------


class TestQuotaLog:
    def test_log_creation(self, tmp_path):
        from organize.extract_projects import _init_quota_log, _log_call
        log_path = tmp_path / "quota_log.csv"
        _init_quota_log(log_path)
        assert log_path.exists()
        _log_call(log_path, "groq", "openai/gpt-oss-120b", "extract", "WP-3-extract", True, 100, 50)
        lines = log_path.read_text().splitlines()
        assert len(lines) == 2  # header + 1 data line
        assert "groq" in lines[1]


# ---------------------------------------------------------------------------
# Tests for retry cap (step 1: at most 2 calls per page)
# ---------------------------------------------------------------------------


class TestRetryCap:
    """A page that always fails must make exactly 2 calls (1 initial + 1 retry)."""

    def test_always_failing_page_makes_exactly_2_calls(self, tmp_path):
        """When the LLM always returns None, _extract_with_retry makes exactly 2 calls."""
        from organize.extract_projects import _extract_with_retry

        call_count = 0

        async def fake_extract_one(client, model, prompt_text, page_text,
                                   tpm_state, log_path, call_counter,
                                   response_format="json_object",
                                   validation_error=None):
            nonlocal call_count
            call_count += 1
            call_counter["total"] += 1
            call_counter["failed"] += 1
            return None  # noqa: RET501, PLR1711

        page = {"text": "some text", "slug": "fail-page"}
        call_counter = {"total": 0, "ok": 0, "failed": 0,
                        "prompt_tokens": 0, "completion_tokens": 0}
        tpm_state = {"remaining": 8000, "reset_at": 0.0,
                     "tpm_limit": 8000, "waits": 0, "wait_total_s": 0.0}
        log_path = tmp_path / "quota_log.csv"
        log_path.write_text("ts_utc,provider,model,purpose,wp,ok,prompt_tokens,completion_tokens\n")

        async def _run():
            with patch("organize.extract_projects._extract_one", side_effect=fake_extract_one):
                return await _extract_with_retry(
                    client=None,
                    model="test-model",
                    prompt_text="test prompt",
                    page=page,
                    tpm_state=tpm_state,
                    log_path=log_path,
                    call_counter=call_counter,
                    max_calls=100,
                )

        result = asyncio.run(_run())

        assert result is None
        assert call_count == 2, f"Expected exactly 2 calls, got {call_count}"

    def test_success_on_first_call_makes_1_call(self, tmp_path):
        """When the LLM succeeds on the first call, only 1 call is made."""
        from organize.extract_projects import _extract_with_retry

        call_count = 0

        async def fake_extract_one(client, model, prompt_text, page_text,
                                   tpm_state, log_path, call_counter,
                                   response_format="json_object",
                                   validation_error=None):
            nonlocal call_count
            call_count += 1
            call_counter["total"] += 1
            call_counter["ok"] += 1
            return dict(EXTRACT_RESPONSE)

        page = {"text": "some text", "slug": "ok-page"}
        call_counter = {"total": 0, "ok": 0, "failed": 0,
                        "prompt_tokens": 0, "completion_tokens": 0}
        tpm_state = {"remaining": 8000, "reset_at": 0.0,
                     "tpm_limit": 8000, "waits": 0, "wait_total_s": 0.0}
        log_path = tmp_path / "quota_log.csv"
        log_path.write_text("ts_utc,provider,model,purpose,wp,ok,prompt_tokens,completion_tokens\n")

        async def _run():
            with patch("organize.extract_projects._extract_one", side_effect=fake_extract_one):
                return await _extract_with_retry(
                    client=None,
                    model="test-model",
                    prompt_text="test prompt",
                    page=page,
                    tpm_state=tpm_state,
                    log_path=log_path,
                    call_counter=call_counter,
                    max_calls=100,
                )

        result = asyncio.run(_run())

        assert result is not None
        assert call_count == 1

    def test_missing_keys_does_not_add_third_call(self, tmp_path):
        """When the LLM returns partial keys, the retry with validation error
        is the second call, not a third."""
        from organize.extract_projects import _extract_with_retry

        call_count = 0
        partial = {"name": "X"}  # missing most keys

        async def fake_extract_one(client, model, prompt_text, page_text,
                                   tpm_state, log_path, call_counter,
                                   response_format="json_object",
                                   validation_error=None):
            nonlocal call_count
            call_count += 1
            call_counter["total"] += 1
            call_counter["ok"] += 1
            return dict(partial)

        page = {"text": "some text", "slug": "partial-page"}
        call_counter = {"total": 0, "ok": 0, "failed": 0,
                        "prompt_tokens": 0, "completion_tokens": 0}
        tpm_state = {"remaining": 8000, "reset_at": 0.0,
                     "tpm_limit": 8000, "waits": 0, "wait_total_s": 0.0}
        log_path = tmp_path / "quota_log.csv"
        log_path.write_text("ts_utc,provider,model,purpose,wp,ok,prompt_tokens,completion_tokens\n")

        async def _run():
            with patch("organize.extract_projects._extract_one", side_effect=fake_extract_one):
                return await _extract_with_retry(
                    client=None,
                    model="test-model",
                    prompt_text="test prompt",
                    page=page,
                    tpm_state=tpm_state,
                    log_path=log_path,
                    call_counter=call_counter,
                    max_calls=100,
                )

        asyncio.run(_run())

        # At most 2 calls regardless of missing keys
        assert call_count <= 2, f"Expected at most 2 calls, got {call_count}"


# ---------------------------------------------------------------------------
# Tests for sub-project name disambiguation (step 3)
# ---------------------------------------------------------------------------


class TestDisambiguateNames:
    def test_duplicate_names_get_suffix(self):
        """Two projects sharing a name get sub-project suffixes."""
        from organize.extract_projects import disambiguate_project_names

        projects = [
            {
                "slug": "trump-plaza__executive-residences",
                "name": "Trump Plaza Jeddah",
                "page_title": "Trump Plaza Executive Residences",
            },
            {
                "slug": "trump-plaza__park-residences",
                "name": "Trump Plaza Jeddah",
                "page_title": "Trump Plaza Park Residences",
            },
            {
                "slug": "the-astera",
                "name": "The Astera",
                "page_title": "The Astera",
            },
        ]
        result = disambiguate_project_names(projects)
        # The two Trump Plaza entries should have different names
        trump_names = [r["name"] for r in result if "trump-plaza" in r["slug"]]
        assert len(trump_names) == 2
        assert trump_names[0] != trump_names[1]
        # The unique name should be unchanged
        astera = next(r for r in result if r["slug"] == "the-astera")
        assert astera["name"] == "The Astera"

    def test_sub_project_from_page_title(self):
        """Sub-project name comes from page title when available."""
        from organize.extract_projects import disambiguate_project_names

        projects = [
            {
                "slug": "trump-plaza__executive-residences",
                "name": "Trump Plaza Jeddah",
                "page_title": "Trump Plaza Executive Residences",
            },
            {
                "slug": "trump-plaza__park-residences",
                "name": "Trump Plaza Jeddah",
                "page_title": "Trump Plaza Park Residences",
            },
        ]
        result = disambiguate_project_names(projects)
        exec_proj = next(r for r in result if "executive" in r["slug"])
        park_proj = next(r for r in result if "park" in r["slug"])
        # Names should include sub-project info
        assert "Executive" in exec_proj["name"]
        assert "Park" in park_proj["name"]

    def test_sub_project_from_slug_fallback(self):
        """When page title matches the base name, sub-project comes from URL slug."""
        from organize.extract_projects import disambiguate_project_names

        projects = [
            {
                "slug": "parent__sub-a",
                "name": "Same Name",
                "page_title": "Same Name",  # no extra info in title
            },
            {
                "slug": "parent__sub-b",
                "name": "Same Name",
                "page_title": "Same Name",
            },
        ]
        result = disambiguate_project_names(projects)
        names = [r["name"] for r in result]
        assert names[0] != names[1]
        # Slug-based: should include the sub-part from slug
        assert "Sub A" in names[0] or "sub-a" in names[0].lower()
        assert "Sub B" in names[1] or "sub-b" in names[1].lower()

    def test_no_change_when_names_unique(self):
        """When all names are unique, nothing changes."""
        from organize.extract_projects import disambiguate_project_names

        projects = [
            {"slug": "a", "name": "Project A", "page_title": "Project A"},
            {"slug": "b", "name": "Project B", "page_title": "Project B"},
        ]
        result = disambiguate_project_names(projects)
        assert result[0]["name"] == "Project A"
        assert result[1]["name"] == "Project B"

    def test_parent_page_keeps_base_name(self):
        """A slug without __ is the parent page and keeps the base name unsuffixed.
        Only sub-pages (slugs with __) get the suffix."""
        from organize.extract_projects import disambiguate_project_names

        projects = [
            {
                "slug": "trump-international-hotel-and-tower-dubai",
                "name": "Trump International Hotel & Tower Dubai",
                "page_title": "Trump International Hotel & Tower Dubai | DarGlobal",
            },
            {
                "slug": "one-of-one__trump-penthouse",
                "name": "Trump International Hotel & Tower Dubai",
                "page_title": "Buy Branded Homes | Million Dollar Houses for Sale | Darglobal",
            },
        ]
        result = disambiguate_project_names(projects)
        parent = next(r for r in result if "__" not in r["slug"])
        child = next(r for r in result if "__" in r["slug"])
        # Parent keeps the base name exactly
        assert parent["name"] == "Trump International Hotel & Tower Dubai"
        assert parent["name_source"] == "parent"
        # Child gets a suffix
        assert "–" in child["name"]
        assert child["name"].startswith("Trump International Hotel & Tower Dubai")


# ---------------------------------------------------------------------------
# Tests for project name cleanup (D2)
# ---------------------------------------------------------------------------


class TestCleanProjectName:
    def test_strips_darglobal_suffix(self):
        from organize.extract_projects import _clean_project_name

        name = _clean_project_name(
            "Sunrise Haven Villas | Luxury Beach Villa In Muscat, Oman | DarGlobal",
            "Sunrise-Haven-Luxury-Villas",
            "Sunrise Haven Villas | Luxury Beach Villa In Muscat, Oman | DarGlobal",
        )
        assert "DarGlobal" not in name
        assert "Sunrise" in name

    def test_generic_darglobal_uses_slug(self):
        from organize.extract_projects import _clean_project_name

        name = _clean_project_name(
            "DarGlobal",
            "aston-martin-media",
            "DarGlobal",
        )
        assert name != "DarGlobal"
        assert "Aston" in name or "aston" in name.lower()

    def test_great_escape_seo_title(self):
        from organize.extract_projects import _clean_project_name

        name = _clean_project_name(
            "Ultra Luxury Apartments | Flats for Sale in Muscat, Oman - Great Escape 2 | DarGlobal",
            "the-great-escape-2-aida-oman",
            "Ultra Luxury Apartments | Flats for Sale in Muscat, Oman - Great Escape 2 | DarGlobal",
        )
        assert name == "The Great Escape 2"

    def test_normal_name_unchanged(self):
        from organize.extract_projects import _clean_project_name

        name = _clean_project_name("The Astera", "the-astera", "The Astera")
        assert name == "The Astera"
