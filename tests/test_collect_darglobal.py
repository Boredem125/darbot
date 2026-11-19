"""Tests for the DarGlobal collector.

Verifies URL classification, slug extraction, blog scope filtering,
PII stripping, and sitemap parsing.
"""

from __future__ import annotations

from pathlib import Path

from collect.darglobal import (
    blog_in_scope,
    classify_url,
    parse_sitemap_urls,
    slug_from_url,
    strip_pii_from_text,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------

class TestClassifyUrl:
    def test_project_root_slug(self) -> None:
        assert classify_url("https://darglobal.co.uk/trump-tower-jeddah") == "project"

    def test_project_root_slug_www(self) -> None:
        assert classify_url("https://www.darglobal.co.uk/the-astera") == "project"

    def test_press_page(self) -> None:
        url = "https://darglobal.co.uk/press/darglobal-breaks-ground-on-trump-tower-jeddah"
        assert classify_url(url) == "press"

    def test_insight_page(self) -> None:
        url = "https://darglobal.co.uk/insights/mouawad-branded-villas-riyadh"
        assert classify_url(url) == "insight"

    def test_blog_page(self) -> None:
        url = "https://darglobal.co.uk/blog/luxury-living-in-dubai"
        assert classify_url(url) == "blog"

    def test_press_index_skipped(self) -> None:
        assert classify_url("https://darglobal.co.uk/press") is None

    def test_blog_index_skipped(self) -> None:
        assert classify_url("https://darglobal.co.uk/blog") is None

    def test_insights_index_skipped(self) -> None:
        assert classify_url("https://darglobal.co.uk/insights") is None

    def test_homepage_skipped(self) -> None:
        assert classify_url("https://darglobal.co.uk/") is None
        assert classify_url("https://darglobal.co.uk") is None

    def test_non_content_page_skipped(self) -> None:
        assert classify_url("https://darglobal.co.uk/about") is None
        assert classify_url("https://darglobal.co.uk/careers") is None
        assert classify_url("https://darglobal.co.uk/faq") is None
        assert classify_url("https://darglobal.co.uk/privacy-policy") is None

    def test_partner_page_skipped(self) -> None:
        assert classify_url("https://darglobal.co.uk/partners/elie-saab") is None

    def test_sub_project_classified(self) -> None:
        assert classify_url("https://darglobal.co.uk/rayana/trump-mansions") == "project"
        assert classify_url("https://darglobal.co.uk/one-of-one/trump-penthouse") == "project"
        assert classify_url("https://darglobal.co.uk/trump-plaza/park-residences") == "project"

    def test_marketing_page_skipped(self) -> None:
        url = "https://darglobal.co.uk/DarGlobal-exclusive-Off-Plan-Investment-Catalogue"
        assert classify_url(url) is None

    def test_pay_online_skipped(self) -> None:
        assert classify_url("https://darglobal.co.uk/pay-online/status") is None


# ---------------------------------------------------------------------------
# Slug extraction
# ---------------------------------------------------------------------------

class TestSlugFromUrl:
    def test_root_slug(self) -> None:
        assert slug_from_url("https://darglobal.co.uk/trump-tower-jeddah") == "trump-tower-jeddah"

    def test_press_slug(self) -> None:
        slug = slug_from_url("https://darglobal.co.uk/press/some-press-release")
        assert slug == "press__some-press-release"

    def test_sub_project_slug(self) -> None:
        slug = slug_from_url("https://darglobal.co.uk/rayana/trump-mansions")
        assert slug == "rayana__trump-mansions"

    def test_special_chars_sanitised(self) -> None:
        # URLs with special characters get sanitised
        slug = slug_from_url("https://darglobal.co.uk/press/Dar Global-Test")
        assert " " not in slug


# ---------------------------------------------------------------------------
# Blog scope filtering
# ---------------------------------------------------------------------------

class TestBlogInScope:
    def test_project_mention(self) -> None:
        assert blog_in_scope("experience-the-extraordinary-tierra-viva-launch-event")

    def test_saudi_city_mention(self) -> None:
        assert blog_in_scope("luxury-real-estate-riyadh-market")

    def test_gulf_market_mention(self) -> None:
        assert blog_in_scope("investing-in-dubai-real-estate-2024")

    def test_generic_blog_out_of_scope(self) -> None:
        assert not blog_in_scope("five-types-of-insurance-coverage-every-homeowner-needs")

    def test_oman_in_scope(self) -> None:
        assert blog_in_scope("unlocking-the-doors-to-tax-free-real-estate-investment-in-oman")


# ---------------------------------------------------------------------------
# PII stripping
# ---------------------------------------------------------------------------

class TestStripPii:
    def test_email_removed(self) -> None:
        text = "Contact us at info@darglobal.co.uk for more details."
        result = strip_pii_from_text(text)
        assert "info@darglobal.co.uk" not in result

    def test_saudi_phone_removed(self) -> None:
        text = "Call us: +966 800 118 8081 for enquiries."
        result = strip_pii_from_text(text)
        assert "800 118 8081" not in result

    def test_uae_phone_removed(self) -> None:
        text = "Contact: +971 4 562 9666"
        result = strip_pii_from_text(text)
        assert "562 9666" not in result

    def test_clean_text_unchanged(self) -> None:
        text = "DarGlobal launches new project in Jeddah."
        result = strip_pii_from_text(text)
        assert result == text


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------

class TestSitemapParsing:
    def test_parse_chromium_wrapped_xml(self) -> None:
        """The spike saved sitemap.xml.html which is Chromium's HTML wrapper."""
        fixture_path = FIXTURES.parent.parent / "data" / "raw" / "darglobal" / "sitemap.xml.html"
        if not fixture_path.exists():
            # Use a minimal inline fixture if the real file is not available
            xml = (
                '<html><body>'
                '<div id="webkit-xml-viewer-source-xml">'
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                '<url><loc>https://darglobal.co.uk/trump-tower-jeddah</loc></url>'
                '<url><loc>https://darglobal.co.uk/press/test-press</loc></url>'
                '</urlset>'
                '</div></body></html>'
            )
            urls = parse_sitemap_urls(xml)
            assert len(urls) == 2
            assert "https://darglobal.co.uk/trump-tower-jeddah" in urls
            return

        html = fixture_path.read_text(encoding="utf-8")
        urls = parse_sitemap_urls(html)
        # The real sitemap has hundreds of URLs
        assert len(urls) > 100
        # Check that project URLs are present
        assert any("trump-tower-jeddah" in u for u in urls)

    def test_parse_raw_xml(self) -> None:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            '<url><loc>https://darglobal.co.uk/the-astera</loc></url>'
            '</urlset>'
        )
        urls = parse_sitemap_urls(xml)
        assert urls == ["https://darglobal.co.uk/the-astera"]


# ---------------------------------------------------------------------------
# Integration: fixture-based extraction test
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_extract_from_fixture(self) -> None:
        """Test text extraction from a press page fixture using trafilatura."""
        fixture_path = FIXTURES / "darglobal_press_page.html"
        html = fixture_path.read_text(encoding="utf-8")

        from collect.darglobal import extract_text

        text, _title = extract_text(html)
        # trafilatura should extract some text
        assert text is not None
        assert len(text) > 20
        # Key content should be present
        assert "Trump Tower Jeddah" in text or "Dar Global" in text

    def test_pii_stripped_from_fixture(self) -> None:
        """PII in the fixture text should be removed."""
        fixture_path = FIXTURES / "darglobal_press_page.html"
        html = fixture_path.read_text(encoding="utf-8")

        from collect.darglobal import extract_text

        text, _ = extract_text(html)
        assert text is not None
        cleaned = strip_pii_from_text(text)
        assert "info@example.com" not in cleaned
