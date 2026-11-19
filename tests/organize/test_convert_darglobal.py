"""Tests for organize.convert_darglobal."""

import gzip
import json


class TestSlugFromUrl:
    def test_simple_press(self):
        from organize.convert_darglobal import _slug_from_url
        url = "https://www.darglobal.co.uk/press/Some-Press-Release"
        slug = _slug_from_url(url)
        assert slug == "press-Some-Press-Release"
        assert slug[0].isalnum()

    def test_blog(self):
        from organize.convert_darglobal import _slug_from_url
        url = "https://www.darglobal.co.uk/blog/trump-tower-latest"
        slug = _slug_from_url(url)
        assert slug == "blog-trump-tower-latest"

    def test_no_double_hyphens(self):
        from organize.convert_darglobal import _slug_from_url
        url = "https://www.darglobal.co.uk/press/a--b"
        slug = _slug_from_url(url)
        assert "--" not in slug

    def test_max_length(self):
        from organize.convert_darglobal import _slug_from_url
        url = "https://www.darglobal.co.uk/press/" + "a" * 300
        slug = _slug_from_url(url)
        assert len(slug) <= 200


class TestConvertArticles:
    def test_converts_press_to_article(self, tmp_path):
        from organize.convert_darglobal import convert_articles

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        processed_dir = tmp_path / "processed"

        record = {
            "url": "https://www.darglobal.co.uk/press/test-article",
            "title": "Test Article About DarGlobal",
            "type": "press",
            "text": "This is a test article about DarGlobal real estate projects. " * 5,
            "fetched_at": "2026-11-18T12:00:00+00:00",
        }
        with gzip.open(raw_dir / "test-article.json.gz", "wt", encoding="utf-8") as f:
            json.dump(record, f)

        counts = convert_articles(raw_dir=raw_dir, processed_dir=processed_dir)
        assert counts["press"] == 1

        out_path = processed_dir / "darglobal_articles.jsonl"
        assert out_path.exists()
        line = out_path.read_text(encoding="utf-8").strip()
        article = json.loads(line)
        assert article["kind"] == "press"
        assert article["source"] == "darglobal"
        assert "press-test-article" in article["slug"]

    def test_skips_project_type(self, tmp_path):
        from organize.convert_darglobal import convert_articles

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        processed_dir = tmp_path / "processed"

        record = {
            "url": "https://www.darglobal.co.uk/trump-tower-jeddah",
            "title": "Trump Tower Jeddah",
            "type": "project",
            "text": "Trump Tower is a luxury project. " * 10,
            "fetched_at": "2026-11-18T12:00:00+00:00",
        }
        with gzip.open(raw_dir / "trump.json.gz", "wt", encoding="utf-8") as f:
            json.dump(record, f)

        counts = convert_articles(raw_dir=raw_dir, processed_dir=processed_dir)
        assert counts["skipped_project"] == 1
        assert counts["press"] == 0

    def test_skips_short_text(self, tmp_path):
        from organize.convert_darglobal import convert_articles

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        processed_dir = tmp_path / "processed"

        record = {
            "url": "https://www.darglobal.co.uk/press/short",
            "title": "Short",
            "type": "press",
            "text": "Too short",
            "fetched_at": "2026-11-18T12:00:00+00:00",
        }
        with gzip.open(raw_dir / "short.json.gz", "wt", encoding="utf-8") as f:
            json.dump(record, f)

        counts = convert_articles(raw_dir=raw_dir, processed_dir=processed_dir)
        assert counts["press"] == 0
        assert counts["errors"] == 1

    def test_deterministic_output(self, tmp_path):
        """Two runs produce identical JSONL."""
        from organize.convert_darglobal import convert_articles

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        p1 = tmp_path / "out1"
        p2 = tmp_path / "out2"

        for name, kind in [("b-article", "blog"), ("a-article", "press")]:
            record = {
                "url": f"https://www.darglobal.co.uk/{kind}/{name}",
                "title": f"Article {name}",
                "type": kind,
                "text": f"Content about {name} in the DarGlobal real estate market. " * 5,
                "fetched_at": "2026-11-18T12:00:00+00:00",
            }
            with gzip.open(raw_dir / f"{name}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(record, f)

        convert_articles(raw_dir=raw_dir, processed_dir=p1)
        convert_articles(raw_dir=raw_dir, processed_dir=p2)

        content1 = (p1 / "darglobal_articles.jsonl").read_text(encoding="utf-8")
        content2 = (p2 / "darglobal_articles.jsonl").read_text(encoding="utf-8")
        assert content1 == content2
