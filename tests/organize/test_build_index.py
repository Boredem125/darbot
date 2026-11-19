"""Tests for organize.build_index CLI options."""

from __future__ import annotations

import json
from pathlib import Path


class TestBuildIndexOutDir:
    """The build_index function accepts --out-dir to write to a custom directory."""

    def test_out_dir_argument_accepted(self):
        """build_index CLI parser accepts --out-dir."""
        from organize.build_index import _parse_args

        args = _parse_args(["--out-dir", "/tmp/test_index"])
        assert args.out_dir == "/tmp/test_index"

    def test_out_dir_defaults_to_data_index(self):
        """When --out-dir is omitted, the default is data/index."""
        from organize.build_index import _parse_args

        args = _parse_args([])
        assert args.out_dir == "data/index"


class TestExtraListings:
    """The --extra-listings option appends Listing records."""

    def test_extra_listings_argument_accepted(self):
        from organize.build_index import _parse_args

        args = _parse_args(["--extra-listings", "eval/sets/injection_listings.jsonl"])
        assert args.extra_listings == "eval/sets/injection_listings.jsonl"

    def test_extra_listings_defaults_to_none(self):
        from organize.build_index import _parse_args

        args = _parse_args([])
        assert args.extra_listings is None

    def test_load_extra_listings(self, tmp_path: Path):
        from organize.build_index import _load_extra_listings

        listing = {
            "id": "wasalt:test-extra",
            "source": "wasalt",
            "source_id": "test-extra",
            "url": "https://example.com/test",
            "title": "Test extra listing",
            "purpose": "sale",
            "property_type": "apartment",
            "city": "Riyadh",
            "price_sar": 500000,
            "scraped_at": "2026-11-19T00:00:00+00:00",
        }
        p = tmp_path / "extra.jsonl"
        p.write_text(json.dumps(listing) + "\n", encoding="utf-8")

        result = _load_extra_listings(p)
        assert len(result) == 1
        assert result[0].id == "wasalt:test-extra"
