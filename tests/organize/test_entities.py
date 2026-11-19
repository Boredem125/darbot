"""Tests for organize.entities project alias generation (D1)."""

from __future__ import annotations


class TestProjectAliases:
    """D1: project entity aliases are generated correctly."""

    def test_head_before_comma(self):
        from organize.entities import _project_aliases

        aliases = _project_aliases("Trump International Simaisma, Qatar")
        assert "Trump International Simaisma" in aliases

    def test_head_before_endash(self):
        from organize.entities import _project_aliases

        aliases = _project_aliases("Trump Plaza Jeddah – Executive Residences")
        assert "Trump Plaza Jeddah" in aliases

    def test_no_the(self):
        from organize.entities import _project_aliases

        aliases = _project_aliases("The Astera")
        assert "Astera" in aliases

    def test_trailing_location_word(self):
        from organize.entities import _project_aliases

        aliases = _project_aliases("Marriott Residences Aida")
        lower_aliases = [a.lower() for a in aliases]
        assert "marriott residences" in lower_aliases

    def test_trailing_location_oman(self):
        from organize.entities import _project_aliases

        aliases = _project_aliases("Trump International Hotel Oman")
        lower_aliases = [a.lower() for a in aliases]
        assert "trump international hotel" in lower_aliases

    def test_no_alias_for_short_name(self):
        from organize.entities import _project_aliases

        aliases = _project_aliases("DG1")
        # Head is same as canonical, no commas, no "The", no location tail
        # Only meaningful alias would be if there was a structure to split
        assert all(len(a) >= 2 for a in aliases)

    def test_collision_check_in_build(self):
        """An alias that matches another project's canonical is skipped."""
        from datetime import UTC, datetime

        from common.schemas import Project, project_id
        from organize.entities import build_entities

        scraped = datetime.now(UTC)
        p1 = Project(
            id=project_id("proj-a"),
            slug="proj-a",
            url="https://example.com/a",
            name="The Neptune",
            body=None,
            scraped_at=scraped,
        )
        p2 = Project(
            id=project_id("proj-b"),
            slug="proj-b",
            url="https://example.com/b",
            name="Neptune",  # This IS the alias of p1
            body=None,
            scraped_at=scraped,
        )
        entities = build_entities([], projects=[p1, p2])
        # Find p1's entity
        p1_ent = next(e for e in entities.entities if e.canonical == "The Neptune")
        # "Neptune" should NOT be in p1's aliases since it collides with p2's canonical
        assert "Neptune" not in p1_ent.aliases
