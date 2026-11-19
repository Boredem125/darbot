"""Build the entities dictionary (data/index/entities.json).

Sources:
  - cities and districts from listings (parent = city), plus countries
  - projects and brands from Project records (if present)
  - seed_aliases.json: hand-written Arabic and English aliases

D1 (WP-3-extract2): project entity aliases.
  For each project add:
  - the canonical up to the first "," or "|"
  - that head without a leading "The "
  - the head with a trailing master-project or location word removed
    (e.g. "Marriott Residences Aida" -> "Marriott Residences")
  Skip an alias that collides with another project's canonical or alias.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from common.schemas import (
    Article,
    Entities,
    Entity,
    EntityKind,
    Listing,
    Project,
)

logger = logging.getLogger(__name__)

SEED_ALIASES_PATH = Path(__file__).parent / "seed_aliases.json"


def _load_seed_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """Load seed aliases from JSON file. Keys are canonical names."""
    p = path or SEED_ALIASES_PATH
    if not p.exists():
        logger.warning("No seed_aliases.json found at %s", p)
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# Location and master-project words that can be stripped from the tail of a name
_LOCATION_TAIL_WORDS = frozenset({
    "oman", "qatar", "dubai", "jeddah", "riyadh", "muscat",
    "aida", "rayana", "amaya", "wadi", "safar",
})


def _project_aliases(canonical: str) -> list[str]:
    """Generate aliases for a project canonical name per D1 rules.

    Returns a list of candidate aliases (may contain duplicates or the
    canonical itself; the caller deduplicates and collision-checks).
    """
    aliases: list[str] = []

    # 1. Head: canonical up to the first "," or "|" or " -- " (en-dash)
    head = re.split(r"[,|]|\s–\s", canonical)[0].strip()
    if head and head != canonical:
        aliases.append(head)
    else:
        head = canonical

    # 2. Head without leading "The "
    if head.lower().startswith("the ") and len(head) > 4:
        no_the = head[4:].strip()
        if no_the:
            aliases.append(no_the)

    # 3. Head with trailing master-project or location word removed
    words = head.split()
    if len(words) >= 2:
        last_word = words[-1].lower()
        if last_word in _LOCATION_TAIL_WORDS:
            shorter = " ".join(words[:-1]).strip()
            if shorter and len(shorter) >= 3:
                aliases.append(shorter)

    return aliases


def build_entities(
    listings: list[Listing],
    projects: list[Project] | None = None,
    articles: list[Article] | None = None,
    seed_aliases_path: Path | None = None,
) -> Entities:
    """Build the Entities dictionary."""
    projects = projects or []
    articles = articles or []

    seed = _load_seed_aliases(seed_aliases_path)

    entities: list[Entity] = []
    seen: set[tuple[EntityKind, str, str | None]] = set()

    def _add(kind: EntityKind, canonical: str, parent: str | None = None,
             doc_id: str | None = None, extra_aliases: list[str] | None = None) -> None:
        key = (kind, canonical, parent)
        if key in seen:
            return
        seen.add(key)
        aliases = list(seed.get(canonical, []))
        if extra_aliases:
            for a in extra_aliases:
                if a not in aliases and a != canonical:
                    aliases.append(a)
        entities.append(Entity(
            kind=kind,
            canonical=canonical,
            aliases=aliases,
            parent=parent,
            doc_id=doc_id,
        ))

    # Countries
    _add(EntityKind.country, "Saudi Arabia")

    # Cities and districts from listings
    for listing in listings:
        city = listing.city
        if city:
            _add(EntityKind.city, city, parent="Saudi Arabia")
        district = listing.district
        if district and city:
            _add(EntityKind.district, district, parent=city)

    # D1: collect all project canonicals and aliases to check collisions
    all_project_canonicals: set[str] = set()
    all_project_aliases: set[str] = set()
    project_alias_map: dict[str, list[str]] = {}  # canonical -> candidate aliases

    for proj in projects:
        can = proj.name.lower()
        all_project_canonicals.add(can)
        candidates = _project_aliases(proj.name)
        project_alias_map[proj.name] = candidates
        for a in candidates:
            all_project_aliases.add(a.lower())

    # Projects
    for proj in projects:
        if proj.country:
            _add(EntityKind.country, proj.country)
        if proj.city:
            parent = proj.country or None
            _add(EntityKind.city, proj.city, parent=parent)

        # D1: filter aliases to avoid collisions with other projects
        candidates = project_alias_map.get(proj.name, [])
        safe_aliases: list[str] = []
        for alias in candidates:
            alias_lower = alias.lower()
            # Skip if it matches another project's canonical
            if alias_lower in all_project_canonicals and alias_lower != proj.name.lower():
                logger.debug("Skipping alias %r for %s: collides with another canonical",
                             alias, proj.name)
                continue
            # Skip if alias is too short or generic
            if len(alias) < 3:
                continue
            safe_aliases.append(alias)

        _add(
            EntityKind.project,
            proj.name,
            parent=proj.city,
            doc_id=proj.id,
            extra_aliases=safe_aliases,
        )
        if proj.brand:
            _add(EntityKind.brand, proj.brand)

    result = Entities(entities=entities)
    logger.info("Built entities: %d total", len(entities))
    return result
