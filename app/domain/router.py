"""Rule-based router: language detection, filter extraction, route selection.

Imports only common.* and app.domain.* (R4). No pydantic import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.domain.ports import AggregateOp, ChatMessage
from common.schemas import (
    DocKind,
    Entities,
    Entity,
    EntityKind,
    Language,
    PropertyType,
    Purpose,
    Route,
    SearchFilters,
    SortKey,
)
from common.text import (
    arabic_ratio,
    canonical_number,
    detect_negation,
    normalise_alias,
    parse_numbers,
)

# ---------------------------------------------------------------------------
# SessionSlots (frozen dataclass defined here per the brief)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionSlots:
    filters: SearchFilters | None = None
    last_prices: tuple[int, ...] = ()
    last_areas: tuple[float, ...] = ()
    history: tuple[ChatMessage, ...] = ()


# ---------------------------------------------------------------------------
# RouteDecision
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteDecision:
    language: Language
    route: Route
    op: AggregateOp | None = None
    example_limit: int = 3
    filters: SearchFilters = field(default_factory=SearchFilters)
    carried_slots: frozenset[str] = frozenset()
    numbers: tuple[str, ...] = ()
    negated: bool = False
    follow_up: bool = False


# ---------------------------------------------------------------------------
# EntityIndex: alias -> Entity, longest-match-first on word boundaries
# ---------------------------------------------------------------------------


class EntityIndex:
    """An index of entity aliases for fast lookup.

    Built once from an Entities object. Aliases are normalised with normalise_alias.
    Lookup matches the longest alias first on word boundaries.
    """

    def __init__(self, entities: Entities) -> None:
        # Map normalised alias -> Entity, longest first
        self._entries: list[tuple[str, Entity]] = []
        seen: dict[str, Entity] = {}
        for entity in entities.entities:
            for name in [entity.canonical] + list(entity.aliases):
                normed = normalise_alias(name)
                if normed and normed not in seen:
                    seen[normed] = entity
        # Sort by length descending for longest-match-first
        self._entries = sorted(seen.items(), key=lambda x: len(x[0]), reverse=True)
        # Build a quick set of all normalised aliases per entity canonical
        self._all_aliases: dict[str, set[str]] = {}
        for entity in entities.entities:
            names = {normalise_alias(entity.canonical)}
            for a in entity.aliases:
                names.add(normalise_alias(a))
            names.discard("")
            self._all_aliases[entity.canonical] = names

    def find_all(self, text: str) -> list[tuple[str, Entity]]:
        """Return all (alias, Entity) matches in text, longest first."""
        normed = normalise_alias(text)
        results: list[tuple[str, Entity]] = []
        matched_spans: list[tuple[int, int]] = []
        for alias, entity in self._entries:
            # Word boundary match
            pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.UNICODE)
            for m in pattern.finditer(normed):
                start, end = m.start(), m.end()
                # Avoid overlapping matches
                if any(s <= start < e or s < end <= e for s, e in matched_spans):
                    continue
                results.append((alias, entity))
                matched_spans.append((start, end))
        return results

    def find_in_text(self, text: str) -> list[tuple[str, Entity]]:
        """Alias for find_all, used by grounding."""
        return self.find_all(text)

    def has_alias(self, text: str) -> bool:
        """Check if text matches any entity alias or canonical."""
        normed = normalise_alias(text)
        for alias, _entity in self._entries:
            if alias == normed:
                return True
        return False

    def aliases_for(self, canonical: str) -> set[str]:
        """All normalised aliases for the entity with the given canonical name."""
        return self._all_aliases.get(canonical, set())

    def find_fuzzy(self, text: str) -> list[tuple[str, Entity]]:
        """Fuzzy-match unmatched tokens against city/district aliases.

        Distance threshold: <= 1 for 5-7 char tokens, <= 2 for 8+ chars.
        Tokens < 5 chars are never fuzzy-matched.
        Only city and district entities are candidates.
        """
        normed = normalise_alias(text)
        # Exact matches first -- skip tokens that already matched
        exact_results = self.find_all(text)
        exact_spans: set[str] = set()
        for alias, _ent in exact_results:
            exact_spans.add(alias)

        # Build candidate aliases for city/district only
        candidates: list[tuple[str, Entity]] = [
            (alias, ent)
            for alias, ent in self._entries
            if ent.kind in (EntityKind.city, EntityKind.district)
        ]

        tokens = re.findall(r"\b\w+\b", normed, re.UNICODE)
        results: list[tuple[str, Entity]] = []
        matched_tokens: set[str] = set()
        for token in tokens:
            tlen = len(token)
            if tlen < 5:
                continue
            # Skip if this token is part of an exact match
            if any(token in span for span in exact_spans):
                continue
            max_dist = 1 if tlen <= 7 else 2
            best: tuple[str, Entity] | None = None
            best_dist = max_dist + 1
            for alias, ent in candidates:
                d = _damerau_levenshtein(token, alias)
                if d <= max_dist and d < best_dist:
                    best = (alias, ent)
                    best_dist = d
            if best is not None and best[0] not in matched_tokens:
                results.append(best)
                matched_tokens.add(best[0])
        return results


def _damerau_levenshtein(s: str, t: str) -> int:
    """Optimal string alignment distance (restricted Damerau-Levenshtein).

    Pure Python, O(len(s)*len(t)) time and space. Sufficient for short
    city/district names (< 20 chars).
    """
    ls, lt = len(s), len(t)
    # Quick reject: if length diff > 2, distance is at least that
    if abs(ls - lt) > 2:
        return abs(ls - lt)
    d = [[0] * (lt + 1) for _ in range(ls + 1)]
    for i in range(ls + 1):
        d[i][0] = i
    for j in range(lt + 1):
        d[0][j] = j
    for i in range(1, ls + 1):
        for j in range(1, lt + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            # Transposition
            if i > 1 and j > 1 and s[i - 1] == t[j - 2] and s[i - 2] == t[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)
    return d[ls][lt]


# ---------------------------------------------------------------------------
# Dictionaries (English and Arabic keywords -> filter values)
# ---------------------------------------------------------------------------

_PROPERTY_TYPE_MAP: dict[str, PropertyType] = {
    # villa
    "villa": PropertyType.villa, "villas": PropertyType.villa,
    "فيلا": PropertyType.villa,  # فيلا
    "فلل": PropertyType.villa,  # فلل
    "الفلل": PropertyType.villa,  # الفلل (B5)
    "الفيلا": PropertyType.villa,  # الفيلا (B5)
    "فلة": PropertyType.villa,  # فلة (B5)
    # apartment
    "apartment": PropertyType.apartment, "apartments": PropertyType.apartment,
    "flat": PropertyType.apartment, "flats": PropertyType.apartment,
    "شقة": PropertyType.apartment,  # شقة
    "شقق": PropertyType.apartment,  # شقق
    "الشقق": PropertyType.apartment,  # الشقق (B5)
    "الشقة": PropertyType.apartment,  # الشقة (B5)
    # townhouse
    "townhouse": PropertyType.townhouse, "townhouses": PropertyType.townhouse,
    # duplex
    "duplex": PropertyType.duplex,
    "دوبلكس": PropertyType.duplex,  # دوبلكس
    # floor
    "floor": PropertyType.floor,
    "دور": PropertyType.floor,  # دور
    # land
    "land": PropertyType.land, "plot": PropertyType.land, "plots": PropertyType.land,
    "أرض": PropertyType.land,  # أرض
    "ارض": PropertyType.land,  # ارض
    "الأراضي": PropertyType.land,  # الأراضي (B5)
    "أراضي": PropertyType.land,  # أراضي (B5)
    "الأرض": PropertyType.land,  # الأرض (B5)
    "الارض": PropertyType.land,  # الارض (B5)
    # building
    "building": PropertyType.building, "buildings": PropertyType.building,
    "عمارة": PropertyType.building,  # عمارة
    "مبنى": PropertyType.building,  # مبنى
    # office
    "office": PropertyType.office, "offices": PropertyType.office,
    "مكتب": PropertyType.office,  # مكتب
    # shop
    "shop": PropertyType.shop, "shops": PropertyType.shop, "retail": PropertyType.shop,
    "محل": PropertyType.shop,  # محل
    # warehouse
    "warehouse": PropertyType.warehouse,
    "مستودع": PropertyType.warehouse,  # مستودع
    # chalet
    "chalet": PropertyType.chalet,
    "شاليه": PropertyType.chalet,  # شاليه
    # farm
    "farm": PropertyType.farm,
    "مزرعة": PropertyType.farm,  # مزرعة
}

_PURPOSE_RENT_CUES = {
    "rent", "rental", "lease", "to let", "for rent",
    "إيجار",  # إيجار
    "للإيجار",  # للإيجار
    "ايجار",  # ايجار
}

_PURPOSE_SALE_CUES = {
    "buy", "sale", "for sale", "purchase",
    "للبيع",  # للبيع
    "شراء",  # شراء
    "بيع",  # بيع
}

# Monthly rent cues: the amount * 12 (filters use annual rent)
_MONTHLY_RENT_CUES = {
    "per month", "monthly", "a month",
    "شهري",  # شهري
    "شهريا",  # شهريا
}

_YEARLY_RENT_CUES = {
    "per year", "yearly", "annual",
    "سنوي",  # سنوي
}

# Price bound keywords
_PRICE_MAX_CUES = {
    "under", "below", "less than", "max", "up to",
    "أقل من",  # أقل من
    "تحت",  # تحت
}

_PRICE_MIN_CUES = {
    "over", "above", "more than", "at least", "from",
    "أكثر من",  # أكثر من
    "فوق",  # فوق
}

# Area bound keywords (same bound words as price)
_AREA_UNITS = {
    "sqm", "m2", "square metres", "square meters", "sq m",
    "متر",  # متر
    "م٢",  # م2 (but after digit normalisation this is م2)
    "م2",  # م2 literal
}

# Price-intent cues (non-aggregate: set sort on lookup, not an aggregate op)
_PRICE_LOW_INTENT = {
    "cheap", "affordable", "budget", "low price", "inexpensive",
    "رخيص", "رخيصة", "بسعر مناسب",
}
_PRICE_HIGH_INTENT = {
    "expensive", "luxury", "high price",
    "غالي", "غالية", "فاخر", "فاخرة",
}

# Aggregate cues
_COUNT_CUES = {"how many", "number of", "count", "كم عدد"}  # كم عدد
_AVG_PRICE_CUES = {"average", "mean", "avg", "typical price", "متوسط"}  # متوسط; B3: "avg"
_MEDIAN_PRICE_CUES = {"median", "الوسيط", "وسيط"}  # B4: Arabic median
_MIN_PRICE_CUES = {
    "cheapest", "lowest price", "least expensive",
    "أرخص",  # أرخص
}
_MAX_PRICE_CUES = {
    "most expensive", "highest price", "priciest",
    "أغلى",  # أغلى
}
_AVG_AREA_CUES = {
    "average size", "average area", "mean size", "avg size",  # B2
    "متوسط مساحة", "متوسط المساحة",  # B2: Arabic
}
_NEWEST_CUES = {
    "newest", "latest",
    "أحدث",  # أحدث
}

# Project info cues
_PROJECT_CUES = {
    "darglobal", "dar global", "project", "development",
    "handover", "payment plan", "off-plan", "launch", "press", "news",
    "مشروع",  # مشروع
    "دار جلوبال",  # دار جلوبال
    "تسليم",  # تسليم
}

# Domain cues for not-out-of-scope
_DOMAIN_CUES = {
    "property", "properties", "real estate", "home", "homes", "house", "houses",
    "عقار",  # عقار
    "منزل",  # منزل
    "بيت",  # بيت
    "listing", "listings", "apartment", "villa", "townhouse", "land",
    "rent", "sale", "buy", "bedroom", "bedrooms", "price",
    "sqm", "area",
}

# B7: OOS topic cues -- presence of a topic cue + no real-estate slots -> out_of_scope
_OOS_TOPIC_CUES = {
    "weather", "forecast", "temperature",
    "football", "soccer", "match",
    "recipe",
    "python", "javascript", "code",
    "joke", "poem",
    "capital of", "president",
    "bitcoin", "stock price",
    "translate",
    "طقس", "الطقس", "حرارة",  # weather
    "مباراة",  # match
    "وصفة",  # recipe
    "عاصمة",  # capital
    "نكتة",  # joke
}

# Follow-up cues
_FOLLOWUP_CUES_EN = {
    "cheaper", "less expensive", "more expensive",
    "bigger", "larger", "smaller",
    "more bedrooms", "fewer bedrooms",
    "what about", "how about", "and in",
    "those", "these", "them", "ones", "same but",
}

_FOLLOWUP_CUES_AR = {
    "أرخص",  # أرخص
    "أكبر",  # أكبر
    "أصغر",  # أصغر
    "ماذا عن",  # ماذا عن
    "نفس",  # نفس
}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def _detect_language(text: str, hint: Language | None) -> Language:
    """Arabic if arabic_ratio(letters) >= 0.30. If no letters, use hint, then en."""
    letters = "".join(ch for ch in text if ch.isalpha())
    if not letters:
        return hint if hint is not None else Language.en
    if arabic_ratio(letters) >= 0.30:
        return Language.ar
    return Language.en


# ---------------------------------------------------------------------------
# Filter extraction helpers
# ---------------------------------------------------------------------------


def _cue_found(text: str, cues: set[str]) -> bool:
    """Check if any cue appears in text (case-insensitive, word boundary for single words)."""
    lower = text.lower()
    for cue in sorted(cues, key=len, reverse=True):
        if " " in cue:
            if cue in lower:
                return True
        else:
            if re.search(r"\b" + re.escape(cue) + r"\b", lower, re.UNICODE):
                return True
    return False


def _extract_property_type(text: str) -> PropertyType | None:
    lower = text.lower()
    for cue, ptype in sorted(_PROPERTY_TYPE_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"\b" + re.escape(cue) + r"\b", lower, re.UNICODE):
            return ptype
    return None


def _extract_purpose(text: str) -> Purpose | None:
    if _cue_found(text, _PURPOSE_RENT_CUES) or _cue_found(text, _MONTHLY_RENT_CUES) or _cue_found(text, _YEARLY_RENT_CUES):
        return Purpose.rent
    if _cue_found(text, _PURPOSE_SALE_CUES):
        return Purpose.sale
    return None


def _extract_bedrooms(text: str) -> tuple[int | None, int | None]:
    """Extract bedroom constraints. Returns (min, max)."""
    lower = text.lower()
    # Studio
    if re.search(r"\bstudio\b", lower):
        return (0, 0)
    # Arabic bedroom patterns: "3 غرف" or "ثلاث غرف"
    # English: "3-bed", "3 bed", "3 bedrooms", "3br", "three bedroom"
    # "3+", "at least 3", "3 or more"
    bed_pattern = re.compile(
        r"(\d+)\s*(?:-?\s*bed(?:room)?s?|br|غرف)",
        re.UNICODE,
    )
    m = bed_pattern.search(lower)
    if m:
        n = int(m.group(1))
        # Check for "3+" or "at least" or "or more" after the number
        rest = lower[m.start():]
        if re.search(r"\d+\s*\+", rest) or "at least" in lower or "or more" in lower:
            return (n, None)
        return (n, n)

    # Arabic dual: "غرفتين" / "غرفتين نوم" / "غرفتي نوم" = 2 bedrooms
    if re.search(r"غرفت[يى]ن(?:\s+نوم)?", lower):
        return (2, 2)
    # Arabic singular: "غرفة واحدة" = 1 bedroom
    if re.search(r"غرفة\s+واحدة", lower):
        return (1, 1)

    # Arabic word-based bedroom: "ثلاث غرف" etc.
    from common.text import _AR_NUMBER_WORDS
    for word, value in sorted(_AR_NUMBER_WORDS.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = re.compile(
            re.escape(word) + r"\s+غرف",
            re.UNICODE,
        )
        if pattern.search(lower):
            return (value, value)

    # English word-based: "three bedroom"
    from common.text import _EN_NUMBER_WORDS
    for word, value in _EN_NUMBER_WORDS.items():
        pattern = re.compile(
            r"\b" + re.escape(word) + r"\s+bed(?:room)?s?\b",
            re.UNICODE,
        )
        if pattern.search(lower):
            return (value, value)

    # "at least N" bedrooms
    atleast = re.search(r"at least (\d+)\s*(?:bed|br|غرف)", lower)
    if atleast:
        return (int(atleast.group(1)), None)

    return (None, None)


def _is_monthly_context(text: str) -> bool:
    return _cue_found(text, _MONTHLY_RENT_CUES)


def _is_yearly_context(text: str) -> bool:
    return _cue_found(text, _YEARLY_RENT_CUES)


def _find_price_number(text: str, cue: str, numbers: list[str]) -> int | None:
    """Find the number closest (after) to a price/area cue word in the text.

    Falls back to the largest non-bedroom number if cue position isn't informative.
    """
    if not numbers:
        return None
    lower = text.lower()

    # Find position of the cue in text
    if " " in cue:
        pos = lower.find(cue)
    else:
        m = re.search(r"\b" + re.escape(cue) + r"\b", lower)
        pos = m.start() if m else -1

    if pos < 0:
        return None

    # Find all number positions in the lower text using the magnitude-aware pattern
    from common.text import _MAGNITUDES
    mag_keys = sorted(_MAGNITUDES.keys(), key=len, reverse=True)
    mag_alt = "|".join(re.escape(k) for k in mag_keys)
    num_pattern = re.compile(
        r"(\d[\d,]*(?:\.\d+)?)(?:\s*(" + mag_alt + r"))?\b",
        re.UNICODE,
    )

    # Find the number that appears closest after the cue
    best_num: str | None = None
    best_dist = float("inf")
    for m in num_pattern.finditer(lower):
        num_str = m.group(1).replace(",", "")
        mag_str = m.group(2)
        try:
            from decimal import Decimal, InvalidOperation
            val = Decimal(num_str)
            if mag_str and mag_str in _MAGNITUDES:
                val = val * _MAGNITUDES[mag_str]
            canon = str(int(val)) if val == int(val) else str(val)
        except (InvalidOperation, ValueError, ArithmeticError):
            continue
        # Check this number is in our canonical list (not a bedroom number)
        if canon not in numbers:
            continue
        dist = abs(m.start() - pos)
        if dist < best_dist:
            best_dist = dist
            best_num = canon

    if best_num is not None:
        return int(best_num)
    return None


def _get_price_numbers(text: str, numbers: list[str]) -> list[str]:
    """Get numbers that are likely price/area numbers (not bedroom numbers).

    Bedroom numbers appear directly before bed/bedroom/br/غرف patterns.
    """
    lower = text.lower()
    bed_numbers: set[str] = set()

    bed_pattern = re.compile(
        r"(\d+)\s*(?:-?\s*bed(?:room)?s?|br|غرف)",
        re.UNICODE,
    )
    for m in bed_pattern.finditer(lower):
        bed_numbers.add(m.group(1))

    # Also handle studio as 0
    if re.search(r"\bstudio\b", lower):
        bed_numbers.add("0")

    # Filter out bedroom numbers from the number list
    result: list[str] = []
    for n in numbers:
        if n in bed_numbers:
            continue
        result.append(n)
    return result


def _extract_price_bounds(text: str, numbers: list[str]) -> tuple[int | None, int | None]:
    """Extract price_min and price_max from text and parsed numbers."""
    lower = text.lower()
    price_min: int | None = None
    price_max: int | None = None

    # Filter out bedroom numbers
    price_nums = _get_price_numbers(text, numbers)
    if not price_nums:
        return (None, None)

    # B6: Arabic "بين X و Y" price range
    if (
        re.search(r"(?:^|\s)بين\s", text)
        and re.search(r"\sو\s?", text)
        and len(price_nums) >= 2
    ):
        vals = sorted(int(n) for n in price_nums[:2])
        return (vals[0], vals[1])

    # "between X and Y"
    between = re.search(
        r"\b(?:between)\s+[\d,]+[kKmMbB]*\s+(?:and)\s+[\d,]+[kKmMbB]*",
        lower,
    )
    if between and len(price_nums) >= 2:
        vals = sorted(int(n) for n in price_nums[:2])
        return (vals[0], vals[1])

    # "around/about X" -> [0.9X, 1.1X]
    around = re.search(r"\b(?:around|about)\s+", lower)
    if around and price_nums:
        val = int(price_nums[0])
        return (int(val * 0.9), int(val * 1.1))

    # Check for max cues
    for cue in sorted(_PRICE_MAX_CUES, key=len, reverse=True):
        found = False
        if " " in cue:
            found = cue in lower
        else:
            found = bool(re.search(r"\b" + re.escape(cue) + r"\b", lower))
        if found and price_nums:
            price_max = int(price_nums[0])
            break

    # Check for min cues
    for cue in sorted(_PRICE_MIN_CUES, key=len, reverse=True):
        found = False
        if " " in cue:
            found = cue in lower
        else:
            found = bool(re.search(r"\b" + re.escape(cue) + r"\b", lower))
        if found and price_nums:
            price_min = int(price_nums[0])
            break

    return (price_min, price_max)


def _extract_area_bounds(text: str, numbers: list[str]) -> tuple[float | None, float | None]:
    """Extract area_min and area_max from text and parsed numbers."""
    lower = text.lower()
    has_area_unit = any(
        (re.search(r"\b" + re.escape(u) + r"\b", lower, re.UNICODE) if " " not in u else u in lower)
        for u in _AREA_UNITS
    )
    price_nums = _get_price_numbers(text, numbers)
    if not has_area_unit or not price_nums:
        return (None, None)

    area_min: float | None = None
    area_max: float | None = None
    val = float(price_nums[0])

    for cue in sorted(_PRICE_MAX_CUES, key=len, reverse=True):
        if " " in cue:
            if cue in lower:
                area_max = val
                return (area_min, area_max)
        else:
            if re.search(r"\b" + re.escape(cue) + r"\b", lower):
                area_max = val
                return (area_min, area_max)

    for cue in sorted(_PRICE_MIN_CUES, key=len, reverse=True):
        if " " in cue:
            if cue in lower:
                area_min = val
                return (area_min, area_max)
        else:
            if re.search(r"\b" + re.escape(cue) + r"\b", lower):
                area_min = val
                return (area_min, area_max)

    return (None, None)


def _apply_rent_period_multiplier(
    price_min: int | None, price_max: int | None, text: str
) -> tuple[int | None, int | None]:
    """Multiply by 12 if monthly context is present (filters use annual rent)."""
    if _is_monthly_context(text):
        if price_min is not None:
            price_min = price_min * 12
        if price_max is not None:
            price_max = price_max * 12
    # Yearly: factor is 1, no change
    return (price_min, price_max)


# ---------------------------------------------------------------------------
# Aggregate detection
# ---------------------------------------------------------------------------


def _detect_aggregate(text: str) -> tuple[AggregateOp | None, int, SortKey]:
    """Return (op, example_limit, sort) or (None, 3, relevance)."""
    # Order matters: B2 -- check avg_area BEFORE avg_price because
    # "average size" contains the word "average" which is an avg_price cue.
    if _cue_found(text, _COUNT_CUES):
        return (AggregateOp.count, 3, SortKey.relevance)
    if _cue_found(text, _AVG_AREA_CUES):
        return (AggregateOp.avg_area, 3, SortKey.relevance)
    if _cue_found(text, _AVG_PRICE_CUES):
        return (AggregateOp.avg_price, 3, SortKey.relevance)
    if _cue_found(text, _MEDIAN_PRICE_CUES):
        return (AggregateOp.median_price, 3, SortKey.relevance)
    if _cue_found(text, _MIN_PRICE_CUES):
        return (AggregateOp.min_price, 3, SortKey.price_asc)
    if _cue_found(text, _MAX_PRICE_CUES):
        return (AggregateOp.max_price, 3, SortKey.price_desc)
    if _cue_found(text, _NEWEST_CUES):
        return (AggregateOp.count, 5, SortKey.newest)
    return (None, 3, SortKey.relevance)


# ---------------------------------------------------------------------------
# Follow-up detection and comparative handling
# ---------------------------------------------------------------------------


def _is_followup(text: str, session: SessionSlots | None) -> bool:
    """A turn is a follow-up iff the session has slots AND a follow-up cue is present."""
    if session is None or session.filters is None:
        return False
    lower = text.lower()
    all_cues = _FOLLOWUP_CUES_EN | _FOLLOWUP_CUES_AR
    for cue in sorted(all_cues, key=len, reverse=True):
        if " " in cue:
            if cue in lower:
                return True
        else:
            if re.search(r"\b" + re.escape(cue) + r"\b", lower, re.UNICODE):
                return True
    return False


def _apply_comparatives(
    text: str, session: SessionSlots,
    price_min: int | None, price_max: int | None,
    area_min: float | None, area_max: float | None,
    bed_min: int | None, bed_max: int | None,
) -> tuple[int | None, int | None, float | None, float | None, int | None, int | None]:
    """Apply comparative modifiers relative to previous session values."""
    lower = text.lower()

    # cheaper
    if re.search(r"\b(?:cheaper|less expensive|أرخص)\b", lower, re.UNICODE):
        if session.last_prices:
            price_max = min(session.last_prices) - 1
        elif session.filters and session.filters.price_max_sar:
            price_max = int(session.filters.price_max_sar * 0.8)

    # more expensive
    if re.search(r"\bmore expensive\b", lower) and session.last_prices:
        price_min = max(session.last_prices) + 1

    # bigger / larger
    if re.search(r"\b(?:bigger|larger|أكبر)\b", lower, re.UNICODE) and session.last_areas:
        area_min = max(session.last_areas) + 1

    # smaller
    if re.search(r"\b(?:smaller|أصغر)\b", lower, re.UNICODE) and session.last_areas:
        area_max = min(session.last_areas) - 1

    # more bedrooms
    if (
        re.search(r"\bmore bedrooms\b", lower)
        and session.filters
        and session.filters.bedrooms_max is not None
    ):
        bed_min = session.filters.bedrooms_max + 1
        bed_max = None

    # fewer bedrooms
    if (
        re.search(r"\bfewer bedrooms\b", lower)
        and session.filters
        and session.filters.bedrooms_min is not None
    ):
        bed_max = session.filters.bedrooms_min - 1
        bed_min = None

    return (price_min, price_max, area_min, area_max, bed_min, bed_max)


# ---------------------------------------------------------------------------
# Main route function
# ---------------------------------------------------------------------------


def route(
    message: str,
    *,
    session: SessionSlots | None = None,
    entities: EntityIndex,
    lang_hint: Language | None = None,
) -> RouteDecision:
    """Route a user message to a handler with extracted filters."""
    language = _detect_language(message, lang_hint)
    negated = detect_negation(message)

    # Parse numbers from the raw message
    raw_numbers = parse_numbers(message)
    canonical_nums = tuple(canonical_number(n) for n in raw_numbers)

    # Extract entities
    entity_matches = entities.find_all(message)

    # Extract filters
    property_type = _extract_property_type(message)
    purpose = _extract_purpose(message)
    bed_min, bed_max = _extract_bedrooms(message)

    # Price extraction using canonical numbers
    # We need numbers as strings of their integer/decimal values
    num_strs = [canonical_number(n) for n in raw_numbers]
    price_min, price_max = _extract_price_bounds(message, num_strs)
    area_min, area_max = _extract_area_bounds(message, num_strs)

    # Rent period multiplier
    if _is_monthly_context(message) or _is_yearly_context(message):
        if purpose is None:
            purpose = Purpose.rent
        price_min, price_max = _apply_rent_period_multiplier(price_min, price_max, message)

    # Place entities
    city: str | None = None
    district: str | None = None
    country: str | None = None
    project_id: str | None = None
    has_project_entity = False
    has_brand_entity = False
    has_country_entity = False

    for _alias, entity in entity_matches:
        if entity.kind == EntityKind.city:
            city = entity.canonical
        elif entity.kind == EntityKind.district:
            district = entity.canonical
            if entity.parent:
                city = entity.parent
        elif entity.kind == EntityKind.country:
            country = entity.canonical
            has_country_entity = True
        elif entity.kind == EntityKind.project:
            has_project_entity = True
            if project_id is None and entity.doc_id:  # B8a: first (longest) match wins
                project_id = entity.doc_id
        elif entity.kind == EntityKind.brand:
            has_brand_entity = True

    # Fuzzy match: if no city or district found by exact match, try fuzzy
    if city is None and district is None and not has_project_entity:
        for _alias, entity in entities.find_fuzzy(message):
            if entity.kind == EntityKind.city and city is None:
                city = entity.canonical
            elif entity.kind == EntityKind.district and district is None:
                district = entity.canonical
                if entity.parent:
                    city = entity.parent

    # B8b: when a project entity is matched, place filters are redundant
    # (the project id is the stricter constraint) and can cause 0-hit refusals.
    if project_id is not None:
        city = None
        district = None
        country = None

    # Aggregate detection
    agg_op, example_limit, sort = _detect_aggregate(message)

    # Price intent: "cheap"/"affordable" -> price_asc, "expensive"/"luxury" -> price_desc
    # Only on lookup (agg_op is None); "cheapest" already routes to aggregate.
    if agg_op is None:
        if _cue_found(message, _PRICE_LOW_INTENT):
            sort = SortKey.price_asc
        elif _cue_found(message, _PRICE_HIGH_INTENT):
            sort = SortKey.price_desc

    # Follow-up handling
    follow_up = _is_followup(message, session)
    carried_slots: set[str] = set()

    if follow_up and session and session.filters:
        prev = session.filters
        # Carry slots from previous turn unless current message sets them
        if property_type is None and prev.property_type is not None:
            property_type = prev.property_type
            carried_slots.add("property_type")
        if purpose is None and prev.purpose is not None:
            purpose = prev.purpose
            carried_slots.add("purpose")
        if (
            bed_min is None
            and bed_max is None
            and (prev.bedrooms_min is not None or prev.bedrooms_max is not None)
        ):
            bed_min = prev.bedrooms_min
            bed_max = prev.bedrooms_max
            carried_slots.add("bedrooms")
        if (
            price_min is None
            and price_max is None
            and (prev.price_min_sar is not None or prev.price_max_sar is not None)
        ):
            price_min = prev.price_min_sar
            price_max = prev.price_max_sar
            carried_slots.add("price")
        if (
            area_min is None
            and area_max is None
            and (prev.area_min_sqm is not None or prev.area_max_sqm is not None)
        ):
            area_min = prev.area_min_sqm
            area_max = prev.area_max_sqm
            carried_slots.add("area")
        if city is None and prev.city is not None:
            city = prev.city
            carried_slots.add("city")
        if district is None and prev.district is not None:
            district = prev.district
            carried_slots.add("district")
        if country is None and prev.country is not None:
            country = prev.country
            carried_slots.add("country")

        # Apply comparatives
        price_min, price_max, area_min, area_max, bed_min, bed_max = _apply_comparatives(
            message, session, price_min, price_max, area_min, area_max, bed_min, bed_max,
        )

    # Default: price intent without explicit purpose defaults to sale
    # Covers both numeric price bounds AND sort-by-price intent ("cheap", "expensive")
    has_price_intent = (
        price_min is not None
        or price_max is not None
        or sort in (SortKey.price_asc, SortKey.price_desc)
    )
    if (
        has_price_intent
        and purpose is None
        and not (follow_up and "purpose" in carried_slots)
    ):
        purpose = Purpose.sale

    # CR-1: price aggregates (avg, median, min, max of price) default to sale
    _PRICE_AGG_OPS = {
        AggregateOp.avg_price, AggregateOp.median_price,
        AggregateOp.min_price, AggregateOp.max_price,
    }
    if (
        agg_op in _PRICE_AGG_OPS
        and purpose is None
        and not (follow_up and "purpose" in carried_slots)
    ):
        purpose = Purpose.sale

    # Determine kinds for route
    kinds: tuple[DocKind, ...] | None = None

    # Determine route (order: aggregate, project_info, lookup, out_of_scope)
    # 1. Aggregate
    if agg_op is not None:
        filters = SearchFilters(
            purpose=purpose,
            property_type=property_type,
            city=city,
            district=district,
            country=country,
            project_id=project_id,
            bedrooms_min=bed_min,
            bedrooms_max=bed_max,
            price_min_sar=price_min,
            price_max_sar=price_max,
            area_min_sqm=area_min,
            area_max_sqm=area_max,
            sort=sort,
        )
        return RouteDecision(
            language=language,
            route=Route.aggregate,
            op=agg_op,
            example_limit=example_limit,
            filters=filters,
            carried_slots=frozenset(carried_slots),
            numbers=canonical_nums,
            negated=negated,
            follow_up=follow_up,
        )

    # 2. Project info
    if has_project_entity or has_brand_entity or _cue_found(message, _PROJECT_CUES):
        project_kinds: tuple[DocKind, ...] = (DocKind.project, DocKind.article)
        filters = SearchFilters(
            kinds=project_kinds,
            purpose=purpose,
            property_type=property_type,
            city=city,
            district=district,
            country=country,
            project_id=project_id,
            bedrooms_min=bed_min,
            bedrooms_max=bed_max,
            price_min_sar=price_min,
            price_max_sar=price_max,
            area_min_sqm=area_min,
            area_max_sqm=area_max,
            sort=sort,
        )
        return RouteDecision(
            language=language,
            route=Route.project_info,
            filters=filters,
            carried_slots=frozenset(carried_slots),
            numbers=canonical_nums,
            negated=negated,
            follow_up=follow_up,
        )

    # 3. Out of scope check
    has_entity = len(entity_matches) > 0
    has_filter = (
        property_type is not None
        or purpose is not None
        or bed_min is not None
        or bed_max is not None
        or price_min is not None
        or price_max is not None
        or area_min is not None
        or area_max is not None
        or city is not None
        or district is not None
        or country is not None
        or project_id is not None
    )
    has_domain_cue = _cue_found(message, _DOMAIN_CUES)
    has_project_cue = _cue_found(message, _PROJECT_CUES)

    # B7: topic-cue OOS -- a topic cue forces OOS unless a real-estate slot,
    # an aggregate op, a domain cue, or a project cue is present.  A city or
    # country entity alone does not override the topic cue.
    has_real_estate_slot = (
        purpose is not None
        or property_type is not None
        or bed_min is not None
        or bed_max is not None
        or price_min is not None
        or price_max is not None
        or area_min is not None
        or area_max is not None
        or project_id is not None
    )
    if (
        _cue_found(message, _OOS_TOPIC_CUES)
        and not has_real_estate_slot
        and agg_op is None
        and not has_domain_cue
        and not has_project_cue
    ):
        return RouteDecision(
            language=language,
            route=Route.out_of_scope,
            numbers=canonical_nums,
            negated=negated,
        )

    if not has_entity and not has_filter and not has_domain_cue and not has_project_cue and not follow_up:
        return RouteDecision(
            language=language,
            route=Route.out_of_scope,
            numbers=canonical_nums,
            negated=negated,
        )

    # 4. Lookup
    # kinds=(listing,) if any listing-only slot is set
    listing_only_slots = (
        purpose is not None
        or property_type is not None
        or bed_min is not None or bed_max is not None
        or price_min is not None or price_max is not None
        or area_min is not None or area_max is not None
        or district is not None
    )
    if listing_only_slots:
        kinds = (DocKind.listing,)
    elif has_country_entity:
        # A country entity sets kinds=(project,) per the brief
        kinds = (DocKind.project,)

    filters = SearchFilters(
        kinds=kinds,
        purpose=purpose,
        property_type=property_type,
        city=city,
        district=district,
        country=country,
        project_id=project_id,
        bedrooms_min=bed_min,
        bedrooms_max=bed_max,
        price_min_sar=price_min,
        price_max_sar=price_max,
        area_min_sqm=area_min,
        area_max_sqm=area_max,
        sort=sort,
    )

    return RouteDecision(
        language=language,
        route=Route.lookup,
        filters=filters,
        carried_slots=frozenset(carried_slots),
        numbers=canonical_nums,
        negated=negated,
        follow_up=follow_up,
    )
