"""Tests for cache key computation (caching.md section 9)."""


from app.domain.cache_keys import (
    CacheKeyParts,
    build_key,
    build_scope,
    parts_from,
)
from app.domain.router import RouteDecision
from common.schemas import Language, PropertyType, Purpose, Route, SearchFilters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parts(**overrides) -> CacheKeyParts:
    defaults = {
        "index_version": "0123456789abcdef",
        "prompt_version": "system_v1@1a2b3c4d",
        "language": Language.en,
        "route": Route.lookup,
        "op": None,
        "filters_json": SearchFilters().canonical_json(),
        "numbers": (),
        "negated": False,
        "query_key_text": "villas in riyadh",
        "amenity_tokens": (),
    }
    defaults.update(overrides)
    return CacheKeyParts(**defaults)


# ---------------------------------------------------------------------------
# Tests from caching.md section 9
# ---------------------------------------------------------------------------


class TestCacheKeyDifferences:
    def test_3bed_vs_4bed(self):
        """3-bed and 4-bed give different keys."""
        f3 = SearchFilters(bedrooms_min=3, bedrooms_max=3)
        f4 = SearchFilters(bedrooms_min=4, bedrooms_max=4)
        p3 = _parts(filters_json=f3.canonical_json(), query_key_text="3 bed villas")
        p4 = _parts(filters_json=f4.canonical_json(), query_key_text="4 bed villas")
        assert build_key(p3) != build_key(p4)

    def test_riyadh_vs_jeddah(self):
        fr = SearchFilters(city="Riyadh")
        fj = SearchFilters(city="Jeddah")
        pr = _parts(filters_json=fr.canonical_json(), query_key_text="villas in riyadh")
        pj = _parts(filters_json=fj.canonical_json(), query_key_text="villas in jeddah")
        assert build_key(pr) != build_key(pj)

    def test_sale_vs_rent(self):
        fs = SearchFilters(purpose=Purpose.sale)
        fr = SearchFilters(purpose=Purpose.rent)
        ps = _parts(filters_json=fs.canonical_json())
        pr = _parts(filters_json=fr.canonical_json())
        assert build_key(ps) != build_key(pr)

    def test_min_price_vs_max_price(self):
        fmin = SearchFilters(price_min_sar=1_000_000)
        fmax = SearchFilters(price_max_sar=1_000_000)
        pmin = _parts(filters_json=fmin.canonical_json())
        pmax = _parts(filters_json=fmax.canonical_json())
        assert build_key(pmin) != build_key(pmax)

    def test_prompt_version_change(self):
        p1 = _parts(prompt_version="system_v1@1a2b3c4d")
        p2 = _parts(prompt_version="system_v1@deadbeef")
        assert build_key(p1) != build_key(p2)

    def test_index_version_change(self):
        p1 = _parts(index_version="0123456789abcdef")
        p2 = _parts(index_version="fedcba9876543210")
        assert build_key(p1) != build_key(p2)

    def test_language_change(self):
        pen = _parts(language=Language.en)
        par = _parts(language=Language.ar)
        assert build_key(pen) != build_key(par)


class TestNormalisedKeyEquality:
    """Under normalise_for_key, number representations give the same key."""

    def test_2m_equals_2000000(self):
        from common.text import normalise_for_key
        q1 = normalise_for_key("villas under 2M")
        q2 = normalise_for_key("villas under 2,000,000")
        p1 = _parts(query_key_text=q1)
        p2 = _parts(query_key_text=q2)
        assert build_key(p1) == build_key(p2)

    def test_2_million_equals_2m(self):
        from common.text import normalise_for_key
        q1 = normalise_for_key("villas under 2 million")
        q2 = normalise_for_key("villas under 2m")
        p1 = _parts(query_key_text=q1)
        p2 = _parts(query_key_text=q2)
        assert build_key(p1) == build_key(p2)

    def test_arabic_digits(self):
        from common.text import normalise_for_key
        q1 = normalise_for_key("villas under 2000000")
        q2 = normalise_for_key("villas under ٢٠٠٠٠٠٠")
        p1 = _parts(query_key_text=q1)
        p2 = _parts(query_key_text=q2)
        assert build_key(p1) == build_key(p2)

    def test_three_equals_3(self):
        from common.text import normalise_for_key
        q1 = normalise_for_key("three bedrooms")
        q2 = normalise_for_key("3 bedrooms")
        p1 = _parts(query_key_text=q1)
        p2 = _parts(query_key_text=q2)
        assert build_key(p1) == build_key(p2)


class TestNegationScope:
    def test_with_vs_without_pool_different_scopes(self):
        """'with a pool' vs 'without a pool' gives different scopes (negation cue)."""
        p1 = _parts(negated=False, query_key_text="villas with a pool in riyadh")
        p2 = _parts(negated=True, query_key_text="villas without a pool in riyadh")
        assert build_scope(p1) != build_scope(p2)


class TestDeterminism:
    def test_same_parts_same_key(self):
        """Same parts always give the same key."""
        p = _parts()
        assert build_key(p) == build_key(p)

    def test_same_parts_same_scope(self):
        p = _parts()
        assert build_scope(p) == build_scope(p)


class TestSchemaPrefix:
    def test_key_prefix(self):
        p = _parts()
        assert build_key(p).startswith("ck2:k:")

    def test_scope_prefix(self):
        p = _parts()
        assert build_scope(p).startswith("ck2:s:")


class TestPartsFrom:
    def test_roundtrip(self):
        d = RouteDecision(
            language=Language.en,
            route=Route.lookup,
            filters=SearchFilters(city="Riyadh", property_type=PropertyType.villa),
            numbers=("2000000",),
            negated=False,
        )
        p = parts_from(d, "villas under 2M in Riyadh", "0123456789abcdef", "system_v1@1a2b3c4d")
        assert p.language == Language.en
        assert p.route == Route.lookup
        assert p.numbers == ("2000000",)
        key = build_key(p)
        assert key.startswith("ck2:k:")


# ---------------------------------------------------------------------------
# B9 tests: amenity/view tokens in scope
# ---------------------------------------------------------------------------


class TestB9AmenityScope:
    """B9: pool vs gym -> different scopes; sea view vs park view -> different scopes."""

    def test_pool_vs_gym_different_scopes(self):
        p1 = _parts(query_key_text="villas with a pool in riyadh", amenity_tokens=("pool",))
        p2 = _parts(query_key_text="villas with a gym in riyadh", amenity_tokens=("gym",))
        assert build_scope(p1) != build_scope(p2)

    def test_sea_view_vs_park_view_different_scopes(self):
        p1 = _parts(query_key_text="apartments with sea view", amenity_tokens=("sea view",))
        p2 = _parts(query_key_text="apartments with park view", amenity_tokens=("park view",))
        assert build_scope(p1) != build_scope(p2)

    def test_furnished_vs_unfurnished_different_scopes(self):
        """'unfurnished' must not match 'furnished'."""
        p1 = _parts(query_key_text="furnished apartments", amenity_tokens=("furnished",))
        p2 = _parts(query_key_text="unfurnished apartments", amenity_tokens=("unfurnished",))
        assert build_scope(p1) != build_scope(p2)

    def test_same_amenity_same_scope(self):
        """Same amenity tokens -> same scope."""
        p1 = _parts(query_key_text="villas with pool in riyadh", amenity_tokens=("pool",))
        p2 = _parts(query_key_text="riyadh villas pool", amenity_tokens=("pool",))
        assert build_scope(p1) == build_scope(p2)

    def test_anything_but_villas_negated(self):
        """'anything but villas' triggers negation."""
        from common.text import detect_negation

        assert detect_negation("anything but villas") is True

    def test_3bed_vs_4bed_never_share_answer(self):
        """3-bed and 4-bed must never share an answer (caching.md section 9)."""
        f3 = SearchFilters(bedrooms_min=3, bedrooms_max=3)
        f4 = SearchFilters(bedrooms_min=4, bedrooms_max=4)
        p3 = _parts(filters_json=f3.canonical_json(), query_key_text="3 bed villas")
        p4 = _parts(filters_json=f4.canonical_json(), query_key_text="4 bed villas")
        # Different scopes because filters differ
        assert build_scope(p3) != build_scope(p4)
        # Different keys
        assert build_key(p3) != build_key(p4)
