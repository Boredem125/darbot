"""Tests for the rule-based router (app/domain/router.py).

Covers PLAN section 12 probes and the brief section 2 requirements.
"""

import pytest

from app.domain.ports import AggregateOp
from app.domain.router import (
    EntityIndex,
    SessionSlots,
    route,
)
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

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_entities() -> Entities:
    return Entities(entities=[
        Entity(kind=EntityKind.city, canonical="Riyadh", aliases=["الرياض"]),
        Entity(kind=EntityKind.city, canonical="Jeddah", aliases=["جدة", "Jiddah"]),
        Entity(kind=EntityKind.district, canonical="Al Narjis", parent="Riyadh", aliases=["al narjis"]),
        Entity(kind=EntityKind.district, canonical="Al Rawdah", parent="Jeddah"),
        Entity(kind=EntityKind.country, canonical="Saudi Arabia", aliases=["السعودية"]),
        Entity(
            kind=EntityKind.project,
            canonical="Trump Tower Jeddah",
            aliases=["Trump Tower", "trump tower jeddah"],
            doc_id="darglobal:project:trump-tower-jeddah",
        ),
        Entity(kind=EntityKind.brand, canonical="Trump", aliases=[]),
    ])


@pytest.fixture
def entity_index() -> EntityIndex:
    return EntityIndex(_make_entities())


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


class TestLanguageDetection:
    def test_english(self, entity_index):
        d = route("villas in Riyadh", entities=entity_index)
        assert d.language == Language.en

    def test_arabic(self, entity_index):
        d = route("فلل في الرياض", entities=entity_index)
        assert d.language == Language.ar

    def test_hint_fallback(self, entity_index):
        d = route("123", entities=entity_index, lang_hint=Language.ar)
        assert d.language == Language.ar

    def test_no_letters_no_hint(self, entity_index):
        d = route("123", entities=entity_index)
        assert d.language == Language.en


# ---------------------------------------------------------------------------
# Property type extraction
# ---------------------------------------------------------------------------


class TestPropertyType:
    @pytest.mark.parametrize("text,expected", [
        ("villas in Riyadh", PropertyType.villa),
        ("فيلا in Riyadh", PropertyType.villa),
        ("apartment for rent", PropertyType.apartment),
        ("flat for sale", PropertyType.apartment),
        ("شقة Riyadh", PropertyType.apartment),
        ("townhouse in Riyadh", PropertyType.townhouse),
        ("duplex in Riyadh", PropertyType.duplex),
        ("دوبلكس in Riyadh", PropertyType.duplex),
        ("land for sale in Riyadh", PropertyType.land),
        ("أرض Riyadh", PropertyType.land),
        ("building in Riyadh", PropertyType.building),
        ("عمارة Riyadh", PropertyType.building),
        ("office for rent in Riyadh", PropertyType.office),
        ("مكتب Riyadh", PropertyType.office),
        ("shop for sale in Riyadh", PropertyType.shop),
        ("warehouse in Riyadh", PropertyType.warehouse),
        ("chalet in Riyadh", PropertyType.chalet),
        ("شاليه Riyadh", PropertyType.chalet),
        ("farm for sale in Riyadh", PropertyType.farm),
        ("مزرعة Riyadh", PropertyType.farm),
    ])
    def test_property_type(self, text, expected, entity_index):
        d = route(text, entities=entity_index)
        assert d.filters.property_type == expected


# ---------------------------------------------------------------------------
# Purpose extraction
# ---------------------------------------------------------------------------


class TestPurpose:
    def test_rent(self, entity_index):
        d = route("apartment for rent in Jeddah", entities=entity_index)
        assert d.filters.purpose == Purpose.rent

    def test_sale(self, entity_index):
        d = route("villas for sale in Riyadh", entities=entity_index)
        assert d.filters.purpose == Purpose.sale

    def test_arabic_rent(self, entity_index):
        d = route("شقة للإيجار Jeddah", entities=entity_index)
        assert d.filters.purpose == Purpose.rent

    def test_arabic_sale(self, entity_index):
        d = route("فيلا للبيع Riyadh", entities=entity_index)
        assert d.filters.purpose == Purpose.sale

    def test_price_without_purpose_defaults_to_sale(self, entity_index):
        d = route("villas under 2M in Riyadh", entities=entity_index)
        assert d.filters.purpose == Purpose.sale


# ---------------------------------------------------------------------------
# Bedrooms
# ---------------------------------------------------------------------------


class TestBedrooms:
    def test_exact_3bed(self, entity_index):
        d = route("3-bed villas in Riyadh", entities=entity_index)
        assert d.filters.bedrooms_min == 3
        assert d.filters.bedrooms_max == 3

    def test_3br(self, entity_index):
        d = route("3br apartments in Jeddah", entities=entity_index)
        assert d.filters.bedrooms_min == 3
        assert d.filters.bedrooms_max == 3

    def test_studio(self, entity_index):
        d = route("studio apartment in Riyadh", entities=entity_index)
        assert d.filters.bedrooms_min == 0
        assert d.filters.bedrooms_max == 0

    def test_arabic_bedrooms(self, entity_index):
        d = route("3 غرف Riyadh", entities=entity_index)
        assert d.filters.bedrooms_min == 3
        assert d.filters.bedrooms_max == 3


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------


class TestPrice:
    def test_under_2m(self, entity_index):
        d = route("villas under 2M in Riyadh", entities=entity_index)
        assert d.filters.price_max_sar == 2_000_000

    def test_above_1m(self, entity_index):
        d = route("villas over 1M in Riyadh", entities=entity_index)
        assert d.filters.price_min_sar == 1_000_000

    def test_monthly_rent_multiplied_by_12(self, entity_index):
        d = route("apartment under 5000 per month in Jeddah", entities=entity_index)
        assert d.filters.purpose == Purpose.rent
        assert d.filters.price_max_sar == 60_000  # 5000 * 12


# ---------------------------------------------------------------------------
# Place entities
# ---------------------------------------------------------------------------


class TestPlaces:
    def test_city(self, entity_index):
        d = route("villas in Riyadh", entities=entity_index)
        assert d.filters.city == "Riyadh"

    def test_district_sets_parent_city(self, entity_index):
        d = route("villas in Al Narjis", entities=entity_index)
        assert d.filters.district == "Al Narjis"
        assert d.filters.city == "Riyadh"

    def test_country_sets_kinds(self, entity_index):
        d = route("properties in Saudi Arabia", entities=entity_index)
        assert d.filters.country == "Saudi Arabia"
        assert d.filters.kinds == (DocKind.project,)


# ---------------------------------------------------------------------------
# Aggregate routes
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_cheapest(self, entity_index):
        d = route("cheapest apartment for rent in Jeddah", entities=entity_index)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.min_price
        assert d.filters.purpose == Purpose.rent
        assert d.filters.city == "Jeddah"
        assert d.filters.property_type == PropertyType.apartment

    def test_how_many(self, entity_index):
        d = route("how many villas in Riyadh", entities=entity_index)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.count

    def test_average_price(self, entity_index):
        d = route("average price of villas in Riyadh", entities=entity_index)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_price

    def test_newest(self, entity_index):
        d = route("newest listings in Riyadh", entities=entity_index)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.count
        assert d.example_limit == 5
        assert d.filters.sort == SortKey.newest


# ---------------------------------------------------------------------------
# Project info
# ---------------------------------------------------------------------------


class TestProjectInfo:
    def test_project_entity(self, entity_index):
        d = route("Trump Tower Jeddah handover date?", entities=entity_index)
        assert d.route == Route.project_info
        assert d.filters.project_id == "darglobal:project:trump-tower-jeddah"
        assert d.filters.kinds == (DocKind.project, DocKind.article)

    def test_project_cue(self, entity_index):
        d = route("what about darglobal projects?", entities=entity_index)
        assert d.route == Route.project_info


# ---------------------------------------------------------------------------
# Out of scope
# ---------------------------------------------------------------------------


class TestOutOfScope:
    def test_irrelevant_query(self, entity_index):
        d = route("what is the weather today?", entities=entity_index)
        assert d.route == Route.out_of_scope

    def test_domain_cue_not_oos(self, entity_index):
        d = route("property listings", entities=entity_index)
        assert d.route != Route.out_of_scope


# ---------------------------------------------------------------------------
# Follow-ups and slot carry-over
# ---------------------------------------------------------------------------


class TestFollowUp:
    def test_cheaper_ones(self, entity_index):
        """PLAN section 12: 'cheaper ones?' follow-up."""
        session = SessionSlots(
            filters=SearchFilters(
                purpose=Purpose.sale,
                property_type=PropertyType.villa,
                city="Riyadh",
                bedrooms_min=3,
                bedrooms_max=3,
                price_max_sar=2_000_000,
            ),
            last_prices=(1_950_000,),
            last_areas=(),
        )
        d = route("cheaper ones?", session=session, entities=entity_index)
        assert d.follow_up is True
        assert d.filters.property_type == PropertyType.villa
        assert d.filters.city == "Riyadh"
        assert d.filters.price_max_sar == 1_949_999  # min(last_prices) - 1
        assert "property_type" in d.carried_slots
        assert "city" in d.carried_slots

    def test_no_followup_without_session(self, entity_index):
        d = route("cheaper ones?", entities=entity_index)
        assert d.follow_up is False


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------


class TestNegation:
    def test_negated_query(self, entity_index):
        d = route("villas without a pool in Riyadh", entities=entity_index)
        assert d.negated is True

    def test_non_negated(self, entity_index):
        d = route("villas with a pool in Riyadh", entities=entity_index)
        assert d.negated is False


# ---------------------------------------------------------------------------
# PLAN section 12 integration probes
# ---------------------------------------------------------------------------


class TestPlanSection12Probes:
    def test_3bed_villas_riyadh_under_2m(self, entity_index):
        """'3-bed villas in Riyadh under 2M SAR'."""
        d = route("3-bed villas in Riyadh under 2M SAR", entities=entity_index)
        assert d.language == Language.en
        assert d.route == Route.lookup
        assert d.filters.property_type == PropertyType.villa
        assert d.filters.bedrooms_min == 3
        assert d.filters.bedrooms_max == 3
        assert d.filters.city == "Riyadh"
        assert d.filters.price_max_sar == 2_000_000
        assert d.filters.purpose == Purpose.sale  # price without purpose defaults to sale

    def test_cheapest_apartment_rent_jeddah(self, entity_index):
        """'cheapest apartment for rent in Jeddah'."""
        d = route("cheapest apartment for rent in Jeddah", entities=entity_index)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.min_price
        assert d.filters.purpose == Purpose.rent
        assert d.filters.city == "Jeddah"
        assert d.filters.property_type == PropertyType.apartment

    def test_trump_tower_handover(self, entity_index):
        """'Trump Tower Jeddah handover date?'."""
        d = route("Trump Tower Jeddah handover date?", entities=entity_index)
        assert d.route == Route.project_info
        assert d.filters.project_id == "darglobal:project:trump-tower-jeddah"

    def test_arabic_query(self, entity_index):
        """فلل في الرياض (villas in Riyadh)."""
        d = route("فلل في الرياض", entities=entity_index)
        assert d.language == Language.ar
        assert d.filters.property_type == PropertyType.villa
        assert d.filters.city == "Riyadh"

    def test_followup_cheaper(self, entity_index):
        """Follow-up 'cheaper ones?'."""
        session = SessionSlots(
            filters=SearchFilters(
                purpose=Purpose.sale,
                property_type=PropertyType.villa,
                city="Riyadh",
                bedrooms_min=3,
                bedrooms_max=3,
                price_max_sar=2_000_000,
            ),
            last_prices=(1_950_000,),
            last_areas=(),
        )
        d = route("cheaper ones?", session=session, entities=entity_index)
        assert d.follow_up is True
        assert d.filters.price_max_sar == 1_949_999


# ---------------------------------------------------------------------------
# Live-probe regressions (2026-11-20)
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_index() -> EntityIndex:
    ents = _make_entities()
    return EntityIndex(Entities(entities=[
        *ents.entities,
        Entity(kind=EntityKind.city, canonical="Dubai"),
        # The scraper stored DarGlobal's /projects index page as a project.
        Entity(kind=EntityKind.project, canonical="Projects", doc_id="darglobal:project:projects"),
        Entity(kind=EntityKind.project, canonical="DG1", parent="Dubai", doc_id="darglobal:project:dg1"),
        Entity(
            kind=EntityKind.project,
            canonical="Nº149 Old Park Lane",
            doc_id="darglobal:project:the-mulliner",
        ),
    ]))


class TestLiveProbeRegressions:
    def test_generic_projects_word_is_not_an_entity(self, probe_index):
        d = route("tell me about Dar Global projects in Dubai", entities=probe_index)
        assert d.route == Route.project_info
        assert d.filters.project_id is None
        assert d.filters.city == "Dubai"

    def test_what_projects_does_dar_global_have(self, probe_index):
        d = route("what projects does Dar Global have?", entities=probe_index)
        assert d.filters.project_id is None

    @pytest.mark.parametrize("q", ["tell me about DG1", "tell me about DG1?", "DG1 details"])
    def test_digit_in_project_name_is_not_a_price(self, probe_index, q):
        d = route(q, entities=probe_index)
        assert d.filters.project_id == "darglobal:project:dg1"
        assert d.filters.price_min_sar is None
        assert d.filters.price_max_sar is None
        assert d.filters.purpose is None

    def test_digit_project_name_keeps_real_price(self, probe_index):
        d = route("DG1 apartments under 3 million", entities=probe_index)
        assert d.filters.project_id == "darglobal:project:dg1"
        assert d.filters.price_max_sar == 3_000_000

    def test_normalised_digit_project_name_is_not_a_price(self, probe_index):
        d = route("tell me about Nº149 Old Park Lane", entities=probe_index)
        assert d.filters.project_id == "darglobal:project:the-mulliner"
        assert d.filters.price_min_sar is None
        assert d.filters.price_max_sar is None

    @pytest.mark.parametrize("q", [
        "شقق للايجار في الرياض غرفتين",  # no hamza
        "شقق للإيجار في الرياض غرفتين",
        "فلل الايجار في الرياض",
    ])
    def test_arabic_rent_spellings(self, probe_index, q):
        d = route(q, entities=probe_index)
        assert d.filters.purpose == Purpose.rent
        assert d.filters.city == "Riyadh"


class TestNoWordlistFallback:
    """Off-wordlist phrasing must reach search, not a router refusal."""

    @pytest.mark.parametrize("q", [
        "show me something spacious with a garden",
        "anything with a pool and a balcony?",
        "مكان واسع مع حديقة",
    ])
    def test_listing_features_search_listings(self, probe_index, q):
        d = route(q, entities=probe_index)
        assert d.route == Route.lookup
        assert d.filters.kinds == (DocKind.listing,)

    @pytest.mark.parametrize("q", ["a place near the sea", "hello", "tell me a story about dragons"])
    def test_no_signal_goes_to_search_with_empty_filters(self, probe_index, q):
        # Empty filters: the service's dense score floor decides refusal.
        d = route(q, entities=probe_index)
        assert d.route == Route.lookup
        assert d.filters.is_empty()

    def test_topic_cue_still_refused_by_router(self, probe_index):
        d = route("what is the weather in riyadh", entities=probe_index)
        assert d.route == Route.out_of_scope

    def test_budget_per_year_is_rent_cap(self, probe_index):
        d = route("I need a place for my family of 5 in Jeddah, budget 80k a year", entities=probe_index)
        assert d.filters.purpose == Purpose.rent
        assert d.filters.price_max_sar == 80_000
        assert d.filters.city == "Jeddah"

    def test_ordinary_phrase_district_not_matched(self):
        idx = EntityIndex(Entities(entities=[
            Entity(kind=EntityKind.district, canonical="The Beach", parent="Jazan"),
        ]))
        d = route("any penthouses by the beach?", entities=idx)
        assert d.filters.district is None
        assert d.filters.city is None
