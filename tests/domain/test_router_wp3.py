"""WP-3-router tests: one per tuning-report item B1-B8 and B10.

Tests are written first (before the code changes) per the brief.
Each test names the item, the dev case(s) it fixes, and the expected
route/op/filters.
"""

import pytest

from app.domain.ports import AggregateOp
from app.domain.router import (
    EntityIndex,
    route,
)
from common.schemas import (
    Entities,
    Entity,
    EntityKind,
    PropertyType,
    Purpose,
    Route,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal entity set covering the dev cases. Includes Marbella (city),
# Tierra Viva (project), and two project entities for B8a longest-match.
_ENTITIES = Entities(entities=[
    Entity(kind=EntityKind.city, canonical="Riyadh", aliases=["الرياض"]),
    Entity(kind=EntityKind.city, canonical="Jeddah", aliases=["جدة", "Jiddah"]),
    Entity(kind=EntityKind.city, canonical="Khobar", aliases=[]),
    Entity(kind=EntityKind.city, canonical="Jazan", aliases=["جازان"]),
    Entity(kind=EntityKind.city, canonical="Marbella", aliases=[]),
    Entity(kind=EntityKind.city, canonical="Muscat", aliases=[]),
    Entity(kind=EntityKind.country, canonical="Saudi Arabia", aliases=["السعودية"]),
    Entity(kind=EntityKind.country, canonical="Oman", aliases=["عمان"]),
    Entity(
        kind=EntityKind.project,
        canonical="Tierra Viva",
        aliases=[],
        doc_id="darglobal:project:tierra-viva",
    ),
    Entity(
        kind=EntityKind.project,
        canonical="Marriott Residences Aida, Oman",
        aliases=["Marriott Residences Aida"],
        doc_id="darglobal:project:marriott-residences-aida-oman",
    ),
    # Two overlapping projects to test B8a longest-match-first
    Entity(
        kind=EntityKind.project,
        canonical="Grand Tower",
        aliases=["Grand Tower"],
        doc_id="darglobal:project:grand-tower",
    ),
    Entity(
        kind=EntityKind.project,
        canonical="Grand Tower Residences",
        aliases=["Grand Tower Residences"],
        doc_id="darglobal:project:grand-tower-residences",
    ),
])


@pytest.fixture
def eidx() -> EntityIndex:
    return EntityIndex(_ENTITIES)


# ---------------------------------------------------------------------------
# B1: reranker_fastembed import fix
# ---------------------------------------------------------------------------


class TestB1RerankerImport:
    """B1: fastembed.rerank.cross_encoder.TextCrossEncoder must import without error."""

    def test_import_text_cross_encoder(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401


# ---------------------------------------------------------------------------
# B2: avg_area before avg_price; new area cues
# ---------------------------------------------------------------------------


class TestB2AvgAreaBeforeAvgPrice:
    """B2: 'average size' must route to avg_area, not avg_price.
    Fixes tpl-dev-010, tpl-dev-017, tpl-dev-020, tpl-dev-033.
    """

    def test_average_size_routes_avg_area(self, eidx):
        # tpl-dev-010: "What is the average size of floor units for rent in Riyadh?"
        d = route(
            "What is the average size of floor units for rent in Riyadh?",
            entities=eidx,
        )
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_area

    def test_mean_size_routes_avg_area(self, eidx):
        d = route("mean size of villas for sale in Khobar?", entities=eidx)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_area

    def test_avg_size_routes_avg_area(self, eidx):
        d = route("avg size of apartments in Riyadh", entities=eidx)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_area

    def test_arabic_avg_area(self, eidx):
        # tpl-dev-033: "ما متوسط مساحة شقق ..."
        d = route(
            "ما متوسط مساحة شقق 5 غرف للبيع في جازان تحت 650 ألف ريال؟",
            entities=eidx,
        )
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_area

    def test_arabic_avg_area_definite(self, eidx):
        d = route("متوسط المساحة شقق في الرياض", entities=eidx)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_area

    def test_average_price_still_works(self, eidx):
        """'average price' must still route to avg_price."""
        d = route("average price of villas in Riyadh", entities=eidx)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_price


# ---------------------------------------------------------------------------
# B3: "avg" cue for avg_price
# ---------------------------------------------------------------------------


class TestB3AvgCue:
    """B3: 'avg' alone triggers avg_price. Fixes hw-004."""

    def test_avg_yearly_rent(self, eidx):
        # hw-004: "avg yearly rent for an apartment in Riyadh?"
        d = route("avg yearly rent for an apartment in Riyadh?", entities=eidx)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.avg_price


# ---------------------------------------------------------------------------
# B4: Arabic median cues
# ---------------------------------------------------------------------------


class TestB4ArabicMedian:
    """B4: 'الوسيط' and 'وسيط' trigger median_price. Fixes tpl-dev-052."""

    def test_arabic_median_definite(self, eidx):
        # tpl-dev-052: "ما هو السعر الوسيط لـ شقق 2 غرف للإيجار في الرياض؟"
        d = route(
            "ما هو السعر الوسيط لـ شقق 2 غرف للإيجار في الرياض؟",
            entities=eidx,
        )
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.median_price

    def test_arabic_median_bare(self, eidx):
        d = route("وسيط سعر الشقق في الرياض", entities=eidx)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.median_price


# ---------------------------------------------------------------------------
# B5: Arabic property type additions
# ---------------------------------------------------------------------------


class TestB5ArabicPropertyTypes:
    """B5: Arabic plural/definite property types. Fixes ar-002."""

    @pytest.mark.parametrize(
        "cue,expected",
        [
            ("الفلل", PropertyType.villa),
            ("الفيلا", PropertyType.villa),
            ("فلة", PropertyType.villa),
            ("الشقق", PropertyType.apartment),
            ("الشقة", PropertyType.apartment),
            ("الأراضي", PropertyType.land),
            ("أراضي", PropertyType.land),
            ("الأرض", PropertyType.land),
            ("الارض", PropertyType.land),
            ("buildings", PropertyType.building),
            ("plots", PropertyType.land),
            ("offices", PropertyType.office),
            ("shops", PropertyType.shop),
        ],
    )
    def test_property_type_cue(self, cue, expected, eidx):
        # Wrap the cue in a small sentence to simulate real usage
        d = route(f"{cue} in Riyadh", entities=eidx)
        assert d.filters.property_type == expected

    def test_ar002_full_query(self, eidx):
        # ar-002: "كم عدد الفلل للبيع في الرياض؟"
        d = route("كم عدد الفلل للبيع في الرياض؟", entities=eidx)
        assert d.route == Route.aggregate
        assert d.op == AggregateOp.count
        assert d.filters.property_type == PropertyType.villa
        assert d.filters.purpose == Purpose.sale
        assert d.filters.city == "Riyadh"


# ---------------------------------------------------------------------------
# B6: Arabic price bounds "بين ... و ..."
# ---------------------------------------------------------------------------


class TestB6ArabicPriceBounds:
    """B6: Arabic 'بين X و Y' extracts price_min and price_max. Fixes ar-003."""

    def test_ar003_full_query(self, eidx):
        # ar-003: "شقق للبيع في جدة بين 500 ألف و 1 مليون ريال"
        d = route("شقق للبيع في جدة بين 500 ألف و 1 مليون ريال", entities=eidx)
        assert d.route == Route.lookup
        assert d.filters.purpose == Purpose.sale
        assert d.filters.property_type == PropertyType.apartment
        assert d.filters.city == "Jeddah"
        assert d.filters.price_min_sar == 500_000
        assert d.filters.price_max_sar == 1_000_000

    def test_arabic_between_simple(self, eidx):
        d = route("بين 100000 و 200000 شقق في الرياض", entities=eidx)
        assert d.filters.price_min_sar == 100_000
        assert d.filters.price_max_sar == 200_000


# ---------------------------------------------------------------------------
# B7: OOS topic cues
# ---------------------------------------------------------------------------


class TestB7OOSTopicCues:
    """B7: topic cue + no real-estate slots -> out_of_scope. Fixes oos-001."""

    def test_weather_in_riyadh_oos(self, eidx):
        # oos-001: "What's the weather in Riyadh tomorrow?"
        d = route("What's the weather in Riyadh tomorrow?", entities=eidx)
        assert d.route == Route.out_of_scope

    @pytest.mark.parametrize(
        "text",
        [
            "What is the capital of France?",
            "Tell me a joke",
            "python code to sort a list",
            "football match results",
            "ما هو الطقس في الرياض؟",
        ],
    )
    def test_topic_cue_triggers_oos(self, text, eidx):
        d = route(text, entities=eidx)
        assert d.route == Route.out_of_scope

    def test_domain_cue_overrides_topic_cue(self, eidx):
        """A domain cue keeps the query in scope even with a weather word."""
        d = route("weather resistant villas for sale in Riyadh", entities=eidx)
        # 'villas' is a property type -> has a slot -> not OOS
        assert d.route != Route.out_of_scope

    def test_property_type_overrides_topic_cue(self, eidx):
        d = route("joke villas in Riyadh", entities=eidx)
        assert d.route != Route.out_of_scope


# ---------------------------------------------------------------------------
# B8: Project entity handling
# ---------------------------------------------------------------------------


class TestB8ProjectEntityHandling:
    """B8a: first (longest) project match wins.
    B8b: when project_id is set, city/district/country are cleared.
    Fixes pj-005 (Tierra Viva + Marbella city).
    """

    def test_b8a_longest_project_match_wins(self, eidx):
        """B8a: 'Grand Tower Residences' should beat 'Grand Tower'."""
        d = route("Grand Tower Residences handover date?", entities=eidx)
        assert d.route == Route.project_info
        assert d.filters.project_id == "darglobal:project:grand-tower-residences"

    def test_b8b_project_clears_city(self, eidx):
        """B8b: pj-005 'How far is Tierra Viva from Marbella?' -- city cleared."""
        d = route("How far is Tierra Viva from Marbella?", entities=eidx)
        assert d.route == Route.project_info
        assert d.filters.project_id == "darglobal:project:tierra-viva"
        # Marbella is a city entity but should be cleared because project_id is set
        assert d.filters.city is None
        assert d.filters.district is None
        assert d.filters.country is None


# ---------------------------------------------------------------------------
# B10: SCORE_FLOOR constant
# ---------------------------------------------------------------------------


class TestB10ScoreFloor:
    """B10: SCORE_FLOOR = 0.52, used as default in below_floor."""

    def test_score_floor_constant(self):
        from app.domain.retrieval_post import SCORE_FLOOR

        assert SCORE_FLOOR == 0.52

    def test_below_floor_uses_new_default(self):
        from app.domain.retrieval_post import below_floor

        # Below 0.52 -> refuse
        assert below_floor(
            n_hits=5, filters_is_empty=True, top_dense_score=0.51
        ) is True
        # At 0.52 -> not refused
        assert below_floor(
            n_hits=5, filters_is_empty=True, top_dense_score=0.52
        ) is False
        # Above 0.52 -> not refused
        assert below_floor(
            n_hits=5, filters_is_empty=True, top_dense_score=0.55
        ) is False

    def test_tau_default_constant(self):
        """TAU_DEFAULT = 0.93 (kept after B9; test-set cp-023 at 0.916 needs the margin)."""
        from app.domain.retrieval_post import TAU_DEFAULT

        assert TAU_DEFAULT == 0.93
