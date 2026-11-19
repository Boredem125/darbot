"""Tests for grounding checks (PLAN section 6, section 12)."""

import pytest

from app.domain.grounding import check, normalise_citations
from app.domain.messages import REFUSAL_AR, REFUSAL_EN
from app.domain.ports import AggregateOp, SqlResult
from app.domain.router import EntityIndex
from common.schemas import Entities, Entity, EntityKind

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_entities() -> EntityIndex:
    return EntityIndex(Entities(entities=[
        Entity(kind=EntityKind.city, canonical="Riyadh", aliases=["الرياض"]),
        Entity(kind=EntityKind.city, canonical="Jeddah", aliases=["جدة"]),
        Entity(kind=EntityKind.district, canonical="Al Narjis", parent="Riyadh"),
        Entity(kind=EntityKind.district, canonical="Al Rawdah", parent="Jeddah"),
        Entity(
            kind=EntityKind.project,
            canonical="Trump Tower Jeddah",
            aliases=["Trump Tower"],
            doc_id="darglobal:project:trump-tower-jeddah",
        ),
    ]))


@pytest.fixture
def entities() -> EntityIndex:
    return _make_entities()


# ---------------------------------------------------------------------------
# Number checks
# ---------------------------------------------------------------------------


class TestNumberGrounding:
    def test_1_95m_equals_1_950_000(self, entities):
        """PLAN section 12: 1.95M == 1,950,000."""
        report = check(
            "The villa costs SAR 1,950,000 [1].",
            context_texts=["3-bed villa in Al Narjis, Riyadh. Price SAR 1.95M."],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.numbers_ok is True

    def test_arabic_digits(self, entities):
        """PLAN section 12: Arabic digits grounding."""
        report = check(
            "The price is SAR 1,950,000 [1].",
            context_texts=["Price: ١٬٩٥٠٬٠٠٠ SAR"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.numbers_ok is True

    def test_sql_average_within_tolerance(self, entities):
        """PLAN section 12: +/- 0.5% average."""
        sql = SqlResult(
            op=AggregateOp.avg_price,
            value=1_000_000,
            unit="SAR",
            n_matched=10,
        )
        # 1_004_000 is within 0.5% of 1_000_000
        report = check(
            "The average price is SAR 1,004,000 [1].",
            context_texts=["various listings"],
            sql=sql,
            query="average price of villas",
            entities=entities,
            k=5,
        )
        assert report.numbers_ok is True

    def test_hallucinated_price_fails(self, entities):
        """PLAN section 12: a hallucinated price fails."""
        report = check(
            "This villa costs SAR 3,000,000 [1].",
            context_texts=["3-bed villa in Al Narjis, Riyadh. Price SAR 1,950,000."],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.numbers_ok is False
        assert any("3000000" in f for f in report.failures)

    def test_citation_markers_excluded(self, entities):
        """Citation markers [n] are not counted as answer numbers."""
        report = check(
            "See source [1] for details.",
            context_texts=["Some context."],
            sql=None,
            query="show me listings",
            entities=entities,
            k=5,
        )
        assert report.numbers_ok is True

    def test_query_number_accepted(self, entities):
        """Numbers from the query are accepted."""
        report = check(
            "Under 2,000,000 there are 3 options [1].",
            context_texts=["Villa in Riyadh for SAR 1,500,000."],
            sql=None,
            query="villas under 2000000",
            entities=entities,
            k=5,
        )
        assert report.numbers_ok is True

    def test_sql_n_matched_accepted(self, entities):
        """ar-004 fix: n_matched from SqlResult is accepted as grounded."""
        sql = SqlResult(
            op=AggregateOp.count,
            value=42,
            unit="count",
            n_matched=351,
        )
        report = check(
            "There are 351 listings matching your criteria, with an average count of 42 [1].",
            context_texts=["various listings"],
            sql=sql,
            query="how many villas",
            entities=entities,
            k=5,
            sql_n_matched=351,
        )
        assert report.numbers_ok is True

    def test_sql_n_matched_not_accepted_without_param(self, entities):
        """Without sql_n_matched, the n_matched value is not grounded."""
        sql = SqlResult(
            op=AggregateOp.count,
            value=42,
            unit="count",
            n_matched=351,
        )
        report = check(
            "There are 351 listings [1].",
            context_texts=["various listings"],
            sql=sql,
            query="how many villas",
            entities=entities,
            k=5,
            # sql_n_matched not passed -> 351 not accepted
        )
        assert report.numbers_ok is False


# ---------------------------------------------------------------------------
# Citation checks
# ---------------------------------------------------------------------------


class TestCitationGrounding:
    def test_bad_citation_7_with_k5(self, entities):
        """PLAN section 12: a bad citation [7] with k=5 fails."""
        report = check(
            "The villa costs SAR 1,950,000 [7].",
            context_texts=["3-bed villa, SAR 1,950,000"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.citations_ok is False
        assert any("[7]" in f for f in report.failures)

    def test_valid_citation(self, entities):
        report = check(
            "The villa costs SAR 1,950,000 [1].",
            context_texts=["3-bed villa, SAR 1,950,000"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.citations_ok is True

    def test_number_without_citation_fails(self, entities):
        """An answer with numbers but no citation fails."""
        report = check(
            "The villa costs SAR 1,950,000.",
            context_texts=["3-bed villa, SAR 1,950,000"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.citations_ok is False

    def test_sql_result_citation_k_plus_1_valid(self, entities):
        """Addendum: [k+1] is valid when sql_result is present."""
        sql = SqlResult(
            op=AggregateOp.count,
            value=42,
            unit="count",
            n_matched=42,
        )
        report = check(
            "There are 42 listings [3].",
            context_texts=["card 1 text", "card 2 text"],
            sql=sql,
            query="how many villas",
            entities=entities,
            k=2,  # 2 cards, so [3] = sql_result citation
        )
        assert report.citations_ok is True

    def test_sql_result_citation_without_cards(self, entities):
        """Addendum: [1] valid for sql_result when k=0."""
        sql = SqlResult(
            op=AggregateOp.avg_price,
            value=500000,
            unit="SAR",
            n_matched=10,
        )
        report = check(
            "The average is SAR 500,000 [1].",
            context_texts=[],
            sql=sql,
            query="average price",
            entities=entities,
            k=0,
        )
        assert report.citations_ok is True

    def test_citation_still_invalid_beyond_k_plus_1_with_sql(self, entities):
        """[k+2] is still invalid even with sql_result."""
        sql = SqlResult(
            op=AggregateOp.count,
            value=5,
            unit="count",
            n_matched=5,
        )
        report = check(
            "There are 5 listings [4].",
            context_texts=["card 1"],
            sql=sql,
            query="count",
            entities=entities,
            k=2,  # valid range 1..3, [4] is invalid
        )
        assert report.citations_ok is False


# ---------------------------------------------------------------------------
# Entity checks
# ---------------------------------------------------------------------------


class TestEntityGrounding:
    def test_hallucinated_district_fails(self, entities):
        """PLAN section 12: a hallucinated district fails."""
        report = check(
            "There are villas in Al Rawdah [1].",
            context_texts=["3-bed villa in Al Narjis, Riyadh. SAR 1,950,000"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.entities_ok is False
        assert any("Al Rawdah" in f for f in report.failures)

    def test_entity_in_context_passes(self, entities):
        report = check(
            "There are villas in Riyadh, Al Narjis [1].",
            context_texts=["3-bed villa in Al Narjis, Riyadh. SAR 1,950,000"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.entities_ok is True


# ---------------------------------------------------------------------------
# Refusal line
# ---------------------------------------------------------------------------


class TestRefusalLine:
    def test_refusal_en_all_null(self, entities):
        """PLAN section 12: the refusal line gives all null."""
        report = check(
            REFUSAL_EN,
            context_texts=[],
            sql=None,
            query="weather today",
            entities=entities,
            k=0,
        )
        assert report.numbers_ok is None
        assert report.entities_ok is None
        assert report.citations_ok is None
        assert report.grounded is True

    def test_refusal_ar_all_null(self, entities):
        report = check(
            REFUSAL_AR,
            context_texts=[],
            sql=None,
            query="ما الطقس",
            entities=entities,
            k=0,
        )
        assert report.numbers_ok is None
        assert report.entities_ok is None
        assert report.citations_ok is None


# ---------------------------------------------------------------------------
# Grounded property
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# normalise_citations
# ---------------------------------------------------------------------------


class TestNormaliseCitations:
    def test_fullwidth_lenticular(self):
        """Groq gpt-oss-120b writes 【n】."""
        assert normalise_citations("price is 1M 【1】") == "price is 1M [1]"

    def test_fullwidth_lenticular_multiple(self):
        assert normalise_citations("【1】 and 【2】") == "[1] and [2]"

    def test_tortoise_shell(self):
        """〔n〕 -> [n]."""
        assert normalise_citations("see 〔3〕") == "see [3]"

    def test_fullwidth_square(self):
        """［n］ -> [n]."""
        assert normalise_citations("ref ［5］") == "ref [5]"

    def test_mixed_brackets(self):
        """Mix of fullwidth and ASCII."""
        text = "text [1] more 【2】 end ［3］"
        assert normalise_citations(text) == "text [1] more [2] end [3]"

    def test_no_change_for_ascii(self):
        """ASCII [n] is left unchanged."""
        text = "some text [1] and [2]"
        assert normalise_citations(text) == text

    def test_empty(self):
        assert normalise_citations("") == ""

    def test_no_brackets(self):
        assert normalise_citations("no citations here") == "no citations here"


class TestFullwidthCitationGrounding:
    """Grounding check accepts fullwidth citations after normalisation."""

    def test_fullwidth_citation_valid(self, entities):
        """Fullwidth 【n】 should be treated as [n] by the grounding check."""
        report = check(
            "The villa costs SAR 1,950,000 【1】.",
            context_texts=["3-bed villa, SAR 1,950,000"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.citations_ok is True

    def test_fullwidth_citation_out_of_range(self, entities):
        """Fullwidth 【7】 out of range should still fail."""
        report = check(
            "The villa costs SAR 1,950,000 【7】.",
            context_texts=["3-bed villa, SAR 1,950,000"],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.citations_ok is False


class TestGroundedProperty:
    def test_all_ok_is_grounded(self, entities):
        report = check(
            "There is a villa [1].",
            context_texts=["Villa in Riyadh."],
            sql=None,
            query="villas in Riyadh",
            entities=entities,
            k=5,
        )
        assert report.grounded is True

    def test_number_fail_is_not_grounded(self, entities):
        report = check(
            "The price is SAR 9,999,999 [1].",
            context_texts=["Villa in Riyadh for SAR 1,000,000"],
            sql=None,
            query="villas",
            entities=entities,
            k=5,
        )
        assert report.grounded is False
