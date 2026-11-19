"""Tests for DuckDB StructuredQuery adapter: every AggregateOp, rent vs sale, known answers."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.ports import AggregateOp, AggregateQuery
from common.schemas import SearchFilters


@pytest.fixture()
def sql_adapter(synthetic_index: Path):
    from app.adapters.sql_duckdb import DuckDBStructuredQuery

    return DuckDBStructuredQuery(synthetic_index / "listings.parquet")


class TestAggregateOps:
    """Every AggregateOp against the 10-row fixture with known answers."""

    @pytest.mark.asyncio
    async def test_count_all(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(op=AggregateOp.count, filters=SearchFilters())
        )
        assert result.op == AggregateOp.count
        assert result.unit == "count"
        assert result.value == 10
        assert result.n_matched == 10

    @pytest.mark.asyncio
    async def test_count_with_city_filter(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(op=AggregateOp.count, filters=SearchFilters(city="Riyadh"))
        )
        # Riyadh: indices 0, 2, 4, 5, 6, 8, 9 = 7
        assert result.value == 7
        assert result.n_matched == 7

    @pytest.mark.asyncio
    async def test_avg_price_sale(self, sql_adapter) -> None:
        """avg_price with purpose=sale uses price_sar."""
        result = await sql_adapter.aggregate(
            AggregateQuery(op=AggregateOp.avg_price, filters=SearchFilters(purpose="sale"))
        )
        assert result.op == AggregateOp.avg_price
        assert result.unit == "SAR"
        # Sale prices: 1000000, 500000, 300000, 400000, 600000, 2000000, 800000
        expected_avg = round((1000000 + 500000 + 300000 + 400000 + 600000 + 2000000 + 800000) / 7)
        assert result.value == expected_avg

    @pytest.mark.asyncio
    async def test_avg_price_rent(self, sql_adapter) -> None:
        """avg_price with purpose=rent uses annual_rent_sar."""
        result = await sql_adapter.aggregate(
            AggregateQuery(op=AggregateOp.avg_price, filters=SearchFilters(purpose="rent"))
        )
        assert result.op == AggregateOp.avg_price
        assert result.unit == "SAR"
        # Rent annual: 5000, 24000, 3000
        expected_avg = round((5000 + 24000 + 3000) / 3)
        assert result.value == expected_avg

    @pytest.mark.asyncio
    async def test_median_price(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.median_price,
                filters=SearchFilters(purpose="sale"),
            )
        )
        assert result.op == AggregateOp.median_price
        assert result.unit == "SAR"
        # Sale prices sorted: 300000, 400000, 500000, 600000, 800000, 1000000, 2000000
        # Median (index 3 of 7) = 600000
        assert result.value == 600000

    @pytest.mark.asyncio
    async def test_min_price(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.min_price,
                filters=SearchFilters(purpose="sale"),
            )
        )
        assert result.op == AggregateOp.min_price
        assert result.unit == "SAR"
        assert result.value == 300000

    @pytest.mark.asyncio
    async def test_max_price(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.max_price,
                filters=SearchFilters(purpose="sale"),
            )
        )
        assert result.op == AggregateOp.max_price
        assert result.unit == "SAR"
        assert result.value == 2000000

    @pytest.mark.asyncio
    async def test_avg_area(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(op=AggregateOp.avg_area, filters=SearchFilters())
        )
        assert result.op == AggregateOp.avg_area
        assert result.unit == "sqm"
        # All areas: 200, 100, 150, 500, 80, 180, 90, 300, 120, 250
        expected = round(sum([200, 100, 150, 500, 80, 180, 90, 300, 120, 250]) / 10, 1)
        assert abs(result.value - expected) < 0.5

    @pytest.mark.asyncio
    async def test_examples_returned(self, sql_adapter) -> None:
        """min_price should return example listings (the cheapest)."""
        result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.min_price,
                filters=SearchFilters(purpose="sale"),
                example_limit=3,
            )
        )
        assert len(result.examples) <= 3
        # Examples should be ordered by price ascending for min_price
        if len(result.examples) >= 2:
            assert result.examples[0].price_sar <= result.examples[1].price_sar

    @pytest.mark.asyncio
    async def test_no_match_returns_none(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.count,
                filters=SearchFilters(city="Nonexistent"),
            )
        )
        assert result.value == 0
        assert result.n_matched == 0

    @pytest.mark.asyncio
    async def test_rent_vs_sale_column(self, sql_adapter) -> None:
        """Rent queries must use annual_rent_sar, sale must use price_sar."""
        rent_result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.min_price,
                filters=SearchFilters(purpose="rent"),
            )
        )
        sale_result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.min_price,
                filters=SearchFilters(purpose="sale"),
            )
        )
        # Min rent annual is 3000, min sale is 300000
        assert rent_result.value == 3000
        assert sale_result.value == 300000

    @pytest.mark.asyncio
    async def test_bedrooms_filter(self, sql_adapter) -> None:
        result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.count,
                filters=SearchFilters(bedrooms_min=3, bedrooms_max=5),
            )
        )
        # bedrooms 3,4,5,3,3 = 5 rows with bedrooms in [3,5]
        assert result.value == 5

    @pytest.mark.asyncio
    async def test_max_price_examples_desc(self, sql_adapter) -> None:
        """max_price examples should be the most expensive, sorted desc."""
        result = await sql_adapter.aggregate(
            AggregateQuery(
                op=AggregateOp.max_price,
                filters=SearchFilters(purpose="sale"),
                example_limit=3,
            )
        )
        if len(result.examples) >= 2:
            assert result.examples[0].price_sar >= result.examples[1].price_sar
