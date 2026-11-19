"""DuckDB StructuredQuery adapter: whitelisted SQL templates over listings.parquet.

Parameterised SQL only (never string-format a value). Price ops use annual_rent_sar
when purpose==rent, else price_sar. Each call runs asyncio.to_thread with a per-call cursor.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import duckdb

from app.domain.ports import (
    AggregateOp,
    AggregateQuery,
    SqlResult,
    StructuredQueryError,
)
from common.schemas import Listing, Purpose, SortKey

logger = logging.getLogger(__name__)


class DuckDBStructuredQuery:
    """StructuredQuery port implementation using DuckDB on listings.parquet."""

    def __init__(self, listings_path: Path) -> None:
        self._con = duckdb.connect(database=":memory:", read_only=False)
        self._con.execute("SET threads = 1")
        # Materialise the parquet into an in-memory table once at startup so
        # queries do not re-read the file on every call.
        self._con.execute(
            f"CREATE TABLE listings AS SELECT * FROM read_parquet('{listings_path.as_posix()}')"
        )

    async def aggregate(self, query: AggregateQuery) -> SqlResult:
        try:
            return await asyncio.to_thread(self._aggregate_sync, query)
        except StructuredQueryError:
            raise
        except Exception as exc:
            raise StructuredQueryError(str(exc)) from exc

    def _aggregate_sync(self, query: AggregateQuery) -> SqlResult:
        op = query.op
        filters = query.filters

        # Determine price column based on purpose
        price_col = (
            "annual_rent_sar"
            if filters.purpose == Purpose.rent
            else "price_sar"
        )

        # Build WHERE clause with positional parameters ($1, $2, ...)
        conditions: list[str] = []
        params: list[object] = []
        param_idx = 0

        def _add_param(value: object) -> str:
            nonlocal param_idx
            param_idx += 1
            params.append(value)
            return f"${param_idx}"

        if filters.purpose is not None:
            p = _add_param(str(filters.purpose))
            conditions.append(f"purpose = {p}")
        if filters.property_type is not None:
            p = _add_param(str(filters.property_type))
            conditions.append(f"property_type = {p}")
        if filters.city is not None:
            p = _add_param(str(filters.city))
            conditions.append(f"city = {p}")
        if filters.district is not None:
            p = _add_param(str(filters.district))
            conditions.append(f"district = {p}")
        if filters.country is not None:
            # listings don't have a country column, so this always matches zero
            conditions.append("1 = 0")
        if filters.bedrooms_min is not None:
            p = _add_param(filters.bedrooms_min)
            conditions.append(f"bedrooms >= {p}")
        if filters.bedrooms_max is not None:
            p = _add_param(filters.bedrooms_max)
            conditions.append(f"bedrooms <= {p}")
        if filters.price_min_sar is not None:
            p = _add_param(filters.price_min_sar)
            conditions.append(f"{price_col} >= {p}")
        if filters.price_max_sar is not None:
            p = _add_param(filters.price_max_sar)
            conditions.append(f"{price_col} <= {p}")
        if filters.area_min_sqm is not None:
            p = _add_param(filters.area_min_sqm)
            conditions.append(f"area_sqm >= {p}")
        if filters.area_max_sqm is not None:
            p = _add_param(filters.area_max_sqm)
            conditions.append(f"area_sqm <= {p}")

        where = " AND ".join(conditions) if conditions else "1 = 1"

        # Execute the appropriate template
        cursor = self._con.cursor()
        try:
            return self._execute_template(
                cursor, op, price_col, where, params, filters, query.example_limit
            )
        finally:
            cursor.close()

    def _execute_template(
        self,
        cursor: duckdb.DuckDBPyConnection,
        op: AggregateOp,
        price_col: str,
        where: str,
        params: list[object],
        filters,
        example_limit: int,
    ) -> SqlResult:
        # Count of matching rows
        count_sql = f"SELECT COUNT(*) FROM listings WHERE {where}"
        cursor.execute(count_sql, params)
        n_matched = cursor.fetchone()[0]

        if op == AggregateOp.count:
            examples = self._get_examples(
                cursor, where, params, example_limit, filters.sort, price_col
            )
            return SqlResult(
                op=op, value=n_matched, unit="count",
                n_matched=n_matched, examples=examples,
            )

        # For price/area ops, we need the aggregate and to exclude nulls in the target
        if op in (
            AggregateOp.avg_price, AggregateOp.median_price,
            AggregateOp.min_price, AggregateOp.max_price,
        ):
            target_col = price_col
            unit = "SAR"
        elif op == AggregateOp.avg_area:
            target_col = "area_sqm"
            unit = "sqm"
        else:
            raise StructuredQueryError(f"Unknown op: {op}")

        not_null_where = f"{where} AND {target_col} IS NOT NULL"

        if op == AggregateOp.avg_price or op == AggregateOp.avg_area:
            agg_sql = f"SELECT AVG({target_col}) FROM listings WHERE {not_null_where}"
            cursor.execute(agg_sql, params)
            raw = cursor.fetchone()[0]
            value = round(raw) if raw is not None else None
            if op == AggregateOp.avg_area and raw is not None:
                value = round(raw, 1)

        elif op == AggregateOp.median_price:
            agg_sql = f"SELECT MEDIAN({target_col}) FROM listings WHERE {not_null_where}"
            cursor.execute(agg_sql, params)
            raw = cursor.fetchone()[0]
            value = round(raw) if raw is not None else None

        elif op == AggregateOp.min_price:
            agg_sql = f"SELECT MIN({target_col}) FROM listings WHERE {not_null_where}"
            cursor.execute(agg_sql, params)
            raw = cursor.fetchone()[0]
            value = int(raw) if raw is not None else None

        elif op == AggregateOp.max_price:
            agg_sql = f"SELECT MAX({target_col}) FROM listings WHERE {not_null_where}"
            cursor.execute(agg_sql, params)
            raw = cursor.fetchone()[0]
            value = int(raw) if raw is not None else None
        else:
            value = None

        # Examples: for min_price the cheapest, for max_price the most expensive
        if op == AggregateOp.min_price:
            example_sort = f"{target_col} ASC"
        elif op == AggregateOp.max_price:
            example_sort = f"{target_col} DESC"
        else:
            example_sort = self._sort_clause(filters.sort, price_col)

        examples = self._get_examples(
            cursor, not_null_where, params, example_limit,
            sort_override=example_sort,
        )

        return SqlResult(
            op=op, value=value, unit=unit,
            n_matched=n_matched, examples=examples,
        )

    def _sort_clause(self, sort: SortKey, price_col: str) -> str:
        if sort == SortKey.price_asc:
            return f"{price_col} ASC NULLS LAST"
        elif sort == SortKey.price_desc:
            return f"{price_col} DESC NULLS LAST"
        elif sort == SortKey.newest:
            return "scraped_at DESC NULLS LAST"
        else:
            return "id ASC"  # stable default

    def _get_examples(
        self,
        cursor: duckdb.DuckDBPyConnection,
        where: str,
        params: list[object],
        limit: int,
        sort: SortKey | None = None,
        price_col: str = "price_sar",
        sort_override: str | None = None,
    ) -> tuple[Listing, ...]:
        if limit <= 0:
            return ()

        order = sort_override or self._sort_clause(sort or SortKey.relevance, price_col)

        sql = f"""
            SELECT id, source, source_id, url, title, description, purpose,
                   property_type, property_type_raw, city, district,
                   price_sar, annual_rent_sar, bedrooms, bathrooms,
                   area_sqm, lat, lon, amenities, rent_period,
                   published_at, scraped_at
            FROM listings
            WHERE {where}
            ORDER BY {order}
            LIMIT {int(limit)}
        """
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        columns = [
            "id", "source", "source_id", "url", "title", "description", "purpose",
            "property_type", "property_type_raw", "city", "district",
            "price_sar", "annual_rent_sar", "bedrooms", "bathrooms",
            "area_sqm", "lat", "lon", "amenities", "rent_period",
            "published_at", "scraped_at",
        ]

        listings: list[Listing] = []
        for row in rows:
            data = dict(zip(columns, row))
            listing = Listing.model_validate(data)
            listings.append(listing)

        return tuple(listings)
