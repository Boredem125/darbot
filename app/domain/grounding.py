"""Grounding check: numbers, entities, citations.

Implements PLAN section 6, ADR-0007. Imports only common.* and app.domain.* (R4).
"""

from __future__ import annotations

import re
from decimal import Decimal

from app.domain.messages import REFUSAL_AR, REFUSAL_EN
from app.domain.ports import SqlResult
from common.schemas import GroundingReport
from common.text import canonical_number, parse_numbers


def normalise_citations(text: str) -> str:
    """Map fullwidth and other bracket citation forms to ASCII ``[n]``.

    Groq gpt-oss-120b writes citations as fullwidth 【n】.  This function
    normalises all three known variants:

    - 【n】 (LEFT/RIGHT BLACK LENTICULAR BRACKET)
    - 〔n〕 (LEFT/RIGHT TORTOISE SHELL BRACKET)
    - ［n］ (FULLWIDTH LEFT/RIGHT SQUARE BRACKET)

    to plain ASCII ``[n]``.

    Signature: ``normalise_citations(text: str) -> str``
    """
    # 【n】 -> [n]
    text = re.sub(r"【(\d+)】", r"[\1]", text)
    # 〔n〕 -> [n]
    text = re.sub(r"〔(\d+)〕", r"[\1]", text)
    # ［n］ -> [n]
    text = re.sub(r"［(\d+)］", r"[\1]", text)
    return text


def check(
    answer: str,
    *,
    context_texts: list[str],
    sql: SqlResult | None,
    query: str,
    entities: object,  # EntityIndex (duck-typed to avoid circular import)
    k: int,
    sql_n_matched: int | None = None,
) -> GroundingReport:
    """Run the three grounding checks on the LLM answer.

    If the answer equals the refusal line, all three checks are null.
    """
    # Refusal: all checks null
    if answer.strip() in (REFUSAL_EN, REFUSAL_AR):
        return GroundingReport()

    failures: list[str] = []

    # Citation range: 1..k normally, 1..k+1 when sql_result is present
    max_cite = k + 1 if sql is not None else k

    # --- Numbers check ---
    numbers_ok = _check_numbers(answer, context_texts=context_texts, sql=sql, query=query, k=k, sql_n_matched=sql_n_matched, failures=failures)

    # --- Entities check ---
    entities_ok = _check_entities(answer, context_texts=context_texts, query=query, entities=entities, failures=failures)

    # --- Citations check ---
    citations_ok = _check_citations(answer, k=max_cite, failures=failures)

    return GroundingReport(
        numbers_ok=numbers_ok,
        entities_ok=entities_ok,
        citations_ok=citations_ok,
        failures=failures[:20],
    )


def _check_numbers(
    answer: str,
    *,
    context_texts: list[str],
    sql: SqlResult | None,
    query: str,
    k: int,
    sql_n_matched: int | None = None,
    failures: list[str],
) -> bool:
    """Every number in the answer must be justified."""
    answer_numbers = parse_numbers(answer)
    if not answer_numbers:
        return True

    # Build the set of accepted canonical numbers
    accepted: set[str] = set()

    # Numbers from the query
    for n in parse_numbers(query):
        accepted.add(canonical_number(n))

    # Numbers from context texts
    for ctx in context_texts:
        for n in parse_numbers(ctx):
            accepted.add(canonical_number(n))

    # SQL values: exact and +/- 0.5%
    sql_values: list[Decimal] = []
    if sql is not None and sql.value is not None:
        sql_values.append(Decimal(str(sql.value)))
        accepted.add(canonical_number(Decimal(str(sql.value))))

    # SQL n_matched: the count of rows matching the filters (ar-004 fix)
    if sql_n_matched is not None:
        accepted.add(canonical_number(Decimal(sql_n_matched)))

    # Integers <= k are accepted (counts like "3 options")
    for i in range(k + 1):
        accepted.add(str(i))

    ok = True
    for num in answer_numbers:
        canon = canonical_number(num)
        if canon in accepted:
            continue

        # Check +/- 0.5% of SQL values (for averages/medians)
        if sql_values and _within_tolerance(num, sql_values):
            continue

        # Fail
        ok = False
        msg = f"number {canon} not in sources"
        if len(msg) > 200:
            msg = msg[:200]
        failures.append(msg)

    return ok


def _within_tolerance(num: Decimal, sql_values: list[Decimal], tol: float = 0.005) -> bool:
    """Check if num is within +/- tol of any SQL value."""
    for sv in sql_values:
        if sv == 0:
            if num == 0:
                return True
            continue
        ratio = abs(float(num - sv) / float(sv))
        if ratio <= tol:
            return True
    return False


def _check_entities(
    answer: str,
    *,
    context_texts: list[str],
    query: str,
    entities: object,  # EntityIndex
    failures: list[str],
) -> bool:
    """Every entity in the answer must appear in context or query."""
    # Find entities in the answer
    answer_entities = entities.find_in_text(answer)  # type: ignore[union-attr]
    if not answer_entities:
        return True

    # Build the set of entity canonicals found in context or query
    context_entity_canonicals: set[str] = set()
    for ctx in context_texts:
        for _alias, ent in entities.find_in_text(ctx):  # type: ignore[union-attr]
            context_entity_canonicals.add(ent.canonical)
    for _alias, ent in entities.find_in_text(query):  # type: ignore[union-attr]
        context_entity_canonicals.add(ent.canonical)

    ok = True
    for alias, entity in answer_entities:
        if entity.canonical in context_entity_canonicals:
            continue
        # Also check if any alias of this entity appears in context/query
        all_aliases = entities.aliases_for(entity.canonical)  # type: ignore[union-attr]
        found = False
        for ctx in context_texts:
            normed_ctx = ctx.lower()
            for a in all_aliases:
                if a in normed_ctx:
                    found = True
                    break
            if found:
                break
        if not found:
            normed_q = query.lower()
            for a in all_aliases:
                if a in normed_q:
                    found = True
                    break
        if not found:
            ok = False
            msg = f"entity '{entity.canonical}' not in sources"
            if len(msg) > 200:
                msg = msg[:200]
            failures.append(msg)

    return ok


def _check_citations(
    answer: str,
    *,
    k: int,
    failures: list[str],
) -> bool:
    """Every [n] must be in 1..k, and an answer with numbers must have >= 1 citation."""
    answer = normalise_citations(answer)
    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations:
        # An answer with numbers must have at least one citation
        answer_numbers = parse_numbers(answer)
        if answer_numbers:
            failures.append("answer contains numbers but no citations")
            return False
        return True

    ok = True
    for cite_str in citations:
        cite_num = int(cite_str)
        if cite_num < 1 or cite_num > k:
            ok = False
            msg = f"citation [{cite_num}] out of range 1..{k}"
            if len(msg) > 200:
                msg = msg[:200]
            failures.append(msg)

    return ok
