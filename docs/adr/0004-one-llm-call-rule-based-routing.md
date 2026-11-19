# ADR-0004: One LLM call per turn, with rule-based routing

- Status: Accepted
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0005, ADR-0006, ADR-0007, `docs/specs/retrieval.md` and `docs/specs/caching.md` (dg-ml-lead)

## Context

The free OpenRouter tier allows **50 requests per day per account, shared across all `:free` models**, and failed calls count. Groq is the fallback, at about 1K requests per day per model but with a low tokens-per-minute limit (about 8K for `gpt-oss-120b`). Reviewers will try the public URL on Friday, and the warm cache and probes also draw on the same 50 (PLAN section 9).

Most questions fall into a few shapes:
- filtered lookups ("3-bed villas in Riyadh under 2M SAR");
- aggregates ("average rent in Jeddah", "cheapest apartment for rent");
- project facts ("Trump Tower Jeddah handover date");
- follow-ups ("cheaper ones?");
- out-of-scope questions.

The filters live in a small, closed vocabulary: cities, districts, projects and brands from `entities.json`, property types, purpose, bedroom counts, and prices in SAR with k/M suffixes, in English and Arabic.

## Decision

1. **Exactly one LLM call per answered turn, and zero for refusals.** The call generates the final answer from the retrieved context, streamed through the `LLM` port. No LLM call is used for routing, query rewriting, filter extraction or reranking.

2. **Routing is deterministic** and lives in `app/domain/router.py`, which is pure Python with no I/O (R4). It decides:
   - **language** (`en` or `ar`), from the script of the message; the `lang` field of `ChatRequest` is only a hint;
   - **route** (`Route`): `lookup`, `aggregate`, `project_info` or `out_of_scope`, from keyword and regex rules;
   - **filters** (`SearchFilters`), from dictionaries built on `entities.json` aliases plus regexes for bedrooms, prices (k, M, thousand, million, commas, Arabic-Indic digits), areas, purpose and property type;
   - **slot carry-over**: the session holds the last two turns and the merged slots, so "cheaper ones?" inherits the previous city, type and purpose and tightens the price bound.

3. **A hybrid-retrieval fallback catches misroutes.** When the router finds no strong signal for `aggregate` or `project_info`, and the query is not clearly out of scope, it routes to `lookup` with whatever filters it found. That route runs dense plus bm25 search, which tolerates odd phrasing. A query is refused only when it matches out-of-scope rules or the best retrieval score falls below the floor.

4. **Refusals use no LLM.** `out_of_scope`, or a top score below the floor, returns the fixed refusal line as tokens with `answer_mode = refusal`, `provider = none` and `finish_reason = no_llm` (SSE contract v1).

5. **Aggregates are computed, not generated.** The `aggregate` route runs a whitelisted DuckDB template (`AggregateOp`) and passes `<sql_result>` plus a few example cards to the LLM. The model only phrases a number it was given, and grounding checks it against the SQL result (ADR-0007).

The exact rules, the score floor and the default for a price bound with no purpose are dg-ml-lead decisions, recorded in `docs/specs/retrieval.md` and `docs/specs/caching.md`.

## Alternatives rejected

- **LLM-based routing or function calling (two calls per turn).** It doubles quota spend, so 50 a day becomes 25 answers. It adds a full TTFT before retrieval can start, and it brings a second failure point per turn.
- **LLM query rewriting for follow-ups.** Same cost problem. Slot carry-over covers the planned follow-up cases ("cheaper ones?", "what about Jeddah?").
- **An agentic loop (plan, tool calls, reflect).** Unbounded calls per turn on a 50-per-day budget, with latency that cannot be held to an SLO.
- **An embedding classifier for intent.** A reasonable later step, but it needs labelled data we do not have. Rules plus the hybrid fallback are auditable and testable today.

## Consequences

- Quota math is simple: one OpenRouter request per uncached, in-scope turn. Cache hits and refusals cost nothing.
- The router is unit-tested on fixtures, including Arabic and multi-turn cases, without keys. Routing adds well under a millisecond.
- Filters are exact, so payload filters and SQL can enforce them. The cache key can include them (ADR-0006), which keeps "3-bed" and "4-bed" apart.
- Unusual phrasings can be misrouted or miss a filter. The eval sets (`handwritten`, `multiturn`, `arabic`, `oos`) measure how often.
- The rules grow with the vocabulary. New cities and projects arrive through `entities.json`, not through code.

## Trade-off accepted

Unusual phrasings get misrouted. The hybrid-retrieval fallback route mitigates this, and the eval gates (false refusals at most 5%, out-of-scope refused n/n) bound it.
