#!/usr/bin/env python3
"""Build the L0 warm cache (``data/index/warm_cache.jsonl``).

Runs queries from ``eval/warm_queries.jsonl`` through the domain pipeline
(router, embedder, retriever, SQL, LLM) and keeps only answers that pass
the caching.md section 5 put policy:

  1. Standalone turn (no carried slots).
  2. Route in {lookup, aggregate, project_info}.
  3. answer_mode == llm.
  4. finish_reason == stop.
  5. grounding.grounded is true.

Writes ``WarmCacheRecord`` per line to a temp file first, then swaps it
into ``data/index/warm_cache.jsonl`` and updates ``manifest.json``
``prompt_version``.  ``index_version`` must not change.

Usage:
    python scripts/warm_cache.py --provider-chain or-gemma --budget 25 --wp WP-4-qa
    python scripts/warm_cache.py --dry-run --provider-chain fake --budget 25 --wp WP-4-qa
    python scripts/warm_cache.py --plan --wp WP-4-qa
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_queries(path: Path) -> list[dict]:
    """Load warm queries from a JSONL file."""
    if not path.exists():
        return []
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            queries.append(json.loads(line))
    return queries


def _get_prompt_version(prompt_stem: str) -> str:
    """Compute prompt_version from the prompt file."""
    from common.schemas import compute_prompt_version

    prompt_path = REPO_ROOT / "app" / "prompts" / f"{prompt_stem}.txt"
    prompt_bytes = prompt_path.read_bytes()
    return compute_prompt_version(prompt_stem, prompt_bytes)


def _get_index_version(index_dir: Path) -> str:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["index_version"]


async def _run_warm(args: argparse.Namespace) -> int:
    """Main warm cache loop."""
    from app.domain import grounding as grounding_mod
    from app.domain import prompting
    from app.domain.cache_keys import build_key, parts_from
    from app.domain.ports import LLMDelta, LLMDone, LLMFailover, LLMUnavailableError
    from app.domain.retrieval_post import below_floor, collapse_to_docs, to_cards
    from app.domain.router import EntityIndex, route
    from common.schemas import (
        CachedAnswer,
        Entities,
        Provider,
        Route,
        WarmCacheRecord,
    )
    from scripts.evallib.llm_step import QuotaLogger

    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = REPO_ROOT / index_dir

    index_version = _get_index_version(index_dir)
    prompt_version = _get_prompt_version(args.prompt)

    # Load queries
    queries_path = REPO_ROOT / "eval" / "warm_queries.jsonl"
    queries = _load_queries(queries_path)
    if not queries:
        print("ERROR: no warm queries found", file=sys.stderr)
        return 1

    print(f"Loaded {len(queries)} warm queries")
    print(f"Index version: {index_version}, Prompt version: {prompt_version}")

    # Plan mode
    if args.plan:
        print(f"Plan: {len(queries)} queries, budget {args.budget}")
        print(f"Worst-case attempts: {len(queries)}")
        return 0

    # Determine output directory
    if args.dry_run:
        output_dir = REPO_ROOT / ".orchestration" / "tmp" / "warm_dry"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = index_dir

    # Build adapters
    from app.adapters.factory import build_read_side

    rs = build_read_side(
        index_dir,
        prompt_version=prompt_version,
        warm_cache_present=False,
    )
    embedder = rs.embedder
    retriever = rs.retriever
    sql = rs.sql

    # Load entities
    entities_data = json.loads((index_dir / "entities.json").read_text(encoding="utf-8"))
    entities = EntityIndex(Entities.model_validate(entities_data))

    # Load system prompt
    prompt_path = REPO_ROOT / "app" / "prompts" / f"{args.prompt}.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    # Build LLM
    if args.fakes:
        from tests.fakes.fake_adapters import FakeLLM
        llm = FakeLLM()
    else:
        from app.adapters.factory import build_llm
        chain_ids = [c.strip() for c in args.provider_chain.split(",") if c.strip()] if args.provider_chain else None
        llm = build_llm(chain_ids=chain_ids)

    # Quota logger
    quota_csv = REPO_ROOT / "eval" / "quota_log.csv"
    quota_logger = QuotaLogger(quota_csv) if not args.dry_run or args.log_quota else None

    # Run queries
    records: list[WarmCacheRecord] = []
    attempts = 0
    skipped = 0

    for i, q in enumerate(queries):
        query_text = q["query"]
        lang_hint = q.get("lang")

        if attempts >= args.budget:
            print(f"  Budget exhausted at query {i}")
            break

        # Route
        decision = route(query_text, session=None, entities=entities, lang_hint=lang_hint)

        # Only standalone routes
        if decision.route == Route.out_of_scope:
            skipped += 1
            continue

        # Build cache key
        key_parts = parts_from(decision, query_text, index_version, prompt_version)
        cache_key = build_key(key_parts)

        # Retrieve
        language = decision.language
        cards = []
        sql_result = None

        if decision.route in (Route.lookup, Route.project_info):
            embedding = await embedder.embed_query(query_text)
            retrieval = await retriever.search(embedding, decision.filters, limit=20)

            if below_floor(
                n_hits=len(retrieval.hits),
                filters_is_empty=decision.filters.is_empty(),
                top_dense_score=retrieval.top_dense_score,
                floor=0.52,
            ):
                skipped += 1
                continue

            hits_for_collapse = [(hit.payload, hit.score) for hit in retrieval.hits]
            project_id = decision.filters.project_id if decision.route == Route.project_info else None
            doc_hits = collapse_to_docs(hits_for_collapse, top_k=5, project_id=project_id)
            cards = to_cards(doc_hits)

        elif decision.route == Route.aggregate:
            from app.domain.ports import AggregateQuery

            agg_query = AggregateQuery(
                op=decision.op,
                filters=decision.filters,
                example_limit=3,
            )
            sql_result = await sql.aggregate(agg_query)

            if sql_result.examples:
                from app.domain.retrieval_post import DocHit
                from common.schemas import ChunkPayload, DocKind, Source

                doc_hits_list = []
                for ex in sql_result.examples:
                    payload = ChunkPayload(
                        doc_id=ex.id, kind=DocKind.listing, source=Source.wasalt,
                        title=ex.title, url=ex.url, scraped_at=ex.scraped_at,
                        purpose=ex.purpose, property_type=ex.property_type,
                        city=ex.city, district=ex.district, price_sar=ex.price_sar,
                        rent_period=ex.rent_period, annual_rent_sar=ex.annual_rent_sar,
                        bedrooms=ex.bedrooms, bathrooms=ex.bathrooms,
                        area_sqm=ex.area_sqm, chunk_index=0, text=ex.title,
                    )
                    doc_hits_list.append(DocHit(ex.id, 1.0, payload))
                cards = to_cards(doc_hits_list)

        # Build prompt
        from common.cards import render_card

        cards_context = [
            render_card(c, mode="context", n=c.n, snippet=c.snippet)
            for c in cards
        ]

        max_tokens = prompting.max_tokens_for(language)
        plan = prompting.build_messages(
            system_prompt=system_prompt,
            cards_context=cards_context,
            sql_block=sql_result,
            question=query_text,
        )

        # Stream LLM
        attempts += 1
        answer_parts: list[str] = []
        done = None
        try:
            import contextlib
            async with contextlib.aclosing(
                llm.stream(plan.messages, max_tokens=max_tokens, temperature=0.0)
            ) as stream:
                async for item in stream:
                    if isinstance(item, LLMDelta):
                        answer_parts.append(item.text)
                    elif isinstance(item, LLMDone):
                        done = item
                    elif isinstance(item, LLMFailover):
                        # Log the failed attempt
                        code_val = item.code.value if hasattr(item.code, "value") else str(item.code)
                        if quota_logger and code_val != "provider_skipped":
                            prov = item.provider.value if hasattr(item.provider, "value") else str(item.provider)
                            quota_logger.log(
                                provider=prov, model=item.model,
                                purpose="warm_cache", wp=args.wp,
                                ok=False, prompt_tokens=0, completion_tokens=0,
                            )
                        # Check for 429 and stop if requested
                        if "rate_limited" in code_val and not args.continue_on_429:
                            print(f"  429 at query {i}, stopping (use --continue-on-429 to override)")
                            attempts = args.budget  # stop loop
                            break
        except LLMUnavailableError as exc:
            # Log each failed attempt to quota
            is_429 = False
            for fa in exc.attempts:
                code_val = fa.code.value if hasattr(fa.code, "value") else str(fa.code)
                if "rate_limited" in code_val:
                    is_429 = True
                if quota_logger and code_val != "provider_skipped":
                    prov = fa.provider.value if hasattr(fa.provider, "value") else str(fa.provider)
                    quota_logger.log(
                        provider=prov,
                        model=fa.model,
                        purpose="warm_cache",
                        wp=args.wp,
                        ok=False,
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
            if is_429 and not args.continue_on_429:
                print(f"  429 at query {i}, stopping (use --continue-on-429 to override)")
                break
            print(f"  All providers unavailable at query {i}: {[a.code.value for a in exc.attempts]}")
            continue

        answer = "".join(answer_parts)
        if not answer or done is None:
            continue

        # Log to quota
        if quota_logger:
            provider_str = done.provider.value if hasattr(done.provider, "value") else str(done.provider)
            quota_logger.log(
                provider=provider_str,
                model=done.model,
                purpose="warm_cache",
                wp=args.wp,
                ok=True,
                prompt_tokens=done.usage.prompt_tokens,
                completion_tokens=done.usage.completion_tokens,
            )

        # Put policy checks
        finish_str = done.finish_reason.value if hasattr(done.finish_reason, "value") else str(done.finish_reason)
        if finish_str != "stop":
            continue

        # Normalise fullwidth citations (Groq writes 【n】) before grounding
        answer = grounding_mod.normalise_citations(answer)

        # Grounding check
        grounding_report = grounding_mod.check(
            answer,
            context_texts=cards_context[:plan.k],
            sql=sql_result,
            query=query_text,
            entities=entities,
            k=plan.k,
        )
        if not grounding_report.grounded:
            print(f"  Query {i} not grounded: {grounding_report.failures[:3]}")
            continue

        # Build the record
        provider_enum = done.provider if isinstance(done.provider, Provider) else Provider(done.provider)
        source_cards_data = [c.model_dump(mode="json") for c in cards]
        from common.schemas import Card as CardSchema
        source_cards = [CardSchema.model_validate(d) for d in source_cards_data]

        cached = CachedAnswer(
            answer=answer,
            route=decision.route,
            language=language,
            filters=decision.filters,
            sources=source_cards,
            provider=provider_enum,
            model=done.model,
            grounding=grounding_report,
            prompt_version=prompt_version,
            index_version=index_version,
            created_at=datetime.now(UTC),
        )

        record = WarmCacheRecord(key=cache_key, query=query_text, entry=cached)
        records.append(record)
        display = query_text[:50].encode("ascii", errors="replace").decode("ascii")
        print(f"  [{i+1}/{len(queries)}] cached: {display}...")

    print(f"\nResults: {len(records)} cached, {skipped} skipped, {attempts} attempts")

    if not records:
        print("No records to write.")
        return 0

    # Write to temp file then swap
    warm_path = output_dir / "warm_cache.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", dir=str(output_dir),
        delete=False, encoding="utf-8",
    ) as tmp:
        for rec in records:
            tmp.write(rec.model_dump_json() + "\n")
        tmp_path = Path(tmp.name)

    # Atomic swap (best effort on Windows)
    if warm_path.exists():
        warm_path.unlink()
    tmp_path.rename(warm_path)
    print(f"Wrote {len(records)} records to {warm_path}")

    # Update manifest prompt_version (only on real run)
    if not args.dry_run:
        manifest_path = index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_iv = manifest["index_version"]
        manifest["prompt_version"] = prompt_version
        # index_version must not change
        assert manifest["index_version"] == old_iv, "index_version must not change"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Updated manifest.json prompt_version to {prompt_version}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build L0 warm cache")
    parser.add_argument("--index-dir", default="data/index", help="Path to index directory")
    parser.add_argument("--provider-chain", default="or-gemma", help="Comma-separated provider ids")
    parser.add_argument("--budget", type=int, default=25, help="Max LLM calls")
    parser.add_argument("--wp", default="", help="WP id for quota log")
    parser.add_argument("--prompt", default="system_v1", help="Prompt file stem")
    parser.add_argument("--plan", action="store_true", help="Print plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="Write to .orchestration/tmp/warm_dry/ only")
    parser.add_argument("--fakes", action="store_true", help="Use fake adapters")
    parser.add_argument("--continue-on-429", action="store_true", help="Continue on upstream 429")
    parser.add_argument("--log-quota", action="store_true", help="Log quota even on dry run")

    args = parser.parse_args()

    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(REPO_ROOT / ".orchestration" / "tmp" / "ml" / "fe_cache"))
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".orchestration" / "tmp" / "ml" / "fe_cache"))

    return asyncio.run(_run_warm(args))


if __name__ == "__main__":
    sys.exit(main())
