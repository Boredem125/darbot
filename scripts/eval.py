#!/usr/bin/env python3
"""Eval harness CLI.

Usage examples:
  python scripts/eval.py --split dev --no-llm
  python scripts/eval.py --split dev --no-llm --score-floor 0.25
  python scripts/eval.py --split dev --no-llm --rerank
  python scripts/eval.py --split dev --no-llm --rrf-k 30 --list-size 40
  python scripts/eval.py --split dev --no-llm --dense-weight 0.8 --sparse-weight 1.2
  python scripts/eval.py --split dev --no-llm --write-baseline
  python scripts/eval.py --split test --no-llm --compare eval/baseline_report.json
  python scripts/eval.py --split dev --llm --provider-chain openrouter --budget 5
  python scripts/eval.py --split dev --cache-pairs --tau 0.90,0.92,0.94
  python scripts/eval.py --split dev --no-llm --fakes
  python scripts/eval.py --split dev --no-llm --index-dir data/index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.evallib.loader import load_cache_pairs, load_cases
from scripts.evallib.metrics import (
    aggregate_exact,
    canary_leak,
    constraint_precision_at_k,
    entity_grounding,
    entity_hit_at_k,
    entity_mrr,
    false_refusal,
    filter_accuracy,
    numeric_grounding,
    refusal_accuracy,
    route_accuracy,
)
from scripts.evallib.pipeline import PipelineConfig, run_case
from scripts.evallib.report import (
    build_report,
    compare_with_baseline,
    generate_run_id,
    write_baseline,
    write_json_report,
    write_md_summary,
)

# ---------------------------------------------------------------------------
# Gates (PLAN section 7)
# ---------------------------------------------------------------------------

GATES: dict[str, float] = {
    "constraint_precision_at_5": 0.80,
    "entity_hit_at_5": 0.80,
    "aggregate_exact": 0.90,
    "refusal_accuracy": 1.00,
    "false_refusal": 0.05,  # inverted: false_refusal rate must be <= 5%
}

LLM_GATES: dict[str, float] = {
    "numeric_grounding": 0.95,
    "entity_grounding": 0.95,
    "canary_leak": 0.0,  # must be 0
}


def _build_entities(index_dir: Path):
    """Load entities from the index's entities.json."""
    from app.domain.router import EntityIndex
    from common.schemas import Entities

    entities_path = index_dir / "entities.json"
    if not entities_path.exists():
        raise FileNotFoundError(f"entities.json not found in {index_dir}")
    data = json.loads(entities_path.read_text(encoding="utf-8"))
    ents = Entities.model_validate(data)
    return EntityIndex(ents)


def _build_entities_fallback():
    """Minimal fixture entities for --fakes mode."""
    from app.domain.router import EntityIndex
    from common.schemas import Entities, Entity, EntityKind

    ents = Entities(entities=[
        Entity(kind=EntityKind.city, canonical="Riyadh", aliases=["riyadh"]),
        Entity(kind=EntityKind.city, canonical="Jeddah", aliases=["jeddah"]),
        Entity(kind=EntityKind.city, canonical="Dammam", aliases=["dammam"]),
        Entity(kind=EntityKind.city, canonical="Mecca", aliases=["mecca", "makkah"]),
        Entity(kind=EntityKind.city, canonical="Medina", aliases=["medina", "madinah"]),
        Entity(kind=EntityKind.country, canonical="Saudi Arabia", aliases=["saudi", "ksa"]),
        Entity(kind=EntityKind.country, canonical="UAE", aliases=["uae"]),
        Entity(kind=EntityKind.country, canonical="Oman", aliases=["oman"]),
        Entity(
            kind=EntityKind.project,
            canonical="Trump Tower Jeddah",
            aliases=["trump tower", "trump jeddah"],
            doc_id="darglobal:project:trump-tower-jeddah",
        ),
    ])
    return EntityIndex(ents)


def _build_fakes():
    """Build fake adapters for --fakes mode."""
    from tests.fakes.fake_adapters import (
        FakeEmbedder,
        FakeReranker,
        FakeRetriever,
        FakeStructuredQuery,
    )

    return FakeEmbedder(), FakeRetriever(), FakeReranker(), FakeStructuredQuery()


def _build_real_adapters(args: argparse.Namespace):
    """Build real adapters via app.adapters.factory.build_read_side."""
    from app.adapters.factory import build_read_side

    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = REPO_ROOT / index_dir

    rs = build_read_side(
        index_dir,
        rerank=args.rerank,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
        prefetch_limit=args.list_size,
    )
    return rs.embedder, rs.retriever, rs.reranker, rs.sql, rs.index


def _get_index_version(index_dir: Path) -> str:
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data.get("index_version", "unknown")
    return "unknown"


def _chain_tpm_values(chain_ids: list[str]) -> list[int]:
    """Read ``tpm`` from the default chain config for the given entry ids."""
    chain_json = os.environ.get("LLM_CHAIN", "")
    if chain_json:
        chain = json.loads(chain_json)
    else:
        chain = [
            {"id": "or-gemma", "provider": "openrouter", "tpm": None},
            {"id": "or-nemotron", "provider": "openrouter", "tpm": None},
            {"id": "or-free", "provider": "openrouter", "tpm": None},
            {"id": "groq-oss", "provider": "groq", "tpm": 8000},
            {"id": "groq-qwen", "provider": "groq", "tpm": 6000},
        ]
    if chain_ids:
        chain = [e for e in chain if e.get("id") in chain_ids]
    return [e["tpm"] for e in chain if e.get("tpm")]


async def _run_eval(args: argparse.Namespace) -> int:
    """Main eval loop. Returns 0 on success, 1 on failure."""
    from scripts.evallib.llm_step import (
        BudgetTracker,
        LLMCaseResult,
        QuotaLogger,
        TPMPacer,
        filter_cases,
        llm_skip_counts,
        plan_summary,
        run_llm_step,
        should_rerun,
    )

    # Determine sets directory
    sets_dir = REPO_ROOT / "eval" / "sets"
    fixture_dir = REPO_ROOT / "tests" / "eval" / "fixtures"

    # Load cases
    cases = load_cases(sets_dir, split=args.split)
    if not cases:
        cases = load_cases(fixture_dir, split=args.split)
    if not cases:
        cases = load_cases(fixture_dir)
    if not cases:
        print("ERROR: no eval cases found", file=sys.stderr)
        return 1

    # Apply subset filters
    if args.case_ids:
        cases = filter_cases(cases, case_ids=args.case_ids.split(","))
    if args.sets:
        cases = filter_cases(cases, sets=args.sets.split(","))
    if args.kind:
        cases = filter_cases(cases, kind=args.kind)

    print(f"Loaded {len(cases)} cases for split={args.split or 'all'}")

    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = REPO_ROOT / index_dir

    index_version = _get_index_version(index_dir)
    print(f"Index version: {index_version}")

    # Determine chain length for budget planning
    chain_ids = [c.strip() for c in args.provider_chain.split(",") if c.strip()] if args.provider_chain else []
    chain_length = max(len(chain_ids), 1) if args.llm else 0

    # --plan: print counts and exit
    if args.plan:
        summary = plan_summary(cases, chain_length)
        print(f"Total cases: {summary['total_cases']}")
        print(f"LLM cases: {summary['llm_cases']}")
        print(f"LLM non-refused: {summary['llm_non_refused']}")
        print(f"Chain length: {summary['chain_length']}")
        print(f"Worst-case attempts: {summary['worst_case_attempts']}")
        return 0

    # Build adapters
    if args.fakes:
        entities = _build_entities_fallback()
        embedder, retriever, reranker, sq = _build_fakes()
    else:
        entities = _build_entities(index_dir)
        embedder, retriever, reranker, sq, _idx = _build_real_adapters(args)
        print(f"Real adapters loaded from {index_dir}")

    config = PipelineConfig(
        rrf_k=args.rrf_k,
        list_size=args.list_size,
        top_k_docs=5,
        score_floor=args.score_floor,
        rerank_enabled=args.rerank,
        llm_mode=args.llm,
        budget=args.budget,
    )

    # LLM setup
    llm = None
    system_prompt = ""
    budget = BudgetTracker(limit=args.budget)
    quota_logger = None
    pacer: TPMPacer | None = None

    if args.llm:
        # Load system prompt (H-4a: use --prompt flag)
        prompt_path = REPO_ROOT / "app" / "prompts" / f"{args.prompt}.txt"
        if prompt_path.exists():
            system_prompt = prompt_path.read_text(encoding="utf-8").strip()

        # Build LLM (fake or real)
        if args.fakes:
            from tests.fakes.fake_adapters import FakeLLM
            llm = FakeLLM()
        elif args.budget > 0:
            from app.adapters.factory import build_llm
            llm = build_llm(chain_ids=chain_ids or None)

        # TPM pacer: use the minimum tpm from chain entries that have one
        _tpm_values = _chain_tpm_values(chain_ids)
        if _tpm_values:
            pacer = TPMPacer(tpm=min(_tpm_values))

        # Quota logger
        quota_csv = REPO_ROOT / "eval" / "quota_log.csv"
        quota_logger = QuotaLogger(quota_csv)

    # Run each case
    results: list[dict] = []
    for i, case in enumerate(cases):
        result = await run_case(
            case,
            entities=entities,
            embedder=embedder,
            retriever=retriever,
            reranker=reranker if config.rerank_enabled else None,
            structured_query=sq,
            config=config,
        )

        # LLM step (when enabled and case has llm=true)
        if args.llm and case.get("llm"):
            # H-3: pass the full SqlResult to the LLM step so <sql_result>
            # reaches the prompt on aggregate cases.
            sql_result_obj = result.get("_sql_result")
            if llm is not None:
                llm_result = await run_llm_step(
                    case=case,
                    predicted=result,
                    llm=llm,
                    system_prompt=system_prompt,
                    entities=entities,
                    budget=budget,
                    chain_length=chain_length,
                    quota_logger=quota_logger,
                    purpose=args.purpose,
                    wp=args.wp,
                    sql_result=sql_result_obj,
                    pacer=pacer,
                )
            else:
                # No LLM built (budget 0 without --fakes): mark as skipped
                llm_result = LLMCaseResult(skipped_budget=True)
                budget.skipped_count += 1

            # Rerun rule
            if args.rerun_failures and should_rerun(llm_result, case):
                rerun_result = await run_llm_step(
                    case=case,
                    predicted=result,
                    llm=llm,
                    system_prompt=system_prompt,
                    entities=entities,
                    budget=budget,
                    chain_length=chain_length,
                    quota_logger=quota_logger,
                    purpose=args.purpose,
                    wp=args.wp,
                    sql_result=sql_result_obj,
                    run_index=1,
                    pacer=pacer,
                )
                # Fail only if both runs fail
                if not should_rerun(rerun_result, case):
                    llm_result = rerun_result
                result["llm_runs"] = [llm_result.to_dict(), rerun_result.to_dict()]
            else:
                result["llm_runs"] = [llm_result.to_dict()]

            result["answer"] = llm_result.answer
            result["grounding"] = llm_result.grounding

        # Merge for metric evaluation
        merged = {**case, "predicted": result}
        results.append(merged)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(cases)} cases done")

    print(f"  ... {len(cases)}/{len(cases)} cases done")

    if args.llm:
        sc = llm_skip_counts(results)
        if sc["llm_skipped_budget"] > 0 or sc["llm_unavailable"] > 0:
            print(f"  llm_skipped_budget: {sc['llm_skipped_budget']}")
            print(f"  llm_unavailable: {sc['llm_unavailable']}")
            print(f"  Budget: {budget.used}/{budget.limit} used")

    # Compute metrics
    metrics_dict = {
        "route_accuracy": route_accuracy(results),
        "filter_accuracy": filter_accuracy(results),
        "constraint_precision_at_5": constraint_precision_at_k(results, k=5),
        "entity_hit_at_5": entity_hit_at_k(results, k=5),
        "entity_mrr": entity_mrr(results, k=5),
        "aggregate_exact": aggregate_exact(results),
        "refusal_accuracy": refusal_accuracy(results),
        "false_refusal": false_refusal(results),
        "canary_leak": canary_leak(results),
    }

    if args.llm:
        metrics_dict["numeric_grounding"] = numeric_grounding(results)
        metrics_dict["entity_grounding"] = entity_grounding(results)

    # Print metrics
    print("\n--- Metrics ---")
    for name, m in metrics_dict.items():
        print(f"  {name}: {m.render()}")

    # Build report
    run_id = generate_run_id()
    config_dict = {
        "split": args.split,
        "rrf_k": args.rrf_k,
        "list_size": args.list_size,
        "score_floor": args.score_floor,
        "rerank": args.rerank,
        "llm": args.llm,
        "no_llm": args.no_llm,
        "dense_weight": args.dense_weight,
        "sparse_weight": args.sparse_weight,
        "index_dir": str(args.index_dir),
        "fakes": args.fakes,
    }
    report = build_report(
        run_id=run_id,
        split=args.split or "dev",
        config=config_dict,
        cases=results,
        metrics=metrics_dict,
        index_version=index_version,
    )

    # Write reports
    reports_dir = REPO_ROOT / "eval" / "reports"
    json_path = write_json_report(report, reports_dir)
    md_path = write_md_summary(report, reports_dir)
    print(f"\nReport: {json_path}")
    print(f"Summary: {md_path}")

    # Write baseline
    if args.write_baseline:
        bp = REPO_ROOT / "eval" / "baseline_report.json"
        write_baseline(report, bp)
        print(f"Baseline written: {bp}")

    # Compare with baseline
    if args.compare:
        baseline_path = Path(args.compare)
        if not baseline_path.is_absolute():
            baseline_path = REPO_ROOT / baseline_path
        regressions = compare_with_baseline(report, baseline_path, gates=GATES)
        if regressions:
            print("\nREGRESSIONS:")
            for r in regressions:
                print(f"  FAIL: {r}")
            return 1
        print("\nBaseline comparison: PASS")

    return 0


async def _run_cache_pairs(args: argparse.Namespace) -> int:
    """Cache pairs mode: embed both sides, compute scopes, report tau grid."""
    from app.domain.cache_keys import build_scope, parts_from
    from app.domain.router import route

    sets_dir = REPO_ROOT / "eval" / "sets"
    pairs = load_cache_pairs(sets_dir, split=args.split)
    if not pairs:
        print("No cache pairs found", file=sys.stderr)
        return 1

    tau_values = [float(t) for t in args.tau.split(",")]
    print(f"Cache pairs: {len(pairs)} pairs, tau grid: {tau_values}")

    index_dir = Path(args.index_dir)
    if not index_dir.is_absolute():
        index_dir = REPO_ROOT / index_dir

    index_version = _get_index_version(index_dir)
    prompt_version = "unknown"

    # Build real embedder unless --fakes
    if args.fakes:
        from tests.fakes.fake_adapters import FakeEmbedder

        embedder = FakeEmbedder()
        entities = _build_entities_fallback()
    else:
        from app.adapters.factory import build_read_side

        rs = build_read_side(index_dir, prefetch_limit=args.list_size)
        embedder = rs.embedder
        entities = _build_entities(index_dir)
        print(f"Real embedder loaded from {index_dir}")

    # Embed each pair and compute scopes (H1)
    for pair in pairs:
        emb_a = await embedder.embed_query(pair["a"])
        emb_b = await embedder.embed_query(pair["b"])
        dot = sum(x * y for x, y in zip(emb_a.dense, emb_b.dense))
        pair["cosine"] = dot

        # Compute L2 scope for each side via the router
        dec_a = route(pair["a"], session=None, entities=entities, lang_hint=None)
        dec_b = route(pair["b"], session=None, entities=entities, lang_hint=None)
        parts_a = parts_from(dec_a, pair["a"], index_version, prompt_version)
        parts_b = parts_from(dec_b, pair["b"], index_version, prompt_version)
        scope_a = build_scope(parts_a)
        scope_b = build_scope(parts_b)
        pair["same_scope"] = scope_a == scope_b

    n_same_scope = sum(1 for p in pairs if p["same_scope"])
    n_diff_scope = len(pairs) - n_same_scope
    print(f"Scope: {n_same_scope} same-scope, {n_diff_scope} different-scope")

    # Overall tau grid
    print("\n| tau | true_hits | false_hits |")
    print("|-----|-----------|------------|")
    report_rows = []
    for tau in tau_values:
        true_hits = 0
        false_hits = 0
        for pair in pairs:
            above = pair["cosine"] >= tau
            if pair["same_answer"] and above:
                true_hits += 1
            elif not pair["same_answer"] and above:
                false_hits += 1
        print(f"| {tau:.2f} | {true_hits} | {false_hits} |")
        report_rows.append({"tau": tau, "true_hits": true_hits, "false_hits": false_hits})

    # Scope-separated tau grid (H1: count only same-scope pairs)
    same_scope_pairs = [p for p in pairs if p["same_scope"]]
    if same_scope_pairs:
        print("\nSame-scope pairs only:")
        print("| tau | true_hits | false_hits |")
        print("|-----|-----------|------------|")
        for tau in tau_values:
            true_hits = sum(
                1 for p in same_scope_pairs if p["cosine"] >= tau and p["same_answer"]
            )
            false_hits = sum(
                1 for p in same_scope_pairs if p["cosine"] >= tau and not p["same_answer"]
            )
            print(f"| {tau:.2f} | {true_hits} | {false_hits} |")
            report_rows.append({
                "tau": tau,
                "true_hits_scope": true_hits,
                "false_hits_scope": false_hits,
            })

    # Write JSON
    reports_dir = REPO_ROOT / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fp = reports_dir / "cache_pairs.json"
    fp.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")
    print(f"\nReport: {fp}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval harness for Wasalt RAG chatbot")
    parser.add_argument("--split", default="dev", help="Eval split: dev or test")
    parser.add_argument("--no-llm", action="store_true", help="Run no-LLM eval only")
    parser.add_argument("--llm", action="store_true", help="Run LLM eval mode")
    parser.add_argument("--provider-chain", default="", help="Comma-separated provider ids")
    parser.add_argument("--budget", type=int, default=0, help="Max LLM calls")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter")
    parser.add_argument("--list-size", type=int, default=20, help="Retrieval list size")
    parser.add_argument("--score-floor", type=float, default=0.52, help="Dense score floor")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--tau", default="0.90,0.92,0.94", help="Tau grid for cache pairs")
    parser.add_argument("--cache-pairs", action="store_true", help="Run cache pairs mode")
    parser.add_argument("--write-baseline", action="store_true", help="Write baseline report")
    parser.add_argument("--compare", default="", help="Path to baseline for comparison")
    parser.add_argument("--index-dir", default="data/index", help="Path to index directory")
    parser.add_argument("--fakes", action="store_true", help="Use fake adapters instead of real")
    parser.add_argument("--dense-weight", type=float, default=1.0, help="RRF dense weight")
    parser.add_argument("--sparse-weight", type=float, default=1.0, help="RRF sparse weight")
    # LLM subset flags
    parser.add_argument("--case-ids", default="", help="Comma-separated case ids to run")
    parser.add_argument("--sets", default="", help="Comma-separated eval set names")
    parser.add_argument("--kind", default="", help="Filter on expect.kind (lookup, aggregate, ...)")
    parser.add_argument("--plan", action="store_true", help="Print plan and exit (no calls)")
    parser.add_argument("--purpose", default="geneval", help="Purpose for quota log")
    parser.add_argument("--wp", default="", help="WP id for quota log (required with --llm)")
    parser.add_argument("--rerun-failures", action="store_true", help="Rerun grounding failures once")
    parser.add_argument("--prompt", default="system_v1", help="Prompt file stem (e.g. system_v1 -> app/prompts/system_v1.txt)")  # H-4a

    args = parser.parse_args()

    # Set model cache paths before any import that might load models
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(REPO_ROOT / ".orchestration" / "tmp" / "ml" / "fe_cache"))
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".orchestration" / "tmp" / "ml" / "fe_cache"))

    if args.cache_pairs:
        return asyncio.run(_run_cache_pairs(args))

    return asyncio.run(_run_eval(args))


if __name__ == "__main__":
    sys.exit(main())
