# Tuning record (Wave 3, dev split only)

Owner: dg-ml-lead. Date: 19 Nov 2026. Index `ca2305e19fe1610d` (`data/index/`, 3,657 points: 1,398 sale, 688 rent, 41 projects, 174 articles; built 2026-11-18T19:27Z).
Every number comes from one run of `.orchestration/tmp/tuning/grid.py` on the 84 dev cases and 18 dev cache pairs, with no LLM calls. Raw output: `.orchestration/tmp/tuning/grid_run_ca2305.txt` (grid §1) and `grid_run2_ca2305.txt` (§2 to §7), JSON in `grid_out_ca2305e19fe1610d.json`. The script drives `scripts/evallib/pipeline.run_case` and the `scripts/evallib/metrics` functions, so the numbers are the harness numbers. Its baseline row reproduces T0's 07:14 run exactly (route 81/84, filters 72/74, CP@5 272/278, entity 17/19, MRR 0.855, aggregate 40/47, refusal 3/4, false refusals 1/80). The test split was not loaded. The loaders assert `split == "dev"`.

Metric names: CP@5 is constraint precision@5, hit@5 is entity hit@5, and primary is the mean of CP@5 and hit@5 in points (retrieval.md §3.4). Rates are x/n with a Wilson 95% CI where given.

## 1. RRF k, list size, dense/sparse weights

Rule (retrieval.md §3.4): move off (k 60, weights (1, 1)) only for a primary gain of at least 3 points. Tie-break on MRR.
Grid: k ∈ {10, 30, 60, 100} × weights (dense, bm25) ∈ {(1,1), (1,0.5), (0.5,1), (1,0), (0,1)} × list size ∈ {20, 40}, 40 configs.

| Config | CP@5 | hit@5 | MRR | Primary |
|---|---|---|---|---|
| k 60, (1,1), list 20 (baseline) | 272/278 | 17/19 | 0.855 | 93.66 |
| any k, (1,1) or (1,0.5) | 272/278 | 17/19 | 0.855 to 0.860 | 93.66 |
| any k, (0.5,1) | 272/278 | 17/19 | 0.868 (k ≥ 30) | 93.66 |
| any k, (1,0), dense only (best) | 273/278 | 17/19 | 0.895 | 93.84 |
| any k, (0,1), bm25 only | 271/277 | 17/19 | 0.855 | 93.65 |

List size 40 gave the same numbers as 20 for every row. The best gain is 0.18 points, below the 3-point rule.
**Chosen: k 60, weights (1.0, 1.0), list 20 (unchanged).** 13 of the 19 entity cases are aggregate example cards, which retrieval does not touch. The 6 project cases carry all the variation. The MRR spread comes from pj-004 alone (the project card at rank 4), and the router/alias fixes in §7 move it to rank 1 at every config. After those fixes, the dev set cannot separate the RRF configs.

## 2. Score floor on `top_dense_score`

Rule (retrieval.md §3.6): if in-scope and out-of-scope top-1 cosines are separable, set the floor to the midpoint, rounded down to 0.01.
Measured on the real routing: 36 dev cases reach retrieval and **0 of them have empty filters**, so the floor (which applies only to empty filters) never fires on dev, at any floor from 0.20 to 0.60. The only false refusal (pj-005) is a zero-hit case (§7). The only OOS case that reaches retrieval (oos-001) carries a city filter, so the floor does not apply to it either (its filtered top-1 is 0.468).
To get usable evidence, I also scored every dev query against the whole index with no filters, which is the situation when the router extracts nothing:

| Set | n | top-1 cosine |
|---|---|---|
| in-scope | 80 | min 0.537 (pj-001), then 0.568, 0.574, 0.619 |
| OOS | 4 | 0.219, 0.241, 0.442, 0.514 (oos-001, "weather in Riyadh") |

| Floor | False refusals (unfiltered) | OOS refused (unfiltered) |
|---|---|---|
| 0.30 (old) | 0/80 | 2/4 |
| 0.45 | 0/80 | 3/4 |
| 0.52 | 0/80 | 4/4 |
| 0.55 | 1/80 | 4/4 |
| 0.575 | 3/80 | 4/4 |

The sets are separable, with a gap between 0.514 and 0.537. **Chosen: `SCORE_FLOOR = 0.52`** (midpoint 0.5255, rounded down). Actual-rule dev false refusals stay at 0 from the floor (CI [0, 0.046] over 80). The margin is thin (0.006 above the OOS max, 0.017 below the in-scope min). Re-measure on every index rebuild. The OOS topic cue in §7 takes oos-001 off this path anyway.

## 3. Rerank keep or drop

Rule (retrieval.md §3.7): enable only with a gain of at least 3 points on primary **and** on MRR, English only, and an HF p95 below 150 ms.
Measured on 64 English dev cases, rerank of the fused top 10 docs keeping 5 (spec mode), `Xenova/ms-marco-MiniLM-L-6-v2`, threads=1:

| | CP@5 | hit@5 | MRR | Primary |
|---|---|---|---|---|
| no rerank | 205/205 | 13/15 | 0.817 | 93.33 |
| rerank 10 → 5 | 205/205 | 13/15 | 0.867 | 93.33 |

Primary gains 0 points and MRR gains 5.0 points (pj-004 moves from rank 4 to 1). The harness mode (rerank of the collapsed top 5) gave identical numbers. Latency was p50 293 ms per call over 27 calls on the shared dev box, about twice the whole 150 ms path budget.
**Chosen: `RERANK_ENABLED=0` (unchanged).** Rule (1) fails on primary, and the latency fails as well. The MRR gain disappears once the project-alias fix in §7 lands.

## 4. τ (L2 semantic cache)

Rule (caching.md §4): τ = max same-scope near-miss cosine + 0.01, rounded up to 0.01, never below 0.88. Disable L2 if τ would be above 0.98.
Dev pairs: 9 hit pairs (all same scope) and 9 near-misses. Of the near-misses, 3 are scope-separated (cp-022 newest vs oldest, cp-027 bedrooms, cp-028 city) and 6 are same-scope.
Same-scope near-miss cosines: cp-018 view 0.916, cp-016 amenity 0.905, cp-019 negation without a cue word 0.902, cp-017 furnished/unfurnished 0.840, cp-021 gym/spa 0.716, cp-020 payment plan/handover 0.602.
Hit-pair cosines: 0.959, 0.950, 0.925, 0.911, 0.902, 0.902, 0.897, 0.866, 0.837.

| τ | 0.88 | 0.90 | 0.91 | 0.92 | 0.93 | 0.94 | 0.95 |
|---|---|---|---|---|---|---|---|
| true hits (scoped) | 7/9 | 6/9 | 4/9 | 3/9 | 2/9 | 2/9 | 1/9 |
| false hits (scoped) | 3/6 | 3/6 | 1/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| false hits, scope ignored (harness `--cache-pairs`) | 5 | 4 | 2 | 1 | 1 | 1 | 1 |

The lowest τ with 0 false hits is 0.92, but its margin over cp-018 is 0.004. The §4 rule adds 0.01: 0.916 + 0.01 = 0.926, which rounds up to **τ = 0.93 (chosen; was 0.92)**. At τ 0.93, true hits are 2/9 (22%, CI [0.063, 0.547]) and false hits are 0/6 (CI [0, 0.39]). L2 stays on, because τ is not above 0.98, but its recall is low. The residual "1 false hit" in the harness table is cp-027 (cosine 0.972), which the scope already separates. The harness ignores the scope.
The dev part has 6 same-scope near-misses, short of the 10 that caching.md asks for. The whole set has 11.
Recall upside (computed from the cosines above, not re-run through code): if the L2 scope also keyed on an amenity/view token set and "anything but" became a negation cue, only cp-020 (0.602) would stay same-scope. The rule would then give τ 0.88, with 7/9 true hits. See change item B9 in the WP report.

## 5. Plausibility floors (organize/normalise.py)

Current floors: rent ≥ 3,000 SAR/year, sale ≥ 50,000 SAR. The index distribution below that is still present:

| Yearly-rent apartments | n | median area | median rent/yr |
|---|---|---|---|
| < 6,000 | 51 | 100 sqm | 3,500 |
| 6,000–12,000 | 3 | 118 sqm | 6,000 |
| 12,000–24,000 | 22 | 325 sqm | 19,250 |
| ≥ 24,000 | 466 | 128 sqm | 42,000 |

Below 12,000 SAR/year there are 57 rent listings: 54 apartments, 2 buildings and 1 villa. The low cluster (100 sqm at 3,500 a year, about 290 SAR a month) matches monthly rents labelled as yearly. The rent distribution is bimodal, with almost nothing between 6,000 and 12,000. The cluster moves the average yearly apartment rent in Riyadh from 55,906 (floor 12,000, n 391) to 50,225 (floor 3,000, n 439), a shift of 11%, far outside the 0.5% aggregate tolerance. The median moves from 42,000 to 40,000.
**Chosen: rent floor raised to 12,000 SAR/year (1,000 a month); sale floor kept at 50,000.** All 11 sale listings under 200,000 are land at a median of 260 SAR/sqm, which is plausible for regional plots. The lowest is 80,010 for 315 sqm.
Consequence: the index must be rebuilt and every rent `sql_truth` regenerated (`eval/sets/gen_templated.py`, `gen_sets.py`), because the truths come from `listings.parquet`. This is a data change for dg-data-engineer and dg-prompt-eval-lead, not a tuning knob that the eval can score.

## 6. Language threshold (`arabic_ratio` ≥ 0.30)

All 86 dev turns: the Arabic turns have a minimum ratio of 1.00 and the English turns a maximum of 0.00. The number of misclassified turns is 0 at every threshold from 0.1 to 0.6. **Chosen: 0.30 (unchanged).** Dev has no mixed-script turns, so the threshold is untested between 0.1 and 0.9.

## 7. Remaining dev failures, classified

Baseline at the chosen config: 12 cases fail at least one metric. None is a wrong expectation and none is a tunable parameter. All are code or data gaps:

| Case(s) | Symptom | Class |
|---|---|---|
| tpl-dev-010, -017, -020, -033 | "average size" / "متوسط مساحة" routed to avg_price | router code (cue order, Arabic cue) |
| hw-004 | "avg" not an average cue | router dictionary |
| tpl-dev-052 | Arabic median "الوسيط" missing | router dictionary |
| ar-002 | "الفلل" (definite plural) not mapped | router dictionary |
| ar-003 | Arabic "بين X و Y" not parsed | router regex |
| oos-001 | "weather in Riyadh" routed lookup (city filter) | router OOS cue list |
| pj-005 | project_id plus city Marbella (project city is Benahavís): 0 hits, refusal | router code |
| pj-003 | "Marriott Residences at AIDA" → project_id of AIDA (last match wins; no alias) | router code + entity aliases (data) |
| pj-004 | "The Astera by Aston Martin" not matched (no alias) → rank 4 | entity aliases (data) |

I simulated the fixes in the grid script by monkeypatching, not in `app/`. Dev at the chosen config:

| Metric | Now | Router fixes | Router + alias fixes |
|---|---|---|---|
| route | 81/84 | 84/84 | 84/84 |
| filters | 72/74 | 74/74 | 74/74 |
| CP@5 | 272/278 | 274/274 | 274/274 |
| hit@5 | 17/19 | 18/19 | 19/19 |
| MRR | 0.855 | 0.908 | 1.000 |
| aggregate exact | 40/47 | 47/47 | 47/47 |
| refusal | 3/4 | 4/4 | 4/4 |
| false refusals | 1/80 | 0/80 | 0/80 |

The exact change list is in `.orchestration/WP-3-tuning.report.md`.

## Index versions
- `ca2305e19fe1610d`: all seven decisions above.
- Next index (`.orchestration/tmp/index_next/`): pending. Re-run only the chosen config and its neighbours: `grid.py --index-dir <dir> --only-chosen 60,1,1,20`.

## Re-run on 77284224a04cf674 (19 Nov 2026)
Index `77284224a04cf674`: 631 rent listings (D3), 44 projects (D1/D2 aliases), 3,623 points. Router fixes B1–B8 and B10 are in `app/`. I ran `grid.py --only-chosen 60,1,1,20 --no-sim` once: the simulation section and the reranker import workaround are removed, because both fixes are now real. Output is in `.orchestration/tmp/tuning/grid_run_772842.txt` and `grid_out_77284224a04cf674.json`. The neighbour set de-duplicates to 4 configs (k 30, 60, 120 at (1,1)/list 20, plus k 60/list 40). Weight variants were not re-run.

| Metric (dev, chosen config) | ca2305 | 77284224 |
|---|---|---|
| route | 81/84 | 84/84 |
| filters | 72/74 | 74/74 |
| CP@5 | 272/278 | 274/274 |
| hit@5 / MRR | 17/19 / 0.855 | 19/19 / 1.000 |
| aggregate exact | 40/47 | 43/47 (stale rent truths, see below) |
| refusal | 3/4 | 4/4 |
| false refusals | 1/80 | 0/80 |

These numbers equal the §7 simulated "router + alias" column, except for the aggregate. The 4 aggregate misses are hw-004, tpl-dev-027, tpl-dev-030 and tpl-dev-052. All are rent scopes whose `sql_truth` still reflects the 3,000 floor; for example, hw-004 computes 55,906 against a stale truth of 50,225, which is the §5 value. Nothing was tuned on them.
- §1 RRF: all 4 configs score primary 100.0 and MRR 1.000. The gain is 0, so the config is kept.
- §2 Floor: the separation is unchanged. Unfiltered in-scope min is 0.537 (pj-001) and OOS max is 0.514 (oos-001). The midpoint rule still gives **0.52, kept**. Under the actual rule, 0 of the 33 retrieval cases have empty filters, and the OOS leak is now 0 because oos-001 routes out_of_scope.
- §3 Rerank: no rerank and rerank 10 → 5 are identical on English (CP@5 203/203, hit@5 15/15, MRR 1.000). Latency p50 is 303 ms. **Kept off.**
- §4 τ: the scoped cosines are identical (embedder and router scope unchanged for these pairs). The max same-scope near-miss is 0.916, so **τ 0.93 is kept** (2/9 true hits, 0/6 false hits). The scope-ignoring count at 0.93 is still 1 (cp-027).
- §5 Plausibility: 0 rent listings remain below 12,000. The average yearly apartment rent in Riyadh is 55,906 (n 391), as predicted.
- §6 Language: 0 misclassified, unchanged.
No chosen value changes on this index.
