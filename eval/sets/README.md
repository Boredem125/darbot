# Eval sets (WP-3, format in `.orchestration/WP-3-common-rules.md`, "Eval-set format")

Regenerate: `.venv/Scripts/python.exe eval/sets/gen_templated.py` and `eval/sets/gen_sets.py`. Add `--check` to either one to regenerate in memory and diff against the files. Validate the format with `eval/sets/check_sets.py`. No LLM produced any truth value.

- `templated_dev.jsonl`: 60 cases, dev, 44 en / 16 ar, 4 LLM. Seeded generator (seed 20261119) with paraphrased EN/AR templates. `sql_truth` is computed with DuckDB from `data/index/listings.parquet` using the `AggregateOp` semantics. Each (op, filters) scope is used once across both splits.
- `templated_test.jsonl`: 40 cases, test, 28 en / 12 ar, 0 LLM. Same generator and truth, with scopes disjoint from dev.
- `handwritten.jsonl`: 15 cases, 9 dev / 6 test, en, 6 LLM. Realistic queries written by hand. Aggregate truth comes from the same DuckDB code. hw-014 and hw-015 are no-data refusals (0 matching rows, checked).
- `projects.jsonl`: 10 cases, 6 dev / 4 test, en, 4 LLM. Facts are copied from `data/raw/darglobal/<slug>.json.gz`, and the generator asserts that each fact string is on the page. The id is `darglobal:project:<URL slug>` until the extraction WP fixes the `doc_id`. pj-009 lists both slugs of a page that was crawled twice.
- `multiturn.jsonl`: 4 two-turn cases, 2 dev / 2 test, 3 en / 1 ar, 2 LLM. The expected filters are the slots after carry-over. mt-002 truth comes from DuckDB.
- `arabic.jsonl`: 5 cases, 3 dev / 2 test, ar, 3 LLM. Truth comes from DuckDB.
- `oos.jsonl`: 7 cases, 4 dev / 3 test, 5 en / 2 ar, 0 LLM. Expected route `out_of_scope`, refuse true.
- `provider_subset.jsonl`: 5 cases, test, one per kind (lookup, aggregate, project, arabic, refusal), all LLM.
- `injection_fixtures.jsonl`: 3 cases, test, all LLM. Each case names its canary token and the canary listing id.
- `injection_listings.jsonl`: 3 canary `Listing` records (validated). They belong in a separate test index only, never in `data/index/`.
- `cache_pairs.jsonl`: 30 pairs, 18 dev / 12 test. 15 are `same_answer` true (paraphrases) and 15 are near-misses: 11 same-scope (amenity, view, negation without a cue word, project attribute, newest vs oldest, proximity) and 4 different-scope.
- `prompt_analysis/*_baseline.json`: `prompt_optimizer.py --analyze --json` output for `system_v1.txt` and `extract_project_v1.txt`, the baseline for later prompt changes.

Known router-gap probes, kept so tuning can measure them (deterministic router run on 19 Nov):
- `avg_area` phrasing routes to `avg_price`: tpl-dev-010, tpl-dev-017, tpl-dev-020, tpl-dev-033, tpl-test-001, tpl-test-004, tpl-test-023, tpl-test-024, hw-013.
- No Arabic median cue: tpl-dev-052, tpl-test-007.
- Plural "buildings" is not mapped: tpl-test-002.
- "avg" is not an average cue: hw-004.
- Arabic definite plural (الفلل, الشقق) is not mapped: ar-002, ar-004, ps-004.
- Arabic "between X and Y": ar-003. The English "between" case hw-008 passes.
- OOS query that contains a city ("weather in Riyadh"): oos-001.
- Negation without a cue word: hw-009, plus the same-scope cache pair "anything but villas".
- Project questions route `out_of_scope` or `lookup` while the index has 0 projects: pj-*, ps-003. Re-measure after the extraction WP.
