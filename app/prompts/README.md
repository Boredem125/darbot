# Prompts

## Frozen prompt (WP-3-geneval, 19 Nov 2026)

- **Frozen generation prompt: `system_v1.txt`**
- sha256: `fb85fafebd184bd456d96f18e07c9df9b5404af596643cbb0ca82a9d5d997084`
- About 334 tokens by `prompt_optimizer.py` (limit 350). Zero-shot, model-agnostic, temperature 0.
- Analysis baseline: `eval/sets/prompt_analysis/system_v1_baseline.json`.

Any change to a frozen prompt now needs T0 approval. Follow PLAN §6: run the analysis first, change one thing at a time, pass the dev no-regression check, then rebuild the warm cache.

## Rejected revision

- `system_v2.txt` (sha256 `ff82a8d5768b01657418b934a95fce946d669db21713f9b38650ab97d9d0b4ee`, 345 tokens) changed only rule 2. It added "in plain ASCII square brackets, never 【n】" to stop gpt-oss-120b from writing fullwidth citations.
- On the 12 dev LLM cases on groq-oss (first run each), citations went from 6/12 to 4/12. pj-001 and pj-003 regressed because naming the 【n】 form primed the model to use it.
- It was rejected under the no-regression rule. It is kept only as a record. Do not load it.
- Analysis: `eval/sets/prompt_analysis/system_v2_analysis.json`.

## Other prompts

- `extract_project_v1.txt` (offline project extraction, Groq). sha256 `3d224a1d9b7697610c7100310ca6c294b7a80b1e42aee39a313d7d2aed5947b5`.
