# ADR-0007: Deterministic grounding check, with an LLM judge as a secondary signal only

- Status: Accepted
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0004, ADR-0006, `common/schemas.py` (`GroundingReport`, `AnswerFlags`, `ChatEventEnd`), `docs/contracts/sse-v1.md`

## Context

The worst failure for a real-estate assistant is a confident wrong number: a price, a bedroom count, a handover date, or a district that is not in the sources. Free models at temperature 0 still invent such numbers now and then, more so in Arabic.

Every answer has to be checked, and the check has to:
- run on every request without spending quota (50 OpenRouter requests a day);
- finish within a few milliseconds so it does not hurt the SLOs;
- give a yes or no that the cache can rely on (ADR-0006).

The prompt already asks for `[n]` citations after each fact, prices written as `SAR 1,950,000`, and answers drawn only from `<source>` and `<sql_result>` blocks. That makes the most dangerous claims (numbers, names, citations) machine-checkable.

## Decision

1. **`app/domain/grounding.py` runs after the stream ends,** on the full answer text, and fills `GroundingReport` with three checks:
   - **`numbers_ok`**: every number in the answer must appear in the cited sources or the `<sql_result>`. Both sides are normalised first: k, M, thousand and million (English and Arabic words), thousands separators, Arabic-Indic digits, and `SAR` placement. SQL averages may differ by ±0.5% to allow rounding. Counts up to k (the number of cards) are allowed.
   - **`entities_ok`**: every city, district, country, project and brand that the answer names (found by matching `entities.json` canonical names and aliases) must also appear in the context the model was given: the cards, the `<sql_result>`, or the active filters.
   - **`citations_ok`**: every `[n]` must fall within 1..k, where k is the number of cards in `start.sources`.
   
   A check that does not apply is `null`, for example on a refusal. `failures` lists up to 20 short reasons for logs and `/stats`.

2. **The result is enforced by the contract.** `end.flags.unverified` must equal `not grounding.grounded`, where grounded means no check is `false`. The `ChatEventEnd` model rejects any other combination. The UI shows an "unverified" badge when the flag is set.

3. **Ungrounded answers are never cached.** The service skips `Cache.put`, and `CachedAnswer` rejects them as well (ADR-0006).

4. **The answer is still shown when grounding fails.** Tokens have already streamed. The check labels the answer; it does not retract it. Grounding runs once per answer and is timed as `grounding_ms`.

5. **The LLM judge is secondary and off the request path.** A Groq model other than the generator scores faithfulness during evaluation (`scripts/eval.py`), and that score is reported, not gated. `JUDGE_ENABLED` exists so the judge can be cut (first item on the cut list). The judge never changes `flags`, the cache decision, or what the user sees.

6. **Gates.** Numeric plus entity grounding must be at least 95% on the LLM eval subset, and no injection canary may leak (PLAN section 7). Unit tests cover the normalisers, for example `1.95M == 1,950,000` and Arabic-Indic digits.

## Alternatives rejected

- **An LLM judge on every answer.** It doubles quota use, adds a second TTFT to every request, and its verdicts vary between runs. It would also be the only thing standing between a wrong number and the cache.
- **An NLI or cross-encoder entailment model at runtime.** Another ONNX model in the image, tens of milliseconds on 2 vCPU per answer, and weak on Arabic and on numbers, which are the claims we care about most.
- **Constrained or structured output (JSON with value fields).** Free models support `json_schema` unevenly, and it makes streaming prose awkward. It would also not stop the model from putting a wrong number in a valid field.
- **No check, trusting the prompt.** The cache would then replay any invented number, and the "unverified" badge would have no basis.

## Consequences

- Every answer carries a cheap, explainable grounding report, visible in the `end` event and counted in `/stats`.
- The cache holds only answers whose numbers, names and citations trace back to sources.
- The check depends on the prompt's output format. A model that writes "about two million" instead of `SAR 2,000,000` may fail `numbers_ok` even when it is right. Those cases show up as false "unverified" badges and lost cache entries, and the per-model eval measures them.
- `entities.json` becomes a runtime dependency of grounding as well as routing, so its quality matters twice.

## Trade-off accepted

The check misses paraphrased claims that carry no number or named entity, such as "close to the beach" or "family-friendly". In return it is cheap, deterministic, and runs on every answer.
