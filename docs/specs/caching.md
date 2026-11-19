# Caching spec (ADR-0006): tiers L0 to L4, cache key, L2 partition, τ

Status: Wave 1b, 18 Nov 2026. Owner: dg-ml-lead. Implementers: dg-backend-engineer (`common/text.py` normalisers and `app/domain/cache_keys.py` in Wave 2; `app/adapters/caches.py` and the service policy in Wave 4).
This spec decides WP-1a open issue 1a (cache key function and L2 scope string).

## 0. Measured facts

| # | Fact | Value | Source |
|---|---|---|---|
| C1 | Paraphrase pairs (same meaning, 7 pairs), cosine with the multilingual MiniLM | 0.64–0.97 (EN–EN 0.76–0.97; AR–EN 0.64) | `.orchestration/tmp/ml/probe_cos.py` |
| C2 | Same-scope near-miss pairs (different meaning, same filters and numbers, 7 pairs) | 0.74–**0.89**. The max is "villas in Riyadh with a pool" vs "… without a pool". "villas in Riyadh" vs "villas not in Riyadh" = 0.79 | same |
| C3 | Dense query embedding cost (the L2 lookup needs one) | p95 29.3 ms, threads=1 | `probe_models.py` |
| C4 | Qdrant local mode filtered dense search at 6k points | p95 164 ms | `probe_stages.py` |

C2 shows that near-misses overlap paraphrases (0.89 is above three of the seven paraphrase cosines). A cosine threshold alone cannot be safe; the scope partition carries the exactness, and τ only has to separate what the scope cannot see.

## 1. Tiers

| Tier | What | Store | Key | Capacity / TTL | Serves |
|---|---|---|---|---|---|
| L0 | Warm answers built offline | `data/index/warm_cache.jsonl` loaded into a read-only dict | exact key (§2) | whole file / none | whole answer |
| L1 | Exact answer cache | `cachetools.TTLCache` | exact key (§2) | 2,000 / 24 h (PLAN §5) | whole answer |
| L2 | Semantic answer cache | in-process numpy, partitioned by scope (§4) | scope + cosine ≥ τ | 2,000 entries / 24 h | whole answer, standalone turns only |
| L3 | Query-embedding LRU | `cachetools.LRUCache` | `normalise_for_embedding(query)` | 1,024 / none | `QueryEmbedding` (dense + sparse), `cached=True` |
| L4 | Retrieval-result LRU | `cachetools.LRUCache` | `sha256(index_version, normalise_for_embedding(query), filters.canonical_json(), limit)` | 512 / none | `RetrievalResult`, `cached=True` |

- L3 and L4 have no TTL. Their values are pure functions of the query, the filters and the immutable in-process index, and a new index means a new process.
- The `cachetools` caches are **not thread-safe**. Every cache read and write happens on the event-loop thread, never inside an executor callable. The single uvicorn worker (ADR-0008) makes a lock unnecessary under that rule.
- **Lookup order** (service): L0, then L1 (`Cache.get_exact`, no embedding needed, about 0 ms), then L2 (`Cache.get_semantic`, which needs the dense vector from L3 or the embedder; the vector is reused by retrieval).
- On a hit: `start.cache_tier` and `end.cache_tier` are set to the tier; `answer_mode=cache`; `finish_reason=cached`; `provider` and `model` are those of the original author, as the contract says.

## 2. Exact key and L2 scope (decision)

Both are computed by pure functions in `app/domain/cache_keys.py` (stdlib only):

```python
CACHE_KEY_SCHEMA = "ck1"   # bump on any change to this recipe

@dataclass(frozen=True, slots=True)
class CacheKeyParts:
    index_version: str        # manifest.index_version
    prompt_version: str       # compute_prompt_version("system_v1", bytes)
    language: Language        # router decision (reply language)
    route: Route              # lookup | aggregate | project_info
    op: str | None            # AggregateOp value for aggregate, else None
    filters_json: str         # SearchFilters.canonical_json()  (includes sort)
    numbers: tuple[str, ...]  # canonical numbers in order of appearance (§3)
    negated: bool             # negation cue present (§3)
    query_key_text: str       # normalise_for_key(query) (§3)

def scope_preimage(p) -> str:
    return "\x1f".join([CACHE_KEY_SCHEMA, p.index_version, p.prompt_version, p.language,
                        p.route, p.op or "-", p.filters_json, ",".join(p.numbers),
                        "neg" if p.negated else "pos"])

def build_scope(p) -> str:  return "ck1:s:" + sha256(scope_preimage(p).encode()).hexdigest()[:32]
def build_key(p) -> str:    return "ck1:k:" + sha256((scope_preimage(p) + "\x1e" + p.query_key_text).encode()).hexdigest()[:32]
```

Why each part is there:
- `index_version` and `prompt_version`: answers are invalidated by any data or prompt change. This is "never cut" in PLAN §11.
- `language`: the same retrieval answered in Arabic is a different answer.
- `route` and `op`: "cheapest" vs "most expensive" share filters except `sort`, and `op` makes the difference explicit.
- `filters_json`: 3-bed ≠ 4-bed, Riyadh ≠ Jeddah, sale ≠ rent. This is the PLAN §12 test.
- `numbers`: numbers the router did not turn into filters, such as "handover in 2027" vs "2028", or "2% down payment". Kept in order of appearance, not sorted: a different order only costs a miss, never a wrong hit.
- `negated`: C2 measured "with a pool" vs "without a pool" at 0.89, and the router does not model negation. Putting the cue in the scope moves those pairs into different partitions.
- 32 hex chars (128 bits) make collisions negligible. Hashing keeps L1 keys and L2 scope labels short and keeps raw user text out of the key.
- `CachedAnswer.index_version` and `prompt_version` are also checked on read; a mismatch is a miss and is logged. This is a belt-and-braces check.

**L0 uses the same `build_key`.** `scripts/warm_cache.py` (Wave 4) imports `app.domain.cache_keys`. `scripts/` may import anything under the dependency rules.

## 3. Normalisation (pure, `common/text.py`, shared by router, grounding and cache)

`normalise_for_embedding(q)` (light; feeds the embedder and the L3/L4 keys):
1. Unicode NFKC.
2. Arabic-Indic digits (U+0660–0669) and extended digits (U+06F0–06F9) become ASCII; U+066B becomes "."; U+066C becomes ",".
3. Collapse whitespace and strip.

`normalise_for_key(q)` (aggressive; exact-key text only):
1. Apply `normalise_for_embedding`.
2. `casefold()`.
3. Remove Arabic tatweel U+0640 and diacritics U+064B–U+065F and U+0670. Map أ, إ and آ to ا, and ى to ي.
4. Canonicalise numbers with the shared `parse_numbers()`. Remove thousands separators, then expand magnitudes: k, thousand and ألف/الف are ×1e3; m, mn, million and مليون are ×1e6; bn, billion and مليار are ×1e9. Write each result as a plain integer string, or a decimal with no trailing zeros. So "2M", "2 million", "2,000,000" and "2.0m" all become "2000000".
5. Number words zero to twelve (English, plus Arabic واحد to اثنا عشر, including the feminine forms) become digits.
6. Currency tokens sar, sr, riyal(s), ريال and ر.س become "sar".
7. Strip trailing `?!.؟` and whitespace.

Stopwords are not removed; paraphrase is L2's job.

`numbers` is the list of canonical number strings from step 4/5, in order of appearance, excluding citation-like `[n]`.

`negated` is true when any cue appears as a whole word:
- English: not, no, without, except, excluding, exclude, "other than", non, never, isn't, aren't, don't, doesn't.
- Arabic: لا, ليس, ليست, غير, بدون, دون, ماعدا, عدا, باستثناء.

## 4. L2 semantic cache (in-process numpy)

- **Store**: `dict[scope] -> (matrix float32 [n, 384], list[entry_id])`, plus an entry table `entry_id -> (CachedAnswer, created_at, last_hit_at)`.
- **Capacity** is 2,000 entries in total, evicting the oldest `last_hit_at`. TTL is 24 h, the same as L1.
- **Lookup**: dot the L2-normalised query vector with the scope's matrix and take the best match. It is a hit iff `cos ≥ τ` and the entry's `index_version` and `prompt_version` are current. The cost is O(entries in scope), which is microseconds.
- **Why not Qdrant `qa_cache`** (PLAN §5): local-mode filtered search measured p95 164 ms at 6k points (C4), roughly 55 ms at 2k by linear scaling. That alone is over the 50 ms hit SLO, and it blocks the event loop. See `retrieval.md` §1. The AI-SPEC researcher's proposal to keep Qdrant for L2 is declined on this number.
- **Standalone turns only** (§6). L2 is disabled entirely by `SEMANTIC_CACHE_ENABLED=0` (cut list #6), in which case `get_semantic` returns None.

**τ (initial and tuning).**
- Initial **τ = 0.92**. That is the C2 same-scope near-miss max (0.89) plus a 0.03 margin. On C1 it would serve 3 of 7 paraphrases (0.92, 0.93, 0.97): modest recall, zero false hits on the probe.
- With the negation cue in the scope, the remaining near-miss max on the probe is 0.85, which leaves extra margin.

**Tuning rule (Wave 3, `cache_pairs.jsonl` dev part, no LLM).**
1. Compute scopes and cosines for every pair.
2. Discard pairs whose two sides land in different scopes; count them as "scope-separated" and report the count.
3. τ = max(near-miss cosine within the same scope) + 0.01, rounded up to 0.01, but **never below 0.88**. That floor keeps a margin above the measured 0.89 class of near-misses being ignored by a small sample.
4. Report hit recall on the hit pairs at that τ.
5. If τ would exceed 0.98, set `SEMANTIC_CACHE_ENABLED=0` and record why. L2 would then be all risk for no recall.
6. Report false hits as x/n with a Wilson 95% CI. 0/15 has an upper bound of about 20%, so the eval set should hold as many same-scope near-misses as possible.

**Requirement on the eval set** (for dg-prompt-eval-lead): near-miss pairs that differ only in filters or numbers never reach the cosine comparison, so they test the scope function, not τ. `cache_pairs.jsonl` needs **at least 10 same-scope near-misses**: amenities (pool, gym, furnished), sea/park view, negation without a cue word ("anything but villas"), project attributes (gym vs spa, payment plan vs handover), and newest vs oldest within the same op. It also needs a dev/test split, so τ is chosen on dev and confirmed on test.

## 5. Put policy (what may be cached)

The service calls `Cache.put` only when **all** of these hold:
1. The turn is standalone (§6).
2. `route ∈ {lookup, aggregate, project_info}`.
3. `answer_mode == llm`.
4. `LLMDone.finish_reason == stop` (not length, error or content_filter).
5. `grounding.grounded` is true, so `flags.unverified` is false. `CachedAnswer` enforces this a second time.

Never cached: refusals (they are already instant), retrieval-only and static-busy answers (degraded), and anything served from a cache (it is not re-put).

L2 put happens alongside L1 when a dense vector exists for the turn. The service always has one for standalone turns, because the L2 lookup computed it. Groq-authored answers are cached and keep `provider` and `model`, so the label survives a cache hit.

## 6. Standalone turns and history

- A turn is **standalone** iff the router carried no slot from the session (`carried_slots` is empty).
- Only standalone turns read from or write to L0, L1 and L2.
- To make the key complete, the prompt for a standalone turn **omits conversation history**; history (≤ 300 tokens) is included only when slots were carried. This affects `prompting.py` (Wave 2) and is flagged to dg-prompt-eval-lead.
- Follow-up turns ("cheaper ones?") always go to retrieval and the LLM. The multiturn set is 4 cases, so the cost is small.

## 7. Single-flight (quota protection)

If a standalone request misses all answer tiers while an identical exact key is already in flight, the follower awaits the leader's `asyncio.Future` for up to `SINGLE_FLIGHT_WAIT_S = 30`.
- If the leader produced a cacheable answer, the follower serves it as an L1 hit and streams it as one token.
- Otherwise, including timeout, the follower proceeds on its own.

This stops a burst of identical questions, such as the bench or a reviewer double-clicking, from spending several free-quota calls. The map is held on the event loop, and its entries are removed in `finally`.

## 8. Parameters

| Name | Value | Rule or evidence |
|---|---|---|
| L1 size / TTL | 2,000 / 24 h | PLAN §5; not tuned |
| L2 size / TTL | 2,000 / 24 h | Same as L1 |
| τ | **0.93** (Wave 3, was 0.92) | §4 rule (floor 0.88, disable above 0.98); C1, C2; `docs/tuning.md` §4 |
| L3 size | 1,024 | 1,024 × 384 × 4 B = 1.5 MB. Revisit only if `/stats` shows an L3 hit rate below the L1 hit rate (that would point to a key bug, not to size) |
| L4 size | 512 | Payload copies are about 5 KB × 20 hits, so about 50 MB worst case. Revisit if RSS on HF exceeds 2 GB |
| `SINGLE_FLIGHT_WAIT_S` | 30 | Above the `CHAIN_FIRST_TOKEN_BUDGET_S` (20 s) plus a typical 150-word stream |
| `CACHE_KEY_SCHEMA` | ck1 | Bump on any recipe change, which invalidates L0 |

## 9. Tests (written first by dg-backend-engineer, Wave 2)
- 3-bed vs 4-bed keys differ; Riyadh vs Jeddah; sale vs rent; `min_price` vs `max_price`; prompt_version or index_version change; language change.
- Under `normalise_for_key`, "2M", "2 million", "2,000,000" and "٢٠٠٠٠٠٠" give the same key, and "three" equals "3".
- "with a pool" vs "without a pool" gives different scopes (negation cue).
- Determinism: the same parts give the same key across processes (no `hash()`, no dict ordering).
- `filters_json` uses `SearchFilters.canonical_json()` only, never `model_dump_json()` (whose field order is not part of the contract).
