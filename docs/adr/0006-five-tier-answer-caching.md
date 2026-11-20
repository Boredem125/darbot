# ADR-0006: Five cache tiers, with versioned keys and grounded-only storage

- Status: Accepted
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0003, ADR-0004, ADR-0007, ADR-0008, `app/domain/ports.py` (`Cache`, `CacheHit`), `common/schemas.py` (`CachedAnswer`, `WarmCacheRecord`), `docs/specs/caching.md` (dg-ml-lead)

## Context

Latency and caching must be measured, not claimed. The OpenRouter quota (50 per day) makes every avoided LLM call valuable. Reviewers are likely to ask the same demo questions, and the probe list in PLAN section 12 is fixed.

A cache in a RAG system can also do harm. A wrong cached answer gets served again and again, quickly and with confidence. Two queries that differ only in a number ("3-bed" and "4-bed", "under 2M" and "under 3M") are near-identical as embeddings but need different answers. A prompt or index change makes old answers stale.

## Decision

1. **Five tiers.** Three are answer tiers (`AnswerCacheTier`); two are inner tiers (`InnerCacheTier`):

   | Tier | Store | Holds | Hit rule | Reported in |
   |---|---|---|---|---|
   | L0 | `data/index/warm_cache.jsonl`, loaded at startup, read-only | Whole answers | Exact key | `start.cache_tier`, `end.cache_tier` |
   | L1 | In-memory `TTLCache` | Whole answers | Exact key; TTL 24 h, at most 2,000 entries | same |
   | L2 | In-process numpy matrix per `scope` (`app/adapters/caches.py`; ADR-0002 amendment 1) | Whole answers with their query embedding | Cosine at least τ, within the same `scope`, standalone turns only | same, plus `CacheHit.similarity` |
   | L3 | LRU | Query embeddings (dense + sparse) | Exact normalised query | `end.cache_hits` contains `L3` |
   | L4 | LRU | Retrieval results | Exact (query, filters) | `end.cache_hits` contains `L4` |

   The lookup order is L0, L1, L2. It runs after the router has produced the language, route and filters (the key needs them), and before any retrieval or LLM call. L2 needs the dense query embedding, so an L2 lookup goes through the L3-wrapped embedder first. A hit on L0 to L2 streams the stored answer as tokens with `answer_mode = cache` and `finish_reason = cached`. L3 and L4 wrap the `Embedder` and `Retriever` adapters, and their hits are reported through the `cached` flags on `QueryEmbedding` and `RetrievalResult`.

2. **The exact key includes everything that changes the answer:** `index_version` + `prompt_version` + normalised query + `SearchFilters.canonical_json()` + the numbers extracted from the query. The key is built in `app/domain/` (pure). The `Cache` port treats it as opaque. dg-ml-lead fixes the exact function and the normaliser in `docs/specs/caching.md`. L0 and L1 must use the same function, because `WarmCacheRecord.key` is looked up as an L1 key.

3. **The semantic tier is partitioned.** The L2 `scope` is the exact key minus the normalised query. A semantic hit can match only an entry with the same index, prompt, filters and numbers, so "3-bed under 2M" can never return the answer for "4-bed under 3M". L2 is used only for standalone turns. A turn that inherited slots from the session is not semantically cached, because the visible query does not carry its full meaning.

4. **Only grounded, untruncated answers are stored.** The service calls `Cache.put` only when the grounding check passes (ADR-0007) and `finish_reason == stop`. `CachedAnswer` rejects an ungrounded report as a second layer, so the rule holds even if a caller gets it wrong. Refusals, retrieval-only and busy answers are never cached.

5. **τ is tuned, not guessed.** It is set on `cache_pairs.jsonl` (15 true paraphrase pairs and 15 near-misses) as **the lowest value that gives zero false hits**. The initial value and the tuning run belong to dg-ml-lead. `SEMANTIC_CACHE_ENABLED=0` turns L2 off entirely; `get_semantic` then returns `None`.

6. **Invalidation is by versioning, not by purging.** A new index or prompt changes the key prefix, so old L1 and L2 entries can never match and expire on their own. L0 is checked at startup (ADR-0003, `prompt_version` lifecycle).

## Alternatives rejected

- **Exact-match cache only.** Safe, but it misses the common case of the same question in different words. L2 with partitioning and a zero-false-hit τ captures paraphrases without the number problem.
- **An unpartitioned semantic cache (GPTCache style).** Embeddings of "3-bed villas under 2M in Riyadh" and "4-bed villas under 3M in Riyadh" are very close. Without the scope, a high τ still risks serving the wrong numbers.
- **Caching every answer, flagged or not.** Faster to fill, but an unverified answer would be replayed many times. The SSE contract would also have to carry "cached but unverified", which the UI cannot explain well.
- **Redis as the cache store.** A second process to run in the Space, and a network hop on the hit path. The single-process runtime (ADR-0008) makes it unnecessary for now.
- **Time-based invalidation of all entries on deploy.** Versioned keys give the same safety without a purge step, and they cover the warm cache baked into the image.

## Consequences

- The cache-hit path (L0 or L1) skips embedding, retrieval and the LLM. It is the path that must meet the p95 < 50 ms SLO.
- Every hit is attributable. The `end` event names the tier, the inner hits, and the original provider and model. `/stats` reports hit rates per tier.
- L0 makes the first requests after a cold start fast and quota-free for the probe questions.
- Some answers that were correct are never cached, because grounding has false negatives (ADR-0007) or the turn was not standalone.
- L1 and L2 are lost on restart (ADR-0008). L0 survives because it ships in the image.

## Trade-off accepted

Some cacheable answers are lost, because ungrounded, truncated and follow-up answers are not stored. Wrong cached answers are never served.
