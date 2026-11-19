# Architecture decision records

Each record follows the same shape: context, decision, alternatives rejected, consequences, and the trade-off accepted. The source is PLAN section 2. Owner: dg-lead-architect. All eight were accepted on 18 Nov 2026.

| ADR | Title | Decision in one line | Trade-off accepted |
|---|---|---|---|
| [0001](0001-modular-monolith-ports-and-adapters.md) | Modular monolith, ports and adapters | One container; packages map to the AI Ladder; six Protocol ports; `app.api` is the composition root; dependency rules enforced by a test | Scale-out later needs Redis plus a Qdrant server |
| [0002](0002-frameworkless-retrieval.md) | Retrieval without a framework | qdrant-client Query API + fastembed + openai SDK at runtime; llama-index-core only at build time | We hand-write the glue in exchange for exact per-stage timing and a smaller image |
| [0003](0003-offline-online-split-versioned-artifacts.md) | Offline/online split via versioned artifacts | Index built offline, shipped in the image, loaded into in-memory Qdrant + DuckDB; float16 vectors; files under 10 MB; content-hashed `index_version`; `prompt_version` lifecycle; fail fast | Fresher data needs a rebuild and redeploy; listings shown "as of `scraped_at`" |
| [0004](0004-one-llm-call-rule-based-routing.md) | One LLM call per turn, rule-based routing | Regex and dictionary router for language, route, filters and slots; refusals and aggregates computed without extra LLM calls | Unusual phrasings get misrouted; a hybrid-retrieval fallback route mitigates it |
| [0005](0005-llm-router-provider-chain.md) | LLMRouter provider chain | OpenRouter free models first, then Groq; failover only before the first token; circuit breaker; quota ledger | Groq answers, labelled, when the free OpenRouter cap is hit |
| [0006](0006-five-tier-answer-caching.md) | Five cache tiers | L0 warm file, L1 exact TTL, L2 partitioned semantic, L3 embedding LRU, L4 retrieval LRU; versioned keys; grounded, untruncated answers only | Some cacheable answers are lost; wrong cached answers are never served |
| [0007](0007-deterministic-grounding-check.md) | Deterministic grounding check | Numbers, entities and citations checked on every answer; LLM judge offline only | Misses paraphrased claims with no number or entity |
| [0008](0008-single-process-stateful-runtime.md) | Single-process stateful runtime | One uvicorn worker; sessions, caches, limits and quota in memory; ONNX on a 2-thread executor | State is lost on restart; the baked L0 warm cache softens it |

## Related contracts and specs

- `docs/contracts/sse-v1.md`: the wire contract between the API and the UI.
- `docs/contracts/manifest.md`: the file contract between the offline and online halves.
- `docs/contracts/dependency-rules.md`: the import rules from ADR-0001, as a test spec.
- `docs/specs/` (dg-ml-lead): retrieval, LLM router, caching, and concurrency/SLO specs. They own the tunable numbers that the ADRs only name.

## Changing a decision

Write a new ADR that supersedes the old one, and mark the old one `Superseded by ADR-NNNN`. If the change touches a frozen contract (`common/schemas.py`, `app/domain/ports.py`, `docs/contracts/*`), record it in `.orchestration/CONTRACT_CHANGES.md` and get the orchestrator to broadcast it first.
