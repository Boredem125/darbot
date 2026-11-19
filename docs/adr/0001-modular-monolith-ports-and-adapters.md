# ADR-0001: Modular monolith in one container, with ports and adapters

- Status: Accepted
- Date: 18 Nov 2026
- Owner: dg-lead-architect
- Related: ADR-0002, ADR-0003, ADR-0008, `docs/contracts/dependency-rules.md`, `app/domain/ports.py`

## Context

The system scrapes two sites, builds a search index, and answers questions over it with a free LLM. One developer builds it in three days and ships it as a single Docker image on a Hugging Face Space (2 vCPU, 16 GB, one container, port 7860). The reviewers asked for the architecture to be laid out as the IBM AI Ladder: Collect, Organize, Analyze, Infuse.

Five things outside our code can fail or change under us: the LLM providers (OpenRouter, Groq), the vector store (Qdrant), the SQL engine (DuckDB), the caches, and the scraped sites. Each of them needs a fake in tests, because CI has no keys and no network access to providers. The latency work also needs a clean place to time every stage.

## Decision

1. **One deployable unit.** The runtime is a single Python process in a single container (see ADR-0008). Scraping and index building run offline in separate entry points and images (see ADR-0003).

2. **Modules map to the AI Ladder.**

   | Ladder rung | Package | Runs |
   |---|---|---|
   | Collect | `collect/` | Offline, in the scraper image (`Dockerfile.collect`) |
   | Organize | `organize/` | Offline, build time; writes `data/index/` |
   | Analyze | `app/domain/` (router, grounding, prompting), `app/adapters/` (retrieval, SQL) | Runtime |
   | Infuse | `app/service.py`, `app/api.py`, `app/static/` | Runtime |
   | Shared kernel | `common/` (pydantic contracts and pure helpers, no I/O) | Both halves |

3. **Ports and adapters for every external system.** `app/domain/ports.py` defines six `typing.Protocol` ports: `Embedder`, `Retriever`, `Reranker`, `StructuredQuery`, `LLM` and `Cache`. Adapters in `app/adapters/` implement them structurally. `ChatService` (`app/service.py`) is the only place where ports meet. `Reranker` is its own port, a sixth beyond the five in PLAN section 2, because whether to rerank is domain logic (flag-gated, skipped for Arabic) and the service must time it as `rerank_ms`.

4. **`app.api` is the composition root.** It is the only module the runtime starts (`uvicorn app.api:app`), so it constructs the adapters from `app.config` settings and hands them to `ChatService`. No other module constructs an adapter.

5. **The dependency rules are enforced by a test, not by convention.** `tests/test_dependency_rules.py` parses every module with `ast` and fails on any violation. The full spec is `docs/contracts/dependency-rules.md`. In short:

   | Id | Importer | May import (internal) |
   |---|---|---|
   | R1 | `common` | nothing internal; third-party limited to pydantic; no I/O modules and no `open(` |
   | R2, R3 | `collect`, `organize` | `common` |
   | R4 | `app.domain` (incl. `ports`) | `common`, `app.domain`; stdlib only, no third-party |
   | R5 | `app.adapters` | `common`, `app.domain.ports` (not the rest of `domain`) |
   | R6 | `app.service` | `common`, `app.domain`, `app.telemetry`, `app.config` |
   | R7 | `app.api` | `common`, `app.service`, `app.adapters`, `app.domain.ports`, `app.telemetry`, `app.config` |
   | R8 | `app.telemetry`, `app.config`, other `app.*` | `common` |
   | R9 | any | `qdrant_client`, `fastembed`, `openai`, `duckdb`, `onnxruntime` only from `adapters`, `organize`, `collect` |
   | R10 | any `app.*` | never `collect`, `organize`, `scripts`, `eval`, `tests` |
   | R11 | `common`, `collect`, `organize` | never `app` |
   | R12 | `collect`, `organize` | not each other; they hand over files |

   R6's `telemetry` and `config`, all of R7, R9 and R12 are clarifications added in Wave 1a. PLAN section 4 leaves them open. They were approved by T0 when the contracts froze.

6. **Contracts are frozen before parallel work.** `common/schemas.py`, `app/domain/ports.py` and `docs/contracts/*` change only through `.orchestration/CONTRACT_CHANGES.md`, with architect approval and an orchestrator broadcast.

## Alternatives rejected

- **Microservices (API, retriever, LLM gateway as separate services).** One developer, one region, one 2-vCPU box, three days. Network hops would add latency we are trying to measure away, and HF Spaces runs one container per Space.
- **A layered monolith without ports.** Cheaper to write, but provider SDKs would leak into the service. Fakes would then need monkeypatching, and the CI tests that run without keys would be brittle.
- **An agent framework as the backbone (LangChain, LlamaIndex query engine).** See ADR-0002. It hides the stage boundaries that the latency targets need.
- **Letting `service` construct adapters.** That breaks R6 (the service would import vendor SDKs through the adapters) and makes fakes harder to inject. A composition root is the standard answer.

## Consequences

- Every external system can be swapped for a fake with no patching. Each rung of the degradation ladder (PLAN section 5) is testable in CI without keys.
- Vendor SDK imports are confined to `adapters`, `organize` and `collect`. A provider or store swap touches one adapter file.
- `app.domain` cannot import pydantic directly, only through `common.schemas`. Domain code stays plain Python and is fast to test.
- The rules cost one test file and some care when a worker wants a shortcut import. There is no `# noqa` escape hatch; an exception is a contract change.
- The README diagram and the package tree say the same thing, because the ladder is the package layout.

## Trade-off accepted

Scaling out later needs a Redis instance (sessions, caches, rate limits, quota) and a Qdrant server in place of the in-memory store (ADR-0008). For a demo with one developer and one region, that is acceptable. The ports keep that migration to new adapters, with no change to the service.
