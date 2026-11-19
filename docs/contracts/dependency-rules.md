# Dependency rules: test spec

Status: frozen in Wave 1a (18 Nov 2026). Owner: dg-lead-architect. Implementer:
dg-qa-engineer, in `tests/test_dependency_rules.py`. Source: PLAN section 4 and ADR-0001.

The test enforces the hexagonal layering statically. It must run in under 1 s, need no
network and no keys, and import none of the code it checks. It parses with `ast` only.

## What the test walks

- Every `*.py` file under `common/`, `collect/`, `organize/` and `app/`, recursively.
- It skips `__pycache__`, `.venv` and anything outside those four roots. `tests/`, `scripts/` and `eval/` are tooling: they may import anything, and nothing may import them.
- File path to module name: `app/domain/router.py` becomes `app.domain.router`, and `app/domain/__init__.py` becomes `app.domain`.

For each file it collects every import edge `(importer_module, imported_module, lineno)` from:
- `import a.b.c` gives `a.b.c`.
- `from a.b import c` gives `a.b` and also `a.b.c`. The second edge is there because `c` may be a submodule; the rule check uses the longest prefix that matches a layer.
- Relative imports (`from . import x`, `from ..ports import P`) are resolved against the importer's package before checking.
- Imports nested in functions, in `if TYPE_CHECKING:` blocks and in `try/except ImportError` all count. **Type-only imports are not exempt.**
- `importlib.import_module("<literal>")` and `__import__("<literal>")` with a string literal also count as edges. A non-literal argument inside `common` or `app.domain` is itself a violation (rule R0).

Each imported module is then classified:
- **internal**: its top-level package is `common`, `collect`, `organize`, `app`, `scripts`, `eval` or `tests`.
- **stdlib**: its top-level name is in `sys.stdlib_module_names`.
- **third-party**: everything else.

## Layers

The most specific prefix wins:

| Layer | Module prefix |
|---|---|
| `common` | `common` |
| `collect` | `collect` |
| `organize` | `organize` |
| `domain.ports` | `app.domain.ports` |
| `domain` | `app.domain` (all other modules) |
| `adapters` | `app.adapters` |
| `service` | `app.service` |
| `api` | `app.api` |
| `telemetry` | `app.telemetry` |
| `config` | `app.config` |
| `app.root` | `app` (the package `__init__` and any other `app.*` module not listed above) |

## Rules (what the test asserts)

`allowed internal` lists the layers the importer may import. A layer may always import
itself. Any internal edge not listed is a violation.

| Id | Importer | Allowed internal | Third-party allowed | Source |
|---|---|---|---|---|
| R1 | `common` | none (only itself) | `pydantic`, `pydantic_core`, `typing_extensions` | PLAN §4 |
| R2 | `collect` | `common` | any | PLAN §4 |
| R3 | `organize` | `common` | any | PLAN §4 |
| R4 | `domain`, `domain.ports` | `common`, `domain`, `domain.ports` | **none** (stdlib only) | PLAN §4 |
| R5 | `adapters` | `common`, `domain.ports` (only `domain.ports`, not the rest of `domain`) | any | PLAN §4 |
| R6 | `service` | `common`, `domain`, `domain.ports`, `telemetry`, `config` | none of the vendor SDKs in R9 | PLAN §4; telemetry and config are architect clarifications |
| R7 | `api` (composition root) | `common`, `service`, `adapters`, `telemetry`, `config`, `domain.ports` | any except the vendor SDKs in R9 | Architect clarification: something has to construct the adapters, and `api` is the only module the runtime starts (`uvicorn app.api:app`) |
| R8 | `telemetry`, `config`, `app.root` | `common` | any except the vendor SDKs in R9 | Architect clarification |
| R9 | every module | | `qdrant_client`, `fastembed`, `openai`, `duckdb` and `onnxruntime` may be imported only from `adapters`, `organize` and `collect` | Vendor containment (ADR-0001, ADR-0002) |
| R10 | every `app.*` module | must never import `collect`, `organize`, `scripts`, `eval` or `tests` | | PLAN §4 (in bold there) |
| R11 | `common`, `collect`, `organize` | must never import `app` | | Offline/online split (ADR-0003) |
| R12 | `collect` and `organize` | must not import each other | | Architect addition: stages hand over files (`data/raw` → `data/processed`). Shared helpers go in `common`. |

R1 adds an I/O ban for `common`. Even stdlib imports from this set are violations there:
`socket`, `ssl`, `http`, `urllib`, `subprocess`, `sqlite3`, `shutil`, `tempfile`, `asyncio`
and `multiprocessing`. A call to the builtin `open(` anywhere under `common/` is also a
violation, found by walking `ast.Call` nodes whose `func` is `Name(id="open")`.

R4 means `app/domain` may not import `pydantic` directly. It uses pydantic models only
through `common.schemas`.

## Test shape

```python
# tests/test_dependency_rules.py  (dg-qa-engineer writes this)
def collect_edges(root: Path) -> list[Edge]: ...          # pure; takes a repo root
def check(edges: list[Edge]) -> list[Violation]: ...      # pure; returns every violation

def test_repo_obeys_dependency_rules():
    violations = check(collect_edges(REPO_ROOT))
    assert not violations, "\n".join(v.render() for v in violations)

@pytest.mark.parametrize("case", SYNTHETIC_CASES)
def test_checker_catches(case, tmp_path): ...             # self-test: a fake tree per rule
```

- `Violation.render()` gives `"R5 app/adapters/caches.py:12 app.adapters.caches -> app.domain.router"`. The output lists **all** violations, not just the first.
- **Self-test (required).** For each rule from R1 to R12, plus R0, write a minimal fake tree under `tmp_path` that breaks exactly that rule, and assert that `check` reports exactly that rule id. Also include one clean tree that must report nothing. This proves the checker is not vacuous: an empty walk would pass the main test.
- Also assert that the walk found at least `common/schemas.py` and `app/domain/ports.py`, so the main test is never vacuous either.
- No allow-list or `# noqa` escape hatch. An exception needs a contract change (`.orchestration/CONTRACT_CHANGES.md`) and an edit to this spec.

## Current state (Wave 1a)

`common/schemas.py` imports stdlib (`hashlib`, `json`, `uuid`, `collections.abc`, `enum`,
`typing`) and `pydantic`, so R1 passes. `app/domain/ports.py` imports stdlib and
`common.schemas`, so R4 passes.
