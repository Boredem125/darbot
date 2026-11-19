"""Dependency-rule enforcement (docs/contracts/dependency-rules.md).

Uses *only* `ast` and stdlib. Runs in < 1 s, needs no network or keys.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Edge(NamedTuple):
    importer: str  # module name
    imported: str  # module name
    lineno: int
    filepath: str  # relative to REPO_ROOT


@dataclass(frozen=True)
class Violation:
    rule: str
    edge: Edge

    def render(self) -> str:
        return f"{self.rule} {self.edge.filepath}:{self.edge.lineno} {self.edge.importer} -> {self.edge.imported}"


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------

_LAYER_PREFIXES: list[tuple[str, str]] = [
    ("app.domain.ports", "domain.ports"),
    ("app.domain", "domain"),
    ("app.adapters", "adapters"),
    ("app.service", "service"),
    ("app.api", "api"),
    ("app.telemetry", "telemetry"),
    ("app.config", "config"),
    ("common", "common"),
    ("collect", "collect"),
    ("organize", "organize"),
]


def _layer(module: str) -> str | None:
    """Return the layer name for a module, or None if not internal."""
    for prefix, layer in _LAYER_PREFIXES:
        if module == prefix or module.startswith(prefix + "."):
            return layer
    if module == "app" or module.startswith("app."):
        return "app.root"
    return None


def _is_internal(module: str) -> bool:
    top = module.split(".")[0]
    return top in {"common", "collect", "organize", "app", "scripts", "eval", "tests"}


def _is_stdlib(module: str) -> bool:
    top = module.split(".")[0]
    return top in sys.stdlib_module_names


def _is_tooling(module: str) -> bool:
    top = module.split(".")[0]
    return top in {"scripts", "eval", "tests"}


# ---------------------------------------------------------------------------
# AST walker: collect import edges
# ---------------------------------------------------------------------------

_IO_STDLIB_BANNED_IN_COMMON = {
    "socket", "ssl", "http", "urllib", "subprocess", "sqlite3",
    "shutil", "tempfile", "asyncio", "multiprocessing",
}


def _resolve_relative(importer_package: str, name: str, level: int) -> str:
    """Resolve a relative import to an absolute module name."""
    parts = importer_package.split(".")
    if level > len(parts):
        return name  # malformed, let it pass
    base_parts = parts[: len(parts) - level + 1] if level <= len(parts) else []
    # for `from . import x`, level=1, name might be empty or x
    # for `from ..ports import P`, level=2, name="ports"
    # base is the package minus (level-1) components
    base_parts = parts[: max(0, len(parts) - level + 1)]
    if level > 0:
        base_parts = parts[: max(0, len(parts) - (level - 1))]
        # from . -> same package; from .. -> parent
        base_parts = parts[: len(parts) - level]
    if name:
        return ".".join(base_parts + [name]) if base_parts else name
    return ".".join(base_parts) if base_parts else ""


def _file_to_module(filepath: Path, root: Path) -> str:
    rel = filepath.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _module_package(module: str) -> str:
    parts = module.rsplit(".", 1)
    return parts[0] if len(parts) > 1 else module


def collect_edges(root: Path) -> list[Edge]:
    """Walk all .py files in common/, collect/, organize/, app/ and collect import edges."""
    edges: list[Edge] = []
    source_dirs = ["common", "collect", "organize", "app"]
    for sdir in source_dirs:
        dirpath = root / sdir
        if not dirpath.is_dir():
            continue
        for pyfile in sorted(dirpath.rglob("*.py")):
            if "__pycache__" in pyfile.parts:
                continue
            if ".venv" in pyfile.parts:
                continue
            module = _file_to_module(pyfile, root)
            package = _module_package(module)
            relpath = str(pyfile.relative_to(root)).replace("\\", "/")
            try:
                source = pyfile.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(pyfile))
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        edges.append(Edge(module, alias.name, node.lineno, relpath))

                elif isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        mod_base = ""
                    else:
                        mod_base = node.module

                    if node.level and node.level > 0:
                        resolved_base = _resolve_relative(package, mod_base, node.level)
                    else:
                        resolved_base = mod_base

                    if resolved_base:
                        edges.append(Edge(module, resolved_base, node.lineno, relpath))

                    # Also add resolved_base.name for each imported name
                    for alias in (node.names or []):
                        if alias.name == "*":
                            continue
                        full = f"{resolved_base}.{alias.name}" if resolved_base else alias.name
                        edges.append(Edge(module, full, node.lineno, relpath))

                elif isinstance(node, ast.Call):
                    # importlib.import_module("literal") or __import__("literal")
                    func = node.func
                    is_import_module = (
                        isinstance(func, ast.Attribute)
                        and func.attr == "import_module"
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "importlib"
                    )
                    is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
                    if (is_import_module or is_dunder_import) and node.args:
                        arg = node.args[0]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            edges.append(Edge(module, arg.value, node.lineno, relpath))
                        else:
                            # Non-literal: R0 violation if in common or app.domain
                            importer_layer = _layer(module)
                            if importer_layer in ("common", "domain", "domain.ports"):
                                edges.append(Edge(module, "__dynamic_import__", node.lineno, relpath))

                    # open() calls banned in common/
                    if isinstance(func, ast.Name) and func.id == "open":
                        layer = _layer(module)
                        if layer == "common":
                            edges.append(Edge(module, "__builtin_open__", node.lineno, relpath))

    return edges


# ---------------------------------------------------------------------------
# Rule checker
# ---------------------------------------------------------------------------

_VENDOR_SDKS = {"qdrant_client", "fastembed", "openai", "duckdb", "onnxruntime"}

# R1: common third-party whitelist
_COMMON_3P = {"pydantic", "pydantic_core", "typing_extensions"}

# R6: service must not use vendor SDKs (checked via R9)
# R7: api must not use vendor SDKs (checked via R9)

# allowed[layer] = set of internal layers the importer may import
_ALLOWED_INTERNAL: dict[str, set[str]] = {
    "common": set(),  # only itself
    "collect": {"common"},
    "organize": {"common"},
    "domain.ports": {"common", "domain", "domain.ports"},
    "domain": {"common", "domain", "domain.ports"},
    "adapters": {"common", "domain.ports"},
    "service": {"common", "domain", "domain.ports", "telemetry", "config"},
    "api": {"common", "service", "adapters", "telemetry", "config", "domain.ports"},
    "telemetry": {"common"},
    "config": {"common"},
    "app.root": {"common"},
}


def check(edges: list[Edge]) -> list[Violation]:
    """Check all edges against the dependency rules. Returns all violations."""
    violations: list[Violation] = []

    for edge in edges:
        importer_layer = _layer(edge.importer)
        if importer_layer is None:
            continue  # not internal source

        imported = edge.imported

        # R0: dynamic import detection
        if imported == "__dynamic_import__":
            violations.append(Violation("R0", edge))
            continue

        # open() in common
        if imported == "__builtin_open__":
            violations.append(Violation("R1", edge))
            continue

        # Classify the imported module
        imported_top = imported.split(".")[0]
        is_internal = _is_internal(imported)
        is_stdlib_mod = _is_stdlib(imported)
        is_tooling_mod = _is_tooling(imported)
        imported_layer = _layer(imported)

        # --- R10: app.* must never import collect, organize, scripts, eval, tests ---
        if (
            importer_layer not in ("common", "collect", "organize")
            and imported_top in ("collect", "organize", "scripts", "eval", "tests")
        ):
            violations.append(Violation("R10", edge))
            continue

        # --- R11: common, collect, organize must never import app ---
        if importer_layer in ("common", "collect", "organize") and imported_top == "app":
            violations.append(Violation("R11", edge))
            continue

        # --- R12: collect and organize must not import each other ---
        if importer_layer == "collect" and imported_top == "organize":
            violations.append(Violation("R12", edge))
            continue
        if importer_layer == "organize" and imported_top == "collect":
            violations.append(Violation("R12", edge))
            continue

        # --- R9: vendor containment ---
        if imported_top in _VENDOR_SDKS:
            if importer_layer not in ("adapters", "organize", "collect"):
                violations.append(Violation("R9", edge))
                continue
            else:
                continue  # vendor in allowed layer, ok

        # --- R1: common I/O ban (stdlib) ---
        if importer_layer == "common" and is_stdlib_mod and imported_top in _IO_STDLIB_BANNED_IN_COMMON:
            violations.append(Violation("R1", edge))
            continue

        # Internal import rules
        if is_internal and not is_tooling_mod:
            imported_l = imported_layer
            if imported_l is None:
                # imported is an internal module but can't be classified to a layer
                # (shouldn't happen for valid imports)
                continue

            # Self-import always allowed
            if imported_l == importer_layer:
                continue

            allowed = _ALLOWED_INTERNAL.get(importer_layer, set())
            if imported_l not in allowed:
                # Determine the most specific rule
                rule = _rule_for(importer_layer)
                violations.append(Violation(rule, edge))
                continue

        # Third-party rules
        if not is_internal and not is_stdlib_mod:
            # R1: common third-party whitelist
            if importer_layer == "common" and imported_top not in _COMMON_3P:
                violations.append(Violation("R1", edge))
                continue

            # R4: domain and domain.ports allow only stdlib
            if importer_layer in ("domain", "domain.ports") and imported_top not in _COMMON_3P:
                # Actually R4 says no third-party at all (stdlib only), but pydantic
                # is not allowed either because domain uses it through common.schemas
                violations.append(Violation("R4", edge))
                continue

        # R1 for common: no third-party except whitelist (already above)
        # R4 for domain: no third-party (already above)

    return violations


def _rule_for(layer: str) -> str:
    """Map layer to its primary rule id."""
    return {
        "common": "R1",
        "collect": "R2",
        "organize": "R3",
        "domain.ports": "R4",
        "domain": "R4",
        "adapters": "R5",
        "service": "R6",
        "api": "R7",
        "telemetry": "R8",
        "config": "R8",
        "app.root": "R8",
    }.get(layer, "R0")


# ---------------------------------------------------------------------------
# Main test: the real repo
# ---------------------------------------------------------------------------


def test_repo_obeys_dependency_rules():
    edges = collect_edges(REPO_ROOT)
    # Non-vacuous: must find at least these two files
    importers = {e.importer for e in edges}
    assert "common.schemas" in importers, "walk did not find common/schemas.py"
    assert "app.domain.ports" in importers, "walk did not find app/domain/ports.py"

    violations = check(edges)
    assert not violations, "\n".join(v.render() for v in violations)


# ---------------------------------------------------------------------------
# Synthetic self-tests: one per rule
# ---------------------------------------------------------------------------


def _write_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Write a mini Python tree under tmp_path and return the root."""
    root = tmp_path / "repo"
    for relpath, code in files.items():
        fp = root / relpath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(textwrap.dedent(code), encoding="utf-8")
    return root


SYNTHETIC_CASES: list[tuple[str, dict[str, str], str | None]] = [
    # (description, files, expected_rule_id or None for clean)

    # Clean tree: should report nothing
    (
        "clean",
        {
            "common/__init__.py": "",
            "common/schemas.py": "import hashlib\nimport json\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import Route\n",
            "app/domain/router.py": "from common.schemas import Route\nfrom app.domain.ports import Embedder\n",
            "app/adapters/__init__.py": "",
            "app/adapters/embed.py": "from app.domain.ports import Embedder\nimport fastembed\n",
        },
        None,
    ),

    # R0: dynamic import in domain
    (
        "R0_dynamic_import",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import x\n",
            "app/domain/router.py": "import importlib\nmod = importlib.import_module(some_var)\n",
        },
        "R0",
    ),

    # R1: common imports forbidden stdlib (socket)
    (
        "R1_io_stdlib",
        {
            "common/__init__.py": "",
            "common/schemas.py": "import socket\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import x\n",
        },
        "R1",
    ),

    # R1: common uses open()
    (
        "R1_open_call",
        {
            "common/__init__.py": "",
            "common/schemas.py": "f = open('x.txt')\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import x\n",
        },
        "R1",
    ),

    # R1: common imports disallowed third-party
    (
        "R1_third_party",
        {
            "common/__init__.py": "",
            "common/schemas.py": "import requests\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import x\n",
        },
        "R1",
    ),

    # R2: collect imports app (violates R11 which applies to collect)
    # Actually R2 says collect may import common. R11 says collect must not import app.
    # We use R12 for collect->organize.
    (
        "R2_collect_imports_domain",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "collect/__init__.py": "",
            "collect/scraper.py": "from app.domain.router import route\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/router.py": "x = 1\n",
            "app/domain/ports.py": "x = 1\n",
        },
        "R11",
    ),

    # R3: organize imports app (violates R11)
    (
        "R3_organize_imports_app",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "organize/__init__.py": "",
            "organize/builder.py": "from app.domain.router import route\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/router.py": "x = 1\n",
            "app/domain/ports.py": "x = 1\n",
        },
        "R11",
    ),

    # R4: domain imports third-party (pydantic directly)
    (
        "R4_domain_third_party",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import x\n",
            "app/domain/router.py": "import httpx\n",
        },
        "R4",
    ),

    # R5: adapters imports domain (not domain.ports)
    (
        "R5_adapters_imports_domain",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "x = 1\n",
            "app/domain/router.py": "x = 1\n",
            "app/adapters/__init__.py": "",
            "app/adapters/embed.py": "from app.domain.router import route\n",
        },
        "R5",
    ),

    # R6: service imports adapters
    (
        "R6_service_imports_adapters",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "x = 1\n",
            "app/service/__init__.py": "",
            "app/service/chat.py": "from app.adapters.embed import x\n",
            "app/adapters/__init__.py": "",
            "app/adapters/embed.py": "x = 1\n",
        },
        "R6",
    ),

    # R7: api imports domain (non-ports)
    (
        "R7_api_imports_domain",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "x = 1\n",
            "app/domain/router.py": "x = 1\n",
            "app/api/__init__.py": "",
            "app/api/main.py": "from app.domain.router import route\n",
        },
        "R7",
    ),

    # R8: config imports adapters
    (
        "R8_config_imports_adapters",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "x = 1\n",
            "app/config/__init__.py": "",
            "app/config/settings.py": "from app.adapters.embed import x\n",
            "app/adapters/__init__.py": "",
            "app/adapters/embed.py": "x = 1\n",
        },
        "R8",
    ),

    # R9: domain imports vendor SDK
    (
        "R9_domain_vendor",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import x\n",
            "app/domain/router.py": "import qdrant_client\n",
        },
        "R9",
    ),

    # R10: app imports tests
    (
        "R10_app_imports_tests",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "from common.schemas import x\n",
            "app/domain/router.py": "from tests.fakes import fake_adapters\n",
        },
        "R10",
    ),

    # R11: common imports app
    (
        "R11_common_imports_app",
        {
            "common/__init__.py": "",
            "common/schemas.py": "from app.domain.ports import Embedder\n",
            "app/__init__.py": "",
            "app/domain/__init__.py": "",
            "app/domain/ports.py": "x = 1\n",
        },
        "R11",
    ),

    # R12: collect imports organize
    (
        "R12_collect_imports_organize",
        {
            "common/__init__.py": "",
            "common/schemas.py": "x = 1\n",
            "collect/__init__.py": "",
            "collect/scraper.py": "from organize.builder import x\n",
            "organize/__init__.py": "",
            "organize/builder.py": "x = 1\n",
        },
        "R12",
    ),
]


@pytest.mark.parametrize(
    "desc,files,expected_rule",
    SYNTHETIC_CASES,
    ids=[c[0] for c in SYNTHETIC_CASES],
)
def test_checker_catches(desc: str, files: dict[str, str], expected_rule: str | None, tmp_path: Path):
    root = _write_tree(tmp_path, files)
    edges = collect_edges(root)
    violations = check(edges)

    if expected_rule is None:
        assert not violations, "Expected clean but got:\n" + "\n".join(v.render() for v in violations)
    else:
        rule_ids = {v.rule for v in violations}
        assert expected_rule in rule_ids, (
            f"Expected rule {expected_rule} but got {rule_ids}.\n"
            + "\n".join(v.render() for v in violations)
        )
