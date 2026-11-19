#!/usr/bin/env python3
"""Deploy to Hugging Face Space via upload_folder (Gradio SDK / ZeroGPU).

Reads HF_TOKEN and HF_SPACE_ID from environment only. No secrets in code.
Generates main.py and requirements.txt for the Space on the fly.

Usage:
  python scripts/deploy_space.py             # real deploy
  python scripts/deploy_space.py --dry-run   # list files and total size
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patterns that match what the Gradio SDK Space needs
ALLOW_PATTERNS = [
    "app/**",
    "common/**",
    "data/index/**",
    "main.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
]

IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
]

DELETE_PATTERNS = [
    "app/**",
    "common/**",
    "data/index/**",
    "main.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
]

GRADIO_SDK_VERSION = "6.32.0"

HF_FRONT_MATTER = (
    "---\n"
    "title: Wasalt RAG Chatbot\n"
    'emoji: "\U0001f3e0"\n'
    "sdk: gradio\n"
    f'sdk_version: "{GRADIO_SDK_VERSION}"\n'
    'python_version: "3.12"\n'
    "app_file: main.py\n"
    "pinned: false\n"
    "---\n"
)

MAIN_PY = '''\
"""HF Space wrapper: boots the FastAPI app under Gradio SDK + ZeroGPU."""

import os
import sys
import traceback

os.environ["GRADIO_SSR_MODE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/fastembed_cache")
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("RATE_LIMIT_XFF_HOP", "last")

try:
    import gradio as gr
    import spaces
    import uvicorn

    @spaces.GPU
    def _dummy_gpu():
        """Never called; keeps ZeroGPU scheduler happy."""
        return "ok"

    from app.api import app as fastapi_app

    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Wasalt RAG Chatbot")

    app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

    if __name__ == "__main__":
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=7860,
            workers=1,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )

except Exception:
    traceback.print_exc()
    sys.exit(1)
'''


def _build_requirements() -> str:
    """Build requirements.txt from pyproject.toml runtime deps plus Gradio."""
    import tomllib

    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]

    lines = [
        f"gradio=={GRADIO_SDK_VERSION}",
        "spaces>=0.52.0",
    ]
    for dep in deps:
        # Loosen pydantic pin for gradio[mcp] compat (requires <=2.12.5)
        if dep.startswith("pydantic=="):
            lines.append("pydantic>=2.11.10,<=2.12.5")
        elif dep.startswith("pydantic-settings=="):
            lines.append("pydantic-settings>=2.16.0,<3")
        else:
            lines.append(dep)
    return "\n".join(lines) + "\n"


def _collect_files(root: Path) -> list[Path]:
    """Collect files matching ALLOW_PATTERNS, respecting IGNORE_PATTERNS."""
    import fnmatch

    matched: list[Path] = []
    for pattern in ALLOW_PATTERNS:
        if "**" in pattern:
            glob_pat = pattern.replace("**", "**/*")
            for p in root.glob(glob_pat):
                if p.is_file():
                    matched.append(p)
        else:
            candidate = root / pattern
            if candidate.is_file():
                matched.append(candidate)

    filtered: list[Path] = []
    for p in matched:
        rel = str(p.relative_to(root)).replace("\\", "/")
        skip = False
        for ig in IGNORE_PATTERNS:
            if fnmatch.fnmatch(rel, ig):
                skip = True
                break
        if not skip:
            filtered.append(p)

    return sorted(set(filtered))


def _write_generated_files(root: Path) -> list[Path]:
    """Write main.py, requirements.txt, and update README.md. Returns paths."""
    written = []

    # main.py
    main_path = root / "main.py"
    main_path.write_text(MAIN_PY, encoding="utf-8")
    written.append(main_path)

    # requirements.txt
    req_path = root / "requirements.txt"
    req_path.write_text(_build_requirements(), encoding="utf-8")
    written.append(req_path)

    # README.md with front matter
    readme = root / "README.md"
    body = ""
    if readme.exists():
        body = readme.read_text(encoding="utf-8")
        if body.startswith("---"):
            end = body.find("---", 3)
            if end != -1:
                body = body[end + 3:].lstrip("\n")
    readme.write_text(HF_FRONT_MATTER + body, encoding="utf-8")
    written.append(readme)

    return written


def _cleanup_generated_files(root: Path) -> None:
    """Remove generated files that should not persist in the repo."""
    for name in ("main.py", "requirements.txt"):
        p = root / name
        if p.exists():
            p.unlink()


def dry_run(root: Path) -> None:
    """List files that would be uploaded and their total size."""
    _write_generated_files(root)
    try:
        files = _collect_files(root)
        total = 0
        for f in files:
            size = f.stat().st_size
            total += size
            rel = f.relative_to(root)
            print(f"  {rel}  ({size:,} bytes)")
        print(f"\nTotal: {len(files)} files, {total:,} bytes ({total / 1024 / 1024:.2f} MB)")
    finally:
        _cleanup_generated_files(root)


def deploy(root: Path) -> None:
    """Upload to HF Space."""
    from huggingface_hub import HfApi, upload_folder

    token = os.environ.get("HF_TOKEN", "")
    space_id = os.environ.get("HF_SPACE_ID", "")

    if not token or not space_id:
        print("ERROR: HF_TOKEN and HF_SPACE_ID must be set", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    _write_generated_files(root)
    try:
        upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(root),
            token=token,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            delete_patterns=DELETE_PATTERNS,
            commit_message="deploy from local",
        )
        print(f"Deployed to https://huggingface.co/spaces/{space_id}")
    finally:
        _cleanup_generated_files(root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy to HF Space")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and size without uploading",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run(REPO_ROOT)
    else:
        deploy(REPO_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
