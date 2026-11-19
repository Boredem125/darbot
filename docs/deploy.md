# Deployment Guide

## Prerequisites

The application deploys as a Gradio SDK Hugging Face Space with ZeroGPU
hardware. CI runs on GitHub Actions; deploy is triggered automatically
after CI passes on main. The deploy script generates `main.py` and
`requirements.txt` for the Space on the fly from `pyproject.toml`.

## GitHub Repository Settings

### Secrets (Settings > Secrets and variables > Actions > Secrets)

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token with write access to the Space repo. Used by the deploy workflow and the local deploy script. |

### Variables (Settings > Secrets and variables > Actions > Variables)

| Variable | Example | Purpose |
|---|---|---|
| `HF_SPACE_ID` | `DarGlobal/wasalt-rag` | The Hugging Face Space repository id (org/name). |
| `SPACE_URL` | `https://darglobal-wasalt-rag.hf.space` | Public URL of the Space, used by the keepalive workflow. |

## HF Space Secrets (Space Settings > Variables and secrets)

These keys are set only in the HF Space runtime environment. They never
appear in GitHub, CI logs, or the Docker image.

| Secret | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key for the LLM chain. |
| `GROQ_API_KEY` | Groq API key for the fallback LLM provider. |

## Local Deploy

Set the environment variables and run:

```bash
export HF_TOKEN="hf_..."
export HF_SPACE_ID="DarGlobal/wasalt-rag"
python scripts/deploy_space.py
```

To preview which files would be uploaded without deploying:

```bash
python scripts/deploy_space.py --dry-run
```

## Docker (local)

```bash
docker compose up app
```

The app listens on port 7860. An `.env` file is loaded if present but is
not required; the image defaults are sufficient for retrieval-only mode.
LLM features require `OPENROUTER_API_KEY` or `GROQ_API_KEY` in the
environment or `.env` file.
