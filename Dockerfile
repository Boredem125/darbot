# Dockerfile — runtime image for DarGlobal/Wasalt RAG chatbot
# Multi-stage build on python:3.12-slim. See ADR-0003, ADR-0008, PLAN section 10.
# No secrets in any ARG, ENV, or layer.

# ---------------------------------------------------------------------------
# Stage 1: builder — install deps, bake FastEmbed models
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Model ids as build args; defaults must equal common.schemas constants (CC-6).
ARG EMBED_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ARG SPARSE_MODEL="Qdrant/bm25"
ARG RERANK_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"

ENV FASTEMBED_CACHE_PATH=/opt/fastembed

# Create a venv for clean copy to final stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only pyproject.toml first for layer caching of dependencies
COPY pyproject.toml /tmp/pyproject.toml

# Extract runtime dependencies from pyproject.toml and install them.
# We install only [project].dependencies, not build/collect/dev extras,
# and not the project package itself.
RUN python -c "\
import tomllib, pathlib;\
data = tomllib.loads(pathlib.Path('/tmp/pyproject.toml').read_text());\
deps = data['project']['dependencies'];\
pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(deps) + '\n')\
" && pip install --no-cache-dir -r /tmp/requirements.txt \
   && rm /tmp/requirements.txt /tmp/pyproject.toml

# Bake FastEmbed models: run one inference with each model so ONNX files are
# complete and verified. This proves the models are usable offline.
RUN python -c "\
import os;\
os.environ['FASTEMBED_CACHE_PATH'] = '/opt/fastembed';\
from fastembed import TextEmbedding;\
m = TextEmbedding(model_name='${EMBED_MODEL}');\
list(m.embed(['warm-up']));\
print('Dense model baked.');\
from fastembed import SparseTextEmbedding;\
s = SparseTextEmbedding(model_name='${SPARSE_MODEL}');\
list(s.embed(['warm-up']));\
print('Sparse model baked.');\
from fastembed.rerank.cross_encoder import TextCrossEncoder;\
r = TextCrossEncoder(model_name='${RERANK_MODEL}');\
list(r.rerank('query', ['document']));\
print('Reranker model baked.');\
"

# ---------------------------------------------------------------------------
# Stage 2: final — slim runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Re-declare ARGs so they are available in this stage for ENV
ARG EMBED_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ARG SPARSE_MODEL="Qdrant/bm25"
ARG RERANK_MODEL="Xenova/ms-marco-MiniLM-L-6-v2"

# Create non-root user (HF Spaces convention: UID 1000)
RUN useradd -m -u 1000 -s /bin/bash user

# Copy venv and fastembed cache from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed /opt/fastembed

# Ensure fastembed cache is readable by UID 1000
RUN chmod -R a+rX /opt/fastembed

# Environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    HF_HOME=/opt/fastembed \
    HF_HUB_OFFLINE=1 \
    PORT=7860 \
    EMBED_MODEL="${EMBED_MODEL}" \
    SPARSE_MODEL="${SPARSE_MODEL}" \
    RERANK_MODEL="${RERANK_MODEL}" \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /home/user/app

# Copy only what the runtime needs: common/, app/, data/index/, pyproject.toml
# The .dockerignore excludes collect/, organize/, tests/, .env*, etc.
COPY --chown=1000:1000 common/ ./common/
COPY --chown=1000:1000 app/ ./app/
COPY --chown=1000:1000 data/index/ ./data/index/
COPY --chown=1000:1000 pyproject.toml ./pyproject.toml

USER 1000

EXPOSE 7860

# Health check using stdlib (no curl in slim image).
# Interval 30s, timeout 5s, start-period 60s (index load + model warm-up), retries 3.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)" || exit 1

# ADR-0008: exec form, one worker, proxy headers.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
