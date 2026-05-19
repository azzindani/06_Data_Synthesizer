# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# Data Synthesizer — production container.
#
# Long-running worker that drives LLM providers (Gemini / OpenRouter / OpenAI)
# in a continuous synthesis loop, plus an HTTP health/progress endpoint on :8080.
#
# Build:        docker build -t data-synthesizer:latest .
# Run (CLI):    docker run --rm --env-file .env data-synthesizer:latest --dry-run
# Run (stack):  docker compose up -d --build
#
# Persistence (bind-mount these to keep state across restarts):
#   ./output     /app/output       generated chunks + synthesis_progress.json
#   ./logs       /app/logs         rotating log files
#   ./config.yaml + ./configs/     read-only config (edit on host, no rebuild)
# ─────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11

# ── Stage 1: builder — install deps into a virtualenv ──────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build deps for pyarrow / scikit-learn wheels. Removed in the runtime stage.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HEALTH_PORT=8080 \
    CONFIG_FILE=config.yaml

# tini for clean signal handling (SIGTERM → graceful shutdown in run.py);
# curl for the HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tini curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system synth \
 && useradd  --system --gid synth --home /app --shell /bin/bash synth

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Source + scripts + configs. Bind-mount config.yaml/configs at runtime to
# override without rebuilding.
COPY --chown=synth:synth src     ./src
COPY --chown=synth:synth scripts ./scripts
COPY --chown=synth:synth configs ./configs
COPY --chown=synth:synth config.yaml ./config.yaml

RUN chmod +x scripts/docker-entrypoint.sh \
 && mkdir -p /app/output /app/logs \
 && chown -R synth:synth /app

USER synth
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${HEALTH_PORT}/health" >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "scripts/docker-entrypoint.sh"]
CMD ["--config", "config.yaml"]
