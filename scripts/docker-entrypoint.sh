#!/usr/bin/env bash
# Container entrypoint: run the health server alongside the synthesis worker.
#
# - Health server listens on $HEALTH_PORT (default 8080) for /health and /progress.
# - The worker is `scripts/run.py` with whatever args were passed to the container.
# - Signals: tini (PID 1) forwards SIGTERM here; we forward to both children.
#   run.py installs its own SIGTERM/SIGINT handler to persist progress before exit.

set -euo pipefail

HEALTH_PORT="${HEALTH_PORT:-8080}"
HEALTH_PID=""
WORKER_PID=""

shutdown() {
  trap - TERM INT
  if [[ -n "${WORKER_PID}" ]] && kill -0 "${WORKER_PID}" 2>/dev/null; then
    kill -TERM "${WORKER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${HEALTH_PID}" ]] && kill -0 "${HEALTH_PID}" 2>/dev/null; then
    kill -TERM "${HEALTH_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit 0
}

trap shutdown TERM INT

echo "[entrypoint] starting health server on :${HEALTH_PORT}"
python scripts/health_check.py --port "${HEALTH_PORT}" &
HEALTH_PID=$!

if [ "${SYNTH_HEALTH_ONLY:-0}" = "1" ]; then
  echo "[entrypoint] SYNTH_HEALTH_ONLY=1 — skipping worker; serving health only"
  echo "[entrypoint] (set SYNTH_HEALTH_ONLY=0 in .env and recreate to start the worker)"
  WORKER_PID="${HEALTH_PID}"  # wait on the health server instead
else
  echo "[entrypoint] starting synthesizer: scripts/run.py $*"
  python scripts/run.py "$@" &
  WORKER_PID=$!
fi

# Exit when the worker exits; propagate its status.
wait "${WORKER_PID}"
WORKER_STATUS=$?

if kill -0 "${HEALTH_PID}" 2>/dev/null; then
  kill -TERM "${HEALTH_PID}" 2>/dev/null || true
  wait "${HEALTH_PID}" 2>/dev/null || true
fi

exit "${WORKER_STATUS}"
