#!/usr/bin/env bash

# Launch the FastAPI app with the project uv environment.
# Usage:
#   scripts/dev_run.sh [extra uvicorn args]
# Env overrides:
#   HOST (default 127.0.0.1)
#   PORT (default 8000)
#   APP_MODULE (default app.main:app)
#   UVICORN_RELOAD (true/false, default true)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
APP_MODULE="${APP_MODULE:-app.main:app}"
UVICORN_RELOAD="${UVICORN_RELOAD:-true}"

cmd=(uv run uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT")

if [[ "${UVICORN_RELOAD,,}" == "true" ]]; then
  cmd+=(--reload)
fi

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

echo "→ Starting Markdown Web Browser API via: ${cmd[*]}"
exec "${cmd[@]}"
