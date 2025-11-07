#!/usr/bin/env bash

# Run the mandatory verification suite (ruff, ty, Playwright smoke).
# Additional arguments are forwarded to `playwright test` so callers can
# narrow the test selection (defaults to tests/smoke_capture.spec.ts).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PLAYWRIGHT_TARGETS=("tests/smoke_capture.spec.ts")
if [[ $# -gt 0 ]]; then
  PLAYWRIGHT_TARGETS=("$@")
fi

run_step() {
  local desc="$1"
  shift
  echo "→ ${desc}"
  "$@"
  echo
}

run_step "ruff check --fix --unsafe-fixes" uv run ruff check --fix --unsafe-fixes
run_step "ty check" uvx ty check
run_step "playwright smoke" uv run playwright test "${PLAYWRIGHT_TARGETS[@]}"
