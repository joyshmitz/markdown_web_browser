#!/usr/bin/env bash

# Replay a capture job manifest against the local API.
# Usage: scripts/replay_job.sh path/to/manifest.json [server_url]
# Defaults to http://localhost:8000 unless MDWB_SERVER is set.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <manifest.json> [server_url]" >&2
  exit 1
fi

MANIFEST_PATH="$1"
shift || true

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

SERVER_URL="${1:-${MDWB_SERVER:-http://localhost:8000}}"
SERVER_URL="${SERVER_URL%/}"

echo "→ Replaying manifest $MANIFEST_PATH to ${SERVER_URL}/replay"

uv run python - "$MANIFEST_PATH" "$SERVER_URL" <<'PY'
import json
import os
import sys
from pathlib import Path

import httpx

manifest_path = Path(sys.argv[1])
server_url = sys.argv[2]
payload = json.loads(manifest_path.read_text(encoding="utf-8"))

headers: dict[str, str] = {}
api_key = os.environ.get("MDWB_API_KEY")
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

request_body = {"manifest": payload}

try:
    response = httpx.post(
        f"{server_url}/replay",
        json=request_body,
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
except httpx.HTTPStatusError as exc:
    print(f"Replay failed with HTTP {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
    raise SystemExit(1) from exc
except httpx.RequestError as exc:
    print(f"Unable to reach {server_url}: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

print(json.dumps(response.json(), indent=2))
PY
