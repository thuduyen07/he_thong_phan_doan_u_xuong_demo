#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Create the project virtual environment first." >&2
  exit 1
fi

exec "$PYTHON" "$ROOT_DIR/server.py" "$@"
