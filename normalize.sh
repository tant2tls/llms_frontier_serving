#!/usr/bin/env bash
# Current offline summary: defaults to the three primary 16K batch arms.
set -eu
WORKSPACE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PY:-python3}" "$WORKSPACE_DIR/tools/summarize_results.py" "$@"
