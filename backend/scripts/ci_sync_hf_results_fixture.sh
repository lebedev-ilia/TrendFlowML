#!/usr/bin/env bash
# Copy latest HF batch results into the committed CI fixture.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$BACKEND_DIR/.e2e/state/hf_videos11_results.json"
DST="$BACKEND_DIR/tests/fixtures/hf_videos11_results.json"

if [[ ! -f "$SRC" ]]; then
  echo "FATAL: no batch results at $SRC — run e2e_run_hf_videos11.py first." >&2
  exit 1
fi

cp "$SRC" "$DST"
echo "Synced $SRC -> $DST"
echo "Run: cd backend && pytest tests/unit/test_hf_videos11_results_fixture.py -v"
