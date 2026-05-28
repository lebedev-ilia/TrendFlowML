#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_PATH="${HF_MANIFEST_PATH:-$REPO_ROOT/configs/hf_artifacts_manifest.json}"

python3 "$REPO_ROOT/DataProcessor/scripts/hf_artifacts_sync.py" \
  download \
  --repo-root "$REPO_ROOT" \
  --manifest "$MANIFEST_PATH" \
  "$@"
