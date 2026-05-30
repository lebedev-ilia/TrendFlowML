#!/usr/bin/env bash
# Run discover + enrich + download + HF uploads in parallel.
#
# Usage:
#   cd Fetcher
#   export HF_TOKEN=hf_...          # token, NOT in dataset_campaign.json
#   ./scripts/run_dataset_workers.sh Sport
#
# Logs: dataset_runs/dataset-100k/logs/workers/*.log

set -euo pipefail
cd "$(dirname "$0")/.."

CATEGORY="${1:-Sport}"
CONFIG="${2:-dataset_campaign.json}"
INTERVAL="${INTERVAL:-120}"

source .fetcher_venv/bin/activate

if [[ -d monitoring ]]; then
  (cd monitoring && docker compose up -d)
fi

exec python -m fetcher.dataset_collector.cli run-workers "$CONFIG" \
  --category "$CATEGORY" \
  --interval "$INTERVAL" \
  --metrics-port 9095
