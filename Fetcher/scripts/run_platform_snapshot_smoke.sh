#!/usr/bin/env bash
# Platform API+SDK migration tests + snapshot-smoke pipeline.
#
# Usage:
#   cd Fetcher
#   chmod +x scripts/run_platform_snapshot_smoke.sh
#
#   # unit tests + credentials check only
#   ./scripts/run_platform_snapshot_smoke.sh unit
#
#   # full local snapshot-smoke (discover → workers → status → audit)
#   export HF_TOKEN=hf_...
#   ./scripts/run_platform_snapshot_smoke.sh smoke
#
#   # individual steps
#   ./scripts/run_platform_snapshot_smoke.sh discover
#   ./scripts/run_platform_snapshot_smoke.sh workers
#   ./scripts/run_platform_snapshot_smoke.sh workers-enrich
#   ./scripts/run_platform_snapshot_smoke.sh snapshot-loop
#   ./scripts/run_platform_snapshot_smoke.sh status
#
# Env overrides:
#   OUTPUT_DIR   — run directory (default: tests/full_results/snapshot-smoke-100/dataset_runs)
#   DISCOVER_LIMIT — default 100
#   HF_REPO_PREFIX — default Ilialebedev
#   ENABLE_PLATFORMS — space-separated: tiktok rutube instagram twitch (all enabled by default)
#   SKIP_UNIT=1    — skip unit/credentials phase in smoke mode

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROLE="${1:-smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-tests/full_results/snapshot-smoke-100/dataset_runs}"
DISCOVER_LIMIT="${DISCOVER_LIMIT:-100}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-Ilialebedev}"
CREDENTIALS_DIR="${FETCHER_CREDENTIALS_DIR:-fetcher/credentials}"
YOUTUBE_KEYS="${YOUTUBE_KEYS_FILE:-$CREDENTIALS_DIR/youtube_keys.txt}"
METRICS_DISCOVER="${METRICS_DISCOVER:-9095}"
METRICS_WORKERS="${METRICS_WORKERS:-9096}"
SNAPSHOT_SLEEP_SECONDS="${SNAPSHOT_SLEEP_SECONDS:-30}"
WORKER_INTERVAL="${WORKER_INTERVAL:-120}"

export FETCHER_CREDENTIALS_DIR="$CREDENTIALS_DIR"
export FETCHER_YOUTUBE_PROVIDER_MODE="${FETCHER_YOUTUBE_PROVIDER_MODE:-api_first}"
export FETCHER_TIKTOK_PROVIDER_MODE="${FETCHER_TIKTOK_PROVIDER_MODE:-api_first}"
export FETCHER_INSTAGRAM_PROVIDER_MODE="${FETCHER_INSTAGRAM_PROVIDER_MODE:-api_first}"
export FETCHER_TWITCH_PROVIDER_MODE="${FETCHER_TWITCH_PROVIDER_MODE:-api_first}"
export FETCHER_RUTUBE_PROVIDER_MODE="${FETCHER_RUTUBE_PROVIDER_MODE:-sdk_only}"

PY="${PYTHON:-python3}"
BOOTSTRAP=("$PY" scripts/colab_20k_bootstrap.py)

_platform_flags() {
  local flags=()
  local enabled="${ENABLE_PLATFORMS:-tiktok rutube instagram twitch}"
  for p in $enabled; do
    flags+=(--enable-"$p")
  done
  printf '%s\n' "${flags[@]}"
}

run_unit_tests() {
  echo "==> Credentials check"
  "$PY" scripts/check_platform_credentials.py --credentials-dir "$CREDENTIALS_DIR" || true
  if [[ -f "$YOUTUBE_KEYS" ]]; then
    "$PY" scripts/check_youtube_keys.py "$YOUTUBE_KEYS" || true
  fi

  echo "==> py_compile (platform modules)"
  "$PY" -m py_compile \
    fetcher/platforms/provider_mode.py \
    fetcher/platforms/dual_provider.py \
    fetcher/platforms/platform_clients.py \
    fetcher/platforms/adapter_utils.py \
    fetcher/services/credentials.py \
    fetcher/dataset_collector/discovery/tiktok.py \
    fetcher/dataset_collector/discovery/instagram.py \
    fetcher/dataset_collector/discovery/twitch.py \
    fetcher/dataset_collector/discovery/rutube.py

  echo "==> pytest (platform migration unit tests)"
  "$PY" -m pytest \
    tests/unit/test_dual_provider.py \
    tests/unit/test_credentials.py \
    tests/unit/test_platform_video_dto.py \
    tests/unit/test_normalize_platforms.py \
    -q --tb=short

  if [[ -n "${FETCHER_POSTGRES_DSN:-}" ]]; then
    echo "==> pytest (registry + tiktok normalize, needs Postgres)"
    "$PY" -m pytest \
      tests/unit/test_platform_registry.py \
      tests/unit/test_tiktok_normalize_source.py \
      -q --no-cov --tb=short || true
  else
    echo "==> SKIP registry tests (set FETCHER_POSTGRES_DSN to run)"
  fi
}

run_discover() {
  mkdir -p "$OUTPUT_DIR"
  local -a flags
  mapfile -t flags < <(_platform_flags)
  "${BOOTSTRAP[@]}" \
    --campaign-profile snapshot-smoke \
    --role discover \
    --limit "$DISCOVER_LIMIT" \
    --hf-repo-prefix "$HF_REPO_PREFIX" \
    --youtube-keys-file "$YOUTUBE_KEYS" \
    --output-dir "$OUTPUT_DIR" \
    --metrics-port "$METRICS_DISCOVER" \
    "${flags[@]}"
}

run_workers() {
  "${BOOTSTRAP[@]}" \
    --campaign-profile snapshot-smoke \
    --role workers \
    --output-dir "$OUTPUT_DIR" \
    --metrics-port "$METRICS_WORKERS"
}

run_workers_enrich() {
  "${BOOTSTRAP[@]}" \
    --campaign-profile snapshot-smoke \
    --role workers-enrich \
    --output-dir "$OUTPUT_DIR" \
    --metrics-port "$METRICS_WORKERS"
}

run_workers_download() {
  "${BOOTSTRAP[@]}" \
    --campaign-profile snapshot-smoke \
    --role workers-download \
    --output-dir "$OUTPUT_DIR" \
    --metrics-port "$METRICS_WORKERS"
}

run_snapshot_loop() {
  local -a flags
  mapfile -t flags < <(_platform_flags)
  "${BOOTSTRAP[@]}" \
    --campaign-profile snapshot-smoke \
    --role snapshot-loop \
    --snapshot-sleep-seconds "$SNAPSHOT_SLEEP_SECONDS" \
    --output-dir "$OUTPUT_DIR" \
    "${flags[@]}"
}

run_status() {
  "${BOOTSTRAP[@]}" \
    --campaign-profile snapshot-smoke \
    --role status \
    --output-dir "$OUTPUT_DIR"
}

run_audit() {
  local runtime="$OUTPUT_DIR/runtime_dataset_campaign_20k.json"
  if [[ -f "$runtime" ]]; then
    "$PY" -m fetcher.dataset_collector.cli validate "$runtime" || true
  fi
  if [[ -d "$OUTPUT_DIR" ]]; then
    "$PY" scripts/compare_hf_run_state.py "$OUTPUT_DIR" 2>/dev/null || true
    "$PY" scripts/audit_dataset_run.py "$OUTPUT_DIR" --check-hf 2>/dev/null || true
  fi
}

run_smoke() {
  if [[ "${SKIP_UNIT:-0}" != "1" ]]; then
    run_unit_tests
  fi
  if [[ -z "${HF_TOKEN:-}" && ! -f "$OUTPUT_DIR/.hf_token" ]]; then
    echo "WARN: HF_TOKEN not set and $OUTPUT_DIR/.hf_token missing — discover/workers may fail on HF upload."
  fi
  run_discover
  run_workers
  run_status
  run_audit
  echo "==> Done. Logs: $OUTPUT_DIR/logs/workers/"
  echo "    tail -f $OUTPUT_DIR/logs/workers/enrich-metadata.log"
  echo "    tail -f $OUTPUT_DIR/logs/workers/upload-hf-enrich.log"
}

case "$ROLE" in
  unit|test)
    run_unit_tests
    ;;
  credentials)
    "$PY" scripts/check_platform_credentials.py --credentials-dir "$CREDENTIALS_DIR"
    ;;
  discover)
    run_discover
    ;;
  workers)
    run_workers
    ;;
  workers-enrich)
    run_workers_enrich
    ;;
  workers-download)
    run_workers_download
    ;;
  snapshot-loop|snapshots)
    run_snapshot_loop
    ;;
  status)
    run_status
    ;;
  audit)
    run_audit
    ;;
  smoke|all)
    run_smoke
    ;;
  help|-h|--help)
    sed -n '2,28p' "$0"
    ;;
  *)
    echo "Unknown role: $ROLE (try: unit, smoke, discover, workers, workers-enrich, snapshot-loop, status, audit, help)"
    exit 1
    ;;
esac
