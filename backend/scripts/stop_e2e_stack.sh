#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Stop the local E2E app stack started by start_e2e_stack.sh.

Usage:
  ./backend/scripts/stop_e2e_stack.sh [--with-infra] [--quiet]

Options:
  --with-infra  Also stop postgres/redis/minio (Fetcher) and prometheus/grafana (DataProcessor E2E override).
  --quiet       Reduce output.
  -h, --help    Show this help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
FETCHER_DIR="$REPO_ROOT/Fetcher"
DATAPROCESSOR_DIR="$REPO_ROOT/DataProcessor"

RUNTIME_DIR="${E2E_RUNTIME_DIR:-$BACKEND_DIR/.e2e}"
PID_DIR="${E2E_PID_DIR:-$RUNTIME_DIR/pids}"
WITH_INFRA=0
QUIET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-infra)
      WITH_INFRA=1
      ;;
    --quiet)
      QUIET=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "FATAL: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

say() {
  if (( ! QUIET )); then
    echo "$@"
  fi
}

terminate_from_pidfile() {
  local name="$1"
  local pid_file="$PID_DIR/$name.pid"
  local cmd_file="$PID_DIR/$name.cmd"
  local pid current_cmd expected_cmd expected_exec

  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi

  pid="$(<"$pid_file")"
  expected_cmd=""
  if [[ -f "$cmd_file" ]]; then
    expected_cmd="$(<"$cmd_file")"
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file" "$cmd_file"
    return 0
  fi

  current_cmd="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  expected_exec="$(basename "$expected_cmd")"
  if [[ -n "$expected_exec" && -n "$current_cmd" && "$current_cmd" != *"$expected_exec"* ]]; then
    say "Skipping $name: pid $pid no longer matches the recorded command."
    rm -f "$pid_file" "$cmd_file"
    return 0
  fi

  say "Stopping $name (pid=$pid)"
  kill "$pid" 2>/dev/null || true

  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file" "$cmd_file"
      return 0
    fi
    sleep 0.5
  done

  say "Force stopping $name (pid=$pid)"
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pid_file" "$cmd_file"
}

if [[ ! -d "$PID_DIR" ]]; then
  say "No E2E pid directory found: $PID_DIR"
else
  for name in \
    backend-api \
    backend-worker \
    backend-beat \
    fetcher-api \
    fetcher-worker \
    dataprocessor-api \
    dataprocessor-worker \
    embedding-service
  do
    terminate_from_pidfile "$name"
  done
fi

if (( WITH_INFRA )); then
  say "Stopping docker infrastructure (postgres, redis, minio)"
  (cd "$FETCHER_DIR" && docker compose stop postgres redis minio) >/dev/null
  if [[ -f "$DATAPROCESSOR_DIR/monitoring/docker-compose.prometheus-override-e2e.yml" ]]; then
    say "Stopping Prometheus and Grafana (DataProcessor)"
    (cd "$DATAPROCESSOR_DIR" && docker compose -f docker-compose.yml -f monitoring/docker-compose.prometheus-override-e2e.yml stop prometheus grafana) >/dev/null 2>&1 || true
  fi
fi

say "E2E app stack stopped."
