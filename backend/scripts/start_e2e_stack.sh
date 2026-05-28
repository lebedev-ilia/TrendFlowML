#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Start the full local E2E stack for Backend -> Fetcher -> DataProcessor.

Usage:
  ./backend/scripts/start_e2e_stack.sh [--with-infra] [--logs-dir DIR] [--no-stop]
  (если shell уже в каталоге backend — только ./scripts/start_e2e_stack.sh …)

Options:
  --with-infra   Run setup_e2e_infra.sh before starting services (Postgres, Redis, MinIO, миграции, Prometheus+Grafana для scrape DP на хосте).
  --logs-dir     Override the log root directory.
  --no-stop      Do not stop a previously started app stack first.
  -h, --help     Show this help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
FETCHER_DIR="$REPO_ROOT/Fetcher"
DATAPROCESSOR_DIR="$REPO_ROOT/DataProcessor"

WITH_INFRA=0
STOP_FIRST=1
LOG_ROOT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-infra)
      WITH_INFRA=1
      ;;
    --logs-dir)
      shift
      [[ $# -gt 0 ]] || { echo "FATAL: --logs-dir requires a value." >&2; exit 1; }
      LOG_ROOT_OVERRIDE="$1"
      ;;
    --no-stop)
      STOP_FIRST=0
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

# shellcheck source=/dev/null
source "$SCRIPT_DIR/e2e_env.sh"

RUNTIME_DIR="${E2E_RUNTIME_DIR:-$BACKEND_DIR/.e2e}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${E2E_LOG_ROOT:-$RUNTIME_DIR/logs}}"
PID_DIR="${E2E_PID_DIR:-$RUNTIME_DIR/pids}"
STATE_DIR="${E2E_STATE_DIR:-$RUNTIME_DIR/state}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$LOG_ROOT/$RUN_ID"

mkdir -p "$LOG_ROOT" "$PID_DIR" "$STATE_DIR" "$RUN_DIR"
mkdir -p "$STORAGE_ROOT/videos" "$STORAGE_ROOT/uploads"
ln -sfn "$RUN_DIR" "$LOG_ROOT/latest"

cleanup_on_error() {
  echo "Startup failed. See logs in: $RUN_DIR" >&2
}
trap cleanup_on_error ERR

require_executable() {
  local path="$1"
  if [[ ! -x "$path" ]]; then
    echo "FATAL: executable not found: $path" >&2
    exit 1
  fi
}

start_process() {
  local name="$1"
  local cwd="$2"
  shift 2

  local process_dir="$RUN_DIR/$name"
  local log_file="$process_dir/process.log"
  local pid_file="$PID_DIR/$name.pid"
  local cmd_file="$PID_DIR/$name.cmd"
  local pid

  mkdir -p "$process_dir"
  {
    printf '[%s] Starting %s\n' "$(date -Is)" "$name"
    printf 'cwd=%s\n' "$cwd"
    printf 'command='
    printf '%q ' "$@"
    printf '\n\n'
  } >"$log_file"

  (
    cd "$cwd"
    nohup "$@" </dev/null >>"$log_file" 2>&1 &
    echo $! >"$pid_file"
  )

  pid="$(<"$pid_file")"
  printf '%s\n' "$1" >"$cmd_file"

  echo "Started $name (pid=$pid)"
}

wait_for_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local log_file="$RUN_DIR/$name/process.log"
  # Cold start on slow disks (e.g. network volumes): Backend import alone can take ~60s+.
  local deadline=$((SECONDS + 300))

  while (( SECONDS < deadline )); do
    if python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    try:
        sock.connect((host, port))
    except OSError:
        raise SystemExit(1)
raise SystemExit(0)
PY
    then
      echo "$name is ready on $host:$port"
      return 0
    fi
    sleep 1
  done

  echo "FATAL: $name did not open $host:$port in time. Last log lines:" >&2
  python3 - "$log_file" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    raise SystemExit(0)

lines = path.read_text(errors="replace").splitlines()
for line in lines[-40:]:
    print(line)
PY
  exit 1
}

wait_for_pid() {
  local name="$1"
  local pid_file="$PID_DIR/$name.pid"
  local pid

  [[ -f "$pid_file" ]] || { echo "FATAL: pid file missing for $name" >&2; exit 1; }
  pid="$(<"$pid_file")"

  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "FATAL: $name exited immediately. See $RUN_DIR/$name/process.log" >&2
    exit 1
  fi
}

require_executable "$BACKEND_DIR/.venv/bin/uvicorn"
require_executable "$BACKEND_DIR/.venv/bin/celery"
require_executable "$FETCHER_DIR/.fetcher_venv/bin/uvicorn"
require_executable "$DATAPROCESSOR_DIR/.data_venv/bin/uvicorn"
require_executable "$DATAPROCESSOR_DIR/.data_venv/bin/python"
require_executable "$FETCHER_DIR/scripts/run_worker_on_host.sh"

if (( STOP_FIRST )); then
  "$SCRIPT_DIR/stop_e2e_stack.sh" --quiet || true
fi

if (( WITH_INFRA )); then
  "$SCRIPT_DIR/setup_e2e_infra.sh"
fi

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="utf-8"

start_process "backend-api" "$BACKEND_DIR" \
  "$BACKEND_DIR/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8001
start_process "backend-worker" "$BACKEND_DIR" \
  "$BACKEND_DIR/.venv/bin/celery" -A app.worker:celery_app worker --loglevel=info
start_process "backend-beat" "$BACKEND_DIR" \
  "$BACKEND_DIR/.venv/bin/celery" -A app.worker:celery_app beat --loglevel=info
start_process "fetcher-api" "$FETCHER_DIR" \
  "$FETCHER_DIR/.fetcher_venv/bin/uvicorn" fetcher.api:app --host 0.0.0.0 --port 8000
start_process "fetcher-worker" "$FETCHER_DIR" \
  "$FETCHER_DIR/scripts/run_worker_on_host.sh"
start_process "dataprocessor-api" "$DATAPROCESSOR_DIR" \
  "$DATAPROCESSOR_DIR/.data_venv/bin/uvicorn" api.main:app --host 0.0.0.0 --port 8002
start_process "dataprocessor-worker" "$DATAPROCESSOR_DIR" \
  "$DATAPROCESSOR_DIR/.data_venv/bin/python" -m api.worker

# Embedding Service (brand/car/place/franchise/face_identity и др.). Нужны: pgvector Postgres,
# faiss-cpu (см. DataProcessor/embedding_service/requirements-e2e.txt), при первом поиске — модели/индексы могут быть пустыми.
EMBED_PY="${DATAPROCESSOR_DIR}/.data_venv/bin/python"
EMBED_PORT="${EMBEDDING_SERVICE_PORT:-8005}"
if [[ -x "$EMBED_PY" ]] && (cd "$DATAPROCESSOR_DIR" && "$EMBED_PY" -c "import faiss; from embedding_service.config.settings import EmbeddingServiceConfig") 2>/dev/null; then
  start_process "embedding-service" "$DATAPROCESSOR_DIR" \
    "$EMBED_PY" -m embedding_service.run_server
else
  echo "WARN: embedding-service not started (need faiss + embedding_service deps). Install:" >&2
  echo "  $EMBED_PY -m pip install -r $DATAPROCESSOR_DIR/embedding_service/requirements-e2e.txt" >&2
fi

wait_for_port "backend-api" "127.0.0.1" "8001"
wait_for_port "fetcher-api" "127.0.0.1" "8000"
wait_for_port "dataprocessor-api" "127.0.0.1" "8002"
if [[ -f "$PID_DIR/embedding-service.pid" ]]; then
  wait_for_port "embedding-service" "127.0.0.1" "$EMBED_PORT"
fi
wait_for_pid "backend-worker"
wait_for_pid "backend-beat"
wait_for_pid "fetcher-worker"
wait_for_pid "dataprocessor-worker"

cat >"$STATE_DIR/latest-run.env" <<EOF
E2E_RUN_ID=$RUN_ID
E2E_LOG_DIR=$RUN_DIR
E2E_PID_DIR=$PID_DIR
EOF

echo
echo "E2E app stack is up."
echo "Run id:   $RUN_ID"
echo "Logs:     $RUN_DIR"
echo "PID dir:  $PID_DIR"
echo
echo "To stop:"
echo "  $SCRIPT_DIR/stop_e2e_stack.sh"
echo "With the same docker infra and monitoring:  $SCRIPT_DIR/stop_e2e_stack.sh --with-infra"
echo ""
if [[ -f "$BACKEND_DIR/.e2e/state/monitoring_ports.env" ]]; then
  # shellcheck source=/dev/null
  source "$BACKEND_DIR/.e2e/state/monitoring_ports.env"
fi
echo "Monitoring (после setup_e2e_infra / --with-infra):  Prometheus http://localhost:${E2E_PROMETHEUS_HOST_PORT:-9090}  Grafana http://localhost:${E2E_GRAFANA_HOST_PORT:-3000}  (admin/admin)"
