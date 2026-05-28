#!/usr/bin/env bash
# Triton Inference Server для полного локального E2E (CLIP / MiDaS / RAFT и т.д.).
#
# ВАЖНО: не используйте проброс 8000:8000 на хост — порт 8000 занят Fetcher,
# 8001 — Backend, 8002 — DataProcessor API. По умолчанию HTTP API Triton на хосте 8010.
#
# Usage (из корня репозитория):
#   ./backend/scripts/e2e_triton_docker.sh start    # поднять в фоне
#   ./backend/scripts/e2e_triton_docker.sh wait     # ждать /v2/health/ready
#   ./backend/scripts/e2e_triton_docker.sh stop     # остановить контейнер
#
# Переменные:
#   TRITON_E2E_HTTP_PORT — хост-порт под Triton HTTP (default 8010); gRPC+2, metrics+2.
#   TRITON_E2E_MODEL_REPO — каталог model repository (default …/DataProcessor/triton/models_t_1)

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_NAME="${TRITON_E2E_CONTAINER_NAME:-trendflow-e2e-triton}"
HTTP_PORT="${TRITON_E2E_HTTP_PORT:-8010}"
GRPC_PORT=$((HTTP_PORT + 1))
METRICS_PORT=$((HTTP_PORT + 2))
MODEL_REPO="${TRITON_E2E_MODEL_REPO:-$REPO_ROOT/DataProcessor/triton/models_t_1}"
IMAGE="${TRITON_E2E_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}"

usage() {
  cat <<EOF
Usage: $0 {start|wait|stop|logs|status}

  start   Run Triton container (detached, --rm compatible with stop)
  wait    Block until http://127.0.0.1:${HTTP_PORT}/v2/health/ready returns 200
  stop    docker stop \`$CONTAINER_NAME\`
  logs    docker logs -f \`$CONTAINER_NAME\`
  status  curl -sS -o /dev/null -w "%{http_code}" health ready URL

Model repo: $MODEL_REPO
Listen:    TRITON_HTTP_URL=http://127.0.0.1:${HTTP_PORT}
EOF
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "FATAL: model repository not found: $1" >&2
    exit 1
  fi
}

cmd_start() {
  require_dir "$MODEL_REPO"
  docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
  echo "Starting $CONTAINER_NAME (HTTP host :${HTTP_PORT} -> container :8000)..."
  # Имя + --rm: при stop контейнер удаляется; модели только read-only mount.
  docker run -d \
    --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=1g \
    -p "${HTTP_PORT}:8000" \
    -p "${GRPC_PORT}:8001" \
    -p "${METRICS_PORT}:8002" \
    -v "${MODEL_REPO}:/models:ro" \
    "$IMAGE" \
    tritonserver --model-repository=/models
  echo "TRITON_HTTP_URL=http://127.0.0.1:${HTTP_PORT}"
}

cmd_wait() {
  local url="http://127.0.0.1:${HTTP_PORT}/v2/health/ready"
  local deadline=$((SECONDS + 600))
  echo "Waiting for Triton ready: $url"
  while (( SECONDS < deadline )); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "Triton is ready."
      return 0
    fi
    sleep 2
  done
  echo "FATAL: timeout waiting for Triton" >&2
  exit 1
}

cmd_stop() {
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
}

cmd_status() {
  curl -sS -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:${HTTP_PORT}/v2/health/ready" || echo "000"
}

cmd_logs() {
  docker logs -f "$CONTAINER_NAME"
}

[[ $# -ge 1 ]] || { usage >&2; exit 1; }
case "$1" in
  start)  cmd_start ;;
  wait)   cmd_wait ;;
  stop)   cmd_stop ;;
  logs)   cmd_logs ;;
  status) cmd_status ;;
  -h|--help) usage ;;
  *) echo "FATAL: unknown command: $1" >&2; usage >&2; exit 1 ;;
esac
