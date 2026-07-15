#!/usr/bin/env bash
# Host-native Triton (обход Docker GPU / nvidia-container-toolkit).
# Извлекает tritonserver + CUDA/cuDNN/Python libs из образа и запускает на хосте с GPU.
#
# Usage:
#   ./backend/scripts/e2e_triton_host.sh setup    # один раз (~5 GB в ~/.local/triton-bundle)
#   ./backend/scripts/e2e_triton_host.sh start
#   ./backend/scripts/e2e_triton_host.sh wait
#   ./backend/scripts/e2e_triton_host.sh stop
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE="${TRITON_HOST_BUNDLE:-$HOME/.local/triton-bundle}"
IMAGE="${TRITON_E2E_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}"
HTTP_PORT="${TRITON_E2E_HTTP_PORT:-8010}"
GRPC_PORT=$((HTTP_PORT + 1))
METRICS_PORT=$((HTTP_PORT + 2))
MODEL_REPO="${TRITON_E2E_MODEL_REPO:-$REPO_ROOT/DataProcessor/triton/models}"
PID_FILE="${REPO_ROOT}/backend/.e2e/state/triton_host.pid"
LOG_FILE="${REPO_ROOT}/backend/.e2e/logs/triton_host.log"

usage() {
  cat <<EOF
Usage: $0 {setup|start|wait|stop|status}

  setup   Extract tritonserver bundle from Docker image (once)
  start   Run tritonserver on host (GPU)
  wait    Wait for /v2/health/ready
  stop    Kill host tritonserver
  status  HTTP health code

TRITON_HTTP_URL=http://127.0.0.1:${HTTP_PORT}
EOF
}

cmd_setup() {
  mkdir -p "$BUNDLE/syslib" "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
  if [[ -x "$BUNDLE/tritonserver/bin/tritonserver" ]]; then
    echo "Bundle exists: $BUNDLE/tritonserver"
  else
    echo "Extracting tritonserver from $IMAGE ..."
    docker rm -f triton_bundle_extract 2>/dev/null || true
    docker create --name triton_bundle_extract "$IMAGE"
    docker cp triton_bundle_extract:/opt/tritonserver "$BUNDLE/tritonserver"
    docker cp triton_bundle_extract:/usr/local/cuda-12.6 "$BUNDLE/cuda-12.6"
    docker rm triton_bundle_extract
  fi
  echo "Copying runtime libs from image..."
  docker rm -f triton_bundle_libs 2>/dev/null || true
  docker create --name triton_bundle_libs "$IMAGE"
  local lib
  for lib in \
    libcudnn.so.9.3.0 libcudnn.so.9 libcudnn.so \
    libcublas.so.12 libcublasLt.so.12 \
    libpython3.10.so.1.0 libpython3.10.so.1 libpython3.10.so \
    libicuuc.so.70.1 libicudata.so.70.1 libicui18n.so.70.1 \
    libb64.so.0d libdcgm.so.3.2.6 libxml2.so.2.9.13; do
    docker cp "triton_bundle_libs:/usr/lib/x86_64-linux-gnu/$lib" "$BUNDLE/syslib/" 2>/dev/null || true
  done
  docker rm triton_bundle_libs
  ln -sf libcudnn.so.9.3.0 "$BUNDLE/syslib/libcudnn.so.9" 2>/dev/null || true
  ln -sf libpython3.10.so.1.0 "$BUNDLE/syslib/libpython3.10.so.1" 2>/dev/null || true
  ln -sf libicuuc.so.70.1 "$BUNDLE/syslib/libicuuc.so.70" 2>/dev/null || true
  ln -sf libicudata.so.70.1 "$BUNDLE/syslib/libicudata.so.70" 2>/dev/null || true
  ln -sf libxml2.so.2.9.13 "$BUNDLE/syslib/libxml2.so.2" 2>/dev/null || true
  ln -sf libdcgm.so.3.2.6 "$BUNDLE/syslib/libdcgm.so.3" 2>/dev/null || true
  echo "Setup done: $BUNDLE"
}

host_ld_path() {
  echo "$BUNDLE/syslib:$BUNDLE/cuda-12.6/targets/x86_64-linux/lib:$BUNDLE/tritonserver/lib"
}

cmd_start() {
  [[ -x "$BUNDLE/tritonserver/bin/tritonserver" ]] || { cmd_setup; }
  require_dir "$MODEL_REPO"
  cmd_stop 2>/dev/null || true
  mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
  export LD_LIBRARY_PATH="$(host_ld_path)"
  echo "Starting host tritonserver (HTTP :${HTTP_PORT})..."
  nohup "$BUNDLE/tritonserver/bin/tritonserver" \
    --model-repository="$MODEL_REPO" \
    --backend-directory="$BUNDLE/tritonserver/backends" \
    --http-port="$HTTP_PORT" \
    --grpc-port="$GRPC_PORT" \
    --metrics-port="$METRICS_PORT" \
    >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "TRITON_HTTP_URL=http://127.0.0.1:${HTTP_PORT} (host pid=$(cat "$PID_FILE"))"
}

require_dir() {
  [[ -d "$1" ]] || { echo "FATAL: not found: $1" >&2; exit 1; }
}

cmd_stop() {
  if [[ -f "$PID_FILE" ]]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
  pkill -f "tritonserver --model-repository.*--http-port=${HTTP_PORT}" 2>/dev/null || true
}

cmd_wait() {
  local url="http://127.0.0.1:${HTTP_PORT}/v2/health/ready"
  local deadline=$((SECONDS + ${TRITON_E2E_CPU_WAIT_SEC:-900}))
  echo "Waiting for host Triton: $url"
  while (( SECONDS < deadline )); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "Host Triton is ready."
      return 0
    fi
    if [[ -f "$PID_FILE" ]] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "FATAL: tritonserver exited" >&2
      tail -30 "$LOG_FILE" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "FATAL: timeout" >&2
  tail -20 "$LOG_FILE" >&2 || true
  return 1
}

cmd_status() {
  curl -sS -o /dev/null -w "%{http_code}\n" "http://127.0.0.1:${HTTP_PORT}/v2/health/ready" || echo "000"
}

[[ $# -ge 1 ]] || { usage >&2; exit 1; }
case "$1" in
  setup)  cmd_setup ;;
  start)  cmd_start ;;
  wait)   cmd_wait ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  -h|--help) usage ;;
  *) echo "FATAL: unknown: $1" >&2; exit 1 ;;
esac
