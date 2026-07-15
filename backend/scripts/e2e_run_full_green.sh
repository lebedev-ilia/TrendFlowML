#!/usr/bin/env bash
# Полный зелёный E2E с защитой от перегрузки (resource guard).
#
# Usage (из корня репо):
#   ./backend/scripts/e2e_run_full_green.sh
#   ./backend/scripts/e2e_run_full_green.sh --example-suite-7
#
# Guard (по умолчанию): аварийный stop при RAM/swap/GPU VRAM/disk >= 99% (3 раза подряд).
# Переопределение: E2E_GUARD_RAM_USED_PCT=95 E2E_GUARD_GPU_MEM_PCT=92 ...
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"

# shellcheck source=/dev/null
source "$SCRIPT_DIR/e2e_env.sh"

GUARD_PID=""
ORCH_PID=""

cleanup() {
  local code=$?
  if [[ -n "$GUARD_PID" ]] && kill -0 "$GUARD_PID" 2>/dev/null; then
    kill "$GUARD_PID" 2>/dev/null || true
    wait "$GUARD_PID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

# Один активный run на 6 GiB GPU — меньше пиков VRAM/RAM
export MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-1}"
# Полный visual: только GPU (toolkit или manual_gpu), без CPU-fallback Triton
export TRITON_E2E_CPU_FALLBACK="${TRITON_E2E_CPU_FALLBACK:-0}"

preflight() {
  local ok=1
  if ! command -v docker >/dev/null; then
    echo "FATAL: docker not found" >&2
    ok=0
  fi
  if [[ " $* " == *" --with-triton-docker "* ]]; then
    if ! "$SCRIPT_DIR/e2e_triton_docker.sh" preflight >/dev/null 2>&1; then
      "$SCRIPT_DIR/e2e_triton_docker.sh" preflight >&2 || true
      echo "FATAL: полный visual (MiDaS/RAFT/CLIP ensemble) требует Docker GPU." >&2
      echo "  Один раз: ./backend/scripts/install_nvidia_container_toolkit.sh --from-debs" >&2
      echo "  Проверка: ./backend/scripts/install_nvidia_container_toolkit.sh --verify-only" >&2
      ok=0
    fi
  fi
  if [[ ! -d "${TRITON_E2E_MODEL_REPO:-$REPO_ROOT/DataProcessor/triton/models}" ]]; then
    echo "FATAL: Triton model repo missing. Run: python DataProcessor/scripts/download_models.py --groups triton" >&2
    ok=0
  fi
  if [[ ! -f "$REPO_ROOT/example/example_videos/sample_0.mp4" ]]; then
    echo "FATAL: mock videos missing. Run: DataProcessor/.data_venv/bin/python example/scripts/create_e2e_sample_videos.py" >&2
    ok=0
  fi
  if [[ ! -f "$REPO_ROOT/example/text_audit_v3_smoke/scenarios/audit_v3_20_scenarios.json" ]]; then
    echo "Creating E2E text fixtures (audit scenarios + video_document_*.json)…" >&2
    python3 "$REPO_ROOT/example/scripts/create_e2e_text_fixtures.py" >&2 || ok=0
  fi
  local ap_py="$REPO_ROOT/DataProcessor/AudioProcessor/.ap_venv/bin/python"
  if [[ ! -x "$ap_py" ]] || ! "$ap_py" -c "import torch" 2>/dev/null; then
    echo "Installing processor venvs (torch, emoji, librosa, cv2) — one-time, ~10–20 min…" >&2
    "$SCRIPT_DIR/setup_e2e_processor_venvs.sh" >&2 || ok=0
  fi
  local vp_py="$REPO_ROOT/DataProcessor/VisualProcessor/.vp_venv/bin/python"
  if [[ ! -x "$vp_py" ]] || ! "$vp_py" -c "import ultralytics, requests" 2>/dev/null; then
    echo "Updating VisualProcessor .vp_venv (ultralytics, requests)…" >&2
    "$SCRIPT_DIR/setup_e2e_processor_venvs.sh" >&2 || ok=0
  fi
  chmod +x "$SCRIPT_DIR/setup_e2e_model_symlinks.sh" "$SCRIPT_DIR/setup_e2e_audio_models.sh" 2>/dev/null || true
  "$SCRIPT_DIR/setup_e2e_model_symlinks.sh" >&2 || ok=0
  "$SCRIPT_DIR/setup_e2e_audio_models.sh" >&2 || ok=0
  # Mock video в MinIO/Fetcher кешируется по platform_video_id — после regen sample_*.mp4 нужен cold ingest.
  if [[ " $* " == *" --cold-ingestion "* ]] || [[ " $* " == *" --offline-example "* ]]; then
    rm -f "${VIDEO_URL_CACHE_DIR:-${STORAGE_ROOT}/videos/_url_cache}"/*.mp4 2>/dev/null || true
    # Segmenter кеширует audio.wav по video_id (не run_id); без сброса остаётся старый silent wav.
    if [[ -d "${STORAGE_ROOT}/frames_dir" ]]; then
      find "${STORAGE_ROOT}/frames_dir" -type f \( -name 'audio.wav' -o -name 'metadata.json' \) -path '*/audio/*' -delete 2>/dev/null || true
    fi
  fi
  local sample="$REPO_ROOT/example/example_videos/sample_0.mp4"
  if [[ -f "$sample" ]] && command -v ffprobe >/dev/null; then
    if ! ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "$sample" 2>/dev/null | grep -q audio; then
      echo "Regenerating mock videos with tone audio track (440 Hz AAC)…" >&2
      "${REPO_ROOT}/DataProcessor/.data_venv/bin/python" "$REPO_ROOT/example/scripts/create_e2e_sample_videos.py" >&2 || ok=0
    fi
  fi
  return $((1 - ok))
}

LOG_DIR="${E2E_LOG_ROOT:-$BACKEND_DIR/.e2e/logs}"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%d_%H%M%S_utc)"
TERMINAL_LOG="${E2E_TERMINAL_LOG:-$LOG_DIR/full_green_${TS}.log}"
ln -sfn "$TERMINAL_LOG" "$LOG_DIR/full_green_latest.log"

echo "Full green E2E log: $TERMINAL_LOG" >&2
echo "Resource guard log: $BACKEND_DIR/.e2e/logs/resource_guard.log" >&2

cd "$BACKEND_DIR"
# shellcheck source=/dev/null
source .venv/bin/activate

# Guard следит за этим shell (дочерний python наследует дерево)
"$BACKEND_DIR/.venv/bin/python" -u "$SCRIPT_DIR/e2e_resource_guard.py" \
  --watch-pid "$$" \
  --log-file "$BACKEND_DIR/.e2e/logs/resource_guard.log" \
  >>"$BACKEND_DIR/.e2e/logs/resource_guard.log" 2>&1 &
GUARD_PID=$!

DEFAULT_ARGS=(
  --with-triton-docker
  --offline-example
  --cold-ingestion
  --e2e-low-vram
  --timeout 7200
)

if [[ $# -eq 0 ]]; then
  set -- "${DEFAULT_ARGS[@]}"
fi

preflight "$@" || exit 2

echo "Memory scrub before full green E2E (no password)…" >&2
"$SCRIPT_DIR/e2e_host_memory_scrub.sh" --aggressive 2>&1 | tail -6 >&2 || true

echo "Starting e2e_full_max_run.py $*" >&2

# Предзагрузка образа Triton до старта guard — иначе docker pull может забить swap.
if [[ " $* " == *" --with-triton-docker "* ]]; then
  echo "Pre-pulling Triton image (reduces RAM/swap spike during E2E)..." >&2
  docker pull "${TRITON_E2E_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}" >&2 || true
fi

# tee в лог; guard убьёт $$ → python получит SIGTERM при emergency
python -u "$SCRIPT_DIR/e2e_full_max_run.py" "$@" 2>&1 | tee -a "$TERMINAL_LOG"
EXIT_CODE=${PIPESTATUS[0]}

if [[ "$EXIT_CODE" -eq 0 ]]; then
  echo "Validating full green criteria (§0.1)…" >&2
  if ! python -u "$SCRIPT_DIR/e2e_validate_full_green.py" --latest-e2e-artifact; then
    EXIT_CODE=1
  fi
fi

if [[ "$EXIT_CODE" -eq 0 ]]; then
  echo "Validating output quality (§0.2)…" >&2
  QUALITY_PY="${REPO_ROOT}/DataProcessor/.data_venv/bin/python"
  if [[ ! -x "$QUALITY_PY" ]]; then
    QUALITY_PY="python"
  fi
  if ! "$QUALITY_PY" -u "$SCRIPT_DIR/e2e_validate_output_quality.py" --latest-e2e-artifact; then
    EXIT_CODE=1
  fi
fi

exit "$EXIT_CODE"
