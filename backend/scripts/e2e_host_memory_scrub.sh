#!/usr/bin/env bash
# Best-effort очистка между прогонами example-suite-7 (честный снимок ОЗУ/кэша).
#
# Ограничения:
# - VRAM занята Triton, DataProcessor worker, embedding-service — без остановки стека её не «обнулить».
# - Здесь: sync, опционально drop_caches (нужен root), nvidia-smi, gc + torch.cuda.empty_cache в .data_venv.
#
# Использование:
#   ./backend/scripts/e2e_host_memory_scrub.sh
#   sudo E2E_SCRUB_DROP_CACHES=1 ./backend/scripts/e2e_host_memory_scrub.sh   # сброс page cache (осторожно)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
DATA_PY="${REPO_ROOT}/DataProcessor/.data_venv/bin/python"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

log "e2e_host_memory_scrub start"
if [[ -r /proc/meminfo ]]; then
  awk '/^MemAvailable:/ {printf "MemAvailable: %d MiB\n", $2/1024}' /proc/meminfo || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
fi

if [[ "${E2E_SCRUB_DROP_CACHES:-0}" == "1" ]]; then
  sync || true
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 >/proc/sys/vm/drop_caches
    log "vm.drop_caches=3 (page cache + dentries + inodes)"
  else
    log "WARN: cannot write /proc/sys/vm/drop_caches — run: sudo E2E_SCRUB_DROP_CACHES=1 $0"
  fi
fi

if [[ -x "$DATA_PY" ]]; then
  (cd "$REPO_ROOT/DataProcessor" && "$DATA_PY" <<'PY') || true
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        ipc = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc):
            ipc()
except Exception:
    pass
PY
  log "torch.cuda.empty_cache in DataProcessor venv (if CUDA)"
else
  log "skip torch scrub (no $DATA_PY)"
fi

if [[ -r /proc/meminfo ]]; then
  awk '/^MemAvailable:/ {printf "MemAvailable after: %d MiB\n", $2/1024}' /proc/meminfo || true
fi
log "e2e_host_memory_scrub done"
