#!/usr/bin/env bash
# Запуск демона-воркера прогонов на твоём ПК (с GPU). Запусти ОДИН РАЗ.
#   foreground:  ./DataProcessor/scripts/ar_run_daemon.sh
#   в фоне:      nohup ./DataProcessor/scripts/ar_run_daemon.sh >/dev/null 2>&1 &
set -Eeuo pipefail
DP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# предпочитаем venv VisualProcessor (там torch+cuda, pytorchvideo, ultralytics)
PY="$DP/VisualProcessor/.vp_venv/bin/python"
[ -x "$PY" ] || PY="$DP/.data_venv/bin/python"
[ -x "$PY" ] || PY="python3"
export DP_MODELS_ROOT="${DP_MODELS_ROOT:-$DP/dp_models}"
echo "[ar-daemon] python=$PY DP_MODELS_ROOT=$DP_MODELS_ROOT"
exec "$PY" "$DP/scripts/ar_run_daemon.py"
