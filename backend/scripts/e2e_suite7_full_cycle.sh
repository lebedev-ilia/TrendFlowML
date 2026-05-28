#!/usr/bin/env bash
# Полный цикл: stop → infra+stack → suite из 7 видео (с Triton).
# Запуск из корня репозитория TrendFlowML:
#   ./backend/scripts/e2e_suite7_full_cycle.sh
#   ./backend/scripts/e2e_suite7_full_cycle.sh --e2e-low-vram
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"

cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$BACKEND_DIR/scripts/e2e_env.sh"
# shellcheck source=/dev/null
source "$BACKEND_DIR/.venv/bin/activate"

"$BACKEND_DIR/scripts/stop_e2e_stack.sh" || true
"$BACKEND_DIR/scripts/start_e2e_stack.sh" --with-infra

cd "$BACKEND_DIR"
exec python -u scripts/e2e_full_max_run.py \
  --example-suite-7 \
  --with-triton-docker \
  --example-suite-force-all \
  "$@"
