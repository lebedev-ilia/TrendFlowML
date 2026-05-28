#!/usr/bin/env bash
set -euo pipefail
ROOT='/media/ilya/Новый том/TrendFlowML'
cd "$ROOT"
source ./backend/scripts/e2e_env.sh
source ./backend/.venv/bin/activate
cd "$ROOT/backend"
./scripts/stop_e2e_stack.sh
./scripts/start_e2e_stack.sh --with-infra

# Только вывод оркестратора (python) — компактный heartbeat; полная история в
# storage/e2e_full_max/<run_tag>/orchestrator_events.jsonl по каждому ролику.
LOG_DIR="$ROOT/backend/.e2e/logs"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%d_%H%M%S_utc)"
TERMINAL_LOG="${E2E_TERMINAL_LOG:-$LOG_DIR/e2e_terminal_${TS}.log}"
ln -sfn "$TERMINAL_LOG" "$LOG_DIR/e2e_terminal_latest.log"

echo "E2E terminal log (stdout+stderr): $TERMINAL_LOG" >&2
# Сколько роликов из встроенного плана (макс. 20: см. builtin_example_suite_items в e2e_full_max_run.py).
SUITE_N=10
exec python -u scripts/e2e_full_max_run.py \
  --example-suite-7 \
  --example-suite-count "$SUITE_N" \
  --with-triton-docker \
  --example-suite-force-all \
  --e2e-low-vram \
  >"$TERMINAL_LOG" 2>&1
