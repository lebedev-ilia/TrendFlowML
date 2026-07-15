#!/usr/bin/env bash
# Открывает терминал с установкой NVIDIA Container Toolkit (нужен пароль sudo/pkexec).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DISPLAY="${DISPLAY:-:0}"
export DISPLAY
exec x-terminal-emulator -e bash -lc "
  cd '$REPO_ROOT'
  echo '=== NVIDIA Container Toolkit для Docker GPU (Triton E2E) ==='
  ./backend/scripts/install_nvidia_container_toolkit.sh --from-debs
  echo
  ./backend/scripts/install_nvidia_container_toolkit.sh --verify-only && echo 'Успех — можно запускать e2e_run_full_green.sh'
  echo
  read -r -p 'Enter для закрытия…'
"
