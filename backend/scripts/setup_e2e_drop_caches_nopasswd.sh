#!/usr/bin/env bash
# Одноразовая настройка: E2E может сбрасывать page cache БЕЗ запроса пароля при каждом прогоне.
#
# Запустите ОДИН РАЗ в терминале (потребуется пароль sudo):
#   ./backend/scripts/setup_e2e_drop_caches_nopasswd.sh
#
# После этого автоматические scrub в e2e_run_full_green.sh смогут вызывать:
#   sudo -n backend/scripts/e2e_drop_caches_root.sh
#
# Без этой настройки E2E всё равно работает — просто без drop_caches (только gc/torch/docker prune).
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_HELPER="$(cd "$SCRIPT_DIR" && pwd)/e2e_drop_caches_root.sh"
SUDOERS_FILE="/etc/sudoers.d/trendflow-e2e-drop-caches"
USER_NAME="${SUDO_USER:-${USER:-ilya}}"

[[ -f "$ROOT_HELPER" ]] || { echo "FATAL: missing $ROOT_HELPER" >&2; exit 1; }
chmod 755 "$ROOT_HELPER"

if ! sudo -v; then
  echo "FATAL: sudo required for one-time setup" >&2
  exit 1
fi

sudo tee "$SUDOERS_FILE" >/dev/null <<EOF
# TrendFlow E2E: passwordless page-cache drop (optional, one-time setup)
${USER_NAME} ALL=(ALL) NOPASSWD: ${ROOT_HELPER}
EOF
sudo chmod 440 "$SUDOERS_FILE"
sudo visudo -cf "$SUDOERS_FILE"

echo "OK: passwordless drop_caches enabled for ${USER_NAME}"
echo "Test: sudo -n ${ROOT_HELPER} && echo drop_caches works"
