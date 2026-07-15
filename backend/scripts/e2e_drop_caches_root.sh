#!/usr/bin/env bash
# Root-only: sync + drop page cache. Вызывается через passwordless sudo (см. setup_e2e_drop_caches_nopasswd.sh).
set -euo pipefail
sync
echo 3 >/proc/sys/vm/drop_caches
