#!/usr/bin/env bash
# Docker-образ OpenFace для micro_emotion в полном E2E.
#
# Официальный openface/openface:latest часто недоступен (pull denied).
# Скрипт пробует альтернативы и при успехе тегирует в OPENFACE_DOCKER_IMAGE.
#
# Usage:
#   ./backend/scripts/setup_e2e_openface.sh
#   OPENFACE_DOCKER_IMAGE=algebr/openface:latest ./backend/scripts/setup_e2e_openface.sh
#   SKIP_NETWORK=1 ./backend/scripts/setup_e2e_openface.sh   # только проверка
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_IMAGE="${OPENFACE_DOCKER_IMAGE:-openface/openface:latest}"
SKIP_NETWORK="${SKIP_NETWORK:-0}"

docker_image_ok() {
  docker image inspect "$1" >/dev/null 2>&1
}

if docker_image_ok "$TARGET_IMAGE"; then
  echo "OK: OpenFace image present: $TARGET_IMAGE"
  exit 0
fi

if [[ "$SKIP_NETWORK" == "1" ]]; then
  echo "WARN: OpenFace image missing: $TARGET_IMAGE (SKIP_NETWORK=1)" >&2
  echo "  micro_emotion will be auto-disabled in e2e_full_max_run.py" >&2
  exit 0
fi

CANDIDATES=(
  "$TARGET_IMAGE"
  "openface/openface:latest"
  "algebr/openface:latest"
  "garyfeng/openface_prod:latest"
)

seen=""
for img in "${CANDIDATES[@]}"; do
  [[ -z "$img" ]] && continue
  if [[ " $seen " == *" $img "* ]]; then
    continue
  fi
  seen="$seen $img"
  if docker_image_ok "$img"; then
    if [[ "$img" != "$TARGET_IMAGE" ]]; then
      echo "Tagging $img → $TARGET_IMAGE"
      docker tag "$img" "$TARGET_IMAGE"
    fi
    echo "OK: OpenFace image ready: $TARGET_IMAGE"
    exit 0
  fi
done

echo "Pulling OpenFace candidates (this may take several minutes)…"
for img in "${CANDIDATES[@]}"; do
  [[ -z "$img" ]] && continue
  if [[ " $seen " != *" $img "* ]]; then
  seen="$seen $img"
  fi
  if docker_image_ok "$img"; then
    continue
  fi
  echo "Trying: docker pull $img"
  if docker pull "$img"; then
    if [[ "$img" != "$TARGET_IMAGE" ]]; then
      echo "Tagging $img → $TARGET_IMAGE"
      docker tag "$img" "$TARGET_IMAGE"
    fi
    echo "OK: OpenFace image ready: $TARGET_IMAGE (from $img)"
    exit 0
  fi
  echo "WARN: pull failed: $img" >&2
done

echo "FAIL: no OpenFace docker image available." >&2
echo "  Set OPENFACE_DOCKER_IMAGE to a local/working image, or build OpenFace manually." >&2
echo "  E2E will auto-disable micro_emotion without this image." >&2
exit 1
