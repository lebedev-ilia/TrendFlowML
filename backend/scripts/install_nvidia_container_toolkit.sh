#!/usr/bin/env bash
# Одноразовая установка NVIDIA Container Toolkit для Docker GPU (Triton E2E).
# Требует root (sudo или pkexec — появится запрос пароля).
#
# Usage:
#   ./backend/scripts/install_nvidia_container_toolkit.sh
#   ./backend/scripts/install_nvidia_container_toolkit.sh --verify-only
#   ./backend/scripts/install_nvidia_container_toolkit.sh --from-debs   # только локальные .deb
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEB_DIR="${NVIDIA_CTK_DEB_DIR:-$REPO_ROOT/backend/.e2e/debs}"
CTK_VERSION="${NVIDIA_CTK_VERSION:-1.17.8-1}"
IMAGE="${TRITON_E2E_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}"

VERIFY_ONLY=0
FROM_DEBS_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
    --from-debs) FROM_DEBS_ONLY=1 ;;
  esac
done

verify_gpu_docker() {
  if ! command -v docker >/dev/null; then
    echo "FATAL: docker not found" >&2
    return 1
  fi
  if docker run --rm --gpus all "$IMAGE" nvidia-smi >/dev/null 2>&1; then
    echo "OK: docker run --gpus all → nvidia-smi inside container"
    docker run --rm --gpus all "$IMAGE" nvidia-smi 2>&1 | tail -8
    return 0
  fi
  echo "FAIL: Docker cannot use GPU (--gpus all)." >&2
  return 1
}

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return $?
  fi
  if sudo -n true 2>/dev/null; then
    sudo "$@"
    return $?
  fi
  if command -v pkexec >/dev/null; then
    echo "Запрос пароля (pkexec)…" >&2
    pkexec "$@"
    return $?
  fi
  if command -v sudo >/dev/null; then
    echo "Запрос пароля (sudo)…" >&2
    sudo "$@"
    return $?
  fi
  echo "FATAL: need root (sudo/pkexec)" >&2
  return 1
}

ensure_debs() {
  mkdir -p "$DEB_DIR"
  local ver="$CTK_VERSION" base="https://nvidia.github.io/libnvidia-container/stable/deb/amd64"
  local pkg f
  for pkg in libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit-base nvidia-container-toolkit; do
    f="${pkg}_${ver}_amd64.deb"
    if [[ ! -f "$DEB_DIR/$f" ]]; then
      echo "Downloading $f …" >&2
      curl -fsSL -o "$DEB_DIR/$f" "$base/$f"
    fi
  done
}

install_from_debs() {
  ensure_debs
  local ver="$CTK_VERSION"
  run_as_root bash -c "
    set -Eeuo pipefail
    dpkg -i \
      '$DEB_DIR/libnvidia-container1_${ver}_amd64.deb' \
      '$DEB_DIR/libnvidia-container-tools_${ver}_amd64.deb' \
      '$DEB_DIR/nvidia-container-toolkit-base_${ver}_amd64.deb' \
      '$DEB_DIR/nvidia-container-toolkit_${ver}_amd64.deb'
    command -v nvidia-ctk >/dev/null
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
  "
}

install_from_apt() {
  run_as_root bash -c "
    set -Eeuo pipefail
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
  "
}

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
  verify_gpu_docker
  exit $?
fi

if verify_gpu_docker 2>/dev/null; then
  echo "NVIDIA Container Toolkit already configured."
  exit 0
fi

echo "Installing NVIDIA Container Toolkit…" >&2
if [[ "$FROM_DEBS_ONLY" -eq 1 ]]; then
  install_from_debs
else
  if ! install_from_apt 2>/dev/null; then
    echo "apt install failed — trying local .deb packages from $DEB_DIR" >&2
    install_from_debs
  fi
fi

echo "Waiting for docker…"
sleep 3
verify_gpu_docker
