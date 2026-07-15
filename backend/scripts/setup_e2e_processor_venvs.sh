#!/usr/bin/env bash
# Processor venvs для полного E2E (Audio/Text/Visual subprocesses в DataProcessor/main.py).
#
# Usage:
#   ./backend/scripts/setup_e2e_processor_venvs.sh
#   ./backend/scripts/setup_e2e_processor_venvs.sh --data-venv-only   # только .data_venv (fallback)
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DP="$REPO_ROOT/DataProcessor"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
DATA_ONLY=0
for arg in "$@"; do
  [[ "$arg" == "--data-venv-only" ]] && DATA_ONLY=1
done

pip_install_torch() {
  local pip="$1"
  if "$pip" show torch >/dev/null 2>&1; then
    echo "  torch already installed in $(dirname "$pip")"
    return 0
  fi
  echo "  installing torch (cu126) …"
  "$pip" install -q -U pip wheel
  "$pip" install torch torchvision torchaudio --index-url "$TORCH_INDEX"
}

setup_tp_venv() {
  local venv="$DP/TextProcessor/.tp_venv"
  [[ "$DATA_ONLY" -eq 1 ]] && return 0
  echo "==> TextProcessor .tp_venv"
  [[ -d "$venv" ]] || python3 -m venv "$venv"
  pip_install_torch "$venv/bin/pip"
  # Align torch/torchvision (mismatch breaks sentence_transformers / TitleEmbedder).
  "$venv/bin/pip" install -q torch==2.12.1 torchvision torchaudio --index-url "$TORCH_INDEX" 2>/dev/null || true
  if [[ -f "$DP/TextProcessor/requirements.txt" ]]; then
    # scipy==1.15.3 из requirements.txt не имеет wheel для Python 3.14 — ставим совместимую версию отдельно.
    "$venv/bin/pip" install -q 'scipy>=1.16' 'numpy>=1.26,<2.1' || "$venv/bin/pip" install -q 'scipy>=1.16'
    grep -v '^scipy==' "$DP/TextProcessor/requirements.txt" | grep -v '^numpy==' > /tmp/tp_req_filtered.txt || true
    "$venv/bin/pip" install -q -r /tmp/tp_req_filtered.txt || true
  fi
  "$venv/bin/pip" install -q sentence-transformers 2>/dev/null || true
  "$venv/bin/python" -c "import torch, emoji; print('tp_venv ok', torch.__version__)"
}

setup_ap_venv() {
  local venv="$DP/AudioProcessor/.ap_venv"
  [[ "$DATA_ONLY" -eq 1 ]] && return 0
  echo "==> AudioProcessor .ap_venv"
  [[ -d "$venv" ]] || python3 -m venv "$venv"
  pip_install_torch "$venv/bin/pip"
  "$venv/bin/pip" install -q \
    'numpy>=1.26' 'scipy>=1.16' PyYAML psutil librosa soundfile soxr numba onnxruntime \
    transformers huggingface_hub emoji pyannote.audio torch-audiomentations speechbrain \
    openai-whisper demucs \
    "git+https://github.com/LAION-AI/CLAP.git"
  "$venv/bin/python" -c "import torch, librosa, laion_clap; print('ap_venv ok', torch.__version__)"
}

setup_vp_venv() {
  local venv="$DP/VisualProcessor/.vp_venv"
  [[ "$DATA_ONLY" -eq 1 ]] && return 0
  echo "==> VisualProcessor .vp_venv"
  [[ -d "$venv" ]] || python3 -m venv "$venv"
  pip_install_torch "$venv/bin/pip"
  "$venv/bin/pip" install -q \
    numpy 'scipy>=1.16' opencv-python-headless Pillow PyYAML psutil onnxruntime \
    tritonclient[http] gevent requests ultralytics pytorchvideo \
    scikit-learn scikit-image pandas librosa soundfile \
    "git+https://github.com/openai/CLIP.git"
  "$venv/bin/python" -c "import torch, cv2, requests, ultralytics, scipy, clip, sklearn, pandas; print('vp_venv ok', torch.__version__)"
}

setup_core_face_landmarks_venv() {
  local venv="$DP/VisualProcessor/core/model_process/core_face_landmarks/.core_face_landmarks_venv"
  [[ "$DATA_ONLY" -eq 1 ]] && return 0
  echo "==> core_face_landmarks .core_face_landmarks_venv (Python 3.12 + mediapipe 0.10.14)"
  local py312=""
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.12 >/dev/null 2>&1 || true
    py312="$(uv python find 3.12 2>/dev/null || true)"
  fi
  if [[ -z "$py312" ]] && command -v python3.12 >/dev/null 2>&1; then
    py312="$(command -v python3.12)"
  fi
  if [[ -z "$py312" ]]; then
    echo "WARN: Python 3.12 not found (uv python install 3.12). core_face_landmarks will fail on Python 3.14." >&2
    return 0
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    "$py312" -m venv "$venv"
  fi
  "$venv/bin/pip" install -q -U pip wheel
  if ! "$venv/bin/pip" install -q mediapipe==0.10.14 opencv-python-headless 'numpy>=1.23' 2>/dev/null; then
    echo "WARN: mediapipe==0.10.14 install failed in face_landmarks venv" >&2
    return 0
  fi
  "$venv/bin/python" -c "import mediapipe as mp; assert hasattr(mp, 'solutions'); print('face_landmarks venv ok')"
}

setup_action_recognition_venv() {
  local venv="$DP/VisualProcessor/modules/action_recognition/.action_recognition_venv"
  [[ "$DATA_ONLY" -eq 1 ]] && return 0
  echo "==> action_recognition .action_recognition_venv"
  [[ -d "$venv" ]] || python3 -m venv "$venv"
  pip_install_torch "$venv/bin/pip"
  "$venv/bin/pip" install -q numpy opencv-python-headless pytorchvideo 'scipy>=1.16' scikit-learn
  "$venv/bin/python" -c "import torch, pytorchvideo; print('action_recognition venv ok')"
}

setup_data_venv_fallback() {
  local venv="$DP/.data_venv"
  echo "==> DataProcessor .data_venv (API + processor fallback)"
  [[ -d "$venv" ]] || python3 -m venv "$venv"
  [[ -f "$DP/requirements-api.txt" ]] && "$venv/bin/pip" install -q -r "$DP/requirements-api.txt"
  [[ -f "$DP/embedding_service/requirements-e2e.txt" ]] && \
    "$venv/bin/pip" install -q -r "$DP/embedding_service/requirements-e2e.txt"
  pip_install_torch "$venv/bin/pip"
  [[ -f "$DP/TextProcessor/requirements.txt" ]] && {
    grep -v '^scipy==' "$DP/TextProcessor/requirements.txt" | grep -v '^numpy==' > /tmp/tp_req_filtered.txt || true
    "$venv/bin/pip" install -q 'scipy>=1.16' emoji || true
    "$venv/bin/pip" install -q -r /tmp/tp_req_filtered.txt || true
  }
  "$venv/bin/pip" install -q librosa soundfile opencv-python-headless tritonclient[http] gevent
  "$venv/bin/python" -c "import torch, emoji, cv2, librosa; print('data_venv ok', torch.__version__)"
}

setup_tp_venv
setup_ap_venv
setup_vp_venv
setup_core_face_landmarks_venv
setup_action_recognition_venv
setup_data_venv_fallback
echo "Processor venvs ready."
