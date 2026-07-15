#!/usr/bin/env bash
# Скачивание аудио/визуальных in-process весов для полного E2E (без HF_TOKEN).
#
# Usage:
#   ./backend/scripts/setup_e2e_audio_models.sh
#   SKIP_NETWORK=1 ./backend/scripts/setup_e2e_audio_models.sh   # только проверка
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DP="$REPO_ROOT/DataProcessor"
BUNDLE="$DP/dp_models/bundled_models"
AP_PY="$DP/AudioProcessor/.ap_venv/bin/python"
VP_PY="$DP/VisualProcessor/.vp_venv/bin/python"
SKIP_NETWORK="${SKIP_NETWORK:-0}"

# HF_TOKEN из backend/.e2e/secrets.env (см. e2e_secrets.env.example)
_E2E_SECRETS="$SCRIPT_DIR/../.e2e/secrets.env"
if [[ -f "$_E2E_SECRETS" ]]; then
  # shellcheck source=/dev/null
  set -a
  source "$_E2E_SECRETS"
  set +a
fi

mkdir -p "$BUNDLE/audio/laion_clap" "$BUNDLE/audio/whisper" "$BUNDLE/audio/source_separation"
mkdir -p "$BUNDLE/visual/action_recognition/slowfast_r50"

download_clap() {
  local dst="$BUNDLE/audio/laion_clap/clap_ckpt.pt"
  if [[ -f "$dst" ]]; then
    echo "OK: CLAP $dst"
    return 0
  fi
  [[ "$SKIP_NETWORK" == "1" ]] && { echo "WARN: missing $dst (SKIP_NETWORK=1)" >&2; return 0; }
  local url="https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt"
  echo "Downloading CLAP → $dst …"
  curl -fL --retry 5 --retry-delay 10 -o "${dst}.tmp" "$url"
  mv "${dst}.tmp" "$dst"
  echo "OK: CLAP saved"
}

download_fba_file() {
  local url="$1" dst="$2"
  if [[ -f "$dst" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  echo "Downloading (curl) → $dst …"
  curl -fL --retry 10 --retry-delay 15 --retry-all-errors -C - -o "${dst}.tmp" "$url"
  mv "${dst}.tmp" "$dst"
}

ensure_ap_deps() {
  [[ -x "$AP_PY" ]] || {
    echo "Installing AudioProcessor .ap_venv …" >&2
    "$SCRIPT_DIR/setup_e2e_processor_venvs.sh"
  }
  if ! "$AP_PY" -c "import whisper" 2>/dev/null; then
    echo "Installing openai-whisper in .ap_venv …"
    "$AP_PY" -m pip install -q openai-whisper
  fi
  if ! "$AP_PY" -c "from demucs.pretrained import get_model" 2>/dev/null; then
    echo "Installing demucs in .ap_venv …"
    "$AP_PY" -m pip install -q demucs
  fi
  if ! "$AP_PY" -c "import laion_clap" 2>/dev/null; then
    echo "Installing laion-clap in .ap_venv …"
    "$AP_PY" -m pip install -q "git+https://github.com/LAION-AI/CLAP.git"
  fi
}

bootstrap_clap_tokenizers() {
  [[ "$SKIP_NETWORK" == "1" ]] && return 0
  echo "Pre-caching CLAP HF backbones (tokenizers + weights for offline E2E) …"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    HF_HOME="$BUNDLE/hf_cache" \
    "$AP_PY" - <<'PY'
from huggingface_hub import snapshot_download

# CLAP loads bert, roberta, bart text encoders offline — tokenizer-only snapshots are insufficient.
repos = [
    "bert-base-uncased",
    "roberta-base",
    "facebook/bart-base",
]
for repo in repos:
    snapshot_download(repo_id=repo)
    print(f"OK: {repo}")
PY
}

download_whisper() {
  local dst="$BUNDLE/audio/whisper/small.pt"
  if [[ -f "$dst" ]]; then
    echo "OK: whisper $dst"
    return 0
  fi
  [[ "$SKIP_NETWORK" == "1" ]] && { echo "WARN: missing $dst (SKIP_NETWORK=1)" >&2; return 0; }
  ensure_ap_deps
  echo "Downloading Whisper small.pt …"
  "$AP_PY" "$DP/scripts/download_whisper_models.py" --models-root "$BUNDLE" --sizes small
}

download_source_separation() {
  local dst="$BUNDLE/audio/source_separation/large.pt"
  if [[ -f "$dst" ]]; then
    echo "OK: source_separation $dst"
    return 0
  fi
  [[ "$SKIP_NETWORK" == "1" ]] && { echo "WARN: missing $dst (SKIP_NETWORK=1)" >&2; return 0; }
  ensure_ap_deps
  local torch_home="$BUNDLE/torch_cache"
  mkdir -p "$torch_home/hub/checkpoints"
  local demucs_ckpt="$torch_home/hub/checkpoints/955717e8-8726e21a.th"
  download_fba_file \
    "https://huggingface.co/iBoostAI/Demucs-v4/resolve/main/955717e8-8726e21a.th" \
    "$demucs_ckpt"
  echo "Building source_separation large.pt (Demucs wrapper) …"
  TORCH_HOME="$torch_home" \
    "$AP_PY" "$DP/scripts/download_source_separation_models.py" --models-root "$BUNDLE" --sizes large
}

download_emotion_diarization() {
  local hp="$BUNDLE/audio/emotion_diarization/wavlm_large/hyperparams.yaml"
  if [[ -f "$hp" ]]; then
    echo "OK: emotion_diarization $hp"
    return 0
  fi
  [[ "$SKIP_NETWORK" == "1" ]] && {
    echo "WARN: missing $hp (SKIP_NETWORK=1; emotion_diarization stays disabled in E2E)" >&2
    return 0
  }
  ensure_ap_deps
  echo "Downloading SpeechBrain emotion-diarization-wavlm-large bundle …"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    HF_HOME="$BUNDLE/hf_cache" \
    "$AP_PY" "$DP/scripts/save_emotion_diarization_bundle.py" --models-root "$BUNDLE"
}

download_slowfast() {
  local dst="$BUNDLE/visual/action_recognition/slowfast_r50/slowfast_r50.pyth"
  if [[ -f "$dst" ]]; then
    echo "OK: slowfast $dst"
    return 0
  fi
  if [[ "$SKIP_NETWORK" == "1" ]]; then
    echo "WARN: missing $dst (SKIP_NETWORK=1; action_recognition stays disabled in E2E)" >&2
    return 0
  fi
  [[ -x "$VP_PY" ]] || {
    echo "Installing VisualProcessor .vp_venv …" >&2
    "$SCRIPT_DIR/setup_e2e_processor_venvs.sh"
  }
  local hub_ckpt="$BUNDLE/torch_cache/hub/checkpoints/SLOWFAST_8x8_R50.pyth"
  if [[ ! -f "$hub_ckpt" ]] && [[ "${SKIP_SLOWFAST:-0}" != "1" ]]; then
    echo "Downloading SlowFast via pytorchvideo hub (may take several minutes) …"
    TORCH_HOME="$BUNDLE/torch_cache" \
      timeout 900 "$VP_PY" "$DP/scripts/save_slowfast_r50_checkpoint.py" --models-root "$BUNDLE" && return 0
  fi
  if [[ -f "$hub_ckpt" ]]; then
    echo "Converting SlowFast hub checkpoint → bundled .pyth …"
    TORCH_HOME="$BUNDLE/torch_cache" \
      "$VP_PY" "$DP/scripts/save_slowfast_r50_checkpoint.py" \
        --models-root "$BUNDLE" \
        --input-checkpoint "$hub_ckpt"
    return 0
  fi
  echo "WARN: missing $dst (SlowFast download skipped; action_recognition disabled in E2E)" >&2
}

download_ppocr_rec() {
  local onnx="$BUNDLE/visual/ocr/ppocr_rec_onnx_v1/model.onnx"
  if [[ -f "$onnx" ]]; then
    echo "OK: ppocr $onnx"
    return 0
  fi
  [[ "$SKIP_NETWORK" == "1" ]] && {
    echo "WARN: missing $onnx (SKIP_NETWORK=1; ocr_extractor stays disabled in E2E)" >&2
    return 0
  }
  [[ -x "$AP_PY" ]] || "$SCRIPT_DIR/setup_e2e_processor_venvs.sh"
  echo "Downloading PP-OCR rec ONNX (monkt/paddleocr-onnx, eslav) …"
  HF_HUB_OFFLINE=0 \
    "$AP_PY" "$DP/scripts/save_ppocr_rec_onnx_bundle.py" --models-root "$BUNDLE" --lang eslav
}

download_pyannote() {
  local cfg="$BUNDLE/audio/pyannote_speaker_diarization/config.yaml"
  if [[ -f "$cfg" ]]; then
    echo "OK: pyannote $cfg"
    return 0
  fi
  if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGINGFACE_TOKEN:-}" ]]; then
    echo "WARN: missing $cfg (no HF_TOKEN; speaker_diarization stays disabled in E2E)" >&2
    return 0
  fi
  [[ "$SKIP_NETWORK" == "1" ]] && return 0
  ensure_ap_deps
  echo "Downloading pyannote/speaker-diarization-community-1 (gated, HF_TOKEN) …"
  HF_HUB_OFFLINE=0 \
    "$AP_PY" "$DP/scripts/save_pyannote_community_bundle.py" --models-root "$BUNDLE"
}

download_clap
bootstrap_clap_tokenizers
download_whisper
download_source_separation
download_emotion_diarization
download_slowfast
download_ppocr_rec
download_pyannote

echo "==> E2E audio/visual model bundle summary"
for f in \
  "$BUNDLE/audio/laion_clap/clap_ckpt.pt" \
  "$BUNDLE/audio/whisper/small.pt" \
  "$BUNDLE/audio/source_separation/large.pt" \
  "$BUNDLE/audio/emotion_diarization/wavlm_large/hyperparams.yaml" \
  "$BUNDLE/audio/pyannote_speaker_diarization/config.yaml" \
  "$BUNDLE/visual/action_recognition/slowfast_r50/slowfast_r50.pyth" \
  "$BUNDLE/visual/ocr/ppocr_rec_onnx_v1/model.onnx"
do
  if [[ -f "$f" ]]; then
    ls -lh "$f"
  else
    echo "MISSING: $f" >&2
  fi
done
