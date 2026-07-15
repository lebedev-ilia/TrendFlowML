#!/usr/bin/env bash
# Настройка пода RunPod для прогонов DataProcessor (запускать НА ПОДЕ после rsync кода+весов).
# Идемпотентно. После него `run_ar_local.py --device cuda` работает.
#
#   ssh <pod> 'bash /workspace/TrendFlowML/automation/runpod_ssh/pod_setup.sh'
set -Eeuo pipefail
REPO="${REPO:-/workspace/TrendFlowML}"
DP="$REPO/DataProcessor"
export DP_MODELS_ROOT="$DP/dp_models"

echo "[pod_setup] apt: update + rsync ffmpeg (на свежем контейнере списки пакетов пусты!)"
DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null 2>&1 || true
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync ffmpeg >/dev/null 2>&1 || true
which ffmpeg >/dev/null 2>&1 && echo "  ffmpeg ok" || echo "  WARN ffmpeg не встал"

echo "[pod_setup] ПЕРСИСТЕНТНЫЙ venv на Network Volume (/workspace/venv) — переживает рестарт пода!"
# --system-site-packages: наследуем torch/cuda из образа (не качаем ~2.5GB), в venv только extras (~200MB).
VENV=/workspace/venv
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
if ! "$VENV/bin/python" -c "import pytorchvideo,ultralytics,clip,librosa" 2>/dev/null; then
  echo "[pod_setup] ставлю extras в $VENV (один раз, дальше персистентно)"
  "$VENV/bin/pip" install -q \
    pytorchvideo ultralytics opencv-python-headless transformers \
    scipy scikit-learn scikit-image pyyaml pillow ftfy regex librosa soundfile \
    timm huggingface_hub "mediapipe<0.10.15" git+https://github.com/openai/CLIP.git \
    || echo "[warn] часть pip не встала"
fi
# mediapipe 0.10.35 удалил mp.solutions (нужно core_face_landmarks) → форсим рабочую версию
if "$VENV/bin/python" -c "import mediapipe as mp; assert hasattr(mp,'solutions')" 2>/dev/null; then :; else
  echo "[pod_setup] чиню mediapipe (нужен mp.solutions) → <0.10.15"
  "$VENV/bin/pip" install -q "mediapipe<0.10.15" || echo "[warn] mediapipe downgrade failed"
fi
"$VENV/bin/python" -c "import pytorchvideo,ultralytics,clip,librosa,cv2;print('  venv deps OK')" 2>&1 | tail -1
echo "[pod_setup] ИСПОЛЬЗУЙ /workspace/venv/bin/python для прогонов (не системный python3)!"
# опц. альтернативные backbone / ReID: "$VENV/bin/pip" install torchreid

echo "[pod_setup] вес SlowFast → путь spec (ModelManager ждёт visual/action_recognition/slowfast_r50/)"
SF_SRC="$DP_MODELS_ROOT/bundled_models/visual/action_recognition/slowfast_r50/slowfast_r50.pyth"
SF_DST="$DP_MODELS_ROOT/visual/action_recognition/slowfast_r50"
if [ -f "$SF_SRC" ] && [ ! -f "$SF_DST/slowfast_r50.pyth" ]; then
  mkdir -p "$SF_DST" && cp "$SF_SRC" "$SF_DST/slowfast_r50.pyth" && echo "  скопирован SlowFast вес"
fi

echo "[pod_setup] проверка"
python3 -c "import torch,pytorchvideo,ultralytics,cv2,sklearn;print('deps OK, cuda=',torch.cuda.is_available())"
echo "[pod_setup] готово. Прогон:"
echo "  cd $DP && DP_MODELS_ROOT=$DP_MODELS_ROOT python3 scripts/run_ar_local.py --video <mp4> --seconds 0 --fps 25 --device cuda --workdir /workspace/ar_out"
