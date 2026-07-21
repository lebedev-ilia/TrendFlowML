#!/bin/bash
# Setup свежего RunPod-пода под DataProcessor pipeline + Triton.
# Эфемерные apt-зависимости (не персистят на Network Volume) — без них Triton не стартует и метрики/segmenter падают.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg time bc libarchive13 python3-numpy
echo "pod_setup: apt deps installed (ffmpeg time bc libarchive13 python3-numpy)"
# start Triton (бандл + модели уже на volume /workspace)
if [ -f /workspace/start_triton.sh ]; then
  echo "pod_setup: starting Triton (background)..."
  nohup bash /workspace/start_triton.sh > /workspace/triton_srv.log 2>&1 &
  echo "pod_setup: Triton launching; check /workspace/triton_srv.log + curl localhost:8000/v2/health/ready"
fi
