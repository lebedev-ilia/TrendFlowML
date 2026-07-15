#!/bin/bash
# Прогон набора B (разные длины) + сбор summary для behavioral.
# Видео берём из /workspace/scene_videos. Запуск: bash beh_batch.sh "vid1 vid2 ..."
set -u
cd /workspace/TrendFlowML/DataProcessor
export DP_MODELS_ROOT=/workspace/TrendFlowML/DataProcessor/dp_models
PY=/workspace/venv/bin/python
VIDS="$1"
for vid in $VIDS; do
  V=/workspace/scene_videos/${vid}.mp4
  if [ ! -f "$V" ]; then echo "SKIP no file $V"; continue; fi
  echo "=== RUN $vid ==="
  $PY scripts/run_behavioral_local.py --video "$V" --video-id "beh_${vid}" \
      --workdir "/workspace/beh_out/${vid}" --device cuda > "/workspace/beh_out/${vid}.log" 2>&1
  echo "  rc=$? summary:"; cat "/workspace/beh_out/${vid}/summary.json" 2>/dev/null; echo
done
echo "=== BATCH DONE ==="
