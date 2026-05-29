#!/bin/bash
# Валидация P0.4 visual minimal: frames_dir + core_object_detections NPZ.
# action_recognition — только при CUDA (на CPU-only — ожидаемый skip).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DP_ROOT="$REPO_ROOT/DataProcessor"
PYTHON="${PYTHON:-$DP_ROOT/VisualProcessor/.vp_venv/bin/python3}"

VIDEO_ID="${VIDEO_ID:--Q6fnPIybEI}"
RUN_ID="${RUN_ID:-ar_minimal_cli_001}"
PLATFORM_ID="${PLATFORM_ID:-youtube}"
FRAMES_DIR="${FRAMES_DIR:-$REPO_ROOT/storage/frames_dir/$VIDEO_ID/video}"
RS_BASE="${RS_BASE:-$REPO_ROOT/storage/result_store_ar_minimal}"
DETECTIONS_NPZ="$RS_BASE/$PLATFORM_ID/$VIDEO_ID/$RUN_ID/core_object_detections/detections.npz"
VALIDATOR="$DP_ROOT/VisualProcessor/core/model_process/core_object_detections/utils/validate_core_object_detections_npz.py"
AR_NPZ="$RS_BASE/$PLATFORM_ID/$VIDEO_ID/$RUN_ID/action_recognition/action_recognition.npz"

HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_CUDA=1
fi

echo "=== Visual minimal validation ==="
echo "frames_dir: $FRAMES_DIR"
echo "detections: $DETECTIONS_NPZ"
echo ""

invalid=0
skipped=0

if [ ! -d "$FRAMES_DIR" ]; then
    echo "❌ frames_dir не найден: $FRAMES_DIR"
    exit 1
fi
if [ ! -f "$FRAMES_DIR/metadata.json" ]; then
    echo "❌ metadata.json отсутствует в frames_dir"
    invalid=1
else
    echo "✅ frames_dir + metadata.json"
fi

if [ ! -f "$DETECTIONS_NPZ" ]; then
    echo "❌ detections.npz не найден: $DETECTIONS_NPZ"
    invalid=1
elif [ ! -x "$PYTHON" ]; then
    echo "⚠️  VisualProcessor venv не найден ($PYTHON), пропуск NPZ schema check"
elif [ ! -f "$VALIDATOR" ]; then
    echo "⚠️  validator не найден: $VALIDATOR"
    invalid=1
else
    result=$("$PYTHON" "$VALIDATOR" "$DETECTIONS_NPZ" 2>&1) || true
    if echo "$result" | grep -q "✅ VALID"; then
        echo "✅ core_object_detections (detections.npz)"
    else
        echo "❌ core_object_detections validation failed"
        echo "$result" | tail -5
        invalid=1
    fi
fi

if [ "$HAS_CUDA" -eq 0 ]; then
    echo "⏭️  action_recognition: SKIP (no CUDA)"
    ((skipped++)) || true
elif [ -f "$AR_NPZ" ]; then
    echo "✅ action_recognition.npz present"
else
    echo "❌ action_recognition.npz не найден (CUDA доступна)"
    invalid=1
fi

echo ""
echo "Итого: ❌ $invalid проблем, ⏭️  $skipped пропущено (GPU-only)"
[ "$invalid" -eq 0 ]
