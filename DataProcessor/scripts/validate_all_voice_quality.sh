#!/bin/bash
# Скрипт для валидации всех результатов voice_quality_extractor (аналог validate_all_action_recognition.sh)

set -e

DP_ROOT="/media/ilya/Новый том/TrendFlowML/DataProcessor"
RS_BASE="$DP_ROOT/dp_results/youtube"
VALIDATOR="$DP_ROOT/AudioProcessor/src/extractors/voice_quality_extractor/utils/validate_voice_quality.py"

cd "$DP_ROOT"

echo "=== Валидация всех результатов voice_quality_extractor ==="
echo ""

total=0
valid=0
invalid=0

# test_voice_quality_shortest + test_voice_quality_2..20
rids=("test_voice_quality_shortest" "test_voice_quality_2" "test_voice_quality_3" "test_voice_quality_4" "test_voice_quality_5" \
      "test_voice_quality_6" "test_voice_quality_7" "test_voice_quality_8" "test_voice_quality_9" "test_voice_quality_10" \
      "test_voice_quality_11" "test_voice_quality_12" "test_voice_quality_13" "test_voice_quality_14" "test_voice_quality_15" \
      "test_voice_quality_16" "test_voice_quality_17" "test_voice_quality_18" "test_voice_quality_19" "test_voice_quality_20")

for rid in "${rids[@]}"; do

    npz_path="$RS_BASE/$rid/$rid/voice_quality_extractor/voice_quality_extractor_features.npz"

    if [ ! -f "$npz_path" ]; then
        echo "⚠️  $rid: файл не найден ($npz_path)"
        continue
    fi

    echo "Валидация $rid..."
    result=$("$DP_ROOT/.data_venv/bin/python3" "$VALIDATOR" "$npz_path" 2>&1)

    if echo "$result" | grep -q "✅ VALID"; then
        echo "  ✅ VALID"
        valid=$((valid + 1))
    else
        echo "  ❌ INVALID"
        echo "$result" | grep -E "(Error|Warning)" | head -5
        invalid=$((invalid + 1))
    fi

    total=$((total + 1))
    echo ""
done

echo "=================================================================================="
echo "Итого: $total файлов, ✅ $valid валидных, ❌ $invalid невалидных"
echo "=================================================================================="
