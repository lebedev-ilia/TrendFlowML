#!/bin/bash
# Скрипт для валидации всех результатов action_recognition

set -e

DP_ROOT="/media/ilya/Новый том/TrendFlowML/DataProcessor"
RS_BASE="$DP_ROOT/dp_results/youtube"

cd "$DP_ROOT"

echo "=== Валидация всех результатов action_recognition ==="
echo ""

total=0
valid=0
invalid=0

# Поддерживаем новую структуру (tests/action_recognition/) и старую (для обратной совместимости)
for i in {1..20}; do
    rid="test_action_recognition_v$i"
    if [ $i -eq 1 ]; then
        rid="test_action_recognition_shortest"
    fi
    
    # Пробуем новую структуру сначала
    ar_dir="$RS_BASE/tests/action_recognition/$rid/$rid/action_recognition"
    npz_path=""
    if [ -f "$ar_dir/action_recognition_features.npz" ]; then
        npz_path="$ar_dir/action_recognition_features.npz"
    elif [ -f "$ar_dir/action_recognition_emb.npz" ]; then
        npz_path="$ar_dir/action_recognition_emb.npz"
    fi
    if [ -z "$npz_path" ]; then
        ar_dir="$RS_BASE/$rid/$rid/action_recognition"
        if [ -f "$ar_dir/action_recognition_features.npz" ]; then
            npz_path="$ar_dir/action_recognition_features.npz"
        elif [ -f "$ar_dir/action_recognition_emb.npz" ]; then
            npz_path="$ar_dir/action_recognition_emb.npz"
        fi
    fi
    
    if [ ! -f "$npz_path" ]; then
        echo "⚠️  v$i ($rid): файл не найден"
        continue
    fi
    
    echo "Валидация v$i ($rid)..."
    result=$("$DP_ROOT/.data_venv/bin/python3" VisualProcessor/modules/action_recognition/utils/validate_action_recognition.py "$npz_path" 2>&1)
    
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

