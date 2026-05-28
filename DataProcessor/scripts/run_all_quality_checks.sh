#!/bin/bash
# Скрипт для запуска всех проверок качества action_recognition

set -e

DP_ROOT="/media/ilya/Новый том/TrendFlowML/DataProcessor"

cd "$DP_ROOT"

echo "=================================================================================="
echo "ПОЛНАЯ ПРОВЕРКА КАЧЕСТВА action_recognition"
echo "=================================================================================="
echo ""

# 1. Валидация всех артефактов
echo "1. Валидация всех артефактов..."
echo "----------------------------------------------------------------------------------"
bash scripts/validate_all_action_recognition.sh
echo ""

# 2. Статистический анализ
echo "2. Статистический анализ всех результатов..."
echo "----------------------------------------------------------------------------------"
"$DP_ROOT/.data_venv/bin/python3" VisualProcessor/modules/action_recognition/utils/analyze_all_results.py
echo ""

# 3. Проверка качества эмбеддингов (на примере нескольких видео)
echo "3. Проверка качества эмбеддингов..."
echo "----------------------------------------------------------------------------------"
for vid in "test_action_recognition_v3" "test_action_recognition_v8" "test_action_recognition_v15"; do
    ar_dir="$DP_ROOT/dp_results/youtube/tests/action_recognition/$vid/$vid/action_recognition"
    npz_path=""
    if [ -f "$ar_dir/action_recognition_features.npz" ]; then
        npz_path="$ar_dir/action_recognition_features.npz"
    elif [ -f "$ar_dir/action_recognition_emb.npz" ]; then
        npz_path="$ar_dir/action_recognition_emb.npz"
    fi
    if [ -z "$npz_path" ]; then
        ar_dir="$DP_ROOT/dp_results/youtube/$vid/$vid/action_recognition"
        if [ -f "$ar_dir/action_recognition_features.npz" ]; then
            npz_path="$ar_dir/action_recognition_features.npz"
        elif [ -f "$ar_dir/action_recognition_emb.npz" ]; then
            npz_path="$ar_dir/action_recognition_emb.npz"
        fi
    fi
    if [ -f "$npz_path" ]; then
        echo "Проверка: $vid"
        "$DP_ROOT/.data_venv/bin/python3" VisualProcessor/modules/action_recognition/utils/validate_action_recognition.py "$npz_path" | grep -E "(Status|Stats|Metrics)" | head -10
        echo ""
    fi
done

echo "=================================================================================="
echo "ПРОВЕРКА ЗАВЕРШЕНА"
echo "=================================================================================="
echo ""
echo "Следующие шаги:"
echo "1. Откройте HTML рендеры в браузере для визуальной проверки"
echo "2. Проверьте отчёт TESTING_REPORT.md"
echo "3. Сравните результаты между похожими видео"
echo ""

