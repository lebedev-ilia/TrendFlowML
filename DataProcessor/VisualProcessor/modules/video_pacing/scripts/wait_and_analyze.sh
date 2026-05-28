#!/bin/bash
# Скрипт для ожидания завершения тестов video_pacing и запуска валидации/анализа.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"}"

RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
VALIDATOR="${VALIDATOR:-"${BASE_DIR}/DataProcessor/VisualProcessor/modules/video_pacing/utils/validate_video_pacing.py"}"
ANALYZER="${ANALYZER:-"${BASE_DIR}/DataProcessor/VisualProcessor/modules/video_pacing/utils/analyze_all_results.py"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"

echo "=========================================="
echo "Ожидание завершения тестов video_pacing"
echo "=========================================="

# Ожидаем появления файлов video_pacing_features.npz
TIMEOUT=3600  # 1 час
CHECK_INTERVAL=30  # проверка каждые 30 секунд
ELAPSED=0

while [ $ELAPSED -lt $TIMEOUT ]; do
    COUNT=$(find "${RESULTS_DIR}/youtube" -type f -path '*test_video_pacing_*/video_pacing/video_pacing_features.npz' 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] Найдено файлов: ${COUNT}/20"
    
    if [ $COUNT -ge 20 ]; then
        echo "✅ Все файлы созданы!"
        break
    fi
    
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "⚠️ Таймаут ожидания. Продолжаем с найденными файлами..."
fi

echo ""
echo "=========================================="
echo "Запуск валидации"
echo "=========================================="

cd "${BASE_DIR}"
PYTHONPATH="DataProcessor/VisualProcessor:${PYTHONPATH}" "${PYTHON}" "${VALIDATOR}" \
    --results-base "${RESULTS_DIR}" \
    --platform-id youtube

echo ""
echo "=========================================="
echo "Запуск анализа"
echo "=========================================="

PYTHONPATH="DataProcessor/VisualProcessor:${PYTHONPATH}" "${PYTHON}" "${ANALYZER}" \
    --results-base "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Готово!"
echo "=========================================="

