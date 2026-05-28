#!/bin/bash
# Скрипт для ожидания завершения тестов и автоматического запуска валидации и анализа (text_scoring).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"}"

RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
VALIDATOR="${VALIDATOR:-"${BASE_DIR}/DataProcessor/VisualProcessor/modules/text_scoring/utils/validate_text_scoring.py"}"
ANALYZER="${ANALYZER:-"${BASE_DIR}/DataProcessor/VisualProcessor/modules/text_scoring/utils/analyze_all_results.py"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"

cd "${BASE_DIR}"

echo "Мониторинг прогресса тестов text_scoring..."
echo "Лог: /tmp/text_scoring_all_tests.log"
echo ""

count_completed() {
    find "${RESULTS_DIR}/youtube/test_text_scoring_*" -name "text_scoring.npz" -size +1k 2>/dev/null | wc -l
}

count_active() {
    ps aux | grep -E "main.py.*test_text_scoring" | grep -v grep | wc -l
}

TOTAL=20
last_completed=0

while true; do
    completed=$(count_completed)
    active=$(count_active)

    if [ "$completed" -ne "$last_completed" ] || [ "$active" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] Активных процессов: $active, Завершено: $completed/$TOTAL"
        last_completed=$completed
    fi

    if [ "$completed" -ge "$TOTAL" ] && [ "$active" -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Все тесты завершены!"
        echo "Завершено: $completed/$TOTAL"
        echo "=========================================="
        echo ""
        break
    fi

    sleep 30
done

echo "Запуск валидации..."
echo "----------------------------------------"
PYTHONPATH="DataProcessor/VisualProcessor:${PYTHONPATH}" "${PYTHON}" "${VALIDATOR}" \
    --results-base "${RESULTS_DIR}" \
    --platform-id youtube \
    2>&1 | tee /tmp/text_scoring_validation.log

echo ""
echo "Запуск анализа..."
echo "----------------------------------------"
PYTHONPATH="DataProcessor/VisualProcessor:${PYTHONPATH}" "${PYTHON}" "${ANALYZER}" \
    --results-base "${RESULTS_DIR}" \
    2>&1 | tee /tmp/text_scoring_analysis.log

echo ""
echo "=========================================="
echo "Валидация и анализ завершены"
echo "=========================================="
echo "Логи:"
echo "  Валидация: /tmp/text_scoring_validation.log"
echo "  Анализ: /tmp/text_scoring_analysis.log"

