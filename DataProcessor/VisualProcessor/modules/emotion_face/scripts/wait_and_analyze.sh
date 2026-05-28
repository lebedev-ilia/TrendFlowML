#!/bin/bash
# Скрипт для ожидания завершения тестов и автоматического запуска валидации и анализа

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"}"

RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
VALIDATOR="${VALIDATOR:-"${BASE_DIR}/DataProcessor/VisualProcessor/modules/emotion_face/validate_emotion_face.py"}"
ANALYZER="${ANALYZER:-"${BASE_DIR}/DataProcessor/VisualProcessor/modules/emotion_face/analyze_all_results.py"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"

cd "${BASE_DIR}"

echo "Мониторинг прогресса тестов emotion_face..."
echo "Лог: /tmp/emotion_face_all_tests.log"
echo ""

# Функция для подсчета завершенных тестов
count_completed() {
    find "${RESULTS_DIR}/youtube/test_emotion_face_*" -name "emotion_face.npz" -size +1k 2>/dev/null | wc -l
}

# Функция для подсчета активных процессов
count_active() {
    ps aux | grep -E "main.py.*test_emotion_face" | grep -v grep | wc -l
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
    
    # Если все тесты завершены и нет активных процессов
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
"${PYTHON}" "${VALIDATOR}" \
    --results-base "${RESULTS_DIR}" \
    --platform-id youtube \
    2>&1 | tee /tmp/emotion_face_validation.log

echo ""
echo "Запуск анализа..."
echo "----------------------------------------"
"${PYTHON}" "${ANALYZER}" \
    --results-base "${RESULTS_DIR}" \
    2>&1 | tee /tmp/emotion_face_analysis.log

echo ""
echo "=========================================="
echo "Валидация и анализ завершены"
echo "=========================================="
echo "Логи:"
echo "  Валидация: /tmp/emotion_face_validation.log"
echo "  Анализ: /tmp/emotion_face_analysis.log"

