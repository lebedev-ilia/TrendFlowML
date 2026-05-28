#!/bin/bash
# Скрипт для проверки прогресса тестирования

RESULTS_DIR="/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/youtube"

echo "Проверка прогресса тестирования color_light..."
echo "=========================================="

# Количество успешных результатов
SUCCESS_COUNT=$(ls -1 ${RESULTS_DIR}/test_color_light_*/test_color_light_*/color_light/color_light_features.npz 2>/dev/null | wc -l)
echo "Успешных результатов: ${SUCCESS_COUNT}"

# Список успешных
if [ ${SUCCESS_COUNT} -gt 0 ]; then
    echo ""
    echo "Успешные тесты:"
    ls -1 ${RESULTS_DIR}/test_color_light_*/test_color_light_*/color_light/color_light_features.npz 2>/dev/null | \
        sed 's|.*/test_color_light_\([^/]*\)/.*|\1|' | sort -V
fi

# Проверка активных процессов
ACTIVE=$(ps aux | grep -E "main.py.*test_color_light" | grep -v grep | wc -l)
echo ""
echo "Активных процессов: ${ACTIVE}"

if [ ${ACTIVE} -gt 0 ]; then
    echo ""
    echo "Текущие процессы:"
    ps aux | grep -E "main.py.*test_color_light" | grep -v grep | awk '{print $NF}' | sed 's|.*test_color_light_\([^ ]*\).*|\1|' | sort -V
fi

echo ""
echo "Последние строки лога:"
tail -5 /tmp/color_light_all_tests_retry.log 2>/dev/null || echo "Лог еще не создан"

