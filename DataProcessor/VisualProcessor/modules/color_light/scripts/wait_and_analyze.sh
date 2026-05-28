#!/bin/bash
# Скрипт для ожидания завершения тестов и автоматического анализа

echo "Ожидание завершения тестов color_light..."
echo "Проверка каждые 30 секунд..."

while true; do
    ACTIVE=$(ps aux | grep -E "run_tests.sh|main.py.*test_color_light" | grep -v grep | wc -l)
    SUCCESS=$(ls -1 /media/ilya/Новый\ том/TrendFlowML/DataProcessor/dp_results/youtube/test_color_light_*/test_color_light_*/color_light/color_light_features.npz 2>/dev/null | wc -l)
    
    echo "[$(date +%H:%M:%S)] Активных: ${ACTIVE}, Успешных: ${SUCCESS}"
    
    if [ ${ACTIVE} -eq 0 ]; then
        echo ""
        echo "Все тесты завершены!"
        echo "Успешных результатов: ${SUCCESS}"
        echo ""
        echo "Запуск финального анализа..."
        echo "=========================================="
        
        cd "/media/ilya/Новый том/TrendFlowML"
        
        # Валидация
        echo ""
        echo "=== Валидация ==="
        DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/VisualProcessor/modules/color_light/utils/validate_color_light.py \
            --results-base DataProcessor/dp_results --platform-id youtube 2>&1
        
        # Анализ
        echo ""
        echo "=== Анализ результатов ==="
        DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/VisualProcessor/modules/color_light/utils/analyze_all_results.py \
            --results-base DataProcessor/dp_results 2>&1
        
        break
    fi
    
    sleep 30
done

