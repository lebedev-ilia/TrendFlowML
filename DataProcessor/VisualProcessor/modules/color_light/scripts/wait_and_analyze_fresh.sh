#!/bin/bash
# Скрипт для ожидания завершения тестов и автоматического анализа

echo "Ожидание завершения тестов color_light (все 19 видео)..."
echo "Проверка каждые 30 секунд..."
echo ""

TOTAL_VIDEOS=19
RESULTS_DIR="/media/ilya/Новый том/TrendFlowML/DataProcessor/dp_results/youtube"

while true; do
    ACTIVE=$(ps aux | grep -E "run_tests.sh|main.py.*test_color_light" | grep -v grep | wc -l)
    SUCCESS=$(ls -1 ${RESULTS_DIR}/test_color_light_*/test_color_light_*/color_light/color_light_features.npz 2>/dev/null | wc -l)
    
    echo "[$(date +%H:%M:%S)] Активных процессов: ${ACTIVE}, Успешных результатов: ${SUCCESS}/${TOTAL_VIDEOS}"
    
    if [ ${ACTIVE} -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Все тесты завершены!"
        echo "Успешных результатов: ${SUCCESS}/${TOTAL_VIDEOS}"
        echo "=========================================="
        echo ""
        
        # Показываем список успешных
        if [ ${SUCCESS} -gt 0 ]; then
            echo "Успешные тесты:"
            ls -1 ${RESULTS_DIR}/test_color_light_*/test_color_light_*/color_light/color_light_features.npz 2>/dev/null | \
                sed 's|.*/test_color_light_\([^/]*\)/.*|\1|' | sort -V
            echo ""
        fi
        
        # Показываем неуспешные
        MISSING=$((TOTAL_VIDEOS - SUCCESS))
        if [ ${MISSING} -gt 0 ]; then
            echo "Неуспешные тесты:"
            for i in shortest {2..20}; do
                if [ ! -f "${RESULTS_DIR}/test_color_light_${i}/test_color_light_${i}/color_light/color_light_features.npz" ]; then
                    echo "  - test_color_light_${i}"
                fi
            done
            echo ""
        fi
        
        echo "Запуск финального анализа..."
        echo "=========================================="
        
        cd "/media/ilya/Новый том/TrendFlowML"
        
        # Валидация
        echo ""
        echo "=== ВАЛИДАЦИЯ РЕЗУЛЬТАТОВ ==="
        echo ""
        DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/VisualProcessor/modules/color_light/utils/validate_color_light.py \
            --results-base DataProcessor/dp_results --platform-id youtube 2>&1
        
        # Анализ
        echo ""
        echo "=== КОМПЛЕКСНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ==="
        echo ""
        DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/VisualProcessor/modules/color_light/utils/analyze_all_results.py \
            --results-base DataProcessor/dp_results 2>&1
        
        echo ""
        echo "=========================================="
        echo "АНАЛИЗ ЗАВЕРШЕН"
        echo "=========================================="
        
        break
    fi
    
    sleep 30
done

