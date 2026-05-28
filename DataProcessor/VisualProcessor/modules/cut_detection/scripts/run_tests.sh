#!/bin/bash
# Скрипт для запуска тестов cut_detection компонента на всех видео

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"}"

VIDEOS_DIR="${VIDEOS_DIR:-"${BASE_DIR}/example/example_videos"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
CONFIG_PROFILE="${CONFIG_PROFILE:-"${BASE_DIR}/DataProcessor/configs/audit_v3/visual/profile_cut_detection.yaml"}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-"${BASE_DIR}/DataProcessor/configs/global_config.yaml"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"
MAIN_SCRIPT="${MAIN_SCRIPT:-"${BASE_DIR}/DataProcessor/main.py"}"

# Список видео по нарастанию размера (уже протестировано: -Q6fnPIybEI.mp4)
VIDEOS=(
    "-7Ei8e05x30.mp4"
    "-5EYUqIlyJU.mp4"
    "-U5ipG4hohY.mp4"
    "-15jH8mtfJw.mp4"
    "-BXwIsW0t9w.mp4"
    "-Ga4edhrfog.mp4"
    "-7pz_DGQPos.mp4"
    "-FOB4jpQIg8.mp4"
    "-Q_Ch-vrvvM.mp4"
    "-OBC82ymkcs.mp4"
    "-2b9IMP1ih0.mp4"
    "-VX009hQoDA.mp4"
    "-ZLHxCNCpdA.mp4"
    "-3GDPu4XLZY.mp4"
    "-T4Rvscu7b4.mp4"
    "-FyF-rDXAOU.mp4"
    "-1eKh7CJbhM.mp4"
    "-BBSE2F58ik.mp4"
    "-Cnn3Nq_Lpk.mp4"
)

cd "${BASE_DIR}"

echo "=========================================="
echo "Запуск тестов cut_detection компонента"
echo "Всего видео: ${#VIDEOS[@]}"
echo "=========================================="

for i in "${!VIDEOS[@]}"; do
    video="${VIDEOS[$i]}"
    video_id="test_cut_detection_$((i+2))"
    video_path="${VIDEOS_DIR}/${video}"
    
    echo ""
    echo "[$((i+1))/${#VIDEOS[@]}] Тестирование: ${video} (${video_id})"
    echo "----------------------------------------"
    
    if [ ! -f "${video_path}" ]; then
        echo "⚠ Пропущено: файл не найден ${video_path}"
        continue
    fi
    
    "${PYTHON}" "${MAIN_SCRIPT}" \
        --video-path "${video_path}" \
        --global-config "${GLOBAL_CONFIG}" \
        --profile-path "${CONFIG_PROFILE}" \
        --platform-id youtube \
        --video-id "${video_id}" \
        --run-id "${video_id}" \
        --output-dir "${RESULTS_DIR}" \
        2>&1 | tee "/tmp/cut_detection_test_${video_id}.log" | tail -20
    
    exit_code=${PIPESTATUS[0]}
    
    if [ ${exit_code} -eq 0 ]; then
        echo "✅ Успешно: ${video_id}"
    else
        echo "❌ Ошибка (exit code: ${exit_code}): ${video_id}"
        echo "Лог сохранен: /tmp/cut_detection_test_${video_id}.log"
    fi
done

echo ""
echo "=========================================="
echo "Тестирование завершено"
echo "=========================================="

