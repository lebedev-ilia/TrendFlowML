#!/bin/bash
# Скрипт для запуска тестов scene_classification компонента на всех видео (последовательно).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"}"

VIDEOS_DIR="${VIDEOS_DIR:-"${BASE_DIR}/example/example_videos"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
CONFIG_PROFILE="${CONFIG_PROFILE:-"${BASE_DIR}/DataProcessor/configs/audit_v3/visual/profile_scene_classification.yaml"}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-"${BASE_DIR}/DataProcessor/configs/global_config.yaml"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"
MAIN_SCRIPT="${MAIN_SCRIPT:-"${BASE_DIR}/DataProcessor/main.py"}"

# Самое короткое видео для быстрого smoke-test
SHORTEST_VIDEO="-Q6fnPIybEI.mp4"

# Список видео по нарастанию размера (19 файлов + shortest = 20)
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
echo "Запуск тестов scene_classification компонента"
echo "Всего видео: $(( ${#VIDEOS[@]} + 1 ))"
echo "Профиль: ${CONFIG_PROFILE}"
echo "=========================================="

# Тест на самом коротком видео
echo ""
echo "[0] Тестирование: ${SHORTEST_VIDEO} (test_scene_classification_shortest)"
echo "----------------------------------------"
video_path="${VIDEOS_DIR}/${SHORTEST_VIDEO}"
if [ ! -f "${video_path}" ]; then
    echo "❌ Видео не найдено: ${video_path}"
    exit 1
fi

LOG_FILE="/tmp/scene_classification_test_test_scene_classification_shortest.log"
if "${PYTHON}" "${MAIN_SCRIPT}" \
    --video-path "${video_path}" \
    --global-config "${GLOBAL_CONFIG}" \
    --profile-path "${CONFIG_PROFILE}" \
    --platform-id youtube \
    --video-id test_scene_classification_shortest \
    --run-id test_scene_classification_shortest \
    --output-dir "${RESULTS_DIR}" \
    > "${LOG_FILE}" 2>&1; then
    echo "✅ Успешно: test_scene_classification_shortest"
else
    echo "❌ Ошибка (exit code: $?): test_scene_classification_shortest"
    echo "Лог сохранен: ${LOG_FILE}"
fi

# Тесты на остальных видео
for i in "${!VIDEOS[@]}"; do
    video="${VIDEOS[$i]}"
    video_id="test_scene_classification_$((i+2))"
    video_path="${VIDEOS_DIR}/${video}"

    echo ""
    echo "[$((i+1))/${#VIDEOS[@]}] Тестирование: ${video} (${video_id})"
    echo "----------------------------------------"

    if [ ! -f "${video_path}" ]; then
        echo "❌ Видео не найдено: ${video_path}"
        continue
    fi

    LOG_FILE="/tmp/scene_classification_test_${video_id}.log"
    if "${PYTHON}" "${MAIN_SCRIPT}" \
        --video-path "${video_path}" \
        --global-config "${GLOBAL_CONFIG}" \
        --profile-path "${CONFIG_PROFILE}" \
        --platform-id youtube \
        --video-id "${video_id}" \
        --run-id "${video_id}" \
        --output-dir "${RESULTS_DIR}" \
        > "${LOG_FILE}" 2>&1; then
        echo "✅ Успешно: ${video_id}"
    else
        echo "❌ Ошибка (exit code: $?): ${video_id}"
        echo "Лог сохранен: ${LOG_FILE}"
    fi
done

echo ""
echo "=========================================="
echo "Тестирование завершено"
echo "=========================================="


