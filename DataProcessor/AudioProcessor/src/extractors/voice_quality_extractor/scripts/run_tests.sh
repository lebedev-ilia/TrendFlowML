#!/bin/bash
# Скрипт для запуска тестов voice_quality_extractor на 20 видео (последовательно).
# Аналогично DataProcessor/VisualProcessor/modules/action_recognition/scripts/run_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BASE_DIR: корень репозитория (TrendFlowML)
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"}"

VIDEOS_DIR="${VIDEOS_DIR:-"${BASE_DIR}/example/example_videos"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-"${BASE_DIR}/DataProcessor/configs/audit_v3/audio/profile_voice_quality.yaml"}"
PROFILE_PATH="${PROFILE_PATH:-"${BASE_DIR}/DataProcessor/configs/audit_v3/audio/profile_voice_quality.yaml"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/.data_venv/bin/python"}"
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
echo "Запуск тестов voice_quality_extractor"
echo "Всего видео: $(( ${#VIDEOS[@]} + 1 ))"
echo "Профиль: ${PROFILE_PATH}"
echo "=========================================="

# Тест на самом коротком видео
echo ""
echo "[0] Тестирование: ${SHORTEST_VIDEO} (test_voice_quality_shortest)"
echo "----------------------------------------"
video_path="${VIDEOS_DIR}/${SHORTEST_VIDEO}"
if [ ! -f "${video_path}" ]; then
    echo "❌ Видео не найдено: ${video_path}"
    exit 1
fi

LOG_FILE="/tmp/voice_quality_test_test_voice_quality_shortest.log"
if "${PYTHON}" "${MAIN_SCRIPT}" \
    --video-path "${video_path}" \
    --global-config "${GLOBAL_CONFIG}" \
    --profile-path "${PROFILE_PATH}" \
    --platform-id youtube \
    --video-id test_voice_quality_shortest \
    --run-id test_voice_quality_shortest \
    --output "${RESULTS_DIR}" \
    --rs-base "${RESULTS_DIR}" \
    --no-run-visual \
    > "${LOG_FILE}" 2>&1; then
    echo "✅ Успешно: test_voice_quality_shortest"
else
    echo "❌ Ошибка (exit code: $?): test_voice_quality_shortest"
    echo "Лог сохранен: ${LOG_FILE}"
fi

# Тесты на остальных видео
for i in "${!VIDEOS[@]}"; do
    video="${VIDEOS[$i]}"
    video_id="test_voice_quality_$((i+2))"
    video_path="${VIDEOS_DIR}/${video}"

    echo ""
    echo "[$((i+1))/${#VIDEOS[@]}] Тестирование: ${video} (${video_id})"
    echo "----------------------------------------"

    if [ ! -f "${video_path}" ]; then
        echo "❌ Видео не найдено: ${video_path}"
        continue
    fi

    LOG_FILE="/tmp/voice_quality_test_${video_id}.log"
    if "${PYTHON}" "${MAIN_SCRIPT}" \
        --video-path "${video_path}" \
        --global-config "${GLOBAL_CONFIG}" \
        --profile-path "${PROFILE_PATH}" \
        --platform-id youtube \
        --video-id "${video_id}" \
        --run-id "${video_id}" \
        --output "${RESULTS_DIR}" \
        --rs-base "${RESULTS_DIR}" \
        --no-run-visual \
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
