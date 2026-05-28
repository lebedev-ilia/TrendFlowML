#!/bin/bash
# Запуск только недостающих тестов shot_quality (3, 4, 5, 11, 12, 14, 17, 19).
# После исправления core_depth_midas (sys.path для utils.frame_manager).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"}"

VIDEOS_DIR="${VIDEOS_DIR:-"${BASE_DIR}/example/example_videos"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
CONFIG_PROFILE="${CONFIG_PROFILE:-"${BASE_DIR}/DataProcessor/configs/audit_v3/visual/profile_shot_quality.yaml"}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-"${BASE_DIR}/DataProcessor/configs/global_config.yaml"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"
MAIN_SCRIPT="${MAIN_SCRIPT:-"${BASE_DIR}/DataProcessor/main.py"}"

# (video_id, filename) для недостающих: 3, 4, 5, 11, 12, 14, 17, 19
MISSING=(
    "3:-5EYUqIlyJU.mp4"
    "4:-U5ipG4hohY.mp4"
    "5:-15jH8mtfJw.mp4"
    "11:-2b9IMP1ih0.mp4"
    "12:-VX009hQoDA.mp4"
    "14:-ZLHxCNCpdA.mp4"
    "17:-1eKh7CJbhM.mp4"
    "19:-BBSE2F58ik.mp4"
)

cd "${BASE_DIR}"
echo "Запуск недостающих тестов shot_quality (${#MISSING[@]} шт.)"
echo ""

for entry in "${MISSING[@]}"; do
    id="${entry%%:*}"
    file="${entry#*:}"
    video_id="test_shot_quality_${id}"
    video_path="${VIDEOS_DIR}/${file}"
    if [ ! -f "${video_path}" ]; then
        echo "❌ Видео не найдено: ${video_path}"
        continue
    fi
    echo "[${video_id}] ${file}"
    LOG_FILE="/tmp/shot_quality_test_${video_id}.log"
    if "${PYTHON}" "${MAIN_SCRIPT}" \
        --video-path "${video_path}" \
        --global-config "${GLOBAL_CONFIG}" \
        --profile-path "${CONFIG_PROFILE}" \
        --platform-id youtube \
        --video-id "${video_id}" \
        --run-id "${video_id}" \
        --output-dir "${RESULTS_DIR}" \
        > "${LOG_FILE}" 2>&1; then
        echo "  ✅ Успешно"
    else
        echo "  ❌ Ошибка (лог: ${LOG_FILE})"
    fi
done
echo ""
echo "Готово."
