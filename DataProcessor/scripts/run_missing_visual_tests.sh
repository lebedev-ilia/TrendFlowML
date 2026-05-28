#!/bin/bash
# Запуск недостающих тестов VisualProcessor (после поднятия Triton).
# story_structure: тесты 18, 19, 20 (ранее падали из-за embedding_service_client);
# color_light: полный прогон 20 видео (ранее 19 падали из-за Triton);
# uniqueness: полный прогон 20 видео (в отчёте был только smoke).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}"

VIDEOS_DIR="${VIDEOS_DIR:-"${BASE_DIR}/example/example_videos"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-"${BASE_DIR}/DataProcessor/configs/global_config.yaml"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"
MAIN_SCRIPT="${MAIN_SCRIPT:-"${BASE_DIR}/DataProcessor/main.py"}"

# Preflight Triton (recommended): set REQUIRE_TRITON_PREFLIGHT=1 for fail-fast.
# You can override Triton URL with TRITON_HTTP_URL or TRITON_ENDPOINT.
PREFLIGHT_TRITON="${PREFLIGHT_TRITON:-"${BASE_DIR}/DataProcessor/scripts/preflight_triton.py"}"
REQUIRE_TRITON_PREFLIGHT="${REQUIRE_TRITON_PREFLIGHT:-1}"
SKIP_TRITON_PREFLIGHT="${SKIP_TRITON_PREFLIGHT:-0}"
TRITON_PREFLIGHT_TIMEOUT_SEC="${TRITON_PREFLIGHT_TIMEOUT_SEC:-3}"
TRITON_PREFLIGHT_ATTEMPTS="${TRITON_PREFLIGHT_ATTEMPTS:-5}"
TRITON_MODELS_PRESET="${TRITON_MODELS_PRESET:-core_low}"
TRITON_MODELS="${TRITON_MODELS:-}"

if [ "${SKIP_TRITON_PREFLIGHT}" != "1" ]; then
    echo "=========================================="
    echo "Preflight: Triton readiness"
    echo "=========================================="
    if [ "${REQUIRE_TRITON_PREFLIGHT}" = "1" ]; then
        "${PYTHON}" "${PREFLIGHT_TRITON}" --require \
            --timeout-sec "${TRITON_PREFLIGHT_TIMEOUT_SEC}" \
            --attempts "${TRITON_PREFLIGHT_ATTEMPTS}" \
            --models-preset "${TRITON_MODELS_PRESET}" \
            --models "${TRITON_MODELS}"
    else
        "${PYTHON}" "${PREFLIGHT_TRITON}" \
            --timeout-sec "${TRITON_PREFLIGHT_TIMEOUT_SEC}" \
            --attempts "${TRITON_PREFLIGHT_ATTEMPTS}" \
            --models-preset "${TRITON_MODELS_PRESET}" \
            --models "${TRITON_MODELS}" || true
    fi
    echo ""
fi

# Видео для story_structure 18, 19, 20 (индексы в VIDEOS 16, 17, 18)
STORY_VIDEOS=("-1eKh7CJbhM.mp4" "-BBSE2F58ik.mp4" "-Cnn3Nq_Lpk.mp4")
STORY_IDS=("test_story_structure_18" "test_story_structure_19" "test_story_structure_20")
STORY_PROFILE="${BASE_DIR}/DataProcessor/configs/audit_v3/visual/profile_story_structure.yaml"

cd "${BASE_DIR}"

echo "=========================================="
echo "1. Story structure: тесты 18, 19, 20"
echo "=========================================="
for i in 0 1 2; do
    video="${STORY_VIDEOS[$i]}"
    video_id="${STORY_IDS[$i]}"
    video_path="${VIDEOS_DIR}/${video}"
    if [ ! -f "${video_path}" ]; then
        echo "⚠ Пропущено: ${video} не найден"
        continue
    fi
    echo "[$((i+1))/3] ${video_id} ..."
    LOG="/tmp/story_structure_${video_id}.log"
    if "${PYTHON}" "${MAIN_SCRIPT}" \
        --video-path "${video_path}" \
        --global-config "${GLOBAL_CONFIG}" \
        --profile-path "${STORY_PROFILE}" \
        --platform-id youtube \
        --video-id "${video_id}" \
        --run-id "${video_id}" \
        --result-dir "${RESULTS_DIR}" \
        > "${LOG}" 2>&1; then
        echo "  ✅ ${video_id}"
    else
        echo "  ❌ ${video_id} (лог: ${LOG})"
    fi
done

echo ""
echo "=========================================="
echo "2. Color_light: полный прогон 20 видео"
echo "=========================================="
"${BASE_DIR}/DataProcessor/VisualProcessor/modules/color_light/scripts/run_tests.sh" || true

echo ""
echo "=========================================="
echo "3. Uniqueness: полный прогон 20 видео"
echo "=========================================="
"${BASE_DIR}/DataProcessor/VisualProcessor/modules/uniqueness/scripts/run_tests.sh" || true

echo ""
echo "=========================================="
echo "4. High_level_semantic: полный прогон 20 видео"
echo "=========================================="
"${BASE_DIR}/DataProcessor/VisualProcessor/modules/high_level_semantic/scripts/run_tests.sh" || true

echo ""
echo "=========================================="
echo "5. Shot_quality: полный прогон 20 видео"
echo "=========================================="
"${BASE_DIR}/DataProcessor/VisualProcessor/modules/shot_quality/scripts/run_tests.sh" || true

echo ""
echo "=========================================="
echo "Недостающие тесты завершены."
echo "Запустите валидаторы и обновите TESTING_REPORT.md в каждом модуле."
echo "=========================================="
