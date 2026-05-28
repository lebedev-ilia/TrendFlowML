#!/bin/bash
# Скрипт для запуска тестов behavioral на всех видео

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"}"

VIDEOS_DIR="${VIDEOS_DIR:-"${BASE_DIR}/example/example_videos"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/VisualProcessor/.vp_venv/bin/python"}"
MAIN_SCRIPT="${MAIN_SCRIPT:-"${BASE_DIR}/DataProcessor/main.py"}"
CONFIG="${CONFIG:-"${BASE_DIR}/DataProcessor/configs/global_config.yaml"}"
PROFILE="${PROFILE:-"${BASE_DIR}/DataProcessor/configs/audit_v3/visual/profile_behavioral.yaml"}"

# Список видео (уже протестированные пропускаем)
declare -a videos=(
    "-7Ei8e05x30.mp4:test_behavioral_3"
    "-5EYUqIlyJU.mp4:test_behavioral_4"
    "-Ga4edhrfog.mp4:test_behavioral_5"
    "-BXwIsW0t9w.mp4:test_behavioral_6"
    "-U5ipG4hohY.mp4:test_behavioral_7"
    "-FOB4jpQIg8.mp4:test_behavioral_8"
    "-15jH8mtfJw.mp4:test_behavioral_9"
    "-7pz_DGQPos.mp4:test_behavioral_10"
    "-OBC82ymkcs.mp4:test_behavioral_11"
    "-VX009hQoDA.mp4:test_behavioral_12"
    "-2b9IMP1ih0.mp4:test_behavioral_13"
    "-Q_Ch-vrvvM.mp4:test_behavioral_14"
    "-3GDPu4XLZY.mp4:test_behavioral_15"
    "-1eKh7CJbhM.mp4:test_behavioral_16"
    "-FyF-rDXAOU.mp4:test_behavioral_17"
    "-T4Rvscu7b4.mp4:test_behavioral_18"
    "-Cnn3Nq_Lpk.mp4:test_behavioral_19"
    "-BBSE2F58ik.mp4:test_behavioral_20"
)

echo "Starting behavioral tests on ${#videos[@]} videos..."
echo "=========================================="

for video_info in "${videos[@]}"; do
    IFS=':' read -r video_file video_id <<< "$video_info"
    video_path="${VIDEOS_DIR}/${video_file}"
    
    if [ ! -f "$video_path" ]; then
        echo "⚠️  Video not found: $video_path"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Testing: $video_file -> $video_id"
    echo "=========================================="
    
    $PYTHON "$MAIN_SCRIPT" \
        --video-path "$video_path" \
        --global-config "$CONFIG" \
        --profile-path "$PROFILE" \
        --platform-id youtube \
        --video-id "$video_id" \
        --run-id "$video_id" \
        --output-dir "$RESULTS_DIR" 2>&1 | tail -30
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $video_id"
    else
        echo "❌ Failed: $video_id"
    fi
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="

