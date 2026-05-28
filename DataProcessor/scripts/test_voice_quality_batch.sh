#!/bin/bash
# Скрипт для тестирования voice_quality_extractor на 20 видео (аналог test_action_recognition_batch.sh)

set -e

DP_ROOT="/media/ilya/Новый том/TrendFlowML/DataProcessor"
VIDEOS_DIR="/media/ilya/Новый том/TrendFlowML/example/example_videos"
RS_BASE="$DP_ROOT/dp_results"
OUTPUT_BASE="$DP_ROOT/dp_results"
GLOBAL_CONFIG="$DP_ROOT/configs/audit_v3/audio/profile_voice_quality.yaml"
PROFILE_PATH="$DP_ROOT/configs/audit_v3/audio/profile_voice_quality.yaml"

# Совпадает с run_tests.sh: shortest + test_voice_quality_2..20
VIDEOS=(
    "-Q6fnPIybEI.mp4:12:test_voice_quality_shortest"
    "-7Ei8e05x30.mp4:14:test_voice_quality_2"
    "-5EYUqIlyJU.mp4:33:test_voice_quality_3"
    "-U5ipG4hohY.mp4:33:test_voice_quality_4"
    "-15jH8mtfJw.mp4:36:test_voice_quality_5"
    "-BXwIsW0t9w.mp4:37:test_voice_quality_6"
    "-Ga4edhrfog.mp4:39:test_voice_quality_7"
    "-7pz_DGQPos.mp4:43:test_voice_quality_8"
    "-FOB4jpQIg8.mp4:49:test_voice_quality_9"
    "-Q_Ch-vrvvM.mp4:57:test_voice_quality_10"
    "-OBC82ymkcs.mp4:59:test_voice_quality_11"
    "-2b9IMP1ih0.mp4:60:test_voice_quality_12"
    "-VX009hQoDA.mp4:74:test_voice_quality_13"
    "-ZLHxCNCpdA.mp4:75:test_voice_quality_14"
    "-3GDPu4XLZY.mp4:135:test_voice_quality_15"
    "-T4Rvscu7b4.mp4:236:test_voice_quality_16"
    "-FyF-rDXAOU.mp4:256:test_voice_quality_17"
    "-1eKh7CJbhM.mp4:494:test_voice_quality_18"
    "-BBSE2F58ik.mp4:726:test_voice_quality_19"
    "-Cnn3Nq_Lpk.mp4:759:test_voice_quality_20"
)

cd "$DP_ROOT/.."

for video_info in "${VIDEOS[@]}"; do
    IFS=':' read -r video_file duration run_id <<< "$video_info"
    video_path="$VIDEOS_DIR/$video_file"

    if [ ! -f "$video_path" ]; then
        echo "⚠️  Video not found: $video_path, skipping..."
        continue
    fi

    echo ""
    echo "=================================================================================="
    echo "Testing: $video_file (duration: ${duration}s, run_id: $run_id)"
    echo "=================================================================================="

    "$DP_ROOT/.data_venv/bin/python3" "$DP_ROOT/main.py" \
        --video-path "$video_path" \
        --output "$OUTPUT_BASE" \
        --rs-base "$RS_BASE" \
        --platform-id youtube \
        --video-id "$run_id" \
        --run-id "$run_id" \
        --global-config "$GLOBAL_CONFIG" \
        --profile-path "$PROFILE_PATH" \
        --sampling-policy-version v1 \
        --dataprocessor-version audit3_test \
        --no-run-visual \
        2>&1 | tee "/tmp/voice_quality_test_${run_id}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Success: $video_file"
    else
        echo "❌ Failed: $video_file"
        echo "Check log: /tmp/voice_quality_test_${run_id}.log"
    fi

    sleep 2
done

echo ""
echo "=================================================================================="
echo "All tests completed!"
echo "=================================================================================="
