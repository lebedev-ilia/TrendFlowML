#!/bin/bash
# Скрипт для тестирования action_recognition на всех видео по нарастанию длительности

set -e

DP_ROOT="/media/ilya/Новый том/TrendFlowML/DataProcessor"
VIDEOS_DIR="/media/ilya/Новый том/TrendFlowML/example/example_videos"
RS_BASE="$DP_ROOT/dp_results"
OUTPUT_BASE="$DP_ROOT/dp_results/_frames_test_action_recognition"

# Список видео по нарастанию длительности (из предыдущего анализа)
VIDEOS=(
    "-Q6fnPIybEI.mp4:12:test_action_recognition_v1"
    "-7Ei8e05x30.mp4:14:test_action_recognition_v2"
    "-5EYUqIlyJU.mp4:33:test_action_recognition_v3"
    "-U5ipG4hohY.mp4:33:test_action_recognition_v4"
    "-15jH8mtfJw.mp4:36:test_action_recognition_v5"
    "-BXwIsW0t9w.mp4:37:test_action_recognition_v6"
    "-Ga4edhrfog.mp4:39:test_action_recognition_v7"
    "-7pz_DGQPos.mp4:43:test_action_recognition_v8"
    "-FOB4jpQIg8.mp4:49:test_action_recognition_v9"
    "-Q_Ch-vrvvM.mp4:57:test_action_recognition_v10"
    "-OBC82ymkcs.mp4:59:test_action_recognition_v11"
    "-2b9IMP1ih0.mp4:60:test_action_recognition_v12"
    "-VX009hQoDA.mp4:74:test_action_recognition_v13"
    "-ZLHxCNCpdA.mp4:75:test_action_recognition_v14"
    "-3GDPu4XLZY.mp4:135:test_action_recognition_v15"
    "-T4Rvscu7b4.mp4:236:test_action_recognition_v16"
    "-FyF-rDXAOU.mp4:256:test_action_recognition_v17"
    "-1eKh7CJbhM.mp4:494:test_action_recognition_v18"
    "-BBSE2F58ik.mp4:726:test_action_recognition_v19"
    "-Cnn3Nq_Lpk.mp4:759:test_action_recognition_v20"
)

cd "$DP_ROOT"

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
    
    "$DP_ROOT/.data_venv/bin/python3" main.py \
        --video-path "$video_path" \
        --output "$OUTPUT_BASE" \
        --rs-base "$RS_BASE" \
        --platform-id youtube \
        --video-id "$run_id" \
        --run-id "$run_id" \
        --sampling-policy-version v1 \
        --dataprocessor-version audit3_test \
        --global-config configs/global_config.yaml \
        --profile-path configs/audit_v3/visual/profile_core_object_detections_and_action_recognition.yaml \
        2>&1 | tee "/tmp/action_recognition_test_${run_id}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Success: $video_file"
    else
        echo "❌ Failed: $video_file"
        echo "Check log: /tmp/action_recognition_test_${run_id}.log"
    fi
    
    # Небольшая пауза между тестами
    sleep 2
done

echo ""
echo "=================================================================================="
echo "All tests completed!"
echo "=================================================================================="

