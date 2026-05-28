#!/bin/bash
# Тесты emotion_diarization_extractor на 20 видео

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-"$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"}"
VIDEOS_DIR="${VIDEOS_DIR:-"${BASE_DIR}/example/example_videos"}"
RESULTS_DIR="${RESULTS_DIR:-"${BASE_DIR}/DataProcessor/dp_results"}"
GLOBAL_CONFIG="${GLOBAL_CONFIG:-"${BASE_DIR}/DataProcessor/configs/audit_v3/audio/profile_emotion_diarization.yaml"}"
PROFILE_PATH="${PROFILE_PATH:-"${BASE_DIR}/DataProcessor/configs/audit_v3/audio/profile_emotion_diarization.yaml"}"
PYTHON="${PYTHON:-"${BASE_DIR}/DataProcessor/.data_venv/bin/python"}"
MAIN_SCRIPT="${MAIN_SCRIPT:-"${BASE_DIR}/DataProcessor/main.py"}"

SHORTEST_VIDEO="-Q6fnPIybEI.mp4"
VIDEOS=("-7Ei8e05x30.mp4" "-5EYUqIlyJU.mp4" "-U5ipG4hohY.mp4" "-15jH8mtfJw.mp4" "-BXwIsW0t9w.mp4" "-Ga4edhrfog.mp4" "-7pz_DGQPos.mp4" "-FOB4jpQIg8.mp4" "-Q_Ch-vrvvM.mp4" "-OBC82ymkcs.mp4" "-2b9IMP1ih0.mp4" "-VX009hQoDA.mp4" "-ZLHxCNCpdA.mp4" "-3GDPu4XLZY.mp4" "-T4Rvscu7b4.mp4" "-FyF-rDXAOU.mp4" "-1eKh7CJbhM.mp4" "-BBSE2F58ik.mp4" "-Cnn3Nq_Lpk.mp4")

cd "${BASE_DIR}"
echo "Тесты emotion_diarization_extractor (20 видео)"
echo "[0/20] test_emotion_diarization_shortest"
"${PYTHON}" "${MAIN_SCRIPT}" --video-path "${VIDEOS_DIR}/${SHORTEST_VIDEO}" --global-config "${GLOBAL_CONFIG}" --profile-path "${PROFILE_PATH}" --platform-id youtube --video-id "test_emotion_diarization_shortest" --run-id "test_emotion_diarization_shortest" --output "${RESULTS_DIR}" --rs-base "${RESULTS_DIR}" --no-run-visual > "/tmp/emotion_diarization_test_shortest.log" 2>&1 && echo "  OK" || echo "  FAIL"
for i in "${!VIDEOS[@]}"; do
  vid="${VIDEOS[$i]}"
  rid="test_emotion_diarization_$((i+2))"
  vp="${VIDEOS_DIR}/$vid"
  [ ! -f "$vp" ] && echo "Skip $vid" && continue
  echo "[$((i+1))/20] $rid"
  "${PYTHON}" "${MAIN_SCRIPT}" --video-path "$vp" --global-config "${GLOBAL_CONFIG}" --profile-path "${PROFILE_PATH}" --platform-id youtube --video-id "$rid" --run-id "$rid" --output "${RESULTS_DIR}" --rs-base "${RESULTS_DIR}" --no-run-visual > "/tmp/emotion_diarization_test_$rid.log" 2>&1 && echo "  OK" || echo "  FAIL"
done
echo "Done"
