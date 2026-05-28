#!/bin/bash
# Полное тестирование: по 20 видео для каждого из 21 компонентов AudioProcessor.
# Результаты: DataProcessor/dp_results/full_test/
# Структура: full_test/youtube/full_<key>_v1/... , full_<key>_v2/... , ... full_<key>_v20/

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DP_ROOT="$REPO_ROOT/DataProcessor"
VIDEOS_DIR="${VIDEOS_DIR:-$REPO_ROOT/example/example_videos}"
RS_BASE="$(cd "$DP_ROOT" && pwd)/dp_results/full_test"
OUTPUT_BASE="$RS_BASE"
PYTHON="${PYTHON:-$DP_ROOT/.data_venv/bin/python3}"
MAIN_SCRIPT="$DP_ROOT/main.py"
CONFIGS_DIR="$REPO_ROOT/configs/audit_v3/audio"

# 20 видео по нарастанию длительности (как в VisualProcessor)
VIDEOS=(
    "-Q6fnPIybEI.mp4:12"
    "-7Ei8e05x30.mp4:14"
    "-5EYUqIlyJU.mp4:33"
    "-U5ipG4hohY.mp4:33"
    "-15jH8mtfJw.mp4:36"
    "-BXwIsW0t9w.mp4:37"
    "-Ga4edhrfog.mp4:39"
    "-7pz_DGQPos.mp4:43"
    "-FOB4jpQIg8.mp4:49"
    "-Q_Ch-vrvvM.mp4:57"
    "-OBC82ymkcs.mp4:59"
    "-2b9IMP1ih0.mp4:60"
    "-VX009hQoDA.mp4:74"
    "-ZLHxCNCpdA.mp4:75"
    "-3GDPu4XLZY.mp4:135"
    "-T4Rvscu7b4.mp4:236"
    "-FyF-rDXAOU.mp4:256"
    "-1eKh7CJbhM.mp4:494"
    "-BBSE2F58ik.mp4:726"
    "-Cnn3Nq_Lpk.mp4:759"
)

COMPONENTS=(
    "asr:profile_asr.yaml"
    "band_energy:profile_band_energy.yaml"
    "chroma:profile_chroma.yaml"
    "clap:profile_clap.yaml"
    "emotion_diarization:profile_emotion_diarization.yaml"
    "hpss:profile_hpss.yaml"
    "key:profile_key.yaml"
    "loudness:profile_loudness.yaml"
    "mel:profile_mel.yaml"
    "mfcc:profile_mfcc.yaml"
    "onset:profile_onset.yaml"
    "pitch:profile_pitch.yaml"
    "quality:profile_quality.yaml"
    "rhythmic:profile_rhythmic.yaml"
    "source_separation:profile_source_separation.yaml"
    "speaker_diarization:profile_speaker_diarization.yaml"
    "spectral:profile_spectral.yaml"
    "spectral_entropy:profile_spectral_entropy.yaml"
    "speech_analysis:profile_speech_analysis.yaml"
    "tempo:profile_tempo.yaml"
    "voice_quality:profile_voice_quality.yaml"
)

mkdir -p "$RS_BASE"
cd "$REPO_ROOT"

echo "=============================================="
echo "Полное тестирование AudioProcessor"
echo "21 компонентов × 20 видео = 420 запусков"
echo "Результаты: $RS_BASE"
echo "=============================================="

total_ok=0
total_fail=0

for item in "${COMPONENTS[@]}"; do
    key="${item%%:*}"
    profile="${item##*:}"
    profile_path="$CONFIGS_DIR/$profile"

    if [ ! -f "$profile_path" ]; then
        echo ""
        echo "⚠️  $key: профиль не найден, пропуск"
        continue
    fi

    echo ""
    echo "========== Компонент: $key (20 видео) =========="
    comp_ok=0
    comp_fail=0

    for i in "${!VIDEOS[@]}"; do
        IFS=':' read -r video_file duration <<< "${VIDEOS[$i]}"
        run_id="full_${key}_v$((i+1))"
        video_path="$VIDEOS_DIR/$video_file"

        if [ ! -f "$video_path" ]; then
            echo "  [$((i+1))/20] ⚠️  Видео не найдено: $video_file"
            ((comp_fail++)) || true
            continue
        fi

        log_file="/tmp/audio_full_${key}_${run_id}.log"
        echo -n "  [$((i+1))/20] $run_id ... "

        if "$PYTHON" "$MAIN_SCRIPT" \
            --video-path "$video_path" \
            --global-config "$profile_path" \
            --profile-path "$profile_path" \
            --platform-id youtube \
            --video-id "$run_id" \
            --run-id "$run_id" \
            --output "$OUTPUT_BASE" \
            --rs-base "$RS_BASE" \
            --no-run-visual \
            > "$log_file" 2>&1; then
            echo "✅"
            ((comp_ok++)) || true
        else
            echo "❌ (лог: $log_file)"
            ((comp_fail++)) || true
        fi
        sleep 1
    done

    echo "  Итого $key: ✅ $comp_ok / ❌ $comp_fail"
    total_ok=$((total_ok + comp_ok))
    total_fail=$((total_fail + comp_fail))
done

echo ""
echo "=============================================="
echo "Полный итог: ✅ $total_ok успешно, ❌ $total_fail с ошибками"
echo "=============================================="
