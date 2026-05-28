#!/bin/bash
# Smoke-тест: по одному короткому видео для каждого из 21 компонентов AudioProcessor.
# Цель: быстро проверить, что все компоненты запускаются без ошибок.
# Результаты: DataProcessor/dp_results/smoke_test/

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DP_ROOT="$REPO_ROOT/DataProcessor"
VIDEOS_DIR="${VIDEOS_DIR:-$REPO_ROOT/example/example_videos}"
SHORTEST_VIDEO="-Q6fnPIybEI.mp4"
RS_BASE="$(cd "$DP_ROOT" && pwd)/dp_results/smoke_test"
OUTPUT_BASE="$RS_BASE"
PYTHON="${PYTHON:-$DP_ROOT/.data_venv/bin/python3}"
MAIN_SCRIPT="$DP_ROOT/main.py"
CONFIGS_DIR="$REPO_ROOT/configs/audit_v3/audio"

# Все 21 компонент: (key, profile_file)
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
video_path="$VIDEOS_DIR/$SHORTEST_VIDEO"
cd "$REPO_ROOT"

if [ ! -f "$video_path" ]; then
    echo "❌ Видео не найдено: $video_path"
    exit 1
fi

echo "=============================================="
echo "Smoke-тест AudioProcessor (21 компонентов)"
echo "Видео: $SHORTEST_VIDEO"
echo "Результаты: $RS_BASE"
echo "=============================================="

ok=0
fail=0

for item in "${COMPONENTS[@]}"; do
    key="${item%%:*}"
    profile="${item##*:}"
    run_id="smoke_${key}_shortest"
    profile_path="$CONFIGS_DIR/$profile"

    if [ ! -f "$profile_path" ]; then
        echo "⚠️  $key: профиль не найден $profile_path, пропуск"
        ((fail++)) || true
        continue
    fi

    echo ""
    echo "[$key] Запуск smoke (run_id=$run_id)..."
    log_file="/tmp/audio_smoke_${key}_shortest.log"

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
        echo "  ✅ OK"
        ((ok++)) || true
    else
        echo "  ❌ FAIL (лог: $log_file)"
        ((fail++)) || true
    fi
done

echo ""
echo "=============================================="
echo "Итого: ✅ $ok успешно, ❌ $fail с ошибками"
echo "=============================================="
[ $fail -eq 0 ]
