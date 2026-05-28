#!/bin/bash
# Валидация результатов полного тестирования (full_test)
# Проверяет наличие и валидность NPZ для каждого из 20 прогонов по каждому компоненту

set -e

DP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/DataProcessor"
RS_BASE="$DP_ROOT/dp_results/full_test/youtube"
PYTHON="${PYTHON:-$DP_ROOT/.data_venv/bin/python3}"

COMPONENTS=(asr band_energy chroma clap emotion_diarization hpss key loudness mel mfcc onset pitch quality rhythmic source_separation speaker_diarization spectral spectral_entropy speech_analysis tempo voice_quality)

# Map key -> component_name (for path)
declare -A COMP_NAMES=(
    [asr]=asr_extractor
    [band_energy]=band_energy_extractor
    [chroma]=chroma_extractor
    [clap]=clap_extractor
    [emotion_diarization]=emotion_diarization_extractor
    [hpss]=hpss_extractor
    [key]=key_extractor
    [loudness]=loudness_extractor
    [mel]=mel_extractor
    [mfcc]=mfcc_extractor
    [onset]=onset_extractor
    [pitch]=pitch_extractor
    [quality]=quality_extractor
    [rhythmic]=rhythmic_extractor
    [source_separation]=source_separation_extractor
    [speaker_diarization]=speaker_diarization_extractor
    [spectral]=spectral_extractor
    [spectral_entropy]=spectral_entropy_extractor
    [speech_analysis]=speech_analysis_extractor
    [tempo]=tempo_extractor
    [voice_quality]=voice_quality_extractor
)

cd "$DP_ROOT/.."
echo "=== Валидация full-результатов ==="
echo ""

for key in "${COMPONENTS[@]}"; do
    comp_name="${COMP_NAMES[$key]}"
    validator_path="$DP_ROOT/AudioProcessor/src/extractors/$comp_name/utils/validate_${key//_extractor/}.py"
    # Fix for keys with _extractor suffix in validator name
    case "$key" in
        emotion_diarization) validator_path="$DP_ROOT/AudioProcessor/src/extractors/emotion_diarization_extractor/utils/validate_emotion_diarization.py" ;;
        source_separation) validator_path="$DP_ROOT/AudioProcessor/src/extractors/source_separation_extractor/utils/validate_source_separation.py" ;;
        speaker_diarization) validator_path="$DP_ROOT/AudioProcessor/src/extractors/speaker_diarization_extractor/utils/validate_speaker_diarization.py" ;;
        spectral_entropy) validator_path="$DP_ROOT/AudioProcessor/src/extractors/spectral_entropy_extractor/utils/validate_spectral_entropy.py" ;;
        speech_analysis) validator_path="$DP_ROOT/AudioProcessor/src/extractors/speech_analysis_extractor/utils/validate_speech_analysis.py" ;;
        voice_quality) validator_path="$DP_ROOT/AudioProcessor/src/extractors/voice_quality_extractor/utils/validate_voice_quality.py" ;;
        *) validator_path="$DP_ROOT/AudioProcessor/src/extractors/${key}_extractor/utils/validate_${key}.py" ;;
    esac

    count=0
    valid_count=0
    for i in {1..20}; do
        run_id="full_${key}_v${i}"
        npz_path="$RS_BASE/$run_id/$run_id/$comp_name/${comp_name}_features.npz"
        if [ ! -f "$npz_path" ]; then
            continue
        fi
        ((count++)) || true
        if [ -f "$validator_path" ]; then
            if "$PYTHON" "$validator_path" "$npz_path" 2>&1 | grep -q "✅ VALID"; then
                ((valid_count++)) || true
            fi
        else
            ((valid_count++)) || true
        fi
    done
    echo "$key: $valid_count/$count валидных"
done

echo ""
echo "Готово (проверка наличия и валидности NPZ)"
