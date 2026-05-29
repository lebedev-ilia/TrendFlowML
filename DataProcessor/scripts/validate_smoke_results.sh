#!/bin/bash
# Валидация результатов smoke-теста (проверка NPZ для каждого компонента)

set -euo pipefail

DP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/DataProcessor"
HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_CUDA=1
fi
GPU_ONLY_KEYS=(emotion_diarization)
RS_BASE="$DP_ROOT/dp_results/smoke_test/youtube"
PYTHON="${PYTHON:-$DP_ROOT/.data_venv/bin/python3}"

# Компонент -> (key, validator_path, npz_name)
# Используем validate_*.py из utils каждого extractor'а
declare -A VALIDATORS=(
    ["asr"]="AudioProcessor/src/extractors/asr_extractor/utils/validate_asr.py:asr_extractor:asr_extractor_features.npz"
    ["band_energy"]="AudioProcessor/src/extractors/band_energy_extractor/utils/validate_band_energy.py:band_energy_extractor:band_energy_extractor_features.npz"
    ["chroma"]="AudioProcessor/src/extractors/chroma_extractor/utils/validate_chroma.py:chroma_extractor:chroma_extractor_features.npz"
    ["clap"]="AudioProcessor/src/extractors/clap_extractor/utils/validate_clap.py:clap_extractor:clap_extractor_features.npz"
    ["emotion_diarization"]="AudioProcessor/src/extractors/emotion_diarization_extractor/utils/validate_emotion_diarization.py:emotion_diarization_extractor:emotion_diarization_extractor_features.npz"
    ["hpss"]="AudioProcessor/src/extractors/hpss_extractor/utils/validate_hpss.py:hpss_extractor:hpss_extractor_features.npz"
    ["key"]="AudioProcessor/src/extractors/key_extractor/utils/validate_key.py:key_extractor:key_extractor_features.npz"
    ["loudness"]="AudioProcessor/src/extractors/loudness_extractor/utils/validate_loudness.py:loudness_extractor:loudness_extractor_features.npz"
    ["mel"]="AudioProcessor/src/extractors/mel_extractor/utils/validate_mel.py:mel_extractor:mel_extractor_features.npz"
    ["mfcc"]="AudioProcessor/src/extractors/mfcc_extractor/utils/validate_mfcc.py:mfcc_extractor:mfcc_extractor_features.npz"
    ["onset"]="AudioProcessor/src/extractors/onset_extractor/utils/validate_onset.py:onset_extractor:onset_extractor_features.npz"
    ["pitch"]="AudioProcessor/src/extractors/pitch_extractor/utils/validate_pitch.py:pitch_extractor:pitch_extractor_features.npz"
    ["quality"]="AudioProcessor/src/extractors/quality_extractor/utils/validate_quality.py:quality_extractor:quality_extractor_features.npz"
    ["rhythmic"]="AudioProcessor/src/extractors/rhythmic_extractor/utils/validate_rhythmic.py:rhythmic_extractor:rhythmic_extractor_features.npz"
    ["source_separation"]="AudioProcessor/src/extractors/source_separation_extractor/utils/validate_source_separation.py:source_separation_extractor:source_separation_extractor_features.npz"
    ["speaker_diarization"]="AudioProcessor/src/extractors/speaker_diarization_extractor/utils/validate_speaker_diarization.py:speaker_diarization_extractor:speaker_diarization_extractor_features.npz"
    ["spectral"]="AudioProcessor/src/extractors/spectral_extractor/utils/validate_spectral.py:spectral_extractor:spectral_extractor_features.npz"
    ["spectral_entropy"]="AudioProcessor/src/extractors/spectral_entropy_extractor/utils/validate_spectral_entropy.py:spectral_entropy_extractor:spectral_entropy_extractor_features.npz"
    ["speech_analysis"]="AudioProcessor/src/extractors/speech_analysis_extractor/utils/validate_speech_analysis.py:speech_analysis_extractor:speech_analysis_extractor_features.npz"
    ["tempo"]="AudioProcessor/src/extractors/tempo_extractor/utils/validate_tempo.py:tempo_extractor:tempo_extractor_features.npz"
    ["voice_quality"]="AudioProcessor/src/extractors/voice_quality_extractor/utils/validate_voice_quality.py:voice_quality_extractor:voice_quality_extractor_features.npz"
)

cd "$DP_ROOT/.."
echo "=== Валидация smoke-результатов ==="
echo ""

valid=0
invalid=0
skipped=0

for key in asr band_energy chroma clap emotion_diarization hpss key loudness mel mfcc onset pitch quality rhythmic source_separation speaker_diarization spectral spectral_entropy speech_analysis tempo voice_quality; do
    if [ "$HAS_CUDA" -eq 0 ]; then
        for gpu_key in "${GPU_ONLY_KEYS[@]}"; do
            if [ "$key" = "$gpu_key" ]; then
                echo "⏭️  $key: SKIP (no CUDA)"
                ((skipped++)) || true
                continue 2
            fi
        done
    fi
    run_id="smoke_${key}_shortest"
    npz_path="$RS_BASE/$run_id/$run_id"
    # Get component name from validators
    info="${VALIDATORS[$key]}"
    comp_name="${info#*:}"
    comp_name="${comp_name%%:*}"
    npz_name="${info##*:}"
    full_npz="$npz_path/$comp_name/$npz_name"

    if [ ! -f "$full_npz" ]; then
        echo "⚠️  $key: NPZ не найден"
        ((invalid++)) || true
        continue
    fi

    validator="${VALIDATORS[$key]%%:*}"
    validator_path="$DP_ROOT/$validator"
    if [ ! -f "$validator_path" ]; then
        echo "⚠️  $key: валидатор не найден"
        ((invalid++)) || true
        continue
    fi

    result=$("$PYTHON" "$validator_path" "$full_npz" 2>&1)
    if echo "$result" | grep -q "✅ VALID"; then
        echo "✅ $key"
        ((valid++)) || true
    else
        echo "❌ $key"
        ((invalid++)) || true
    fi
done

echo ""
echo "Итого: ✅ $valid валидных, ⏭️  $skipped пропущено, ❌ $invalid невалидных/отсутствуют"
[ "$invalid" -eq 0 ]
