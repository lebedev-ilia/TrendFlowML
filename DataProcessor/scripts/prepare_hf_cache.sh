#!/bin/bash
# Подготовка HuggingFace cache для emotion_diarization (WavLM).
# Добавляет preprocessor_config.json в snapshots, если он отсутствует
# (неполная загрузка microsoft/wavlm-large).
#
# Использование:
#   ./DataProcessor/scripts/prepare_hf_cache.sh
#   HF_CACHE_DIR=/path/to/hub ./DataProcessor/scripts/prepare_hf_cache.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DP_ROOT="$REPO_ROOT/DataProcessor"

# Канонический preprocessor_config для WavLM (microsoft/wavlm-large)
# Источник: https://huggingface.co/microsoft/wavlm-large/raw/main/preprocessor_config.json
create_preprocessor_config() {
    cat << 'JSONEOF'
{
  "do_normalize": true,
  "feature_extractor_type": "Wav2Vec2FeatureExtractor",
  "feature_size": 1,
  "padding_side": "right",
  "padding_value": 0.0,
  "return_attention_mask": true,
  "sampling_rate": 16000
}
JSONEOF
}

# Пути для проверки (в порядке приоритета)
declare -a CACHE_ROOTS=()

if [ -n "$HF_CACHE_DIR" ]; then
    CACHE_ROOTS+=("$HF_CACHE_DIR")
fi

# Bundled models hf_cache
BUNDLED_HUB="$DP_ROOT/dp_models/bundled_models/hf_cache/hub"
if [ -d "$BUNDLED_HUB" ]; then
    CACHE_ROOTS+=("$BUNDLED_HUB")
fi

# Default HF cache
DEFAULT_HUB="$HOME/.cache/huggingface/hub"
if [ -d "$DEFAULT_HUB" ]; then
    CACHE_ROOTS+=("$DEFAULT_HUB")
fi

WAVLM_BASE="models--microsoft--wavlm-large"
SNAPSHOTS_SUBDIR="$WAVLM_BASE/snapshots"
PREPROCESSOR_FILE="preprocessor_config.json"

fixed=0
checked=0

echo "=== Подготовка HF cache для emotion_diarization (WavLM) ==="
echo ""

for cache_root in "${CACHE_ROOTS[@]}"; do
    [ -d "$cache_root" ] || continue
    snap_dir="$cache_root/$SNAPSHOTS_SUBDIR"
    [ -d "$snap_dir" ] || continue

    for rev_dir in "$snap_dir"/*/; do
        [ -d "$rev_dir" ] || continue
        ((checked++)) || true
        target="$rev_dir$PREPROCESSOR_FILE"
        if [ ! -f "$target" ]; then
            create_preprocessor_config > "$target"
            echo "  ✅ Добавлен: $target"
            ((fixed++)) || true
        fi
    done
done

echo ""
if [ $checked -eq 0 ]; then
    echo "⚠️  WavLM cache не найден. Скачайте модель:"
    echo "   huggingface-cli download microsoft/wavlm-large"
    echo "   или: python -c \"from transformers import AutoFeatureExtractor; AutoFeatureExtractor.from_pretrained('microsoft/wavlm-large')\""
    exit 1
fi

if [ $fixed -gt 0 ]; then
    echo "Итого: добавлено $fixed файл(ов) preprocessor_config.json"
else
    echo "✅ Cache готов (preprocessor_config.json уже присутствует)"
fi
