#!/usr/bin/env bash
# E2E-фикстуры путей моделей под DP_MODELS_ROOT=.../dp_models/bundled_models
#
# Usage: ./backend/scripts/setup_e2e_model_symlinks.sh
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DP="$REPO_ROOT/DataProcessor/dp_models"
BUNDLE="$DP/bundled_models"

mkdir -p "$BUNDLE/visual/yolo"
YOLO_SRC="$DP/visual/object_detection/yolo11l/yolo11l.pt"
YOLO_DST="$BUNDLE/visual/yolo/yolo11x_41_best.pt"
if [[ -f "$YOLO_SRC" ]]; then
  ln -sfn "../../../visual/object_detection/yolo11l/yolo11l.pt" "$YOLO_DST"
  echo "OK: $YOLO_DST -> yolo11l.pt"
else
  echo "WARN: missing $YOLO_SRC (run: python DataProcessor/scripts/download_models.py --groups visual)" >&2
fi

TEXT_E5="$BUNDLE/text/embeddings/intfloat_multilingual-e5-large/model.safetensors"
if [[ -f "$TEXT_E5" ]]; then
  echo "OK: text embedding model present"
else
  echo "WARN: missing $TEXT_E5" >&2
  echo "  Run: DataProcessor/TextProcessor/.tp_venv/bin/python DataProcessor/scripts/save_sentence_transformer_model.py \\" >&2
  echo "    --output-dir dp_models/bundled_models/text/embeddings/intfloat_multilingual-e5-large" >&2
fi

TOK="$BUNDLE/text/shared_tokenizer_v1/tokenizer.json"
if [[ -f "$TOK" ]]; then
  echo "OK: shared_tokenizer_v1 present"
else
  echo "WARN: missing $TOK" >&2
  echo "  Run: huggingface-cli download bert-base-uncased tokenizer.json --local-dir $BUNDLE/text/shared_tokenizer_v1" >&2
fi

mkdir -p "$BUNDLE/visual/places365"
PLACES_SRC="$DP/visual/places365/categories_places365.txt"
PLACES_DST="$BUNDLE/visual/places365/categories_places365.txt"
if [[ -f "$PLACES_SRC" ]]; then
  ln -sfn "../../../visual/places365/categories_places365.txt" "$PLACES_DST"
  echo "OK: $PLACES_DST -> categories_places365.txt"
else
  echo "WARN: missing $PLACES_SRC" >&2
fi

"$SCRIPT_DIR/setup_e2e_text_assets.sh" 2>/dev/null || true

# content_domain offline DB (required by content_domain semantic head)
CD="$BUNDLE/semantics/content_domain/v1"
mkdir -p "$CD"
if [[ ! -f "$CD/manifest.json" ]]; then
  cat >"$CD/manifest.json" <<'JSON'
{"db_name":"content_domain","db_version":"v1","db_digest":"e2e_stub_v1","files":["domains.jsonl"]}
JSON
  cat >"$CD/domains.jsonl" <<'JSONL'
{"id":0,"name_en":"live_action","prompts_en":["live action video","real world footage"]}
{"id":1,"name_en":"animation","prompts_en":["animated video","cartoon style"]}
JSONL
  echo "OK: content_domain/v1 E2E stub"
fi

mkdir -p "$BUNDLE/audio/emotion_diarization"
EMO_SRC="$DP/audio/emotion_diarization/wavlm_large"
EMO_DST="$BUNDLE/audio/emotion_diarization/wavlm_large"
if [[ -f "$EMO_DST/hyperparams.yaml" ]]; then
  echo "OK: emotion_diarization bundle present ($EMO_DST)"
elif [[ -d "$EMO_SRC" ]] && [[ ! -e "$EMO_DST" ]]; then
  ln -sfn "../../../audio/emotion_diarization/wavlm_large" "$EMO_DST"
  echo "WARN: $EMO_DST -> partial wavlm_large (run setup_e2e_audio_models.sh for full SpeechBrain bundle)"
fi
