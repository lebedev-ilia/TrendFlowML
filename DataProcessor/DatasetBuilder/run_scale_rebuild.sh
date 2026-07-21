#!/usr/bin/env bash
# run_scale_rebuild.sh — one command to rebuild the content+real-targets dataset and
# re-run the full evaluation when Agent A's scale corpus lands. Operationalizes the
# pipeline validated in exp_0005-0009 so the reaction to more videos is instant.
#
# Usage:
#   run_scale_rebuild.sh <tag> [--download] [--spec <feature_spec.yaml>]
#
#   <tag>        artifact subdir under Models/baseline/artifacts/<tag>/
#   --download   first pull corpus_out rs/ NPZ from the S3 volume (skip if already
#                local in storage/corpus_npz, or if data is on HF — see caveat below)
#   --spec       feature spec (default: feature_spec_v0.5.yaml — the scale superset)
#
# ACCESS-PATH CAVEAT: the 300-run NPZ live on the S3 volume (s3_corpus.py). Agent A's
# scale plan streams NPZ to HF and DELETES from the volume, so a scale run's NPZ may
# be on HF instead. If --download finds nothing on the volume, the NPZ are on HF and
# an HF reader is needed (not yet written) — stop and add it rather than proceeding
# on stale/partial local data.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/Models/.venv/bin/python"
DB="$REPO/DataProcessor/DatasetBuilder"
AN="$REPO/Models/state/analysis"
# array (not a plain string) — REPO path can contain spaces ("Рабочий стол")
SHARDS=(
  "$REPO/storage/pre_final_data/data_00.json"
  "$REPO/storage/pre_final_data/data_01.json"
  "$REPO/storage/pre_final_data/data_02.json"
)

TAG="${1:?usage: run_scale_rebuild.sh <tag> [--download] [--spec <spec>]}"; shift || true
SPEC="$DB/feature_spec_v0.5.yaml"
DOWNLOAD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --download) DOWNLOAD=1 ;;
    --spec) SPEC="$2"; shift ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac; shift
done

OUT="$REPO/Models/baseline/artifacts/$TAG"
mkdir -p "$OUT"
echo "== scale rebuild: tag=$TAG spec=$(basename "$SPEC") download=$DOWNLOAD =="

if [ "$DOWNLOAD" = "1" ]; then
  echo "== [1/5] download corpus_out rs/ NPZ from S3 volume =="
  "$PY" "$DB/s3_corpus.py" --dest "$REPO/storage/corpus_npz"
  n=$(find "$REPO/storage/corpus_npz" -name '*.npz' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || { echo "!! no NPZ downloaded — data may be on HF (see caveat). Stopping."; exit 3; }
fi

echo "== [2/5] build content+real-targets dataset =="
"$PY" "$DB/build_corpus_content_dataset.py" \
  --rs-root "$REPO/storage/corpus_npz" --prefinal "${SHARDS[@]}" \
  --feature-spec "$SPEC" --out "$OUT/dataset_corpus_content.parquet"

echo "== [3/5] Phase-1 feature analysis (dist/NaN/const/redundancy/segmenter) =="
"$PY" "$DB/build_training_table.py" --rs-root "$REPO/storage/corpus_npz" \
  --feature-spec "$SPEC" --out "$OUT/content_table.parquet"
"$PY" "$AN/analyze_features.py" --table "$OUT/content_table.parquet" || true

echo "== [4/5] robust CV: S0 vs S0+lean vs S0+full content =="
"$PY" "$AN/v2_cv_experiment.py" --dataset "$OUT/dataset_corpus_content.parquet" \
  --k 5 --topk 25 --max-iter 200 2>&1 | grep -vE "Warning|warn" || true

echo "== [5/5] residual signal: does content add beyond snapshot_0? =="
"$PY" "$AN/residual_content_signal.py" --dataset "$OUT/dataset_corpus_content.parquet" \
  --heads views_21d likes_21d 2>&1 | grep -vE "Warning|warn" || true

echo "== done -> $OUT (dataset_metadata.json has row/target/channel counts) =="
