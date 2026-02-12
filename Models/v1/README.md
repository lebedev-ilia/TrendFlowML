# v1 (Transformers + trainable Encoder)

Реализация плана `Models/docs/plan_dev/V1_DEV_PLAN.md` и контрактов:
- `Models/docs/contracts/V1_TRANSFORMER_MODEL.md`
- `Models/docs/contracts/MODEL_CONTRACTS_V1.md`

## Milestones mapping (что уже реализовано в коде)

- **V0**: build “model-ready” dataset index (`v1_dataset_index.*`) — `Models/v1/data/build_v1_dataset_index.py`
- **V1**: Encoder v0 (deterministic) — `Models/v1/encoder/encoder_v0.py`
- **V2**: v1 model skeleton — `Models/v1/model/v1_skeleton.py`
- **V2 training loop** — `Models/v1/training/train_v1_skeleton.py`
- **V3**: text/comments tokens (no raw) — `Models/v1/text/build_text_embeddings.py`
- **V4**: Encoder v1 (trainable) — `Models/v1/encoder/encoder_v1.py`
- **V5**: quantile heads (p10/p50/p90) + calibration report — `Models/v1/model/v1_skeleton.py` + `Models/v1/training/evaluate_v1.py`

## Input requirements (what must exist on disk)

### v1 dataset index (`v1_dataset_index.parquet`)

Produced by `Models/v1/data/build_v1_dataset_index.py`. Minimum required columns:

- **IDs**: `platform_id`, `video_id`, `run_id`
- **Pointers to run artifacts**:
  - `core_clip_npz_path` — NPZ must contain `frame_embeddings (N,512) float32` and `frame_indices (N,) int`
  - `segmenter_metadata_path` — JSON must contain `union_timestamps_sec (U,) float`
- **Snapshot/meta fields** (used as meta token):
  - `views_0`, `likes_0`, `comments_0`
  - `channel_subscribers_0`, `channel_total_views_0`, `channel_total_videos_0`
  - `duration_sec`, `publishedAt`
  - `video_age_hours_at_snapshot1` (approx = `manifest_created_at - publishedAt`)
- **Targets**:
  - `target_views_{7d|14d|21d}`, `target_likes_{7d|14d|21d}` on `log1p(delta)` scale
  - `mask_{7d|14d|21d}` (7d can be 0)

Optional but recommended:
- `channel_id` (for true channel-group split; otherwise we fall back to `channelTitle`).

### Text index (`v1_text_index.parquet`) (optional)

Produced by `Models/v1/text/build_text_embeddings.py`.
Required columns:
- `video_id`
- `text_npz_path` — NPZ must contain `text_tokens (Kc+1,D)` and `text_mask (Kc+1,)`.

## Quick start (skeleton)

1) Build v1 dataset index:

```bash
python3 Models/v1/data/build_v1_dataset_index.py \
  --rs-base /abs/path/to/result_store \
  --data-json /abs/path/to/data_00.json \
  --out-index /abs/path/to/v1_dataset_index.parquet \
  --out-metadata /abs/path/to/v1_dataset_metadata.json
```

2) Train skeleton (requires torch + numpy + pandas):

```bash
python3 Models/v1/training/train_v1_skeleton.py \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --out-dir Models/v1/artifacts/v1_run_001 \
  --seed 1337
```

## V3: text/comments tokens (no raw)

Build per-video text artifacts and an index mapping:

```bash
python3 Models/v1/text/build_text_embeddings.py \
  --data-json /abs/path/to/data_00.json \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --out-dir /abs/path/to/v1_text_npz \
  --out-index /abs/path/to/v1_text_index.parquet \
  --kc 8 \
  --max-comments 100 \
  --device cpu
```

**No-network note**: `sentence-transformers` weights must be present/cached; runtime downloads are not allowed.

Then train with text tokens:

```bash
python3 Models/v1/training/train_v1_skeleton.py \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --text-index /abs/path/to/v1_text_index.parquet \
  --out-dir Models/v1/artifacts/v1_run_001 \
  --seed 1337
```

## V4: trainable encoder (visual)

Use trainable encoder v1:

```bash
python3 Models/v1/training/train_v1_skeleton.py \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --text-index /abs/path/to/v1_text_index.parquet \
  --encoder v1 \
  --out-dir Models/v1/artifacts/v1_run_002 \
  --seed 1337
```

## V5: quantiles (p10/p50/p90)

Enable quantile heads and pinball loss:

```bash
python3 Models/v1/training/train_v1_skeleton.py \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --text-index /abs/path/to/v1_text_index.parquet \
  --encoder v1 \
  --quantiles 0.1,0.5,0.9 \
  --out-dir Models/v1/artifacts/v1_run_003 \
  --seed 1337
```

Evaluate checkpoint (p50 Spearman/MAE + p10/p90 coverage):

```bash
python3 Models/v1/training/evaluate_v1.py \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --checkpoint Models/v1/artifacts/v1_run_003/checkpoint.pt \
  --out-dir /abs/path/to/v1_eval \
  --eval-set test
```

## Golden sets (holdout/regression_mini)

Generate fixed sets keyed by `v1_dataset_fingerprint`:

```bash
python3 Models/v1/training/generate_v1_golden_sets.py \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --v1-metadata /abs/path/to/v1_dataset_metadata.json
```

Evaluate holdout:

```bash
python3 Models/v1/training/evaluate_v1.py \
  --v1-index /abs/path/to/v1_dataset_index.parquet \
  --checkpoint Models/v1/artifacts/v1_run_003/checkpoint.pt \
  --out-dir /abs/path/to/v1_eval_holdout \
  --eval-set holdout \
  --golden-set-dir Models/v1/training/golden_sets/<v1_dataset_fingerprint>
```


