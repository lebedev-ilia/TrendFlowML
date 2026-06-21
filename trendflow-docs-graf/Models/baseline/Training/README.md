# Training (baseline models)

This folder implements **M5** from `Models/docs/plan_dev/BASELINE_DEV_PLAN.md`:
- load the baseline dataset built by `DatasetBuilder/build_full_dataset.py`
- split (hybrid time + channel-group)
- train a baseline regressor per output (views/likes × 7/14/21; 7d is masked)
- write reproducible model artifacts + metrics reports

## Quick start (example)

From repo root:

```bash
python3 Models/baseline/Training/train_baseline.py \
  --dataset /abs/path/to/dataset.parquet \
  --dataset-metadata /abs/path/to/dataset_metadata.json \
  --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
  --out-dir Models/baseline/Training/artifacts/baseline_run_001 \
  --model-family catboost \
  --model-version baseline_v0 \
  --seed 1337
```

If you don't have CatBoost installed, you can use `--model-family sklearn` as a CPU-only fallback.

## Golden sets (quality gate)

Generate fixed evaluation sets (keyed by `dataset_fingerprint`):

```bash
python3 Models/baseline/Training/generate_golden_sets.py \
  --dataset /abs/path/to/dataset.parquet \
  --dataset-metadata /abs/path/to/dataset_metadata.json
```

Evaluate using a generated golden set:

```bash
python3 Models/baseline/Training/evaluate_baseline.py \
  --dataset /abs/path/to/dataset.parquet \
  --model-dir /abs/path/to/model_artifacts \
  --out-dir /abs/path/to/eval_out \
  --eval-set holdout \
  --golden-set-dir Models/baseline/Training/golden_sets/<dataset_fingerprint>
```

## Evaluation (quality gate)

Produces:
- `metrics.json`
- `report.md`

Example (test split):

```bash
python3 Models/baseline/Training/evaluate_baseline.py \
  --dataset /abs/path/to/dataset.parquet \
  --model-dir Models/baseline/Training/artifacts/baseline_run_001 \
  --out-dir /abs/path/to/eval_out \
  --eval-set test
```

## Smoke E2E

Dataset → Train → Eval(regression_mini) → Predict(one run):

```bash
python3 Models/baseline/Training/smoke_e2e.py \
  --dataset /abs/path/to/dataset.parquet \
  --dataset-metadata /abs/path/to/dataset_metadata.json \
  --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
  --work-dir /tmp/baseline_smoke \
  --model-family sklearn \
  --seed 1337 \
  --rs-base /abs/path/to/result_store \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id>
```
---

## Навигация

[Models](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
