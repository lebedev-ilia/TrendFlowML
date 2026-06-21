# Baseline models (boosting)

Это реализация baseline из `Models/docs/plan_dev/BASELINE_DEV_PLAN.md` и контракта `Models/docs/contracts/BASELINE_MODEL.md`.

Baseline — контрольная точка качества + production fallback (degraded-mode).

## Где что лежит

### Dataset (features + targets)

Dataset собирается из артефактов DataProcessor (manifest + NPZ) и снапшотов счётчиков:

- **Feature spec (single source of truth)**: `DataProcessor/DatasetBuilder/feature_spec.yaml`
- **Features (manifest + NPZ → tabular)**: `DataProcessor/DatasetBuilder/build_training_table.py`
- **Targets (snapshot_0..3 → log1p(delta) + masks)**: `DataProcessor/DatasetBuilder/add_targets.py`
- **Full dataset builder** (features + snapshot_0 fields + targets + metadata + required checks):
  - `DataProcessor/DatasetBuilder/build_full_dataset.py`

Dataset outputs:
- `dataset.parquet` (или CSV/JSONL fallback)
- `dataset_metadata.json` (fingerprint + параметры сборки + статистика required checks)

### Training / Evaluation / Inference

- **Training**: `Models/baseline/Training/train_baseline.py`
- **Evaluation (quality gate)**: `Models/baseline/Training/evaluate_baseline.py`
- **Golden sets generator**: `Models/baseline/Training/generate_golden_sets.py`
- **Smoke E2E**: `Models/baseline/Training/smoke_e2e.py`
- **Inference**: `Models/baseline/Inference/predict_baseline.py`

Shared:
- **NPZ feature extractor** (без зависимости от DataProcessor python-модулей): `Models/baseline/common/npz_features.py`

## Контракты (важное)

- **Targets**: `views` и `likes`, горизонты 7d(masked), 14d, 21d, таргет = `log1p(delta)` относительно `snapshot_0`.
- **Splits**: hybrid time-split по `publishedAt` + channel-group split по `channel_id` (после enrichment).
- **Quality gate**:
  - north star: Spearman на `log1p(delta)`
  - secondary: MAE + Spearman по age buckets
  - golden sets: holdout=2000, regression mini=200 (`Models/docs/contracts/TARGETS_SPLITS_METRICS.md`)

## Quickstart (end-to-end)

Из корня репозитория:

### 1) Собрать датасет

```bash
python3 DataProcessor/DatasetBuilder/build_full_dataset.py \
  --rs-base /abs/path/to/result_store \
  --data-json /abs/path/to/data_00.json \
  --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
  --out-dataset /abs/path/to/dataset.parquet \
  --out-metadata /abs/path/to/dataset_metadata.json \
  --require-14-21 \
  --enforce-required-components \
  --required-policy drop
```

### 2) Обучить baseline

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

### 3) Сгенерировать golden sets (опционально, но рекомендуется)

```bash
python3 Models/baseline/Training/generate_golden_sets.py \
  --dataset /abs/path/to/dataset.parquet \
  --dataset-metadata /abs/path/to/dataset_metadata.json
```

### 4) Оценить (quality gate)

```bash
python3 Models/baseline/Training/evaluate_baseline.py \
  --dataset /abs/path/to/dataset.parquet \
  --model-dir Models/baseline/Training/artifacts/baseline_run_001 \
  --out-dir /abs/path/to/eval_out \
  --eval-set holdout \
  --golden-set-dir Models/baseline/Training/golden_sets/<dataset_fingerprint>
```

Outputs:
- `metrics.json`
- `report.md`

### 5) Инференс по одному run

```bash
python3 Models/baseline/Inference/predict_baseline.py \
  --rs-base /abs/path/to/result_store \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id> \
  --model-dir Models/baseline/Training/artifacts/baseline_run_001 \
  --out-json /tmp/prediction.json \
  --feature-spec DataProcessor/DatasetBuilder/feature_spec.yaml \
  --enforce-required-components \
  --required-policy degraded
```

## Артефакты обучения (формат)

В `Models/baseline/Training/artifacts/<run>/`:
- `training_run_manifest.json`
  - `dataset.dataset_fingerprint` (если передан dataset_metadata)
  - `feature_names[]`
  - `bundles[]`:
    - `views` → `models/views/{7d,14d,21d}.*`
    - `likes` → `models/likes/{7d,14d,21d}.*`
  - `imputation` (если применяли; например median для sklearn)
- `metrics.json`
- `feature_spec.yaml` (snapshot для воспроизводимости)
---

## Навигация

[Models](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
