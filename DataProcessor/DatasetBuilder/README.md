# DatasetBuilder

Цель: собрать **tabular training table** из per-run артефактов (`result_store/.../manifest.json` + NPZ).

Статус:
- baseline v0 — собираем **фичи** из NPZ и статусы компонентов.
- baseline v1 — собираем **полный датасет** (features + snapshot_0 + targets + masks + metadata) под baseline models (см. `Models/baseline/README.md`).

## Быстрый старт

### 1) Сбор фичей (manifest + NPZ → table)

```bash
python DatasetBuilder/build_training_table.py \
  --rs-base /abs/path/to/VisualProcessor/result_store \
  --out-csv /abs/path/to/training_table.csv
```

### 2) YOLO seed (не относится к baseline models)

Seed‑сэмплинг кадров из видео под YOLO‑разметку (images/labels + manifest):

```bash
python DatasetBuilder/sample_video_frames_for_yolo.py \
  --videos-root /abs/path/to/videos_root \
  --out-dir /abs/path/to/yolo_seed_v1 \
  --max-videos 2000 \
  --per-video 3 \
  --mode random \
  --min-spacing-sec 1.5 \
  --seed 1337
```

Seed‑сэмплинг из HuggingFace dataset (video column):

```bash
python DatasetBuilder/sample_video_frames_for_yolo.py \
  --hf-dataset "Ilialebedev/videos1" \
  --hf-split train \
  --hf-video-column video \
  --out-dir /abs/path/to/yolo_seed_v1 \
  --max-videos 20000 \
  --per-video 3 \
  --mode random \
  --min-spacing-sec 1.5 \
  --seed 1337
```

Seed‑сэмплинг из HuggingFace dataset **repo с файлами mp4** (без таблицы/колонок):

```bash
python DatasetBuilder/sample_video_frames_for_yolo.py \
  --hf-repo "Ilialebedev/videos1" \
  --hf-revision main \
  --out-dir /abs/path/to/yolo_seed_v1 \
  --max-videos 20000 \
  --per-video 3
```

## Feature spec (single source of truth)

Baseline feature set описан в:
- `DatasetBuilder/feature_spec.yaml`

Там задаются:
- список baseline компонентов (allowed + required)
- snapshot_0 поля (как baseline features)
- temporal поля (включая `video_age_hours_at_snapshot1`)

## Полный датасет (features + snapshot_0 + targets)

Если у тебя есть снапшоты счётчиков (например `data_00.json`: `video_id -> metadata + snapshot_0..snapshot_3`), можно собрать полный датасет:

```bash
python3 DatasetBuilder/build_full_dataset.py \
  --rs-base /abs/path/to/result_store \
  --data-json /abs/path/to/data_00.json \
  --feature-spec DatasetBuilder/feature_spec.yaml \
  --out-dataset /abs/path/to/dataset.parquet \
  --out-metadata /abs/path/to/dataset_metadata.json \
  --require-14-21 \
  --enforce-required-components \
  --required-policy drop
```

Outputs:
- `dataset.parquet` (или CSV/JSONL fallback)
- `dataset_metadata.json` (fingerprint + версии + статистика фильтраций/required checks)

## Targets builder (отдельно)

Если нужно отдельно посчитать targets по subset video_id:
- `DatasetBuilder/add_targets.py` (читает большой JSON стримингово; см. исходник)

## Enrichment stub (video_id → channel_id)

Для честного channel-group split нужен стабильный `channel_id`:
- `DatasetBuilder/enrichment.py` — утилита для нормализации mapping (JSON/CSV) в JSON dict.

## Что попадает в таблицу

- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `component_status__*` (ok=1, empty=0, error=-1)
- фичи из NPZ:
  - если NPZ содержит `feature_names`/`feature_values` → развернём 1:1
  - иначе возьмём числовые поля и посчитаем агрегаты по массивам (mean/std/min/max/p50/p90)

Дополнительно в полном датасете (`build_full_dataset.py`) добавляются:
- `views_0/likes_0/comments_0/...` из `snapshot_0`
- `publishedAt/language/duration_sec/analysis_fps/...`
- `video_age_hours_at_snapshot1` (приближение: `manifest.created_at - publishedAt`)
- targets:
  - `target_{views|likes}_{7d|14d|21d}` (log1p(delta))
  - `mask_{7d|14d|21d}` (7d masked)


