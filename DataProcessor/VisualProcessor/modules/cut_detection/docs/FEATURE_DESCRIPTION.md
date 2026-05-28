# `cut_detection` — что в NPZ и CSV (audit)

**producer** в `meta` NPZ: `cut_detection`  
**producer_version (код `utils/cut_detection.py`):** `2.0`  
**schema_version (основной артефакт):** `cut_detection_npz_v1`  
**Папка:** `result_store/.../cut_detection/`  
**Файл:** `cut_detection_features_<ts>_<uid>.npz` (timestamped, см. [SCHEMA.md](SCHEMA.md))

## Назначение

Детекция границ монтажа (hard/soft/motion/… cuts) и агрегаты pacing/editing. Зависит от **core_optical_flow**, **core_face_landmarks**, **core_object_detections** (baseline, без fallback). CLIP/глубинные ветки — по конфигу.

## Ключи NPZ (сводка)

| Группа | Ключи |
|--------|--------|
| Ось | `frame_indices` (N), `times_s` (N) из `union_timestamps_sec` (N ≥ 2 по контракту) |
| Результаты | `features` (0-d object → dict), `detections` (0-d object → dict) — см. [FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md) |
| Опц. | `model_facing_npz_path` — путь к доп. NPZ `cut_detection_model_facing_*` ([SCHEMA_MODEL_FACING.md](SCHEMA_MODEL_FACING.md)) |
| Meta | `meta` (dict) |

**Downstream:** в `detections` при **`meta.status=ok`** ожидается **`shot_boundaries_frame_indices`** (см. `shot_quality`, [SCHEMA.md](SCHEMA.md)).

`stage_timings_ms` (в meta, мс) в `run()` задаёт в т.ч. **`process`**, **`total_without_save`** → в плоском CSV: `meta_timing_process`, `meta_timing_total_without_save` (см. `component_feature_qa.flatten_meta`).

## Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|----------|
| `times_s` | Неубывающий ряд |
| `meta.processed_frames` vs `meta.total_frames` | **processed ≤ total** |
| `len(frame_indices)` vs `meta.total_frames` | **N ≤ total** (семпл ⊆ union) |
| `features.*_count` | Скалярные счётчики **≥ 0** (finite) |

## CSV / melt / QA

- Melt: `view_csv_melt_interesting.json` → `cut_detection` (run identity + meta).
- QA: `view_csv_feature_qa.json` → `cut_detection`.
- RU: `view_csv_feature_descriptions_ru.json` — при необходимости.

## Валидатор

Из корня репозитория:

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/cut_detection/utils/validate_cut_detection_npz.py \
  <path/to/cut_detection_features_*.npz> [--struct] [--qa] [--ranges]
```

- **`--struct`** — ключи, N≥2, `features`/`detections` dict, при `status=ok` — наличие `shot_boundaries_frame_indices` в `detections`.
- **`--qa`** — плоский `meta` против `view_csv_feature_qa.json`.
- **`--ranges`** — см. таблицу выше.

Пакетный обход основного артефакта (без `--qa`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/cut_detection/utils/validate_cut_detection_npz.py \
  --results-base /path/to/storage/result_store --platform-id youtube
```

Отдельно: `utils/validate_cut_detection.py` — агрегаты по датасету, не заменяет проверку одного NPZ.

## См. также

- [README.md](../README.md), [SCHEMA.md](SCHEMA.md)
