# `detalize_face` — что в NPZ и CSV (audit)

**schema_version:** `detalize_face_npz_v3`  
**Артефакт (baseline):** `.../detalize_face/detalize_face.npz` (см. [SCHEMA.md](SCHEMA.md))

## Назначение

Производные face-features поверх `core_face_landmarks` (MediaPipe) и `frames_dir`: ось **Segmenter** `metadata[detalize_face].frame_indices` (fallback на `core_face_landmarks.frame_indices`), маски `face_present` / `processed_mask` / `primary_valid`, `primary_compact_features` (40-D), `aggregated`, `faces_agg`, опциональные `primary_*` кривые (эвристики). Зависимость: **landmarks.npz** (no-fallback).

## Ключи NPZ (сводка)

| Группа | Ключи |
|--------|--------|
| Сводка | `summary` (object), `meta` (object) |
| Ось N | `frame_indices`, `times_s`, `face_present`, `processed_mask`, `primary_valid`, `face_count`, `primary_tracking_id` |
| Модель | `primary_compact_features` (N, 40) |
| Агрегаты | `aggregated` (object, `detalize_face_aggregated_v1`), `faces_agg` (object) |
| Опц. | `primary_gaze_at_camera_prob`, `primary_blink_rate`, … — если `write_primary_curves` (см. meta) |

`stage_timings_ms` в **meta** (и дубли в `summary`) включает в run(): **`frame_manager_ms`**, **`process_ms`**, **`total_ms`** → в плоском CSV: `meta_timing_frame_manager_ms`, `meta_timing_process_ms`, `meta_timing_total_ms`.

## Meta (важно для wide CSV)

- `module_sampling_policy_version`, `face_frames_sampling_policy_version`
- `write_primary_curves`, `write_primary_compact_features`
- `ui_payload` (опц.), `status` / `empty_reason` (valid empty: `no_faces_in_video`)

## CSV / melt / QA

- Melt: `view_csv_melt_interesting.json` → `detalize_face`
- QA: `view_csv_feature_qa.json` → `detalize_face`
- Подробные имена внутри `features`/`curves` см. [FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md)

## Нормальные диапазоны (`--ranges`)

| Проверка | Ожидание |
|----------|-----------|
| `times_s` | Не убывает (N>1) |
| `face_count` | **≥ 0** для finite |
| `meta.processed_frames` | **∑(processed_mask)** (не длина оси; см. `summary` в модуле) |
| `sum(processed_mask)` | **≤ N** = len(`frame_indices`) |
| `meta.processed_frames` / `meta.total_frames` (если оба int) | `processed ≤ total` |
| `meta.empty_reason` при `status=empty` | Обычно **`no_faces_in_video`** |
| `meta.stage_timings_ms`, `summary.stage_timings_ms` | **≥ 0** |

`meta.status=error`: в `--struct` / батче — краткое сообщение; в `--ranges` — проверки пропускаются.

## Валидатор (один NPZ)

```bash
cd <repo>
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  DataProcessor/VisualProcessor/modules/detalize_face/utils/validate_detalize_face_npz.py \
  <path/to/detalize_face.npz> --struct --qa --ranges
```

Батч: `--results-base storage/result_store --platform-id youtube` (обход `**/detalize_face/detalize_face.npz`).

Существующий `utils/validate_detalize_face.py` — отдельный сценарий анализа прогонов, не заменяет этот чек.
