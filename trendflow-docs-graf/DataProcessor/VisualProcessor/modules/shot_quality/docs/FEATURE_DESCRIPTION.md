# `shot_quality` — фичи и артефакт (audit)

**schema_version:** `shot_quality_npz_v3`  
**Файл:** `shot_quality/shot_quality.npz` (см. [SCHEMA.md](SCHEMA.md), [README.md](../README.md), [FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md))

## Назначение

**Техническое качество видео** на уровне кадров и шотов: компактные `frame_features` (резкость, шум, экспозиция, …), **zero-shot CLIP**-вероятности по фиксированным quality-промптам (`quality_probs`), агрегаты по шотам из **cut_detection**. **Hard deps:** `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks` (face-ROI могут быть NaN), `cut_detection`; ось `frame_indices` / `times_s` — из Segmenter (`metadata["shot_quality"]["frame_indices"]`, без fallback).

## Топ-уровень NPZ (ось N = |frame_indices|)

| Ключ | Форма | Смысл |
|------|--------|--------|
| `frame_indices` | (N,) int32 | Семпл Segmenter, sorted+unique |
| `times_s` | (N,) float32 | `union_timestamps_sec[...]` |
| `feature_names` | (F,) object | Имена колонок `frame_features` |
| `frame_features` | (N, F) float32 | Метрики качества; NaN = нет/не определено |
| `frame_feature_present_ratio` | (F,) float32 | Доля конечных значений по признаку |
| `quality_probs` | (N, P) float16 | Softmax по P промптам CLIP (строка ≈ сумма 1) |
| `shot_ids` | (N,) int32 | Кадр → id шота 0..S-1 |
| `shot_start_frame` / `shot_end_frame` | (S,) int32 | Границы шотов (union-индексы) |
| `shot_frame_count` | (S,) int32 | Семплированных кадров в шоте |
| `shot_features_mean` / `std` / `min` / `max` | (S, F) | Агрегаты по шотам |
| `shot_frame_feature_present_ratio` | (S, F) float32 | Доля конечных кадр-значений в шоте |
| `shot_quality_topk_ids` / `shot_quality_topk_probs` | (S, K) | Top-K классов по среднему prob по кадрам шота (сумма по K ≠ 1) |
| `shot_quality_conf_mean` / `shot_quality_entropy_mean` | (S,) float32 | Средние confidence / энтропия по кадрам шота |
| `meta` | object | Канонический meta (см. SCHEMA) |

`ui_payload` и отладочный `impl_meta` **не** лежат в корне NPZ: `run()` переносит их в `meta.ui_payload` и `meta.impl_meta` перед `save_results`.

## `meta` (wide CSV / melt)

Плоский вывод: `meta_producer`, `meta_producer_version`, `meta_schema_version`, `meta_status`, `meta_empty_reason`, `meta_model_signature`, run identity (`meta_platform_id`, `meta_video_id`, `meta_run_id`, `meta_config_hash`, `meta_sampling_policy_version`, `meta_dataprocessor_version`), `meta_total_frames`, `meta_processed_frames`, `meta_analysis_fps` / `width` / `height`, `meta_stage_timings_ms` → `meta_timing_frame_manager_ms`, `meta_timing_process_ms`, `meta_timing_total_ms` (см. `BaseModule` + override `run()`). Промпты: только version + sha в `impl_meta`, не полный текст.

## Валидатор

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  modules/shot_quality/utils/validate_shot_quality_npz.py <path/to/shot_quality.npz> \
  [--struct] [--qa] [--ranges]
```

- **`--struct`** — ключи, размеры `N/S/F/P/K`, согласованность, softmax по строкам `quality_probs` (допуск 0.02), `shot_ids` ∈ `[0, S-1]`.
- **`--qa`** — плоский `meta` против `storage/result_store/view_csv_feature_qa.json` (нужен импорт `qa` из `DataProcessor`).
- **`--ranges`** — неубывающий `times_s`, `meta.processed_frames` ≤ `meta.total_frames`, мягкие границы: `shot_quality_topk_probs`, `shot_quality_conf_mean` ∈ `[0,1]`; `shot_quality_entropy_mean` ≥ 0 (по конечным значениям).

Скан всех `shot_quality/shot_quality.npz` под `result_store` (только `--struct`‑эквивалент, без `--qa`):

```bash
PYTHONPATH=DataProcessor:DataProcessor/VisualProcessor \
  DataProcessor/VisualProcessor/.vp_venv/bin/python3 \
  modules/shot_quality/utils/validate_shot_quality_npz.py \
  --results-base /path/to/storage/result_store --platform-id youtube
```

Пример файла: `storage/result_store/.../shot_quality/shot_quality.npz`.

Расшифровка имён фич в `frame_features`: [FEATURES_DESCRIPTION.md](FEATURES_DESCRIPTION.md).
---

## Навигация

[FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
