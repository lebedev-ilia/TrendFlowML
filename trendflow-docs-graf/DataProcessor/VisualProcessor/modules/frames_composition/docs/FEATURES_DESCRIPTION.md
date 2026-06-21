# `frames_composition` — Features

Фичи разделены на **группы**, чтобы их можно было включать/выключать через `--feature-set` / `--features`.

Форматы:
- **Per-frame**: `frame_feature_values[N,D]` + `frame_feature_names[D]`
- **Video-level**: `feature_values[F]` + `feature_names[F]` (агрегаты по времени)

## Groups

### `faces`
- `face_present` (per-frame, 0/1)
- `face_center_x`, `face_center_y` (normalized, 0..1)
- `face_area_ratio` (normalized bbox-area в unit-square; proxy)

### `objects`
- `object_count`
- `object_max_area_ratio` (max bbox area / frame area)
- `object_bbox_coverage_ratio` (sum bbox areas capped at 1.0)

### `anchors`
- `anchor_distance` (0..1): min distance от лица до ближайшего эстетического якоря (thirds/golden/center), нормализовано
- `anchor_type_id` (categorical id): 0=thirds, 1=golden, 2=center (в основном для UI/debug)
- `thirds_alignment` (0..1): heuristic alignment score (UI/debug)

### `balance`
- `saliency_center_offset` (0..1): смещение “центра внимания” (saliency proxy) от центра кадра

### `symmetry`
- `symmetry_score` (≈[-1..1], clipped at consumer): среднее horizontal/vertical корреляции
- `symmetry_h`, `symmetry_v`

### `negative_space`
- `negative_space_ratio` (0..1): \(1 - bbox\_coverage\_ratio\) (cheap proxy)
- `neg_space_balance_lr` (0..1): баланс негативного пространства слева/справа (по bbox coverage proxy)

### `complexity`
- `edge_density` (0..1): доля edge пикселей (Canny)
- `texture_entropy` (≈[0..]): local variance mean (cheap proxy)
- `hue_std` (0..1): std hue / 180
- `saturation_mean` (0..1)

### `leading_lines`
- `line_strength` (0..1): суммарная длина линий / площадь кадра
- `line_count`
- `convergence_score` (0..1): proxy “сходимости” линий
- `dominant_line_id` (categorical id): 0=horizontal,1=vertical,2=diagonal,3=none

### `depth`
Берётся из `core_depth_midas` (no-fallback, всегда должен быть ok):
- `depth_mean`, `depth_std`, `depth_p05`, `depth_p95`

### `style` (UI explainability)
Heuristic probabilities (per-frame, сумма=1):
- `style_minimalist`
- `style_cinematic`
- `style_vlog`
- `style_product_centered`

## Video-level aggregates

По умолчанию модуль агрегирует per-frame фичи в виде статистик:
- `__mean`, `__std`, `__p10`, `__p50`, `__p90`, `__min`, `__max`

Пример:
- `edge_density__mean`
- `saliency_center_offset__p90`

## Empty semantics

Если во всём видео нет лиц:
- NPZ meta: `status="empty"`, `empty_reason="no_faces_in_video"`
- `has_faces=0`
- фичи не “забиваются нулями” (NaN там, где значения не определены)

Структура артефакта, `meta`, диапазоны для QA/melt: **`FEATURE_DESCRIPTION.md`**.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
