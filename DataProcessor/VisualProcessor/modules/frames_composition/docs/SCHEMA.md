# `frames_composition` — NPZ schema (`frames_composition_npz_v1`)

- **producer**: `frames_composition`
- **producer_version**: `2.0.1`
- **schema_version**: `frames_composition_npz_v1`
- **artifact**: `frames_composition.npz`

## Purpose

Извлечение композиционных признаков по кадрам (union-domain) и агрегирование на уровне видео.

- **Per-frame**: `frame_feature_values (N,D)` + `frame_feature_names (D,)`
- **Video-level**: `feature_values (F,)` + `feature_names (F,)`

## Dependencies / axis

Hard deps (no-fallback, aligned frame_indices):

- `core_object_detections` (valid empty allowed)
- `core_face_landmarks` (valid empty allowed → `no_faces_in_video`)
- `core_depth_midas` (**must be status=ok**, no-fallback)

Axis:

- `frame_indices`: строго от Segmenter (`metadata.json["frames_composition"]["frame_indices"]`)
- `times_s = union_timestamps_sec[frame_indices]` (no-fallback)

## Output keys

Обозначения: `N=len(frame_indices)`, `D=len(frame_feature_names)`, `F=len(feature_names)`.

| key | required | tier | dtype | shape | notes |
|---|---:|---|---|---|---|
| `frame_indices` | True | model_facing | int32 | `(N,)` | sorted+unique |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `frame_feature_names` | True | model_facing | object | `(D,)` | имена per-frame фич |
| `frame_feature_values` | True | model_facing | float32 | `(N,D)` | per-frame фичи (NaN если не определено) |
| `frame_feature_present_ratio` | True | model_facing | float32 | `(D,)` | доля finite значений по каждому столбцу (помогает моделям интерпретировать NaN) |
| `feature_names` | True | model_facing | object | `(F,)` | имена video-level агрегатов |
| `feature_values` | True | model_facing | float32 | `(F,)` | значения video-level агрегатов |

## `meta` contract

`meta` — dict (object array), baseline keys + модели + тайминги:

- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- versions: `producer`, `producer_version`, `schema_version`, `dataprocessor_version`
- status: `status` (`ok|empty|error`), `empty_reason`
- models: `models_used` (обычно `[]`), `model_signature`
- timings: `stage_timings_ms` (dict[str,float])

Optional config highlights: `feature_set`, `features`, `num_workers`.

## Notes (robustness)

- Image-based метрики (`edge_density`, `symmetry_*`, `saliency_center_offset`, `leading_lines`) считаются на кадре после **heuristic letterbox-crop** (удаление top/bottom чёрных полос), чтобы снизить шум от black bars.

## Empty / error semantics

- **Error**:
  - missing/empty `frame_indices` in metadata (no-fallback)
  - missing `union_timestamps_sec` (no-fallback)
  - frame_indices mismatch with any hard dependency (no-fallback)
  - `core_depth_midas.meta.status != "ok"` (no-fallback)
- **Valid empty** (`status="empty"`):
  - `empty_reason="no_faces_in_video"` when no faces detected by `core_face_landmarks`
  - все ключи присутствуют; per-frame фичи, зависящие от лиц, будут NaN (кроме `face_present=0`)


