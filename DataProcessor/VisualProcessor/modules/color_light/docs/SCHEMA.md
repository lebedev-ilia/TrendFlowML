# `color_light` — NPZ schema (`color_light_npz_v2`)

- **producer**: `color_light`
- **producer_version**: `2.0.2`
- **schema_version**: `color_light_npz_v2`
- **artifact**: `color_light_features.npz`

## Purpose

Анализ цвета/освещения на оси Segmenter (union-domain). Основные model-facing выходы:

- `frame_compact_features (M,16)` + имена + `frame_compact_frame_indices` (стабильный encoder input)
- `video_features` + `aggregated` (tabular baseline / глобальные метрики)

Тяжёлые debug/analytics объекты (`frames`, `scenes`) могут быть отключены через `store_debug_objects`.

## Dependencies / axis

- **Hard dependency**: `scene_classification` (no-fallback)
  - должен содержать `scenes: dict` (или legacy `scenes_raw`)
  - каждая сцена содержит `indices` (list[int]) и `scene_label`
- **Frame axis**: берётся строго из Segmenter (`metadata.json` → `color_light.frame_indices` via `BaseModule.get_frame_indices()`).
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback).

## Output keys

Обозначения: `N=len(frame_indices)`, `M=len(sequence_frame_indices)`.

| key | required | tier | dtype | shape | notes |
|---|---:|---|---|---|---|
| `frame_indices` | True | model_facing | int32 | `(N,)` | union frame indices, отсортированы/уникальны |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `sequence_frame_indices` | True | analytics | int32 | `(M,)` | compat: порядок соответствует `sequence_inputs["frames"]` |
| `sequence_times_s` | True | analytics | float32 | `(M,)` | compat |
| `sequence_inputs` | True | debug | object | `()` | compat: dict `{"frames": ... , "scenes": ... , "global": ...}`; `scenes/global` имеют variable dims |
| `frame_compact_features` | True | model_facing | float32 | `(M,16)` | **fixed** compact vectors для моделей |
| `frame_compact_feature_names` | True | model_facing | object | `(16,)` | стабильные имена компактных фич |
| `frame_compact_frame_indices` | True | model_facing | int32 | `(M,)` | frame indices для `frame_compact_features` |
| `video_features` | True | model_facing | object | `()` | dict глобальных метрик (style probs, entropy/gini, динамика и т.д.) |
| `aggregated` | True | model_facing | object | `()` | фиксированные tabular stats (mean/std/quantiles) |
| `scenes` | True | analytics | object | `()` | dict scene_key→scene_features (может быть `{}` если `store_debug_objects=false`) |
| `frames` | True | debug | object | `()` | dict scene_key→{frame_idx→frame_features} (может быть `{}` если `store_debug_objects=false`) |

### Notes on `sequence_inputs` (compat)

`sequence_inputs["frames"]` и `sequence_inputs["scenes"]` строятся как **векторизация числовых ключей** (sorted order) соответствующих dict-объектов. Это обеспечивает детерминированность, но размерность может меняться при изменении набора числовых полей → downstream модели должны опираться на версионирование `producer_version/schema_version`.

В Audit v3 `sequence_inputs` оставлен как **debug/compat**. Для моделей используем `frame_compact_features` + `frame_compact_feature_names`.

## `meta` contract (required keys)

`meta` — dict (object array), строго по `docs/contracts/ARTIFACTS_AND_SCHEMAS.md` + schema JSON:

- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- versions: `producer`, `producer_version`, `schema_version`, `dataprocessor_version`
- status: `status` (`ok|empty|error`), `empty_reason`
- models: `models_used` (обычно `[]`), `model_signature` (sha256 от models_used)
- timings: `stage_timings_ms` (dict[str,float])

Опциональные highlights (reproducibility): `store_debug_objects`, `hue_hist_bins`, `palette_*`, `max_frames_per_scene/stride` (deprecated).

## Empty / error semantics

- **Error**:
  - нет `scene_classification` артефакта (hard dependency)
  - нет `union_timestamps_sec` в `frames_dir/metadata.json`
  - `frame_indices` пустой (Segmenter contract violated)
- **Empty**:
  - `status="empty"`, `empty_reason="after_filt_empty"`: после пересечения Segmenter `frame_indices` с `scene_classification.scenes[*].indices` не осталось кадров.
  - при empty все ключи присутствуют, массивы имеют нулевую длину, dict-объекты пустые.


