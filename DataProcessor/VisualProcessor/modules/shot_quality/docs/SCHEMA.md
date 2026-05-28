# `shot_quality` — NPZ schema (`shot_quality_npz_v3`)

- **producer**: `shot_quality`
- **producer_version**: `2.0.2`
- **schema_version**: `shot_quality_npz_v3`
- **artifact**: `shot_quality.npz`
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/shot_quality_npz_v3.json`

## Purpose

Оценка технического качества видео на:
- **frame-level** (компактные признаки качества, `frame_features`)
- **shot-level** (агрегации `frame_features` по шотам из `cut_detection`)
- **CLIP quality** (zero-shot вероятности по фиксированным промптам из `core_clip`)

## Dependencies / axis

Hard deps (no-fallback, aligned `frame_indices`):
- `core_clip`
- `core_depth_midas`
- `core_object_detections`
- `core_face_landmarks` (валидная пустота допустима; face‑ROI фичи становятся `NaN`)
- `cut_detection` (shot boundaries)

Axis:
- `frame_indices`: строго от Segmenter (`metadata.json["shot_quality"]["frame_indices"]`)
- `times_s = union_timestamps_sec[frame_indices]` (no-fallback)

## Empty / error semantics

- `empty` на уровне компонента **не используется**: если нет лиц, это не блокирует non-face метрики (face‑ROI фичи остаются `NaN`).
- Любая отсутствующая зависимость / рассинхрон `frame_indices` / отсутствующие ключи ⇒ **error** (raise).

## Output keys

Обозначения: `N=len(frame_indices)`, `F=len(feature_names)`, `S=len(shot_start_frame)`, `P=число quality-классов (промптов)`.

| key | required | tier | dtype | shape | notes |
|---|---:|---|---|---|---|
| `frame_indices` | True | model_facing | int32 | `(N,)` | sorted+unique |
| `times_s` | True | model_facing | float32 | `(N,)` | `union_timestamps_sec[frame_indices]` |
| `feature_names` | True | model_facing | object | `(F,)` | stable order |
| `frame_features` | True | model_facing | float32 | `(N,F)` | NaN = missing/undefined |
| `frame_feature_present_ratio` | True | model_facing | float32 | `(F,)` | ratio of finite values per feature |
| `quality_probs` | True | model_facing | float16 | `(N,P)` | zero-shot probs from `core_clip` (softmax по `P`; строка **≈** суммируется в 1) |
| `shot_ids` | True | model_facing | int32 | `(N,)` | mapping frame→shot |
| `shot_start_frame` | True | model_facing | int32 | `(S,)` | union frame idx |
| `shot_end_frame` | True | model_facing | int32 | `(S,)` | union frame idx |
| `shot_frame_count` | True | model_facing | int32 | `(S,)` | sampled frames in shot |
| `shot_features_mean` | True | model_facing | float32 | `(S,F)` | shot aggregates (mean) |
| `shot_features_std/min/max` | True | analytics | float32 | `(S,F)` | shot aggregates |
| `shot_frame_feature_present_ratio` | True | model_facing | float32 | `(S,F)` | finite ratio of frame_features per shot/feature |
| `shot_quality_topk_ids` | True | model_facing | int32 | `(S,K)` | per-shot top-K quality class ids |
| `shot_quality_topk_probs` | True | model_facing | float32 | `(S,K)` | per-shot top-K probs (mean over frames); **не** полное распределение — сумма по `K` **не обязана** быть 1 (в отличие от `quality_probs` по полным `P` классам) |
| `shot_quality_conf_mean` | True | model_facing | float32 | `(S,)` | mean frame confidence per shot |
| `shot_quality_entropy_mean` | True | model_facing | float32 | `(S,)` | mean entropy per shot |
| `meta` | True | debug | object | `()` | canonical meta contract |

## Meta

### Required meta keys

- `producer`, `producer_version`, `schema_version`, `created_at`
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `status`, `empty_reason`
- `models_used`, `model_signature` — в `models_used` включается имя весов **CLIP** из `core_clip` (дублируется в `impl_meta.clip_model_name` для отладки)
- `stage_timings_ms`

### Optional meta keys

- `ui_payload` (dict): payload for HTML/UI
- `impl_meta` (dict): debug config + prompts hashes + mappings
- `total_frames`, `processed_frames`, `frames_dir`, `analysis_fps`, `analysis_width`, `analysis_height`


