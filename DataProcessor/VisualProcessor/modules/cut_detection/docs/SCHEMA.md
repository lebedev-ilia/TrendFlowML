# Schema: `cut_detection_npz_v1`

- **producer**: `cut_detection`
- **artifact_kind**: `npz`
- **allow_extra_keys**: `False`
- **schema_system_version**: `vp_schema_v1`

## Meta

### Required meta keys

- `producer`
- `producer_version`
- `schema_version`
- `created_at`
- `platform_id`
- `video_id`
- `run_id`
- `config_hash`
- `sampling_policy_version`
- `dataprocessor_version`
- `status`
- `empty_reason`
- `models_used`
- `model_signature`
- `stage_timings_ms`

### Optional meta keys

- `total_frames`
- `processed_frames`
- `frames_dir`
- `analysis_fps`
- `analysis_width`
- `analysis_height`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` | Frame indices in union-domain (from Segmenter) |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` | `union_timestamps_sec[frame_indices]` (strict time axis) |
| `features` | `True` | `analytics` | `object` | `()` | Boxed dict of aggregate editing/pacing metrics (см. `FEATURES_DESCRIPTION.md`) |
| `detections` | `True` | `analytics` | `object` | `()` | Boxed dict of detected events / boundaries (hard/soft/motion/jump cuts, shot/scene boundaries) |
| `model_facing_npz_path` | `False` | `debug` | `str` | `()` | Optional path to the additional model-facing artifact written by the module (best-effort) |
| `meta` | `True` | `debug` | `object` | `()` | Boxed meta dict |

## Empty/error semantics

`cut_detection` — baseline required module, **empty не ожидается**.

- `frame_indices` отсутствует/пустой → **error**
- `union_timestamps_sec` отсутствует/невалиден/немонотонен → **error**
- `len(frame_indices) < 2` → **error**
- отсутствуют required core deps (baseline): `core_optical_flow`, `core_face_landmarks`, `core_object_detections` → **error**

## Notes

- Для моделей (transformer/encoder) рекомендуется использовать дополнительный артефакт:
  - `schema_version=cut_detection_model_facing_npz_v1`, см. [SCHEMA_MODEL_FACING.md](./SCHEMA_MODEL_FACING.md)
- `features` и `detections` — намеренно “boxed dict”: контракт фиксирует ключи верхнего уровня, а детализация фич описана в документации.


