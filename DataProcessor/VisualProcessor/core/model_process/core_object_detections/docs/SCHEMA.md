# Schema: `core_object_detections_npz_v2`

- **producer**: `core_object_detections`
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

- `impl`
- `model`
- `box_threshold`
- `batch_size`
- `device`
- `total_frames`
- `total_detections`

## Contract notes

- **Нет persistent track id**: детекции адресуются парой «индекс кадра `N` + слот `M`»; идентификатора трека объекта между кадрами в этом артефакте **нет** (модули вроде `action_recognition` строят свои сегменты/треки локально).

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `areas_frac` | `True` | `model_facing` | `float32` | `(N, M)` |  |
| `boxes` | `True` | `model_facing` | `float32` | `(N, M, 4)` |  |
| `boxes_norm` | `True` | `model_facing` | `float32` | `(N, M, 4)` |  |
| `centers_norm` | `True` | `model_facing` | `float32` | `(N, M, 2)` |  |
| `class_ids` | `True` | `model_facing` | `int32` | `(N, M)` |  |
| `class_names` | `True` | `analytics` | `str` | `(41)` | stable mapping 'id:name' for 0..40 |
| `det_count` | `True` | `analytics` | `int32` | `(N)` |  |
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `logo_region_count` | `True` | `analytics` | `int32` | `(N)` |  |
| `max_logo_area_frac` | `True` | `analytics` | `float32` | `(N)` |  |
| `max_person_area_frac` | `True` | `analytics` | `float32` | `(N)` |  |
| `max_text_area_frac` | `True` | `analytics` | `float32` | `(N)` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `meta_json` | `True` | `debug` | `str` | `()` | meta as JSON string (cross-venv safe) |
| `person_count` | `True` | `analytics` | `int32` | `(N)` |  |
| `scores` | `True` | `model_facing` | `float32` | `(N, M)` |  |
| `sum_logo_area_frac` | `True` | `analytics` | `float32` | `(N)` |  |
| `sum_person_area_frac` | `True` | `analytics` | `float32` | `(N)` |  |
| `sum_text_area_frac` | `True` | `analytics` | `float32` | `(N)` |  |
| `text_region_count` | `True` | `analytics` | `int32` | `(N)` |  |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
| `valid_mask` | `True` | `model_facing` | `bool` | `(N, M)` |  |
