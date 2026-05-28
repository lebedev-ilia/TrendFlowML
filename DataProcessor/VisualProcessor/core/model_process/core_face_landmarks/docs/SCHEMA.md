# Schema: `core_face_landmarks_npz_v2`

- **producer**: `core_face_landmarks`
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

- `model_name`
- `total_frames`
- `device`
- `face_empty_reason`
- `pose_empty_reason`
- `hands_empty_reason`
- `person_mask_enabled`
- `person_class_id`
- `person_frames_count`
- `person_window_radius`
- `face_mesh_frames_count`
- `temporal_filter_enabled`
- `temporal_filter_min_cutoff`
- `temporal_filter_beta`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `created_at` | `False` | `debug` | `str | object` | `()` |  |
| `empty_reason` | `True` | `analytics` | `object | str` | `()` |  |
| `face_empty_reason` | `True` | `analytics` | `object | str` | `()` |  |
| `face_landmarks` | `True` | `model_facing` | `float32` | `(N, FACES, 468, 3)` |  |
| `face_landmarks_raw` | `True` | `debug` | `float32` | `(N, FACES, 468, 3)` |  |
| `face_mesh_ran` | `True` | `analytics` | `bool` | `(N)` |  |
| `face_present` | `True` | `model_facing` | `bool` | `(N, FACES)` |  |
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `hands_empty_reason` | `True` | `analytics` | `object | str` | `()` |  |
| `hands_landmarks` | `False` | `analytics` | `float32` | `(N, HANDS, 21, 3)` |  |
| `hands_landmarks_raw` | `False` | `debug` | `float32` | `(N, HANDS, 21, 3)` |  |
| `hands_present` | `False` | `analytics` | `bool` | `(N, HANDS)` |  |
| `has_any_face` | `True` | `analytics` | `bool` | `()` |  |
| `has_any_hands` | `False` | `analytics` | `bool` | `()` |  |
| `has_any_pose` | `False` | `analytics` | `bool` | `()` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `model_name` | `False` | `debug` | `str | object` | `()` |  |
| `person_present` | `True` | `analytics` | `bool` | `(N)` |  |
| `pose_empty_reason` | `True` | `analytics` | `object | str` | `()` |  |
| `pose_landmarks` | `False` | `analytics` | `float32` | `(N, 33, 4)` |  |
| `pose_landmarks_raw` | `False` | `debug` | `float32` | `(N, 33, 4)` |  |
| `pose_present` | `False` | `analytics` | `bool` | `(N)` |  |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
| `total_frames` | `False` | `debug` | `int32 | int64 | object` | `()` |  |
| `version` | `False` | `debug` | `str | float32 | float64 | int32 | int64 | object` | `()` |  |
