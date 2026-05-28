# Schema: `cut_detection_model_facing_npz_v1`

- **producer**: `cut_detection`
- **artifact_kind**: `npz`
- **allow_extra_keys**: `True`
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
- `cut_detection_config`
- `flow_source`
- `flow_mag_units`
- `thresholds`
- `event_type_map`
- `event_contrib_sources`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `deep_cosine_dist` | `True` | `model_facing` | `float32` | `(N-1)` |  |
| `deep_valid_mask` | `True` | `model_facing` | `bool` | `(N-1)` |  |
| `event_contrib_mask` | `True` | `model_facing` | `bool` | `(E, 4)` |  |
| `event_end_time_s` | `True` | `model_facing` | `float32` | `(E)` |  |
| `event_pair_index` | `True` | `model_facing` | `int32` | `(E)` |  |
| `event_start_time_s` | `True` | `model_facing` | `float32` | `(E)` |  |
| `event_strength` | `True` | `model_facing` | `float32` | `(E)` |  |
| `event_times_s` | `True` | `model_facing` | `float32` | `(E)` |  |
| `event_type_id` | `True` | `model_facing` | `int16` | `(E)` |  |
| `flow_mag` | `True` | `model_facing` | `float32` | `(N-1)` |  |
| `flow_valid_mask` | `True` | `model_facing` | `bool` | `(N-1)` |  |
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `hard_score` | `True` | `model_facing` | `float32` | `(N-1)` |  |
| `hist_diff_l1` | `True` | `model_facing` | `float32` | `(N-1)` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `motion_cam_valid_mask` | `False` | `model_facing` | `bool` | `(N-1)` |  |
| `motion_camera_motion_flag` | `False` | `model_facing` | `bool` | `(N-1)` |  |
| `motion_dir_consistency` | `False` | `model_facing` | `float32` | `(N-1)` |  |
| `motion_dir_valid_mask` | `False` | `model_facing` | `bool` | `(N-1)` |  |
| `motion_flow_mag` | `False` | `model_facing` | `float32` | `(N-1)` |  |
| `motion_mag_variance` | `False` | `model_facing` | `float32` | `(N-1)` |  |
| `motion_var_valid_mask` | `False` | `model_facing` | `bool` | `(N-1)` |  |
| `pair_dt_s` | `True` | `model_facing` | `float32` | `(N-1)` |  |
| `pair_times_s` | `True` | `model_facing` | `float32` | `(N-1)` |  |
| `soft_flow_mag` | `False` | `model_facing` | `float32` | `(N-1)` |  |
| `soft_flow_valid_mask` | `False` | `model_facing` | `bool` | `(N-1)` |  |
| `soft_hist_diff_l1` | `False` | `model_facing` | `float32` | `(N-1)` |  |
| `soft_hsv_v` | `False` | `model_facing` | `float32` | `(N)` |  |
| `soft_lab_l` | `False` | `model_facing` | `float32` | `(N)` |  |
| `ssim_drop` | `True` | `model_facing` | `float32` | `(N-1)` |  |
| `ssim_valid_mask` | `True` | `model_facing` | `bool` | `(N-1)` |  |
| `threshold_deep` | `True` | `debug` | `float32` | `(N-1)` |  |
| `threshold_flow` | `True` | `debug` | `float32` | `(N-1)` |  |
| `threshold_hist` | `True` | `debug` | `float32` | `(N-1)` |  |
| `threshold_ssim` | `True` | `debug` | `float32` | `(N-1)` |  |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
| `union_timestamps_sec` | `True` | `model_facing` | `float32` | `(N)` |  |
