# Schema: `core_optical_flow_npz_v3`

- **producer**: `core_optical_flow`
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
- `backend_proxy_version`
- `preview_k`
- `preview_map_size`

### Optional meta keys

- `triton_model_spec`
- `triton_model_name`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `dt_seconds` | `True` | `analytics` | `float32` | `(N)` |  |
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `motion_norm_per_sec_mean` | `True` | `model_facing` | `float32` | `(N)` |  |
| `flow_mag_std_per_sec_norm` | `True` | `model_facing` | `float32` | `(N)` | std magnitude / dt / max(H,W) |
| `flow_mag_p95_per_sec_norm` | `True` | `model_facing` | `float32` | `(N)` | p95 magnitude / dt / max(H,W) |
| `flow_dx_mean_per_sec_norm` | `True` | `model_facing` | `float32` | `(N)` | mean dx / dt / max(H,W) |
| `flow_dy_mean_per_sec_norm` | `True` | `model_facing` | `float32` | `(N)` | mean dy / dt / max(H,W) |
| `flow_dir_sin_mean` | `True` | `model_facing` | `float32` | `(N)` | magnitude-weighted sin(direction) |
| `flow_dir_cos_mean` | `True` | `model_facing` | `float32` | `(N)` | magnitude-weighted cos(direction) |
| `flow_dir_dispersion` | `True` | `model_facing` | `float32` | `(N)` | 1 - resultant length |
| `flow_div_abs_mean` | `True` | `analytics` | `float32` | `(N)` | mean abs divergence proxy |
| `flow_consistency` | `True` | `analytics` | `float32` | `(N)` | 1/(1+div_abs_mean) |
| `cam_affine_scale` | `True` | `model_facing` | `float32` | `(N)` | affine scale (background) |
| `cam_affine_rotation` | `True` | `model_facing` | `float32` | `(N)` | affine rotation (rad) |
| `cam_tx_per_sec_norm` | `True` | `model_facing` | `float32` | `(N)` | affine tx / dt / max(H,W) |
| `cam_ty_per_sec_norm` | `True` | `model_facing` | `float32` | `(N)` | affine ty / dt / max(H,W) |
| `cam_shake_std_norm` | `True` | `model_facing` | `float32` | `(N)` | std(bg magnitude)/dt/max(H,W) |
| `bg_ratio` | `True` | `analytics` | `float32` | `(N)` | fraction of low-motion pixels |
| `preview_cur_frame_indices` | `True` | `analytics` | `int32` | `(K)` |  |
| `preview_cur_times_s` | `True` | `analytics` | `float32` | `(K)` |  |
| `preview_flow_mag_map_norm` | `True` | `analytics` | `float32` | `(K, H_preview, W_preview)` |  |
| `preview_pair_pos` | `True` | `analytics` | `int32` | `(K)` |  |
| `preview_prev_frame_indices` | `True` | `analytics` | `int32` | `(K)` |  |
| `preview_prev_times_s` | `True` | `analytics` | `float32` | `(K)` |  |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
