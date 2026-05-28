# Schema: `core_depth_midas_npz_v3`

- **producer**: `core_depth_midas`
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

### Optional meta keys

- `model_name`
- `total_frames`
- `out_width`
- `out_height`
- `batch_size`
- `runtime`
- `device`
- `triton_preprocess_preset`
- `triton_model_spec`
- `triton_model_name`

## Fields

| key | required | tier | dtype | shape | description |
|---|---:|---|---|---|---|
| `depth_complexity_score` | `True` | `analytics` | `float32` | `(N)` |  |
| `depth_maps` | `True` | `debug` | `float32` | `(N, H, W)` |  |
| `depth_maps_norm` | `True` | `model_facing` | `float32` | `(N, H, W)` |  |
| `depth_mean` | `True` | `analytics` | `float32` | `(N)` |  |
| `depth_p05` | `True` | `analytics` | `float32` | `(N)` |  |
| `depth_p95` | `True` | `analytics` | `float32` | `(N)` |  |
| `depth_range_robust` | `True` | `analytics` | `float32` | `(N)` |  |
| `depth_std` | `True` | `analytics` | `float32` | `(N)` |  |
| `foreground_background_separation_proxy` | `True` | `analytics` | `float32` | `(N)` |  |
| `frame_indices` | `True` | `model_facing` | `int32` | `(N)` |  |
| `meta` | `True` | `debug` | `object` | `()` |  |
| `preview_depth_maps` | `True` | `debug` | `float32` | `(K, H, W)` |  |
| `preview_depth_maps_norm` | `True` | `analytics` | `float32` | `(K, H, W)` |  |
| `preview_frame_indices` | `True` | `analytics` | `int32` | `(K)` |  |
| `preview_times_s` | `True` | `analytics` | `float32` | `(K)` |  |
| `times_s` | `True` | `model_facing` | `float32` | `(N)` |  |
