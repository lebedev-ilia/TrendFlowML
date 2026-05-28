# `video_pacing` — NPZ schema (`video_pacing_npz_v3`)

## Versioning

- **producer**: `video_pacing`
- **producer_version**: `2.0.1`
- **schema_version**: `video_pacing_npz_v3`

## Files

- **Module**: `DataProcessor/VisualProcessor/modules/video_pacing/video_pacing.py`
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/video_pacing_npz_v3.json`
- **Artifact filename**: `video_pacing_features.npz`

## Axis & sampling policy

- **Axis owner**: Segmenter provides `video_pacing.frame_indices` (union-domain).
- **Time axis source-of-truth**: `metadata.json -> union_timestamps_sec` (no-fallback).
- **Hard deps (no-fallback)**:
  - `cut_detection` (shot boundaries)
  - `core_optical_flow` (motion curve)
  - `core_clip` (semantic change)
- **Min frames**: `min_frames` (default 30), fail-fast.

## Output fields (high-level)

### Model-facing time-series (aligned to `frame_indices`)

- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `shot_boundary_frame_indices (S,) int32` (union-domain)
- `motion_norm_per_sec_mean (N,) float32`
- `semantic_change_rate_per_sec (N,) float32`
- `color_change_rate_per_sec (N,) float32`

### Model-facing tabular scalars

- `feature_names (F,) object`
- `feature_values (F,) float32` — fixed order list in code: `_FEATURE_NAMES_V1` (includes flattened histogram bins).

### Debug

- `meta.ui_payload`: small UI hints (curve pointers + shot boundary markers)
- `meta.stage_timings_ms`


