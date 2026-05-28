# `video_pacing` — features (Audit v3)

Канон для NPZ / wide CSV / QA: **`docs/FEATURE_DESCRIPTION.md`**.

**Module**: `DataProcessor/VisualProcessor/modules/video_pacing/utils/video_pacing.py`  
**Schema**: `video_pacing_npz_v3`  
**producer_version**: `2.0.1`

## Model-facing time-series

- **`motion_norm_per_sec_mean (N,)`**: motion curve aligned to Segmenter sampling (from `core_optical_flow`).
- **`semantic_change_rate_per_sec (N,)`**: CLIP cosine distance between consecutive sampled frames, normalized by $dt$.
- **`color_change_rate_per_sec (N,)`**: cheap LAB mean delta, normalized by $dt$ (first value is 0).

## Model-facing scalars (`feature_names/feature_values`)

Fixed list in code: `_FEATURE_NAMES_V1`. Includes:

- **Shot stats**: counts and duration aggregates.
- **Cut density / histograms**: flattened `shot_length_histogram_5bins_*`, `cut_density_map_8bins_*`.
- **Motion / semantic / color pacing**: robust aggregates and burst proxies (where available).
- **Structural pacing**: intro/main/climax speeds and symmetry.

## Notes

- Strict axis alignment and no-fallback dependencies (cut_detection/core_clip/core_optical_flow).
- Some noisy blocks can be gated by config flags; in `feature_names/feature_values` missing features appear as NaN.


