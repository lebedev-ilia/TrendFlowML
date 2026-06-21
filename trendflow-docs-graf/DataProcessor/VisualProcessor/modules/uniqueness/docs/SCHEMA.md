# `uniqueness` — NPZ schema (`uniqueness_npz_v4`)

## Versioning

- **producer**: `uniqueness`
- **producer_version**: `1.0.2`
- **schema_version**: `uniqueness_npz_v4`

## Files

- **Module**: `DataProcessor/VisualProcessor/modules/uniqueness/uniqueness.py`
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/uniqueness_npz_v4.json`
- **Artifact filename**: `uniqueness.npz`

## Axis & sampling policy

- **Axis owner**: Segmenter provides `uniqueness.frame_indices` (union-domain indices).
- **Time axis**: `times_s = union_timestamps_sec[frame_indices]` (no-fallback).
- **Hard dependency**: `core_clip/embeddings.npz` must cover requested `frame_indices` (strict index mapping).
- **Sampling constraint**: $O(N^2)$ pairwise similarity, so module is **fail-fast** if `N > max_frames` (default 200).

## Output fields

### Model-facing (sequence)

- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `max_sim_to_other (N,) float32`
- `cos_dist_next (N-1,) float32`

### Model-facing (tabular scalars)

- `feature_names (F,) object`
- `feature_values (F,) float32` — fixed order (see `_FEATURE_NAMES_V1` in code). Includes threshold used, repetition/diversity aggregates, and `repeat_threshold_is_otsu` flag.

### Debug

- `meta () object` with `meta.ui_payload` (top repeats) and `meta.stage_timings_ms`.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
