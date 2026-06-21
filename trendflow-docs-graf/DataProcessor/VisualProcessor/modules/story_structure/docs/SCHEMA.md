# `story_structure` — NPZ schema (`story_structure_npz_v3`)

## Versioning

- **producer**: `story_structure`
- **producer_version**: `3.0.2`
- **schema_version**: `story_structure_npz_v3`

## Files

- **Module**: `DataProcessor/VisualProcessor/modules/story_structure/story_structure.py`
- **Machine-readable schema**: `DataProcessor/VisualProcessor/schemas/story_structure_npz_v3.json`
- **Artifact filename**: `story_structure.npz`

## Axis & sampling policy

- **Axis owner**: Segmenter provides `story_structure.frame_indices` (union-domain indices).
- **Time axis source-of-truth**: `metadata.json -> union_timestamps_sec` (no-fallback).
- **Dependency alignment**: all dependencies are aligned to requested `frame_indices` via strict index mapping. Missing coverage → `RuntimeError`.
- **Sampling constraints (fail-fast)**:
  - `min_frames` (default 30)
  - `max_frames` (default 200)

## Output fields (high-level)

### Model-facing (sequence)

- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `story_energy_curve (N,) float32` — combined energy proxy (z-score)
- `frame_feature_present_ratio (N,) float32` — доля finite среди model-facing float кривых (energy/motion/emb_rate/topic_shift)
- `motion_norm_per_sec_mean (N,) float32` — motion curve (per-second mean magnitude from `core_optical_flow`)
- `embedding_change_rate_per_sec (N,) float32` — CLIP embedding change rate (/s)
- `any_face_present (N,) bool`
- `topic_shift_curve (N,) float32` — text-derived curve (NaN if text missing)
- `topic_shift_curve_present () bool`

### Model-facing (tabular)

- `feature_names (F,) object`
- `feature_values (F,) float32` — fixed order (see code: `_FEATURE_NAMES_V1`)

### Analytics / debug

- peaks arrays (`story_energy_peaks_*`, `topic_shift_peaks_idx`)
- `story_energy_curve_downsampled_128 (128,) float32`
- `meta () object` with:
  - `meta.stage_timings_ms` (reproducible timings)
  - `meta.ui_payload` (small UI hints, no heavy arrays)
  - config highlights (`min_frames`, `max_frames`, `energy_smoothing_sigma`, `text_mode`, …)
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [FEATURES_DESCRIPTION](FEATURES_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FINAL_TEST_REPORT](FINAL_TEST_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
