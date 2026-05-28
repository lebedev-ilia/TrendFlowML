## `pitch_extractor` schema (Audit v3)

### schema_version

- **Machine schema**: `pitch_extractor_npz_v2` (`AudioProcessor/schemas/pitch_extractor_npz_v2.json`)
- **Producer**: `pitch_extractor`
- **allow_extra_keys**: `false`

### Required (always present in NPZ)

#### Tabular scalars (model_facing)

Stored as parallel arrays:

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Key scalars include:

- `sample_rate`, `hop_length`, `frame_length`, `fmin`, `fmax`, `duration`, `segments_count`, `backend`, `device_used`

#### Segment axis (analytics, canonical)

Canonical time axis for `run_segments()`:

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`

**Aggregation semantics** (no strict alignment):

- Arrays are length \(N = \) number of Segmenter windows.
- Failed/empty segments: `segment_mask[i]=false`.
- Metrics computed on aggregated f0 (valid segments only).

### Optional (feature-gated keys are omitted when disabled)

#### model_facing (basic_stats, default: enabled)

Enabled by `--pitch-enable-basic-stats` (Audit v3 default: enabled):

- `f0_mean`, `f0_std`, `f0_min`, `f0_max`, `f0_median`, `f0_method` (Hz)

#### model_facing (stability_metrics)

Enabled by `--pitch-enable-stability-metrics`:

- `pitch_stability`, `pitch_range`, `pitch_variation`

#### analytics (delta_features)

Enabled by `--pitch-enable-delta-features`:

- `f0_delta_mean`, `f0_delta_std`, `f0_delta_abs_mean`

#### analytics (basic_stats)

- `pitch_contour_smoothness`, `pitch_jump_count`, `pitch_skewness`, `pitch_kurtosis`, `pitch_octave_distribution`

#### analytics (method_stats)

Enabled by `--pitch-enable-method-stats`:

- `f0_mean_pyin`, `f0_std_pyin`, `f0_min_pyin`, `f0_max_pyin`, `f0_median_pyin`, `f0_count_pyin`
- `voiced_fraction_pyin`, `voiced_probability_mean_pyin`
- `f0_mean_yin`, `f0_std_yin`, … (YIN)
- `f0_mean_torchcrepe`, … (torchcrepe, when used)

### Provenance in `meta` (strings)

- `meta.backend`: `"classic"` \| `"torchcrepe"` — не дублируется как число в `feature_values` (избегаем NaN от `as_float`).
- `meta.f0_method`: выбранный метод или `aggregated` в режиме сегментов.

### Debug-only artifacts (meta.extra)

Offline-only pointer to large array stored as `.npy` artifact (not embedded in NPZ):

- `meta.extra.f0_series_npy` — full f0 series (debug only)

### Audit v4.2 observability (meta.extra)

- `meta.extra.stage_timings_ms`: dict, тайминги этапов (ms)
- `meta.extra.pitch_resource_profile`: dict|None, best-effort snapshot RSS/VMS/VRAM (gated by `AP_PITCH_RESOURCE_PROFILE=1`)

### Empty semantics

- `status="empty"` when all segments produce empty pitch
- `empty_reason="pitch_all_segments_empty"`
