## `mfcc_extractor` schema (Audit v3)

### schema_version

- **Machine schema**: `mfcc_extractor_npz_v2` (`AudioProcessor/schemas/mfcc_extractor_npz_v2.json`)
- **Producer**: `mfcc_extractor`
- **allow_extra_keys**: `false`

### Required (always present in NPZ)

#### Tabular scalars (model_facing)

Stored as parallel arrays:

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Key scalars include:

- `sample_rate`, `n_mfcc`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`
- `duration_sec`, `segments_count` (если в payload)
- **`device_used`**: строка — только в **`meta`**, не в `feature_values`
- `mfcc_energy`, `mfcc_centroid`, `mfcc_bandwidth`, `mfcc_stability`

#### Segment axis (analytics, strict alignment)

Canonical time axis for `run_segments()`:

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`

**Strict alignment semantics**:

- Arrays are length \(N = \) number of Segmenter windows.
- Failed segments are **not skipped**.
- For failed segments: `segment_mask[i]=false`; corresponding metric arrays (if present) contain `NaN` at index `i`.

### Optional (feature-gated keys are omitted when disabled)

#### Basic statistics (model_facing)

Enabled by `--mfcc-enable-basic-features` (Audit v3 default: enabled):

- `mfcc_mean`, `mfcc_std`, `mfcc_min`, `mfcc_max`: `float32[M]`, where \(M = n_mfcc\)

#### Deltas (model_facing)

Enabled by `--mfcc-enable-deltas`:

- `delta_mean`, `delta_std`, `delta_delta_mean`, `delta_delta_std`: `float32[M]`

#### Segment-aligned sequences (model_facing)

Enabled by `--mfcc-enable-time-series` (Audit v3 meaning: **segment-aligned**, not per-frame):

- `mfcc_mean_by_segment`: `float32[N, M]`
- `mfcc_energy_by_segment`: `float32[N]`
- `delta_mean_by_segment`: `float32[N, M]` (when deltas enabled)

### Debug-only artifacts (meta.extra)

Offline-only pointers to large arrays stored as `.npy` artifacts (not embedded in NPZ arrays):

- `meta.extra.mfcc_npy` (full MFCC from `run()` or concatenated per-frame from `run_segments()`, if produced)

### Audit v4.2 observability (meta.extra)

- `meta.extra.stage_timings_ms`: dict, тайминги этапов (ms)
- `meta.extra.mfcc_resource_profile`: dict|None, best-effort snapshot RSS/VMS/VRAM (gated by `AP_MFCC_RESOURCE_PROFILE=1`)

### Analytics-only (not model_facing)

Skewness, kurtosis, correlation remain for analytics/debug; not stored in NPZ by default.
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
