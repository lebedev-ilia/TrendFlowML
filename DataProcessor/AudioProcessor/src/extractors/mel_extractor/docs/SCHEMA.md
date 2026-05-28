## `mel_extractor` schema (Audit v3)

### schema_version

- **Machine schema**: `mel_extractor_npz_v2` (`AudioProcessor/schemas/mel_extractor_npz_v2.json`)
- **Producer**: `mel_extractor`
- **allow_extra_keys**: `false`

### Required (always present in NPZ)

#### Tabular scalars (model_facing)

Stored as parallel arrays:

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Key scalars include:

- `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`, `power`
- `duration_sec`, `segments_count` (если в payload)
- **`device_used`**: строка — только в **`meta`**, не в `feature_values`
- `mel_energy`
- `mel_centroid_mean`, `mel_centroid_std`
- `mel_bandwidth_mean`, `mel_bandwidth_std`
- `mel_spectrogram_entropy`, `mel_spectrogram_contrast`
- `mel_rolloff`, `mel_flatness`, `mel_stability`

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

#### Statistics (analytics)

Enabled by `--mel-enable-statistics`:

- `mel_mean`, `mel_std`, `mel_min`, `mel_max`: `float32[M]`, where \(M = n_mels\)

#### Stats vector (model_facing)

Enabled by `--mel-enable-stats-vector` (requires statistics):

- `mel_stats_vector`: `float32[V]` (implementation-defined length)

#### Segment-aligned sequences (model_facing)

Enabled by `--mel-enable-time-series` (Audit v3 meaning: **segment-aligned**, not per-frame):

- `mel_mean_by_segment`: `float32[N, M]`
- `mel_energy_by_segment`: `float32[N]`
- `mel_centroid_mean_by_segment`: `float32[N]`
- `mel_bandwidth_mean_by_segment`: `float32[N]`

### Debug-only artifacts (meta.extra)

Offline-only pointers to large arrays stored as `.npy` artifacts (not embedded in NPZ arrays):

- `meta.extra.mel_spectrogram_npy` (full mel spectrogram from `run()`, if produced)
- `meta.extra.mel_series_npy` (optional, legacy per-frame series for `run()` with time_series enabled)

### Audit v4.2 observability (meta.extra)

- `meta.extra.stage_timings_ms`: dict, тайминги этапов (ms)
- `meta.extra.mel_resource_profile`: dict|None, best-effort snapshot RSS/VMS/VRAM (gated by `AP_MEL_RESOURCE_PROFILE=1`)

