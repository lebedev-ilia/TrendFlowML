## `onset_extractor` schema (Audit v3)

### schema_version

- **Machine schema**: `onset_extractor_npz_v2` (`AudioProcessor/schemas/onset_extractor_npz_v2.json`)
- **Producer**: `onset_extractor`
- **allow_extra_keys**: `false`

### Required (always present in NPZ)

#### Tabular scalars (model_facing)

Stored as parallel arrays:

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Key scalars include:

- `sample_rate`, `hop_length`, `duration`, `segments_count`
- **`backend`** (`"librosa"` \| `"essentia"`): только в **`meta`**, не в `feature_values`

#### Segment axis (analytics, canonical)

Canonical time axis for `run_segments()`:

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`

**Aggregation semantics** (no strict alignment):

- Arrays are length \(N = \) number of Segmenter windows.
- Onset extractor does not fail segments; `segment_mask` is all `true`.
- Metrics are computed on **aggregated** onset stream (merged from all segments).

### Optional (feature-gated keys are omitted when disabled)

#### model_facing (basic_features, default: enabled)

Enabled by `--onset-enable-basic-features` (Audit v3 default: enabled):

- `onset_count`, `onset_density_per_sec`, `insufficient_onsets`
- `onset_tempo_estimate`, `onset_regularity_score` (when rhythmic_metrics enabled)

#### analytics (interval_stats)

Enabled by `--onset-enable-interval-stats`:

- `avg_interval_sec`, `interval_std`, `interval_min`, `interval_max`, `interval_median`

#### analytics (rhythmic_metrics)

Enabled by `--onset-enable-rhythmic-metrics`:

- model_facing: `onset_tempo_estimate`, `onset_regularity_score`
- analytics: `onset_syncopation_score`, `onset_strength_mean`, `onset_strength_std`, `onset_density_variance`, `onset_tempo_consistency`

### Debug-only artifacts (meta.extra)

Offline-only pointer to large array stored as `.npy` artifact (not embedded in NPZ):

- `meta.extra.onset_times_npy` — full onset_times array (debug only)

### Audit v4.2 observability (meta.extra)

- `meta.extra.stage_timings_ms`: dict, тайминги этапов (ms)
- `meta.extra.onset_resource_profile`: dict|None, best-effort snapshot RSS/VMS (gated by `AP_ONSET_RESOURCE_PROFILE=1`)

### Analytics-only (not model_facing)

Syncopation, strength, density variance, tempo consistency remain for analytics/debug.
