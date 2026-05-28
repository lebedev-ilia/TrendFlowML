## mel_extractor — AUDIT V3 REPORT

### TL;DR

- Migrated `mel_extractor` to **Audit v3 contracts** with machine schema `mel_extractor_npz_v2` and `allow_extra_keys=false`.
- Implemented **canonical segment axis + strict alignment**: `segment_start_sec` / `segment_end_sec` / `segment_center_sec` + `segment_mask`. Failed segments are kept (masked) and filled with `NaN` in segment-aligned metrics.
- Implemented **segment-aligned sequences** for time-series mode (`--mel-enable-time-series`): `mel_mean_by_segment`, `mel_energy_by_segment`, `mel_centroid_mean_by_segment`, `mel_bandwidth_mean_by_segment`.
- Enforced **offline renderer** (no CDN / Plotly): vanilla canvas-only HTML.
- Updated defaults to audited preset: **basic + spectral enabled by default**, `statistics/stats_vector/time_series` remain opt-in.
- Disabled CUDA autocast path: **deterministic float32** computations.
- Expanded scalar outputs (model-facing tabular): added/standardized `mel_rolloff`, `mel_flatness`, `mel_stability` and unified centroid/bandwidth naming.

### Ownership / versions

- **Component**: `DataProcessor/AudioProcessor/src/extractors/mel_extractor`
- **Producer**: `mel_extractor`
- **Producer version**: `2.1.0`
- **Schema**: `DataProcessor/AudioProcessor/schemas/mel_extractor_npz_v2.json`
- **Schema mapping**: `DataProcessor/AudioProcessor/run_cli.py` now emits `mel_extractor_npz_v2`

### Inputs

- **Audio**: `audio/audio.wav` (used by `run()`)
- **Segments**: `audio/segments.json` family `mel` (used by `run_segments()`, Segmenter is the only sampling owner)

### Outputs (NPZ)

#### model_facing (tabular scalars: `feature_names`/`feature_values`)

- Extraction parameters: `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`, `power`
- Run identity scalars: `duration_sec`, `segments_count`, `device_used`
- Scalar metrics:
  - `mel_energy`
  - `mel_centroid_mean`, `mel_centroid_std`
  - `mel_bandwidth_mean`, `mel_bandwidth_std`
  - `mel_spectrogram_entropy`, `mel_spectrogram_contrast`
  - `mel_rolloff`, `mel_flatness`, `mel_stability`

#### analytics (arrays)

- Canonical segment axis (always present; may be empty in `run()` mode):
  - `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`

#### model_facing (arrays; gated)

Enabled by `--mel-enable-time-series` (Audit v3 meaning: **segment-aligned sequences**):

- `mel_mean_by_segment` (`N × n_mels`)
- `mel_energy_by_segment` (`N`)
- `mel_centroid_mean_by_segment` (`N`)
- `mel_bandwidth_mean_by_segment` (`N`)

#### analytics (arrays; gated)

Enabled by `--mel-enable-statistics`:

- `mel_mean`, `mel_std`, `mel_min`, `mel_max` (`n_mels`)

Enabled by `--mel-enable-stats-vector` (requires statistics):

- `mel_stats_vector`

#### debug-only (meta.extra)

- `mel_spectrogram_npy`: full mel spectrogram stored as `.npy` artifact (offline-friendly)
- `mel_series_npy`: optional legacy per-frame series for `run()` with time_series enabled

### Empty vs Error semantics

- **Empty**: upstream should mark `status="empty"` (e.g., no audio track). This component follows the pipeline’s status contract and does not silently fallback.
- **Error**: segment failures in `run_segments()` do **not** abort the run; they set `segment_mask=false` and continue. If **all** segments fail, the run fails with `mel_validation_failed`.

### Decisions (from Q&A)

- **Schema rollout**: `mel_extractor_npz_v2`
- **Time axis**: canonical `segment_*_sec` + `segment_mask`
- **Segment failures**: strict alignment with masking (no skipping)
- **Feature gating in NPZ**: omit keys when disabled
- **Model-facing scalars**: expanded set including rolloff/flatness/stability
- **Time series representation**: segment-aligned sequences (not per-frame arrays in NPZ)
- **Full mel storage**: debug-only `.npy` artifact pointers via `meta.extra`
- **GPU determinism**: disabled CUDA autocast (float32)
- **Renderer**: offline-only (vanilla canvas)

### Files changed

- `DataProcessor/AudioProcessor/schemas/mel_extractor_npz_v2.json` (new)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping updated)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (audited defaults + disable flags)
- `DataProcessor/AudioProcessor/src/extractors/mel_extractor/main.py` (strict alignment, new defaults, added metrics, no autocast)
- `DataProcessor/AudioProcessor/src/core/npz_savers/mel.py` (canonical segment keys, omission policy, meta.extra artifact pointers)
- `DataProcessor/AudioProcessor/src/extractors/mel_extractor/render.py` (offline renderer)
- `DataProcessor/AudioProcessor/src/extractors/mel_extractor/README.md` (docs updated)
- `DataProcessor/AudioProcessor/src/extractors/mel_extractor/SCHEMA.md` (new)
- `DataProcessor/AudioProcessor/docs/MAIN_INDEX.md` (docs index updated)

