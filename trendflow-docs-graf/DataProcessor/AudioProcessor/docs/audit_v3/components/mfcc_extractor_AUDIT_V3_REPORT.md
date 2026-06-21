## mfcc_extractor — AUDIT V3 REPORT

### TL;DR

- Migrated `mfcc_extractor` to **Audit v3 contracts** with machine schema `mfcc_extractor_npz_v2` and `allow_extra_keys=false`.
- Implemented **canonical segment axis + strict alignment**: `segment_start_sec` / `segment_end_sec` / `segment_center_sec` + `segment_mask`. Failed segments are kept (masked) and filled with `NaN` in segment-aligned metrics.
- Implemented **segment-aligned sequences** for time-series mode (`--mfcc-enable-time-series`): `mfcc_mean_by_segment`, `mfcc_energy_by_segment`, `delta_mean_by_segment`.
- Enforced **offline renderer** (no CDN / Plotly): vanilla canvas-only HTML.
- Updated defaults to audited preset: **basic_features=True**, `deltas`/`time_series`/`normalization` remain opt-in.
- Fixed **canonical keys** in NPZ saver: `mfcc_mean`, `mfcc_std`, `mfcc_min`, `mfcc_max` (was incorrectly `mean`, `std`, `min`, `max`).
- Full MFCC array **not stored in NPZ**; only in debug-only `.npy` artifact (`meta.extra.mfcc_npy`).
- Explicit **float32** path, no autocast.
- Skewness/kurtosis/correlation remain **analytics-only** (not model_facing in NPZ).

### Ownership / versions

- **Component**: `DataProcessor/AudioProcessor/src/extractors/mfcc_extractor`
- **Producer**: `mfcc_extractor`
- **Producer version**: `2.1.0`
- **Schema**: `DataProcessor/AudioProcessor/schemas/mfcc_extractor_npz_v2.json`
- **Schema mapping**: `DataProcessor/AudioProcessor/run_cli.py` now emits `mfcc_extractor_npz_v2`

### Inputs

- **Audio**: `audio/audio.wav` (used by `run()`)
- **Segments**: `audio/segments.json` family `mfcc` (used by `run_segments()`)

### Outputs (NPZ)

#### model_facing (tabular scalars: `feature_names`/`feature_values`)

- Extraction parameters: `sample_rate`, `n_mfcc`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`
- Run identity scalars: `duration_sec`, `segments_count`, `device_used`
- Scalar metrics: `mfcc_energy`, `mfcc_centroid`, `mfcc_bandwidth`, `mfcc_stability`

#### analytics (arrays)

- Canonical segment axis (always present; may be empty in `run()` mode):
  - `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`

#### model_facing (arrays; gated)

Enabled by `--mfcc-enable-basic-features` (Audit v3 default: enabled):

- `mfcc_mean`, `mfcc_std`, `mfcc_min`, `mfcc_max` (`n_mfcc`)

Enabled by `--mfcc-enable-deltas`:

- `delta_mean`, `delta_std`, `delta_delta_mean`, `delta_delta_std` (`n_mfcc`)

Enabled by `--mfcc-enable-time-series` (segment-aligned sequences):

- `mfcc_mean_by_segment` (`N × n_mfcc`)
- `mfcc_energy_by_segment` (`N`)
- `delta_mean_by_segment` (`N × n_mfcc`, when deltas enabled)

#### debug-only (meta.extra)

- `mfcc_npy`: full MFCC stored as `.npy` artifact (offline-friendly)

### Empty vs Error semantics

- **Empty**: upstream should mark `status="empty"`. This component follows the pipeline's status contract.
- **Error**: segment failures in `run_segments()` do **not** abort the run; they set `segment_mask=false` and continue. If **all** segments fail, the run fails with `mfcc_validation_failed`.

### Decisions (from Q&A)

- **Schema rollout**: `mfcc_extractor_npz_v2`
- **Time axis**: canonical `segment_*_sec` + `segment_mask`
- **Segment failures**: strict alignment with masking (no skipping)
- **Feature gating in NPZ**: omit keys when disabled
- **Time series representation**: segment-aligned sequences (not per-frame arrays in NPZ)
- **Full MFCC storage**: debug-only `.npy` artifact pointers via `meta.extra`
- **Canonical keys**: `mfcc_mean`, `mfcc_std`, `mfcc_min`, `mfcc_max` (fixed saver bug)
- **Audit preset**: `basic_features=True`, `deltas=False`, `time_series=False`, `normalization=False`
- **Renderer**: offline-only (vanilla canvas)
- **Skewness/kurtosis/correlation**: analytics-only, not in NPZ

### Files changed

- `DataProcessor/AudioProcessor/schemas/mfcc_extractor_npz_v2.json` (new)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (audited defaults)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (default basic_features=True)
- `DataProcessor/AudioProcessor/src/extractors/mfcc_extractor/main.py` (strict alignment, preset, float32, no mfcc_features in payload)
- `DataProcessor/AudioProcessor/src/core/npz_savers/mfcc.py` (canonical keys, omit disabled, segment axis, meta.extra)
- `DataProcessor/AudioProcessor/src/extractors/mfcc_extractor/render.py` (offline renderer)
- `DataProcessor/AudioProcessor/src/extractors/mfcc_extractor/SCHEMA.md` (new)
- `DataProcessor/AudioProcessor/docs/MAIN_INDEX.md` (updated)
- `DataProcessor/AudioProcessor/docs/audit_v3/components/mfcc_extractor_AUDIT_V3_REPORT.md` (new)
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/mfcc_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
