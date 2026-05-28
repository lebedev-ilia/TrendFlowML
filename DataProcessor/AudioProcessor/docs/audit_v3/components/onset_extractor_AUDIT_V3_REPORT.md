## onset_extractor — AUDIT V3 REPORT

### TL;DR

- Migrated `onset_extractor` to **Audit v3 contracts** with machine schema `onset_extractor_npz_v2` and `allow_extra_keys=false`.
- Implemented **canonical segment axis** (aggregation semantics): `segment_start_sec` / `segment_end_sec` / `segment_center_sec` + `segment_mask`. No strict alignment — metrics computed on aggregated onset stream.
- **onset_times** not stored in NPZ; only in debug `.npy` artifact (`meta.extra.onset_times_npy`).
- Removed **onset_clustering_score** as redundant with `onset_regularity_score`.
- Fixed **units="frames"** bug: convert frames to time via `librosa.frames_to_time()`.
- Updated preset: **basic_features=True** by default.
- Enforced **offline renderer** (vanilla canvas, no CDN).
- Omit disabled keys in NPZ; model_facing: `onset_count`, `onset_density_per_sec`, `onset_tempo_estimate`, `onset_regularity_score`.

### Ownership / versions

- **Component**: `DataProcessor/AudioProcessor/src/extractors/onset_extractor`
- **Producer**: `onset_extractor`
- **Producer version**: `2.0.0`
- **Schema**: `DataProcessor/AudioProcessor/schemas/onset_extractor_npz_v2.json`
- **Schema mapping**: `DataProcessor/AudioProcessor/run_cli.py` emits `onset_extractor_npz_v2`

### Inputs

- **Audio**: `audio/audio.wav` (used by `run()`)
- **Segments**: `audio/segments.json` family `onset` (used by `run_segments()`)

### Outputs (NPZ)

#### model_facing (tabular scalars: `feature_names`/`feature_values`)

- Extraction parameters: `sample_rate`, `hop_length`, `duration`, `segments_count`, `backend`
- model_facing scalars (when enabled): `onset_count`, `onset_density_per_sec`, `insufficient_onsets`, `onset_tempo_estimate`, `onset_regularity_score`

#### analytics (canonical segment axis)

- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask` — always present (may be empty for `run()`)

#### analytics (interval_stats, when enabled)

- `avg_interval_sec`, `interval_std`, `interval_min`, `interval_max`, `interval_median`

#### analytics (rhythmic_metrics, when enabled)

- `onset_syncopation_score`, `onset_strength_mean`, `onset_strength_std`, `onset_density_variance`, `onset_tempo_consistency`

#### debug-only (meta.extra)

- `onset_times_npy`: path to `.npy` artifact with full onset_times array

### Decisions (from Q&A)

- **Schema rollout**: `onset_extractor_npz_v2`
- **Canonical axis**: added; aggregation semantics preserved (no strict alignment)
- **onset_times storage**: only in debug `.npy`; NPZ only scalar aggregates
- **onset_clustering_score**: removed (redundant)
- **units="frames"**: fixed via `librosa.frames_to_time()`
- **Preset**: `basic_features=True` by default
- **Renderer**: offline-only (vanilla canvas)
- **Omit disabled keys**: yes

### Files changed

- `DataProcessor/AudioProcessor/schemas/onset_extractor_npz_v2.json` (new)
- `DataProcessor/AudioProcessor/src/extractors/onset_extractor/SCHEMA.md` (new)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (audited defaults)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (default basic_features=True)
- `DataProcessor/AudioProcessor/src/core/processor_factory.py` (wire args)
- `DataProcessor/AudioProcessor/src/extractors/onset_extractor/main.py` (canonical axis, preset, units fix, remove clustering, onset_times .npy)
- `DataProcessor/AudioProcessor/src/core/npz_savers/onset.py` (canonical keys, omit disabled, onset_times debug-only)
- `DataProcessor/AudioProcessor/src/extractors/onset_extractor/render.py` (offline renderer, load onset_times from .npy)
- `DataProcessor/AudioProcessor/docs/audit_v3/components/onset_extractor_AUDIT_V3_REPORT.md` (new)
