## pitch_extractor — AUDIT V3 REPORT

### TL;DR

- Migrated `pitch_extractor` to **Audit v3 contracts** with machine schema `pitch_extractor_npz_v2`.
- Implemented **canonical segment axis**: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`.
- **Empty segments**: return `status="empty"`, `empty_reason="pitch_all_segments_empty"` instead of error when all segments have empty pitch.
- **Preset**: `enable_basic_stats=True` by default.
- **Tiers**: model_facing = f0_mean/std/min/max/median, pitch_stability, pitch_range; rest analytics.
- **Removed** `pitch_centroid` (duplicate of f0_mean).
- **f0_series** only in debug `.npy`, path in `meta.extra.f0_series_npy` / `meta.extra.f0_series_torchcrepe_npy`.
- **Renderer**: vanilla canvas, no CDN.
- **voice_quality_extractor**: loads f0 from `.npy` when path is present.

### Ownership / versions

- **Component**: `DataProcessor/AudioProcessor/src/extractors/pitch_extractor`
- **Producer**: `pitch_extractor`
- **Producer version**: `2.0.0`
- **Schema**: `DataProcessor/AudioProcessor/schemas/pitch_extractor_npz_v2.json`
- **Schema mapping**: `DataProcessor/AudioProcessor/run_cli.py` emits `pitch_extractor_npz_v2`

### Inputs

- **Audio**: `audio/audio.wav` (used by `run()`)
- **Segments**: `audio/segments.json` family `pitch` (used by `run_segments()`)

### Outputs (NPZ)

#### model_facing (when basic_stats enabled)

- `f0_mean`, `f0_std`, `f0_min`, `f0_max`, `f0_median`
- `pitch_stability`, `pitch_range`, `pitch_variation` (when stability_metrics enabled)

#### analytics (canonical segment axis)

- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask` — always present

#### analytics (when enabled)

- Delta features: `f0_delta_mean`, `f0_delta_std`, `f0_delta_abs_mean`
- Method stats: `f0_mean_pyin`, `f0_std_pyin`, etc.
- Additional: `pitch_contour_smoothness`, `pitch_jump_count`, `pitch_octave_distribution`, `pitch_skewness`, `pitch_kurtosis`

#### debug-only (meta.extra)

- `f0_series_npy`: path to `.npy` with f0 series (classic backend)
- `f0_series_torchcrepe_npy`: path to `.npy` with torchcrepe f0 series

### Decisions (from Q&A)

- **Q1**: Machine schema `pitch_extractor_npz_v2.json`, SCHEMA.md, `save_pitch_npz`
- **Q2**: Canonical segment axis: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- **Q3**: Empty segments: `status="empty"`, `empty_reason="pitch_all_segments_empty"` (no error)
- **Q4**: Preset: `enable_basic_stats=True` by default
- **Q5**: Tiers: model_facing = f0_mean/std/min/max/median, pitch_stability, pitch_range; rest analytics
- **Q6**: Removed `pitch_centroid` (duplicate of f0_mean)
- **Q7**: f0_series only in debug `.npy`, path in `meta.extra.f0_series_npy`
- **Q8**: Render: vanilla canvas, no CDN
- **Q9**: run_segments: keep current behavior for method_stats/time_series; document limitation
- **Q10**: Added `stage_timings_ms`
- **Q11**: Keep both `run()` and `run_segments()`
- **Q12**: Omit disabled keys in NPZ
- **Q13**: voice_quality_extractor: load f0 from `.npy` when path is present

### Files changed

- `DataProcessor/AudioProcessor/schemas/pitch_extractor_npz_v2.json` (new)
- `DataProcessor/AudioProcessor/src/extractors/pitch_extractor/SCHEMA.md` (new)
- `DataProcessor/AudioProcessor/src/core/npz_savers/pitch.py` (save_pitch_npz)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (--pitch-disable-basic-stats, default=True)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (pitch_enable_basic_stats=True default)
- `DataProcessor/AudioProcessor/src/core/processor_factory.py` (wire pitch args)
- `DataProcessor/AudioProcessor/src/extractors/pitch_extractor/main.py` (axis, preset, empty, remove centroid, timings, f0_series .npy)
- `DataProcessor/AudioProcessor/src/extractors/pitch_extractor/render.py` (vanilla canvas, load f0 from .npy)
- `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/main.py` (_get_f0_from_payload: load from .npy)
- `DataProcessor/AudioProcessor/docs/audit_v3/components/pitch_extractor_AUDIT_V3_REPORT.md` (new)
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/pitch_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
