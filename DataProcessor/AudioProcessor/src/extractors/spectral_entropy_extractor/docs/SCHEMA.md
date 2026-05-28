# `spectral_entropy_extractor` — SCHEMA (Audit v3)

## Status

- **Producer**: `spectral_entropy_extractor`
- **Schema version**: `spectral_entropy_extractor_npz_v2`
- **Audit v3**: NPZ contract as below; **Audit v4** L1 на reference run A — см. `DataProcessor/docs/audit_v4/components/audio_processor/spectral_entropy_extractor_audit_v4.md` + [`RUN_LOG.md`](../../../../../docs/audit_v4/RUN_LOG.md)

## Purpose

Extract spectral entropy (and optional flatness/spread) over Segmenter windows and publish:

- minimal stable tabular scalars for models (`feature_names/feature_values`)
- **per-segment** arrays aligned to Segmenter sampling axis (`segment_*_sec`, `segment_mask`)

Audit v3 decisions for this component:

- **Sampling family**: shared-family `spectral` (Segmenter-owned)
- **Time series**: no concatenated frame series; use **per-segment** aggregates
- **No payload**: NPZ contains only strict keys + meta

## Inputs

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json` (schema `audio_segments_v1`):
  - required: `families.spectral.segments[]`

## Outputs (NPZ = source-of-truth)

### 1) Model-facing (tabular, frozen subset)

Stored as:

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Frozen minimal subset (recommended):

- `spectral_entropy_mean`
- `spectral_entropy_std`

Missing values use **NaN**.

### 2) Canonical axis (Segmenter windows)

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`

### 3) Per-segment aggregates

Required:

- `entropy_mean_by_segment`: `float32[N]`
- `entropy_std_by_segment`: `float32[N]`

Optional (only if enabled/produced):

- `entropy_min_by_segment`: `float32[N]`
- `entropy_max_by_segment`: `float32[N]`
- `flatness_mean_by_segment`: `float32[N]`
- `flatness_std_by_segment`: `float32[N]`
- `spread_mean_by_segment`: `float32[N]`
- `spread_std_by_segment`: `float32[N]`

### 4) Meta

- `meta`: object dict (baseline meta contract)

Important meta extras:

- **`device_used`** (str) — только в **`meta`** (baseline), не в `feature_values`
- `spectral_entropy_contract_version`
- `features_enabled`
- parameter echo: `sample_rate`, `n_fft`, `hop_length`, `use_mel`, `n_mels`, `smoothing_window`, `average_channels`, `duration`, `segments_count`
- observability (audit v4.2, optional): `stage_timings_ms`, `spectral_entropy_resource_profile`

## Empty / Error semantics

- **Short audio** (< 1s): `status="empty"`, `empty_reason="audio_too_short"`
- **All segments failed**: `status="empty"`, `empty_reason="spectral_entropy_all_segments_failed"` with full axis arrays and `segment_mask=false`
- Invalid input / missing Segmenter family: `status="error"` (no-fallback)

