## `source_separation_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `source_separation_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `source_separation_extractor_npz_v2`
- **NPZ** is the source-of-truth. Render is dev-only.

### Inputs

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json` (schema `audio_segments_v1`)
  - **required family**: `source_separation`
  - no-fallback: missing/empty family → error

### Empty / Error semantics

- **Audio absent (no audio stream)**: AudioProcessor writes `status="empty"` (upstream) — extractor is not run.
- **Audio too short (<5s)**: `status="empty"`, `empty_reason="audio_too_short"`.
- **Truly silent audio** (all windows silent): `status="empty"`, `empty_reason="audio_silent"`.
- **Model invalid output (NaN/inf/negative energies)**: `status="error"` (fail-fast).

### Canonical segment axis (analytics)

Always present for `run_segments()`:

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`
  - `false` for silent/zero-energy windows (masked out of aggregates)

### Sources

Contract freezes canonical order:

- `source_order`: object[4] = `["vocals", "drums", "bass", "other"]`

### Outputs

#### 1) Model-facing (tabular) — frozen subset (`feature_names` / `feature_values`)

Order is fixed within `schema_version`:

- `share_vocals_mean`
- `share_drums_mean`
- `share_bass_mean`
- `share_other_mean`
- `dominant_source_id`
- `dominant_source_share`
- `source_balance_score`
- `source_transitions_count`
- `source_stability_score`
- `segments_count`
- `sample_rate`

#### 2) Analytics vectors

- `share_mean`: `float32[4]` (mean over unmasked segments)
- `share_std`: `float32[4]` (optional; emitted only if enabled)
- Structured per-source stats (no dicts):
  - `source_distribution_ratio`: `float32[4]`
  - `source_segments_count`: `int32[4]`
  - `source_duration_sec`: `float32[4]`

#### 3) Token-ready sequences (optional)

Feature-gated:

- `share_sequence`: `float32[N,4]` (emitted only if enabled)
- `energy_sequence`: `float32[N,4]` (emitted only if enabled)

Masked rows are skipped by `segment_mask`.

#### 4) Additional analytics scalars (optional)

Emitted only when computed (typically requires `share_sequence` and/or `quality_metrics`):

- `source_entropy_mean`, `source_entropy_std`, `energy_balance_mean`
- `vocals_presence_ratio`, `drums_flux`, `bass_floor_p20`
- per-source deltas/stability/dominance ratios (see machine schema for exact keys)
- quality scalars: `quality_*`

#### 5) Meta (debug)

- `meta`: baseline meta + **`device_used`** (str), **`model_name`**, **`weights_digest`**, `features_enabled`, `source_separation_contract_version` — строки и идентификаторы модели **только здесь**, не в `feature_values`
- observability (audit v4.2, optional): `stage_timings_ms`, `source_separation_resource_profile`

### Feature flags (effective)

Audit v3 preset:

- default-on: baseline model-facing + `share_mean` + canonical axis + mask + structured per-source stats
- opt-in: `share_sequence`, `energy_sequence`, `share_std`, `quality_metrics`

