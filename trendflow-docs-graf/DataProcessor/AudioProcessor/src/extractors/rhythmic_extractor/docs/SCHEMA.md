## `rhythmic_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `rhythmic_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `rhythmic_extractor_npz_v2`
- **source-of-truth**: NPZ (`feature_names/feature_values` + arrays), render = dev-only

### Inputs (contract)

- **Segmenter output**:
  - `frames_dir/audio/audio.wav`
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `tempo` (shared sampling requirement; rhythmic uses tempo windows)
    - Migration note: legacy `families.rhythmic` may be accepted temporarily; recorded in `meta.sampling_family_used`.

### Outputs (NPZ keys)

#### 1) Tabular (model_facing) — `feature_names` / `feature_values`

Frozen subset (order fixed within `schema_version`):

- `rhythm_tempo_bpm` — tempo from backend beat tracker (BPM); `NaN` if no beats
- `rhythm_beats_count` — beats count
- `rhythm_beat_density` — beats/sec
- `rhythm_regularity` — regularity proxy; `NaN` if no beats
- `rhythm_tempo_variation` — CV of IBI; `NaN` if no beats
- `rhythm_beat_consistency` — \(1/(1+tempo_variation)\); `NaN` if no beats
- `duration_sec` — при **`run_segments()`** это **сумма длительностей сегментных окон** (согласовано с \(\sum_i (segment\_end[i]-segment\_start[i])\)), не обязательно полная длина исходного клипа
- `sample_rate`
- `segments_count`

Missing values are encoded as `NaN` (not zeros).

#### 2) Canonical segment axis (analytics)

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]` (false for failed segments)

#### 3) Beat events (analytics / token-ready)

Optional (when `enable_beat_times`):

- `beat_times_sec`: `float32[M]`
- `beat_segment_index`: `int32[M]` (segment index for each beat, for `run_segments()`)

For large `M`, beats may be stored as `.npy` sub-artifacts; paths stored in meta:
- `beat_times_sec_npy`
- `beat_segment_index_npy`

#### 4) Extra analytics scalars (optional NPZ scalar keys)

If enabled/computed, NPZ may also contain scalar float32 keys:
- interval stats: `rhythm_*_period_sec`
- tempo stats: `rhythm_median_bpm`, `rhythm_ibi_tempo_bpm`, `rhythm_tempo_{mean,std,min,max}`
- regularity extras: `rhythm_syncopation_score`, `rhythm_polyrhythm_score`, `rhythm_beat_strength_{mean,std}`, `rhythm_metrical_stability`

#### 5) Meta

- `meta`: object scalar dict (baseline meta + audit v3 extras)
- required baseline meta keys: see `ARTIFACTS_AND_SCHEMAS.md`
- additional meta keys (optional): `backend`, `hop_length`, `features_enabled`, `sampling_family_used`, `rhythmic_contract_version`, beat `.npy` paths
- observability (audit v4.2, optional): `stage_timings_ms`, `rhythmic_resource_profile`

### Empty vs Error semantics

- **No beats detected**: `status="ok"` (not error), `rhythm_beats_count=0`, tempo/regularity/variation/consistency = `NaN`, beat arrays empty/absent.
- **All segments failed**: `status="empty"`, `empty_reason="rhythmic_all_segments_failed"`, `segment_mask=false` for all segments.
- **Missing required sampling family**: error (no-fallback).

### Tiers

- **model_facing**: `feature_names/feature_values` frozen subset
- **analytics**: canonical axis + beat events + extra scalars
- **debug**: `meta`
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
