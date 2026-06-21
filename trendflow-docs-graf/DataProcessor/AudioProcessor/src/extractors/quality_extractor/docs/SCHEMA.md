## `quality_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `quality_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `quality_extractor_npz_v2`
- **source-of-truth**: NPZ (`feature_names/feature_values` + arrays), render = dev-only

### Inputs (contract)

- **Segmenter output**:
  - `frames_dir/audio/audio.wav`
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `primary`
- **Valid empty**:
  - если все сегменты пустые → `status="empty"`, `empty_reason="quality_all_segments_empty"`

### Outputs (NPZ keys)

#### 1) Tabular — `feature_names` / `feature_values`

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Feature-gated (порядок фиксирован внутри `schema_version`):

**Always** (tabular — только числа):
- `sample_rate`, `average_channels`, `frame_len_ms`, `hop_ms`, `clip_threshold`
- `duration`, `segments_count`
- **`device_used`**: строка — только в **`meta`**, не в `feature_values`

**basic_metrics** (если включено):
- `dc_offset`, `clipping_ratio`, `crest_factor_db`, `clipping_segments_count`, `quality_score`, `crest_factor_median`

**dynamic_metrics** (если включено):
- `dynamic_range_db`, `dynamic_range_stability`

**frame_analysis** (если включено):
- `frame_levels_mean`, `frame_levels_std`, `frame_levels_min`, `frame_levels_max`, `frame_levels_median`

Правила:
- missing значения → `NaN`;
- `snr_db` удалён (Audit v3: был дубликатом `dynamic_range_db`);
- `dc_offset_abs` удалён (дубликат `abs(dc_offset)`).

#### 2) Segment sequences (canonical axis)

- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`

#### 3) Time series

- Хранятся **только** в `.npy` файлах; пути в `meta`:
  - `dc_offset_series_npy`, `clipping_ratio_series_npy`, `crest_factor_db_series_npy`
  - `dynamic_range_db_series_npy`, `frame_levels_db_series_npy`, `frame_rms_series_npy`
  - `clipping_segments_series_npy`

#### 4) Meta

- `meta`: object scalar dict
- Обязательные: `producer`, `producer_version`, `schema_version`, `created_at`, `status`, `empty_reason`, run identity, `stage_timings_ms`

### Audit v4.2 observability (meta.extra)

- `meta.extra.stage_timings_ms`: dict, тайминги этапов (ms)
- `meta.extra.quality_resource_profile`: dict|None, best-effort snapshot RSS/VMS (gated by `AP_QUALITY_RESOURCE_PROFILE=1`)

### Tiers

- **model_facing**: `feature_names/feature_values` (скаляры)
- **analytics**: segment sequences (`segment_*_sec`, `segment_mask`)
- **debug**: `meta`, time series paths
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
