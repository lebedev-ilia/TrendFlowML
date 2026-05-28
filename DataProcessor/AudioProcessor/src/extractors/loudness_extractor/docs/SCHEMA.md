## `loudness_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `loudness_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `loudness_extractor_npz_v2`
- **source-of-truth**: NPZ (`feature_names/feature_values` + arrays), render = dev-only

### Inputs (contract)

- **Segmenter output**:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `primary`
- **Valid empty**:
  - если `audio/segments.json` содержит `audio_present=false` → AudioProcessor **не запускает** экстрактор и пишет `status="empty"` с каноничным `empty_reason` (см. `ARTIFACTS_AND_SCHEMAS.md`).

### Outputs (NPZ keys)

#### 1) Tabular (baseline path) — `feature_names` / `feature_values`

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Стабильный набор (Tier‑0, model_facing):

- `loudness_rms` (float)
- `loudness_peak` (float)
- `loudness_dbfs` (float)
- `loudness_lufs` (float; если LUFS недоступен → `NaN`)
- `duration_sec` (float)
- `sample_rate` (float)
- `frame_rms_mean` (float)
- `frame_rms_std` (float)
- `frame_rms_median` (float)
- `frame_rms_p10` (float)
- `frame_rms_p90` (float)
- `frames_count` (float)
- `segments_count` (float)
- `segment_rms_mean` (float)
- `segment_rms_std` (float)
- `segment_rms_median` (float)
- `segment_rms_p10` (float)
- `segment_rms_p90` (float)

Правила:
- порядок `feature_names` фиксирован внутри `schema_version`;
- missing значения кодируются через `NaN` (а не через “0-заглушки”).

#### 2) Segment sequences (analytics / token readiness)

- `lufs_present`: `bool` (scalar) — true если LUFS вычислен и конечен
- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]` (strict alignment; false для failed сегментов)
- `segment_rms`: `float32[N]`
- `segment_peak`: `float32[N]`
- `segment_dbfs`: `float32[N]`
- `segment_lufs`: `float32[N]` (может содержать `NaN`)

Semantics:
- `N` соответствует числу `families.primary.segments` (если экстрактор запускался);
- **strict alignment**: массивы никогда не “сжимаются”; для failed сегмента `segment_mask=false`, а метрики хранят `NaN`;
- если компонент `status="empty"` → массивы могут быть пустыми.

#### 3) Meta

- `meta`: object scalar dict (см. `ARTIFACTS_AND_SCHEMAS.md`)

Обязательные meta поля (baseline + audit v3):
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- versions: `producer`, `producer_version`, `schema_version`, `created_at`
- status: `status`, `empty_reason` (+ `error` если применимо)
- model system: `models_used=[]`, `model_signature`
- observability: `stage_timings_ms`

Опционально (audit v4.2):
- `loudness_resource_profile` (dict|None): snapshot RSS/VMS/VRAM, включается через `AP_LOUDNESS_RESOURCE_PROFILE=1`

### Tiers

- **model_facing**:
  - `feature_names/feature_values` (скаляры loudness + frame stats + segment stats)
- **analytics**:
  - segment sequences (`segment_*`, `segment_*_sec`), `lufs_present`
- **debug**:
  - `meta`


