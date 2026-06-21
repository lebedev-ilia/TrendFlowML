## `band_energy_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `band_energy_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `band_energy_extractor_npz_v1`
- **source-of-truth**: NPZ (arrays + `feature_names/feature_values`), render = dev-only

### Inputs (contract)

- **Segmenter output**:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `spectral` (shared family for spectral-like extractors; declared requirement, not runtime fallback)
- **Valid empty**:
  - если `audio/segments.json` содержит `audio_present=false` → AudioProcessor **не запускает** экстрактор и пишет `status="empty"` с каноничным `empty_reason`.

### Outputs (NPZ keys)

#### 1) Tabular (baseline path) — `feature_names` / `feature_values` (model_facing)

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Стабильный model_facing набор (в пределах `schema_version`):

- `band_share_low` (float)
- `band_share_mid` (float)
- `band_share_high` (float)

Правила:
- порядок `feature_names` фиксирован внутри `schema_version`;
- missing значения кодируются через `NaN` (а не через “0-заглушки”).

Опционально (analytics через feature-gating `enable_balance_metrics`):
- `band_balance_score` (float)
- `band_contrast` (float)
- `band_dominant_band` (float; исходно int, но хранится как float в feature-векторе)

#### 2) Canonical bands (model_facing + analytics)

- `band_edges_hz`: `float32[B,2]` — границы полос в Hz  
  Audit v3 канон: **B=3** (low/mid/high): \([0,200), [200,2000), [2000, nyq)\)
- `band_energy_shares`: `float32[B]` — доли энергии по полосам, сумма \(\approx 1\)

#### 3) Segment-aligned sequences (optional analytics)

Если включено `enable_time_series=True` (Audit v3 semantics: строгое выравнивание по сегментам Segmenter):

- `segment_centers_sec`: `float32[N]`
- `segment_durations_sec`: `float32[N]`
- `segment_mask`: `bool[N]` — `true` если сегмент был обработан (достаточная длительность и без ошибки)
- `band_shares_by_segment`: `float32[N,B]` — shares по каждому сегменту; для `segment_mask=false` значения = `NaN`

Semantics:
- `N` соответствует числу `families.spectral.segments` в `audio/segments.json`;
- “пропуски” не удаляются (нет изменения длины), а кодируются `segment_mask=false`.

#### 4) Meta (debug)

- `meta`: object scalar dict (см. `ARTIFACTS_AND_SCHEMAS.md`)

Обязательные meta поля (baseline + audit v3):
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- versions: `producer`, `producer_version`, `schema_version`, `created_at`
- status: `status`, `empty_reason` (+ `error` если применимо)
- model system: `models_used=[]`, `model_signature`
- observability: `stage_timings_ms`

Optional (Audit 4.2 engineering):
- `band_energy_resource_profile`: object dict — best-effort RSS/VRAM snapshots (env-gated, may be absent).

### Tiers

- **model_facing**:
  - `feature_names/feature_values` (shares only)
  - `band_energy_shares`
- **analytics**:
  - `band_edges_hz`, segment-aligned sequences (если включены)
- **debug**:
  - `meta`
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
