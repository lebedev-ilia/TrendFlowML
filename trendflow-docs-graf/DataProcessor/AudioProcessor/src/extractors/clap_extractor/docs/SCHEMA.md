## `clap_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `clap_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `clap_extractor_npz_v1`
- **source-of-truth**: NPZ (arrays + `feature_names/feature_values`), render = dev-only

### Inputs (contract)

- **Segmenter output**:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `clap` (`families.clap.segments[]`)
- **Valid empty**:
  - если `audio_present=false` → AudioProcessor не запускает экстрактор и пишет `status="empty"`.

### Model system (offline / reproducibility)

- CLAP weights грузятся **строго локально** через `dp_models` (ModelManager-only).
- Любые неявные загрузки/скачивание в runtime запрещены (no-network).

### Sampling requirements (Audio)

- required family: `clap`
- окна могут быть длиннее, чем эффективная длина CLAP → extractor **обрезает** сигнал до `max_audio_length_sec` и репортит `trimmed_ratio`/`trimmed_segments_count` в meta.

### Outputs (NPZ keys)

#### 1) Tabular (`feature_names` / `feature_values`) — model_facing minimal subset

Frozen subset в пределах `schema_version` (порядок строк = порядок в `feature_names` / реализация `npz_savers/clap.py`):
- `embedding_dim`
- `clap_norm`
- `clap_magnitude_mean`
- `clap_magnitude_std`
- `segments_count`

Примечание:
- `embedding_present` хранится отдельным NPZ scalar (`embedding_present: bool`) и может дублироваться в meta/debug, но не обязателен в frozen subset.

#### 2) Embeddings (TokenStreams readiness)

- `embedding`: `float32[D]` — агрегированный эмбеддинг (robust aggregation по валидным сегментам)
- `embedding_sequence`: `float32[N,D]` — per-segment embeddings (для masked сегментов значения = `NaN`)

Time axis + mask:
- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`

Optional analytics:
- `segment_embedding_norm`: `float32[N]` (NaN для masked)

#### 3) Meta (debug)

- `meta`: object scalar dict (baseline + audit v3)

Обязательные meta поля (baseline + audit v3):
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- versions: `producer`, `producer_version`, `schema_version`, `created_at`
- status: `status`, `empty_reason` (+ `error` если применимо)
- model system: `models_used[]`, `model_signature`
- observability: `stage_timings_ms`

Опционально (audit v4.2):
- `clap_resource_profile` (dict|None): snapshot RSS/VMS/VRAM, включается через `AP_CLAP_RESOURCE_PROFILE=1`

Дополнительно (audit v3):
- `max_audio_length_sec`
- `trimmed_segments_count`
- `trimmed_ratio`

### Empty vs Error semantics

- **empty**: только если upstream `audio_present=false`
- **error**:
  - missing/empty `families.clap.segments` при запуске компонента
  - отсутствуют локальные веса CLAP в `dp_models`

### Tiers

- **model_facing**:
  - `embedding`, `embedding_sequence`, tabular minimal subset
- **analytics**:
  - time axis + mask, segment norms, trim stats
- **debug**:
  - `meta`, render
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
