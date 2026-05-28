## `chroma_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `chroma_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `chroma_extractor_npz_v1`
- **source-of-truth**: NPZ (arrays + `feature_names/feature_values`), render = dev-only

### Inputs (contract)

- **Segmenter output**:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `chroma` (`families.chroma.segments[]`)
- **Valid empty**:
  - если `audio/segments.json` содержит `audio_present=false` → AudioProcessor **не запускает** экстрактор и пишет `status="empty"` с каноничным `empty_reason`.

### Canonical configuration (Audit v3)

- `n_chroma=12` (фиксировано; любое другое значение — fail-fast)
- `normalize="l1"` (канонично)
- `chroma_type ∈ {"cqt","stft"}` (явный выбор; no-fallback)
- `enable_audio_normalization=false` по умолчанию (opt-in)

Tuning policy (Audit v3):
- `tuning_estimate` вычисляется **один раз** на полном аудио.
- если оценка tuning не удалась → используем `tuning=0.0` (детерминированно) и фиксируем `tuning_failed=true` в `meta`.

### Outputs (NPZ keys)

#### 1) Tabular (model_facing) — `feature_names` / `feature_values`

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Стабильный model_facing набор (в пределах `schema_version`):

- `chroma_mean_C`, `chroma_mean_C#`, `chroma_mean_D`, `chroma_mean_D#`, `chroma_mean_E`, `chroma_mean_F`,
  `chroma_mean_F#`, `chroma_mean_G`, `chroma_mean_G#`, `chroma_mean_A`, `chroma_mean_A#`, `chroma_mean_B`
- `chroma_entropy`
- `chroma_harmonic_stability`
- `chroma_contrast`
- `chroma_dominant_energy`

Правила:
- порядок `feature_names` фиксирован внутри `schema_version`;
- missing значения кодируются через `NaN` (а не через “0-заглушки”).

#### 2) Canonical arrays (model_facing + analytics)

Always present:
- `chroma_mean`: `float32[12]`
- `chroma_entropy`: `float32` scalar
- `chroma_harmonic_stability`: `float32` scalar
- `chroma_contrast`: `float32` scalar
- `chroma_dominant_class`: `int32` scalar (analytics)
- `chroma_dominant_energy`: `float32` scalar
- `tuning_estimate`: `float32` scalar (analytics)

#### 3) Segment-aligned sequences (optional analytics; `run_segments()`)

Если экстрактор запускался в режиме `run_segments()` и включено `enable_time_series=True` (Audit v3 semantics: segment-level sequence):

- `segment_centers_sec`: `float32[N]`
- `segment_durations_sec`: `float32[N]`
- `segment_mask`: `bool[N]` — `true` если сегмент был обработан (не пустой, без ошибки)
- `chroma_mean_by_segment`: `float32[N,12]` — per-segment chroma mean; для `segment_mask=false` значения = `NaN`

Semantics:
- `N` соответствует числу `families.chroma.segments` в `audio/segments.json`;
- “пропуски” не удаляются (нет изменения длины), а кодируются `segment_mask=false`.

#### 4) Debug-only (optional)

- `chroma`: `float32[12,T]` — полный chroma spectrogram только в режиме **`run()`**, если time series включён и не превышает лимит; в **`run_segments()`** этого ключа нет (сегментная матрица — `chroma_mean_by_segment`).
- `meta`: object scalar dict (см. `ARTIFACTS_AND_SCHEMAS.md`)

Optional (Audit 4.2 engineering):
- `chroma_resource_profile`: object dict — best-effort RSS/VRAM snapshots (env-gated, may be absent).

### Tiers

- **model_facing**:
  - `feature_names/feature_values` (minimal stable set)
  - `chroma_mean` + основные скаляры
- **analytics**:
  - `tuning_estimate`, `chroma_dominant_class`, segment-aligned sequences
- **debug**:
  - `chroma` (если включено), `meta`

