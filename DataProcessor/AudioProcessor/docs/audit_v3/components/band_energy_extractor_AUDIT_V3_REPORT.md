## Audit v3 report — `band_energy_extractor` (AudioProcessor)

### 0) TL;DR

`band_energy_extractor` после Audit v3 публикует **только scale‑invariant band shares** по 3 фиксированным полосам (low/mid/high) как model_facing сигнал, со строгим Segmenter‑sampling контрактом (shared family `spectral`). Убраны/запрещены soft-fallback режимы (`band_method="auto"`, Essentia), введён per-extractor NPZ контракт `band_energy_extractor_npz_v1`, а HTML render переписан в полностью offline режим (без CDN).

---

### 1) Ownership / Versions

- **component_name**: `band_energy_extractor`
- **owner_processor**: `AudioProcessor`
- **producer**: `band_energy_extractor`
- **producer_version**: `2.1.0`
- **schema_version**: `band_energy_extractor_npz_v1`
- **audit_v3_status**: `passed` *(контракт/схемы/логика обновлены; прогон не выполнялся по запросу “без тестов”)*

Machine schema:
- `DataProcessor/AudioProcessor/schemas/band_energy_extractor_npz_v1.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/band_energy_extractor/SCHEMA.md`

---

### 2) Inputs

- **Primary input**:
  - `frames_dir/audio/segments.json` (`schema_version="audio_segments_v1"`)
  - `frames_dir/audio/audio.wav` *(только если `audio_present=true`)*
- **Required sampling family** (Audit v3 decision):
  - `families.spectral.segments[]`

Почему shared family:
- `band_energy`, `pitch`, `spectral_entropy` используют общую sampling family `spectral` (снижение количества почти-дубликатных окон у Segmenter) без runtime fallback — это объявленный sampling requirement.

---

### 3) Outputs (NPZ = source-of-truth)

Файл артефакта:
- `result_store/<platform_id>/<video_id>/<run_id>/band_energy_extractor/band_energy_extractor_features.npz`

#### 3.1 Model-facing

Tabular scalars (`feature_names/feature_values`) — **shares only**:
- `band_share_low`
- `band_share_mid`
- `band_share_high`

Также как массив:
- `band_energy_shares`: `float32[3]` (сумма \(\approx 1\))

#### 3.2 Analytics

Каноничная геометрия полос:
- `band_edges_hz`: `float32[3,2]` (Hz)

Опциональные segment-aligned sequences (если включено `enable_time_series=True`):
- `segment_centers_sec`: `float32[N]`
- `segment_durations_sec`: `float32[N]`
- `segment_mask`: `bool[N]`
- `band_shares_by_segment`: `float32[N,3]` (для `segment_mask=false` значения = `NaN`)

Опциональные balance-метрики (feature-gated `enable_balance_metrics=True`):
- публикуются как scalars в `feature_names/feature_values` (analytics по смыслу) и дублируются в `meta` для debug.

#### 3.3 Debug-only

- `meta`: object scalar dict (run identity + версии + статус + тайминги + debugging extras)
- Render в `.../band_energy_extractor/_render/` (dev-only, полностью offline)

---

### 4) Empty vs Error semantics

**Valid empty**:
- `segments.json.audio_present=false` → `status="empty"` и экстрактор не запускается (AudioProcessor пишет empty NPZ артефакт).

**Error / fail-fast (Audit v3)**:
- отсутствует/пуст `families.spectral.segments` при запросе `band_energy` → error (no-fallback)
- `band_method != "librosa"` → error (запрещены Essentia/auto)
- `use_mel_bands=True` или `bands` не длины 3 → error (контракт на 3 фиксированные полосы)

---

### 5) Sampling requirements (Audio)

Extractor требует family `spectral` от Segmenter.

Правило: Segmenter — единственный владелец sampling; extractor не “придумывает” окна и не делает runtime fallback на другие families.

---

### 6) Decisions (what changed and why)

Принятые решения (по ответам пользователя):
- **Sampling family**: shared `spectral`
- **Method policy**: `librosa` only
- **Bands**: фиксированные 3 полосы (low/mid/high)
- **Normalization**: включена по умолчанию
- **Time series**: строгое выравнивание через `segment_mask` + `NaN`
- **Feature scope**: минимальный набор (shares; optional balance + optional segment-aligned sequences)
- **Schema rollout**: per-extractor `band_energy_extractor_npz_v1`
- **Render**: полностью offline (без CDN)
- **Output units**: model_facing = shares only

Ключевые изменения в коде:
- синхронизирован sampling policy в документации: `band_energy` теперь **требует** `families.spectral` (shared family)
- удалён/запрещён Essentia/auto fallback (Audit v3 no-fallback)
- `run_segments()` перестал “терять” индексы: короткие/ошибочные сегменты отмечаются mask’ой, а не удаляются
- NPZ saver переведён на явные ключи (без `payload` dict), чтобы NPZ был строгим source-of-truth
- HTML render переписан на простой offline вариант (без внешних JS библиотек)

---

### 7) Files changed (high-level)

- `DataProcessor/AudioProcessor/src/extractors/band_energy_extractor/main.py` (audit v3 logic + fail-fast)
- `DataProcessor/AudioProcessor/src/core/npz_savers/band_energy.py` (new NPZ contract keys, no payload)
- `DataProcessor/AudioProcessor/src/extractors/band_energy_extractor/render.py` (offline HTML, no CDN)
- `DataProcessor/AudioProcessor/schemas/band_energy_extractor_npz_v1.json` (new machine schema)
- `DataProcessor/AudioProcessor/src/extractors/band_energy_extractor/SCHEMA.md` (new human schema)
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping)
- docs sync: `AudioProcessor/README.md`, `AudioProcessor/docs/MAIN_INDEX.md`, `DataProcessor/docs/COMPONENTS_DESC.md`, `band_energy_extractor/README.md`

---

### 8) Open items / follow-ups (без прогонов)

- Добавить запись в `DataProcessor/docs/audit_v3/RUN_LOG.md` после первого прогона `band_energy_extractor_npz_v1`.
- При необходимости: формализовать balance-метрики как отдельные NPZ keys (если захотим сделать их “first-class” analytics массивами, а не только табличными scalar features).

