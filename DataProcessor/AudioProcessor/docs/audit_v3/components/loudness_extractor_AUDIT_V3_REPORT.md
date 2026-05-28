## Audit v3 report — `loudness_extractor` (AudioProcessor)

### 0) TL;DR

`loudness_extractor` доведён до Audit v3 контракта: `run_segments()` работает по Segmenter окнам (`families.primary`) и выдаёт **strict-aligned** сегментные ряды с временной осью `segment_start_sec`/`segment_end_sec`/`segment_center_sec` + `segment_mask`. Ошибки сегментов **не срывают** весь экстрактор: failed сегменты кодируются как `segment_mask=false` и `NaN` в метриках. Машинная схема обновлена до `loudness_extractor_npz_v2`. HTML render переведён в offline-only режим (vanilla canvas, без Chart.js CDN).

---

### 1) Ownership / Versions

- **component_name**: `loudness_extractor`
- **owner_processor**: `AudioProcessor`
- **producer**: `loudness_extractor`
- **producer_version**: `2.1.0`
- **schema_version**: `loudness_extractor_npz_v2`
- **audit_v3_status**: `passed`

Machine schema:
- `DataProcessor/AudioProcessor/schemas/loudness_extractor_npz_v2.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/loudness_extractor/SCHEMA.md`

---

### 2) Inputs

- **Segmenter output**:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - required family: `primary` (`families.primary.segments[]`)

Empty vs error:
- `audio_present=false` → компонент не запускается, `status="empty"`
- `audio_present=true` и отсутствует/пуст `families.primary.segments` → **error** (no-fallback)

---

### 3) Outputs (NPZ = source-of-truth)

Файл артефакта:
- `result_store/<platform_id>/<video_id>/<run_id>/loudness_extractor/loudness_extractor_features.npz`

#### 3.1 Model-facing (tabular)

`feature_names`/`feature_values` содержат стабильный Tier‑0 набор:
- `loudness_rms`, `loudness_peak`, `loudness_dbfs`, `loudness_lufs` (NaN если нет LUFS)
- `duration_sec`, `sample_rate`
- frame-wise RMS stats (`frame_rms_mean/std/median/p10/p90`, `frames_count`)
- segment RMS aggregates (`segment_rms_mean/std/median/p10/p90`, `segments_count`)

#### 3.2 Analytics (segment sequences)

- `lufs_present`: bool scalar
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: float32[N]
- `segment_mask`: bool[N]
- `segment_rms`, `segment_peak`, `segment_dbfs`, `segment_lufs`: float32[N] (NaN для masked)

Semantics:
- strict alignment: размеры массивов всегда = N; failed сегменты не удаляются

#### 3.3 Debug-only

- `meta`: baseline meta + observability

---

### 4) Renderer

- Offline-only HTML (без CDN): line charts на `<canvas>` для RMS/dBFS/LUFS и таблица распределений.

---

### 5) Files changed (high-level)

- `DataProcessor/AudioProcessor/src/extractors/loudness_extractor/__init__.py` (strict alignment + NaN padding, version bump)
- `DataProcessor/AudioProcessor/src/core/npz_savers/loudness.py` (новые canonical keys)
- `DataProcessor/AudioProcessor/src/extractors/loudness_extractor/render.py` (offline render, новые ключи)
- `DataProcessor/AudioProcessor/schemas/loudness_extractor_npz_v2.json` (new machine schema)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping на v2)
- docs sync: `SCHEMA.md`, `README.md`, `docs/MAIN_INDEX.md`

---

### 6) Open items / follow-ups

- Добавить запись в `RUN_LOG.md` после первого реального прогона `loudness_extractor_npz_v2`.

