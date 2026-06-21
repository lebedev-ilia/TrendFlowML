## Audit v3 report — `chroma_extractor` (AudioProcessor)

### 0) TL;DR

`chroma_extractor` переведён на Audit v3 контракт: **фиксированное `n_chroma=12`**, **L1 frame-normalization** (fail-fast для других режимов), **tuning оценивается один раз на полном аудио** (при неудаче → `tuning=0.0` + `meta.tuning_failed=true`). Для `run_segments()` добавлена семантика **strict alignment** по сегментам Segmenter через `segment_mask` + `NaN` padding и компактная последовательность `chroma_mean_by_segment`. Компонент мигрирован на per-extractor schema `chroma_extractor_npz_v1`, а HTML render переписан на полностью offline режим (без CDN/Plotly). Дополнительно улучшена интеграция с `key_extractor`: chroma доступна in-memory как `_shared_chroma` (не сохраняется в NPZ).

---

### 1) Ownership / Versions

- **component_name**: `chroma_extractor`
- **owner_processor**: `AudioProcessor`
- **producer**: `chroma_extractor`
- **producer_version**: `2.1.0`
- **schema_version**: `chroma_extractor_npz_v1`
- **audit_v3_status**: `passed` *(контракт/схемы/логика обновлены; прогон не выполнялся по запросу “без тестов”)*

Machine schema:
- `DataProcessor/AudioProcessor/schemas/chroma_extractor_npz_v1.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/chroma_extractor/SCHEMA.md`

---

### 2) Inputs

- **Primary input**:
  - `frames_dir/audio/segments.json` (`schema_version="audio_segments_v1"`)
  - `frames_dir/audio/audio.wav` *(только если `audio_present=true`)*
- **Required sampling family**:
  - `families.chroma.segments[]`

Принятое решение:
- family `chroma` остаётся отдельной (не объединяем с `spectral`), т.к. окна под chroma могут отличаться по требованиям.

---

### 3) Outputs (NPZ = source-of-truth)

Файл артефакта:
- `result_store/<platform_id>/<video_id>/<run_id>/chroma_extractor/chroma_extractor_features.npz`

#### 3.1 Model-facing

Всегда присутствует:
- `chroma_mean`: `float32[12]`
- `chroma_entropy`: `float32` scalar
- `chroma_harmonic_stability`: `float32` scalar
- `chroma_contrast`: `float32` scalar
- `chroma_dominant_energy`: `float32` scalar

Tabular (`feature_names/feature_values`) — стабильный минимальный набор:
- `chroma_mean_<C..B>` (12 скаляров)
- `chroma_entropy`
- `chroma_harmonic_stability`
- `chroma_contrast`
- `chroma_dominant_energy`

#### 3.2 Analytics

- `tuning_estimate`: `float32` scalar
- `chroma_dominant_class`: `int32` scalar

Опционально для `run_segments()` при `enable_time_series=True`:
- `segment_centers_sec`: `float32[N]`
- `segment_durations_sec`: `float32[N]`
- `segment_mask`: `bool[N]`
- `chroma_mean_by_segment`: `float32[N,12]` (NaN для masked сегментов)

#### 3.3 Debug-only

- `chroma`: `float32[12,T]` (только если включено и не превышает лимит; иначе фиксируем `meta.chroma_time_series_omitted=true`)
- `meta`: baseline + audit v3 поля

---

### 4) Empty vs Error semantics

**Valid empty**:
- `segments.json.audio_present=false` → `status="empty"` и extractor не запускается.

**Error / fail-fast (Audit v3)**:
- отсутствует/пуст `families.chroma.segments` при запросе `chroma` → error (no-fallback)
- `n_chroma != 12` → error
- `normalize != "l1"` → error
- любые ошибки извлечения chroma (CQT/STFT) → error (no-fallback, явный `chroma_type`)

Tuning:
- ошибка `estimate_tuning` **не** приводит к error; используется детерминированный `tuning=0.0`.

---

### 5) Decisions (summary)

Принятые решения (по ответам пользователя):
- **Sampling family**: `chroma`
- **Schema rollout**: `chroma_extractor_npz_v1`
- **Contract**: `n_chroma=12` фиксировано
- **Tuning**: global once; fallback `0.0` if failed
- **Segments**: strict alignment через `segment_mask` + `NaN`
- **Model-facing**: минимальный набор (mean[12] + несколько скаляров)
- **Audio normalization**: оставили opt-in
- **Normalize mode**: фиксируем `l1`
- **Render**: offline-only (без CDN)
- **Key integration**: in-memory `_shared_chroma` (не сохраняется в NPZ)

---

### 6) Files changed (high-level)

- `DataProcessor/AudioProcessor/src/extractors/chroma_extractor/main.py` (audit v3 contract + tuning/segments semantics + shared chroma)
- `DataProcessor/AudioProcessor/src/core/npz_savers/chroma.py` (new NPZ contract keys, minimal model_facing features)
- `DataProcessor/AudioProcessor/src/extractors/chroma_extractor/render.py` (offline HTML, no CDN)
- `DataProcessor/AudioProcessor/src/core/extractor_runner.py` (key→chroma in-memory reuse)
- `DataProcessor/AudioProcessor/schemas/chroma_extractor_npz_v1.json` (new machine schema)
- `DataProcessor/AudioProcessor/src/extractors/chroma_extractor/SCHEMA.md` (new human schema)
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping)
- docs sync: `AudioProcessor/README.md`, `AudioProcessor/docs/MAIN_INDEX.md`, `chroma_extractor/README.md`

---

### 7) Open items / follow-ups (без прогонов)

- Добавить запись в `DataProcessor/docs/audit_v3/RUN_LOG.md` после первого прогона `chroma_extractor_npz_v1`.
- Если понадобится richer сигнал для музыки: обсудить отдельный audited режим для sequence-level chroma (например `chroma_mean_by_segment` всегда включён) и/или добавление key-proxy фич в tabular.
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/chroma_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
