## Audit v3 report — `asr_extractor` (AudioProcessor)

### 0) TL;DR

`asr_extractor` извлекает ASR через Whisper (inprocess, offline через `dp_models`) и публикует **privacy-first** выход: **token IDs** (строгий контракт `shared_tokenizer_v1`) + числовые quality/языковые метрики. В рамках аудита мы усилили интеграцию с TextProcessor через **token-only контракт**, запретили stochastic fallback decode и обновили схему артефакта до `asr_extractor_npz_v2` (добавили `audio_duration_sec` и параметры sampling family `asr`).

---

### 1) Ownership / Versions

- **component_name**: `asr_extractor`
- **owner_processor**: `AudioProcessor`
- **producer**: `asr_extractor`
- **producer_version**: `2.2.0`
- **schema_version**: `asr_extractor_npz_v2`
- **audit_v3_status**: `passed` *(контракт/схемы/интеграции обновлены; прогон не выполнялся по запросу “без тестов”)*  

Machine schema:
- `DataProcessor/AudioProcessor/schemas/asr_extractor_npz_v2.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/asr_extractor/SCHEMA.md`

---

### 2) Inputs

- **Primary input**:
  - `frames_dir/audio/segments.json` (`schema_version="audio_segments_v1"`)
  - `frames_dir/audio/audio.wav` *(только если `audio_present=true`)*
- **Required sampling family**:
  - `families.asr.segments[]`

**Hard dependencies (no-fallback)**:
- `dp_models` ModelManager (offline)
- `shared_tokenizer_v1` artifact (`tokenizer.json`) (offline)
- `whisper_{small|medium|large}_inprocess` (offline)

**Soft dependencies**:
- нет (все критично для запуска ASR в audited режиме)

---

### 3) Outputs (NPZ = source-of-truth)

Файл артефакта:
- `result_store/<platform_id>/<video_id>/<run_id>/asr_extractor/asr_extractor_features.npz`

#### 3.1 Model-facing

- **`token_ids_by_segment`** *(optional, feature-gated)*:
  - dtype: `object[N]`, каждый элемент `int32[T_i]` (variable length)
  - **Tier**: `model_facing`
  - **Usefulness**: 9/10 (основа для text downstream без хранения raw текста)
  - **Risk/noise**: 2/10 (privacy-sensitive, но без PII-строк; стабильность обеспечена strict tokenizer contract)

- **Tabular `feature_names/feature_values`**:
  - Tier: `model_facing` (по формату), но содержимое разделяем на `analytics/model_facing` логически.
  - Текущий статус: **quality агрегаты добавлены как analytics-only** (см. ниже).

#### 3.2 Analytics

Per-segment time axis:
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec` (`float32[N]`)

Language signals (preferred):
- `lang_code_by_segment` (`object[N]`), `""` = unknown
- `lang_conf_by_segment` (`float32[N]`), `NaN` = unknown
- `lang_distribution` (`object` dict: `lang_code -> count`) *(если включено)*

Legacy language id:
- `lang_id_by_segment` (`int32[N]`) — **optional в v2**, best-effort (не считаем стабильным идентификатором)

Quality signals (privacy-safe):
- `segment_quality_by_segment` (`object[N]`): dicts с числовыми полями (`avg_logprob`, `compression_ratio`, `no_speech_prob`, `temperature`)
- Tabular analytics агрегаты (новые, stable names):
  - `asr_quality__avg_logprob_mean/p50/p90/present_rate`
  - `asr_quality__compression_ratio_mean/p50/p90/present_rate`
  - `asr_quality__no_speech_prob_mean/p50/p90/present_rate`

Audit v3 v2 addition: duration + sampling params (Segmenter-owned, дублируем в NPZ для удобства downstream):
- `audio_duration_sec` (`float32` scalar; может быть NaN)
- `asr_sampling_profile` (`object` scalar string; `""` если неизвестно)
- `asr_window_sec` (`float32` scalar; может быть NaN)
- `asr_stride_sec` (`float32` scalar; может быть NaN)
- `asr_max_windows` (`int32` scalar; `-1` если неизвестно)

#### 3.3 Debug-only

- HTML/JSON render в `.../asr_extractor/_render/` (dev-only, offline)
- Raw текст **не публикуем** в `VideoDocument` и не используем как source-of-truth.

---

### 4) Empty vs Error semantics

**Valid empty cases**:
- `segments.json.audio_present=false` → `status="empty"` и extractor **не запускается** (AudioProcessor пишет empty NPZ артефакт).

**Error cases (fail-fast)**:
- `audio_present=true`, но отсутствует/пуст `families.asr.segments` → error
- отсутствуют offline артефакты `dp_models` для whisper/tokenizer → error
- `shared_tokenizer_v1.encode()` падает → error (fallback на whisper tokens запрещён)

---

### 5) Sampling requirements (Audio)

Extractor требует `families.asr` (длинные окна).

**Default для audit/production профилей (рекомендация)**:
- `profile="semantic"`
- `window_sec≈30`, `stride_sec≈25` (пример)

Правило: Segmenter — единственный владелец sampling; extractor не “придумывает” окна.

---

### 6) Reproducibility / Model system

- Все модели и токенизатор резолвятся через `dp_models` (no-network).
- `meta.models_used[]` + `meta.model_signature` присутствуют и должны соответствовать фактически использованным spec’ам.

---

### 7) TextProcessor integration (token-only)

Цель: production path без сохранения raw транскрипта.

Изменения:
- `DataProcessor/main.py` теперь автогенерирует `VideoDocument` из ASR NPZ **без raw текста**:
  - кладёт `transcripts_token_ids["whisper"]` (flatten) и token-only `asr` payload (`asr_payload_v2`) с `token_ids_by_segment` + таймингами.
- `TextProcessor`:
  - `VideoDocument` уже умеет best-effort decode `transcripts_token_ids["whisper"]` → `transcripts["whisper"]` transiently.
  - `transcript_chunk_embedder` разрешён fallback на `transcripts["whisper"]`, если `asr.segments` отсутствует.
  - `asr_text_proxy_audio_features` поддерживает token-only режим: transient decode token IDs в текст, **без сохранения** raw текста в артефактах.

---

### 8) Decisions (summary)

- **Token-only ASR→Text**: принято (privacy-first).
- **`lang_id_by_segment`**: переведён в optional (v2), основной язык = `lang_code/lang_conf`.
- **Quality агрегаты**: добавлены в tabular как analytics-only (не model_facing по умолчанию).
- **Fallback decode**: запрещён в audited режиме (fail-fast при включении).
- **Sampling default**: `semantic`.

---

### 9) Open items / follow-ups (без прогонов)

- Добавить запись в `DataProcessor/docs/audit_v3/RUN_LOG.md` для первого прогона `asr_extractor_npz_v2` после запуска.
- При необходимости: расширить machine schema описанием “frozen subset” табличных фичей (если решим, что часть quality агрегатов становится model_facing).

