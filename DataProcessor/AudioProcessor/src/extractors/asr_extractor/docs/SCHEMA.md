## `asr_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `asr_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `asr_extractor_npz_v2`
- **source-of-truth**: NPZ; render/HTML = dev-only

### Inputs (contract)

- Segmenter output:
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `asr`
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)

Empty semantics:
- Если `segments.json.audio_present=false` → AudioProcessor пишет `status="empty"` и **не запускает** ASR.
- Если `audio_present=true`, но `families.asr.segments` отсутствует/пустой → **error** (no-fallback).

### Models (offline, reproducible)

- Whisper: `dp_models` spec `whisper_{small|medium|large}_inprocess`
- Tokenizer: `dp_models` spec `shared_tokenizer_v1`

**Token contract (FINAL)**:
- `token_ids_by_segment` (если включено) **всегда принадлежит `shared_tokenizer_v1`**.
- Любой encode‑fail → `status="error"` (не допускается fallback на whisper tokens).

### Outputs (NPZ keys)

#### 1) Tabular (baseline)

- `feature_names`: `object[F]`
- `feature_values`: `float32[F]`

Model-facing агрегаты (включаются флагами, порядок внутри `schema_version` фиксирован):
- `segments_count`
- `token_total`
- `token_density_per_sec`
- `speech_rate_wpm`
- `segments_with_speech`
- `avg_segment_duration_sec`
- `token_variance`

#### 2) Sequences / per-segment

- `token_ids_by_segment`: `object[N]` (каждый элемент — `int32[T_i]`, variable length), **optional (feature-gated)**
- `segment_start_sec`: `float32[N]` (required)
- `segment_end_sec`: `float32[N]` (required)
- `segment_center_sec`: `float32[N]` (required)
- `lang_id_by_segment`: `int32[N]` (**optional**; legacy, analytics; best-effort)
- `lang_code_by_segment`: `object[N]` (required; preferred, analytics; `""` = unknown)
- `lang_conf_by_segment`: `float32[N]` (required; analytics; `NaN` = unknown)
- `token_counts`: `int32[N]` (optional)
- `lang_distribution`: `object` (dict-like; optional, analytics)

#### 3) Quality / confidence (privacy-safe)

- `segment_quality_by_segment`: `object[N]` (list of dicts; numeric-only)
  - keys (best-effort, missing → `None`):
    - `avg_logprob`
    - `compression_ratio`
    - `no_speech_prob`
    - `temperature`

#### 4) Debug-only raw text (opt-in)

- `segment_texts_by_segment`: `object[N]` (list[str]) — **только если включён `--asr-save-segment-text`**

#### 5) Meta

- `meta`: object scalar dict (см. `ARTIFACTS_AND_SCHEMAS.md`)
- must include:
  - versions/identity/status
  - `models_used[]`, `model_signature`
  - `stage_timings_ms`
  - `asr_text_contract_version`
  - `features_enabled`
- optional (extractor v2.3.0+, «этап 2» профилирование):
  - `asr_stage_timings_ms`: object dict — фазы внутри ASR (мс), см. `docs/README.md` §«Этап 2»
  - `asr_resource_profile`: object dict — снимки RSS/GPU в контрольных точках (если `AP_ASR_RESOURCE_PROFILE=1`)

#### 6) Audio duration + sampling (v2)

Цель: сделать downstream (TextProcessor) менее зависимым от `manifest.json → frames_dir → audio/segments.json`
в режимах инспекции/интеграции.

- `audio_duration_sec`: `float32` scalar (required; может быть `NaN` если не удалось определить)
- `asr_sampling_profile`: `object` scalar string (required; `""` если неизвестно)
- `asr_window_sec`: `float32` scalar (required; `NaN` если неизвестно)
- `asr_stride_sec`: `float32` scalar (required; `NaN` если неизвестно)
- `asr_max_windows`: `int32` scalar (required; `-1` если неизвестно)

### Sampling requirements

Extractor требует family `asr` (длинные окна). Параметры окон (window/stride/caps) должны быть задокументированы в sampling policy и фиксироваться через `segments.json` + `config_hash`.


