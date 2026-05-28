## Audit v3 report — `speech_analysis_extractor` (AudioProcessor)

### 0) TL;DR

Компонент доведён до Audit v3 контракта: per-extractor схема `speech_analysis_extractor_npz_v1`, empty semantics для audio too short (`status="empty"`, `empty_reason="audio_too_short"` вместо RuntimeError), NaN для missing значений (вместо нулевых заглушек), offline HTML render (vanilla canvas, без Plotly CDN). Feature defaults оставлены все False (opt-in). Canonical axis не добавлен (bundle-агрегатор).

---

### 1) Ownership / Versions

- **component_name**: `speech_analysis_extractor`
- **producer_version**: `2.1.0`
- **schema_version**: `speech_analysis_extractor_npz_v1`
- **audit_v3_status**: `in_progress` (нужен validation run + запись в run-log)

Machine schema:

- `DataProcessor/AudioProcessor/schemas/speech_analysis_extractor_npz_v1.json`

Human schema:

- `DataProcessor/AudioProcessor/src/extractors/speech_analysis_extractor/SCHEMA.md`

---

### 2) Inputs / Sampling

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json`:
  - required: `families.asr.segments`, `families.diarization.segments`
- Результаты зависимых компонентов: `asr_result`, `diarization_result`, `pitch_result` (из extractor_results)

---

### 3) Outputs (NPZ = source-of-truth)

#### 3.1 Model-facing (tabular, frozen)

`feature_names`/`feature_values` — run-level скаляры:

- базовые: `duration_sec`, `sample_rate`
- asr_metrics (feature-gated): `asr_segments_count`, `asr_token_total`, `asr_token_mean`, `asr_token_std`, `asr_token_density_per_sec`, `asr_speech_rate_wpm`
- diarization_metrics (feature-gated): `speaker_count`, `dominant_speaker_share`, `speaker_balance_score`, `speaker_transitions_count`, `diar_segments_count`
- pitch_metrics (feature-gated): `pitch_enabled`, `pitch_f0_mean`, `pitch_f0_std`, `pitch_f0_min`, `pitch_f0_max`, `pitch_f0_range`, `pitch_stability`

Missing → **NaN**.

#### 3.2 Analytics

- `asr_lang_id_by_segment` (int32, per ASR segment)
- `speaker_ids` (int32, per speaker)
- `asr_lang_distribution` (object/dict)
- `pitch_distribution` (object/dict)

Canonical axis не добавлен — компонент bundle-агрегатор, run-level.

#### 3.3 Meta

`meta` включает baseline meta + `speech_analysis_contract_version`, `features_enabled`, `stage_timings_ms`.

---

### 4) Semantics (empty/error)

- **audio too short (<5s)**: `status="empty"`, `empty_reason="audio_too_short"` (вместо RuntimeError)
- **тихое аудио**: `status="empty"`, `empty_reason="audio_missing_or_extract_failed"`
- **пустые сегменты**: `status="error"` (no-fallback)
- **зависимости не предоставлены** (при включённых feature flags): `status="error"`

---

### 5) Renderer

- Offline-only HTML render: без CDN, графики через vanilla `<canvas>`:
  - Language distribution (bar chart)
  - Language ID by segment (timeline)
  - Speaker distribution (bar chart)
  - Pitch distribution by octave (bar chart)

---

### 6) Files changed

- `DataProcessor/AudioProcessor/src/extractors/speech_analysis_extractor/main.py` (audio too short → empty)
- `DataProcessor/AudioProcessor/src/core/npz_savers/speech_analysis.py` (NaN policy)
- `DataProcessor/AudioProcessor/src/extractors/speech_analysis_extractor/render.py` (offline vanilla canvas)
- `DataProcessor/AudioProcessor/schemas/speech_analysis_extractor_npz_v1.json` (machine schema)
- `DataProcessor/AudioProcessor/src/extractors/speech_analysis_extractor/SCHEMA.md` (human schema)
- `DataProcessor/AudioProcessor/src/extractors/speech_analysis_extractor/README.md` (Render section, empty semantics)
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping)

---

### 7) Follow-ups (required to close audit)

- Валидационный прогон на audio-present наборе + запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`:
  - `audio_duration_sec`, `sample_rate`
  - families `asr`, `diarization`: N segments
  - sanity по диапазонам (asr_token_*, speaker_count, pitch_f0_*)
  - проверка empty при dur < 5s и при тишине
