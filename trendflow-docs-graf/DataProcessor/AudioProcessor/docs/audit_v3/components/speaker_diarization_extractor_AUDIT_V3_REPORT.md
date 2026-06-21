## Audit v3 report — `speaker_diarization_extractor` (AudioProcessor)

### 0) TL;DR

Компонент переведён на Audit v3 контракт: **ModelManager-only (no-network)**, diarization-only (без transcript/word alignment), **Segmenter-owned** sampling family `diarization` (одно full-audio окно), строгий per-extractor NPZ контракт `speaker_diarization_extractor_npz_v2` (human+machine schema), token-ready представление turns через плоские массивы `turn_*`, рендер переписан в offline-only режим (vanilla canvas, без Plotly CDN). Также обновлена интеграция `speech_analysis_extractor` для чтения новых structured полей.

---

### 1) Ownership / Versions

- **component_name**: `speaker_diarization_extractor`
- **producer_version**: `3.1.0`
- **schema_version**: `speaker_diarization_extractor_npz_v2`
- **audit_v3_status**: `in_progress` (нужен validation run + запись в run-log)

Machine schema:

- `DataProcessor/AudioProcessor/schemas/speaker_diarization_extractor_npz_v2.json`

Human schema:

- `DataProcessor/AudioProcessor/src/extractors/speaker_diarization_extractor/SCHEMA.md`

---

### 2) Inputs

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json`:
  - required: `families.diarization.segments` (no-fallback)
  - required shape: **ровно 1** сегмент (full-audio окно)

---

### 3) Outputs (NPZ = source-of-truth)

#### 3.1 Model-facing (tabular, frozen)

`feature_names/feature_values` фиксируют минимальный стабильный набор:

- `speaker_count`, `duration_sec`
- `speaker_balance_score`, `dominant_speaker_id`
- `speaker_turns_count`, `speaker_turns_density`, `speaker_transitions_count`

#### 3.2 Analytics

- canonical Segmenter axis (ожидаемо \(N=1\)): `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- token-ready turns (плоские массивы длины \(K\)):
  - `turn_start_sec`, `turn_end_sec`, `turn_speaker_id`, `turn_mask`
- per-speaker structured arrays длины \(S=speaker_count\):
  - `speaker_ids`
  - `speaker_duration_sec`
  - `speaker_time_ratio`
  - `speaker_turns_count_by_speaker`

#### 3.3 Meta

`meta` включает baseline meta + diarization extras:

- `model_name`, `weights_digest`
- `diarization_contract_version`
- `features_enabled`

---

### 4) Semantics (empty/error)

- **audio missing** (upstream): `status="empty"`, `empty_reason="audio_missing_or_extract_failed"`
- **audio silent**: `status="empty"`, `empty_reason="audio_silent"`
- **ModelManager not available / spec missing**: `status="error"` (fail-fast)
- **Network fallback запрещён**: любые попытки загрузки с HuggingFace устранены; extractor требует `dp_models`

---

### 5) Privacy / retention

- Transcript/words **не извлекаются** и **не сохраняются** в audited контракте.
- Не сохраняются `speaker_ids_str` / `speaker_id_map` (debug trace) — только `speaker_id` \(0..S-1\).
- Embeddings (speaker/segment) убраны из контракта.

---

### 6) Renderer

- Offline-only HTML render: без CDN, визуализация через vanilla `<canvas>`:
  - timeline turns (bar-like)
  - per-speaker time ratio (bars)

---

### 7) Integration impact

`speech_analysis_extractor` обновлён: при наличии structured v2 полей (`speaker_duration_sec`, `turn_*`) считает `dominant_speaker_share` и `diar_segments_count` без зависимости от legacy `speaker_segments` list[dict]. При отсутствии — сохраняет обратную совместимость с legacy payload.

---

### 8) Files changed

- `DataProcessor/AudioProcessor/src/extractors/speaker_diarization_extractor/main.py`
- `DataProcessor/AudioProcessor/src/core/npz_savers/speaker_diarization.py`
- `DataProcessor/AudioProcessor/src/extractors/speaker_diarization_extractor/render.py`
- `DataProcessor/AudioProcessor/schemas/speaker_diarization_extractor_npz_v2.json`
- `DataProcessor/AudioProcessor/src/extractors/speaker_diarization_extractor/SCHEMA.md`
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping; models_used for diarization)
- `DataProcessor/AudioProcessor/src/core/model_resolver.py` (diarization-only models_used)
- `DataProcessor/AudioProcessor/src/extractors/speech_analysis_extractor/main.py` (compat with v2 structured diarization)
- `DataProcessor/AudioProcessor/docs/MAIN_INDEX.md`
- `DataProcessor/docs/COMPONENTS_DESC.md`

---

### 9) Follow-ups (required to close audit)

- Сделать validation run на audio-present наборе и добавить запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`:
  - `audio_duration_sec`, `sample_rate`
  - family `diarization`: `N_segments` и длительности (min/p50/p90/max)
  - `speaker_count`, `speaker_turns_count`, `speaker_transitions_count`
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/speaker_diarization_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
