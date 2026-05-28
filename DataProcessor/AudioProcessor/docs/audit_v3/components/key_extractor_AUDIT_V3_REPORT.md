## Audit v3 report — `key_extractor` (AudioProcessor)

### 0) TL;DR

`key_extractor` переведён на Audit v3 контракт: **`run()` отключён** (только `run_segments()`), **strict alignment** по сегментам Segmenter через `segment_start_sec`/`segment_end_sec`/`segment_center_sec`/`segment_mask`, `key_id_by_segment`/`key_confidence_by_segment` (masked → -1/NaN). Добавлен **key_id** (0–23) model-facing при `n_valid > 0`. Default `key_method` изменён на `librosa`. NPZ saver обновлён под canonical keys; optional keys опускаются при отключённых features. HTML render переписан на offline-only (vanilla canvas, без Plotly CDN). Schema `key_extractor_npz_v1`.

---

### 1) Ownership / Versions

- **component_name**: `key_extractor`
- **owner_processor**: `AudioProcessor`
- **producer**: `key_extractor`
- **producer_version**: `2.1.0`
- **schema_version**: `key_extractor_npz_v1`
- **audit_v3_status**: `passed`

Machine schema:
- `DataProcessor/AudioProcessor/schemas/key_extractor_npz_v1.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/key_extractor/SCHEMA.md`

---

### 2) Inputs

- **Primary input**:
  - `frames_dir/audio/segments.json` (`schema_version="audio_segments_v1"`)
  - `frames_dir/audio/audio.wav`
- **Required sampling family**:
  - `families.key.segments[]`

Принятое решение:
- family `key` остаётся отдельной.

---

### 3) Outputs (NPZ = source-of-truth)

Файл артефакта:
- `result_store/<platform_id>/<video_id>/<run_id>/key_extractor/key_extractor_features.npz`

#### 3.1 Model-facing

Всегда присутствует:
- `segment_start_sec`: `float32[N]`
- `segment_end_sec`: `float32[N]`
- `segment_center_sec`: `float32[N]`
- `segment_mask`: `bool[N]`
- `key_id_by_segment`: `int32[N]` (-1 для failed/masked)
- `key_confidence_by_segment`: `float32[N]` (NaN для failed/masked)

При `n_valid > 0`:
- `meta.key_id`: `int32` (0–23: maj_C=0, min_C=1, …, min_B=23)

Tabular (`feature_names`/`feature_values`):
- `key_name`, `key_mode`, `key_confidence`, `method`, и др.

#### 3.2 Analytics (feature-gated)

- `key_scores`: `float32[24]` (при `enable_detailed_scores`)
- `key_names_sequence`, `key_modes_sequence` (при `enable_time_series`)
- `key_top_k`, `key_transitions`, stability metrics (при соответствующих flags)

#### 3.3 Meta.extra

- `chroma_reused`: `bool` (при переиспользовании chroma от chroma_extractor)
- `key_id`: при `n_valid > 0`
- Остальные optional keys только при включённых features

---

### 4) Empty vs Error semantics

**Valid empty**:
- `segments` пуст → error (no-fallback)

**Error / fail-fast**:
- отсутствует/пуст `families.key.segments` → error
- `run()` отключён → возвращает `ExtractorResult(False, error="run() disabled, use run_segments with families.key")`

Сегменты < 0.5s или failed → `segment_mask=false`, `key_id_by_segment=-1`, `key_confidence_by_segment=NaN`.

---

### 5) Decisions (summary)

- **Sampling family**: `key`
- **Schema rollout**: `key_extractor_npz_v1`
- **run()**: disabled; только `run_segments()`
- **Default key_method**: `librosa` (вместо `auto`)
- **key_id**: 0–23 model-facing при `n_valid > 0`
- **chroma_reused**: в meta.extra при shared chroma
- **Optional keys**: опускаются при отключённых features
- **Render**: offline-only (vanilla canvas, без Plotly CDN)

---

### 6) Files changed (high-level)

- `DataProcessor/AudioProcessor/src/extractors/key_extractor/main.py` (run disabled, strict alignment, key_id)
- `DataProcessor/AudioProcessor/src/core/npz_savers/key.py` (canonical keys, chroma_reused, key_id in meta)
- `DataProcessor/AudioProcessor/src/extractors/key_extractor/render.py` (offline HTML, segment_center_sec support)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (--key-method default=librosa)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (music_key_id from key_id)
- `DataProcessor/AudioProcessor/schemas/key_extractor_npz_v1.json` (key_id in optional_keys)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping key_extractor → key_extractor_npz_v1)
- docs: `MAIN_INDEX.md`, `key_extractor/README.md`, `key_extractor/SCHEMA.md`

---

### 7) Open items / follow-ups

- Добавить запись в `RUN_LOG.md` после первого прогона `key_extractor_npz_v1`.
