## Audit v3 report — `quality_extractor` (AudioProcessor)

### 0) TL;DR

`quality_extractor` доведён до Audit v3 контракта: `run_segments()` работает по Segmenter окнам (`families.primary`), выдаёт **canonical segment axis** (`segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`). Удалены дубликаты `snr_db` (был = `dynamic_range_db`) и `dc_offset_abs` (= `abs(dc_offset)`). `enable_basic_metrics=True` по умолчанию. Empty semantics: при всех пустых сегментах → `status="empty"`, `empty_reason="quality_all_segments_empty"`. Time series только в `.npy`, пути в `meta`. Render: `feature_names`/`feature_values`, vanilla canvas (no CDN). Machine schema: `quality_extractor_npz_v2`.

---

### 1) Ownership / Versions

- **component_name**: `quality_extractor`
- **owner_processor**: `AudioProcessor`
- **producer**: `quality_extractor`
- **producer_version**: `2.0.0`
- **schema_version**: `quality_extractor_npz_v2`
- **audit_v3_status**: `passed`

Machine schema:
- `DataProcessor/AudioProcessor/schemas/quality_extractor_npz_v2.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/quality_extractor/SCHEMA.md`

---

### 2) Inputs

- **Segmenter output**:
  - `frames_dir/audio/audio.wav`
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - required family: `primary`

Empty vs error:
- все сегменты пустые → `status="empty"`, `empty_reason="quality_all_segments_empty"`
- иначе — нормальная обработка

---

### 3) Outputs (NPZ = source-of-truth)

Файл артефакта:
- `result_store/<platform_id>/<video_id>/<run_id>/quality_extractor/quality_extractor_features.npz`

#### 3.1 Model-facing (tabular)

`feature_names`/`feature_values` — feature-gated:
- **Always**: `sample_rate`, `device_used`, `average_channels`, `frame_len_ms`, `hop_ms`, `clip_threshold`, `duration`, `segments_count`
- **basic_metrics** (default on): `dc_offset`, `clipping_ratio`, `crest_factor_db`, `clipping_segments_count`, `quality_score`, `crest_factor_median`
- **dynamic_metrics**: `dynamic_range_db`, `dynamic_range_stability`
- **frame_analysis**: `frame_levels_mean/std/min/max/median`

Removed: `snr_db`, `dc_offset_abs`, `snr_stability`, `dc_offset_stability`, `clipping_severity`, `crest_factor_stability`

#### 3.2 Analytics (canonical segment axis)

- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: float32[N]
- `segment_mask`: bool[N]

#### 3.3 Time series

- Только в `.npy`; пути в `meta`: `*_series_npy` keys
- Не в NPZ payload

#### 3.4 Debug-only

- `meta`: baseline meta + observability + time series paths

---

### 4) Renderer

- Offline-only HTML (без CDN): `feature_names`/`feature_values`, arrays из NPZ, vanilla canvas для DC offset, clipping ratio, crest factor, dynamic range.

---

### 5) Files changed (high-level)

- `DataProcessor/AudioProcessor/src/extractors/quality_extractor/main.py` (preset, remove snr/dc_offset_abs, empty semantics, canonical axis)
- `DataProcessor/AudioProcessor/src/core/npz_savers/quality.py` (Audit v3 NPZ)
- `DataProcessor/AudioProcessor/src/extractors/quality_extractor/render.py` (feature_names/feature_values, vanilla canvas)
- `DataProcessor/AudioProcessor/schemas/quality_extractor_npz_v2.json` (new machine schema)
- `DataProcessor/AudioProcessor/src/extractors/quality_extractor/SCHEMA.md` (new)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (--quality-disable-basic-metrics, default True)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (remove quality_snr_db from baseline)
- docs: `README.md`, `docs/MAIN_INDEX.md`

---

### 6) Open items / follow-ups

- Добавить запись в `RUN_LOG.md` после первого реального прогона `quality_extractor_npz_v2`.
