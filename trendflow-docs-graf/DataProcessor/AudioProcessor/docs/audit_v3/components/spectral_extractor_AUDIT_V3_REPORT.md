## Audit v3 report — `spectral_extractor` (AudioProcessor)

### 0) TL;DR

Компонент доведён до Audit v3 контракта: per-extractor схема `spectral_extractor_npz_v2`, canonical segment axis (`segment_start_sec`/`segment_end_sec`/`segment_center_sec`/`segment_mask`), per-segment массивы вместо конкатенированных time series. Убран `payload` из NPZ. `enable_basic_features=True` по умолчанию. Empty semantics: `audio_too_short`, `spectral_all_segments_failed`. NaN для missing значений. Offline HTML render (vanilla canvas, без CDN).

---

### 1) Ownership / Versions

- **component_name**: `spectral_extractor`
- **producer_version**: `2.0.0`
- **schema_version**: `spectral_extractor_npz_v2`
- **audit_v3_status**: `in_progress` (нужен validation run + запись в run-log)

Machine schema:

- `DataProcessor/AudioProcessor/schemas/spectral_extractor_npz_v2.json`

Human schema:

- `DataProcessor/AudioProcessor/src/extractors/spectral_extractor/SCHEMA.md`

---

### 2) Inputs / Sampling

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json`:
  - required: `families.spectral.segments`

---

### 3) Outputs (NPZ = source-of-truth)

#### 3.1 Model-facing (tabular, frozen)

`feature_names`/`feature_values` — скаляры из aggregate stats:

- basic: `spectral_centroid_mean/std/min/max/median`, `spectral_bandwidth_*`, `spectral_flatness_*`, `spectral_rolloff_*`, `zcr_*`
- contrast: `spectral_contrast_mean/std/...`, `spectral_contrast_variance`
- advanced: `spectral_slope_*`, `spectral_slope_stability`, `spectral_flatness_db_*`

Missing → **NaN**.

#### 3.2 Analytics

- canonical axis: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- per-segment arrays:
  - `centroid_mean_by_segment`, `bandwidth_mean_by_segment`, `flatness_mean_by_segment`, `rolloff_mean_by_segment`, `zcr_mean_by_segment` (basic_features)
  - `contrast_mean_by_segment` (contrast)
  - `slope_mean_by_segment` (advanced_features)
- optional: `spectral_contrast_bands` (analytics)

#### 3.3 Meta

`meta` включает baseline meta + `spectral_contract_version`, `features_enabled`, echo параметров.

---

### 4) Semantics (empty/error)

- **audio too short (<1s)**: `status="empty"`, `empty_reason="audio_too_short"`
- **all segments failed**: `status="empty"`, `empty_reason="spectral_all_segments_failed"` (+ axis arrays, mask=false)
- **missing required sampling family**: `status="error"` (no-fallback)

---

### 5) Renderer

- Offline-only HTML render: без CDN, графики через vanilla `<canvas>` (centroid_mean_by_segment, flatness_mean_by_segment vs segment_center_sec).

---

### 6) Files changed

- `DataProcessor/AudioProcessor/src/extractors/spectral_extractor/main.py`
- `DataProcessor/AudioProcessor/src/core/npz_savers/spectral.py`
- `DataProcessor/AudioProcessor/src/extractors/spectral_extractor/render.py`
- `DataProcessor/AudioProcessor/schemas/spectral_extractor_npz_v2.json`
- `DataProcessor/AudioProcessor/src/extractors/spectral_extractor/SCHEMA.md`
- `DataProcessor/AudioProcessor/src/extractors/spectral_extractor/README.md` (обновлён)
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (defaults: basic_features enabled)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (default basic_features enabled)

---

### 7) Follow-ups (required to close audit)

- Валидационный прогон на audio-present наборе + запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`:
  - `audio_duration_sec`, `sample_rate`
  - family `spectral`: N segments + stats длительностей
  - доля `segment_mask=false`
  - sanity по диапазонам (centroid, flatness, etc.)
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/spectral_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
