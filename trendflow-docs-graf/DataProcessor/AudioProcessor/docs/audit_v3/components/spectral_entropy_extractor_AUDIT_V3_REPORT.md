## Audit v3 report — `spectral_entropy_extractor` (AudioProcessor)

### 0) TL;DR

Компонент доведён до Audit v3 контракта: зафиксирован shared sampling family `spectral`, введён per-extractor контракт `spectral_entropy_extractor_npz_v2` (human+machine schema), убран `payload` object из NPZ, time-series по кадрам не используется (исключён риск некорректной конкатенации); вместо этого добавлены **per-segment массивы** (`entropy_*_by_segment`) + `segment_mask`. Принята NaN-политика для missing значений (без нулевых заглушек). Empty semantics: short audio → `empty(audio_too_short)`, all segments failed → `empty(spectral_entropy_all_segments_failed)`. HTML render переписан в offline-only режим (vanilla canvas, без Plotly CDN).

---

### 1) Ownership / Versions

- **component_name**: `spectral_entropy_extractor`
- **producer_version**: `2.0.0`
- **schema_version**: `spectral_entropy_extractor_npz_v2`
- **audit_v3_status**: `in_progress` (нужен validation run + запись в run-log)

Machine schema:

- `DataProcessor/AudioProcessor/schemas/spectral_entropy_extractor_npz_v2.json`

Human schema:

- `DataProcessor/AudioProcessor/src/extractors/spectral_entropy_extractor/SCHEMA.md`

---

### 2) Inputs / Sampling

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json`:
  - required: `families.spectral.segments` (shared-family policy)

---

### 3) Outputs (NPZ = source-of-truth)

#### 3.1 Model-facing (tabular, frozen)

`feature_names/feature_values` фиксируют минимальный набор:

- `spectral_entropy_mean`
- `spectral_entropy_std`

Missing значения кодируются как **NaN**.

#### 3.2 Analytics

- canonical axis: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- per-segment arrays:
  - required: `entropy_mean_by_segment[N]`, `entropy_std_by_segment[N]`
  - optional: `entropy_min_by_segment[N]`, `entropy_max_by_segment[N]` (extended_stats)
  - optional: `flatness_*_by_segment[N]`, `spread_*_by_segment[N]`

#### 3.3 Meta

`meta` включает baseline meta +:

- `spectral_entropy_contract_version`
- `features_enabled`
- echo параметров (sample_rate/n_fft/hop/use_mel/…)

---

### 4) Semantics (empty/error)

- **audio too short (<1s)**: `status="empty"`, `empty_reason="audio_too_short"`
- **all segments failed**: `status="empty"`, `empty_reason="spectral_entropy_all_segments_failed"` (+ axis arrays, mask=false)
- **missing required sampling family**: `status="error"` (no-fallback)

---

### 5) Renderer

- Offline-only HTML render: без CDN, графики через vanilla `<canvas>` (line plot entropy_mean_by_segment vs segment_center_sec).

---

### 6) Files changed

- `DataProcessor/AudioProcessor/src/extractors/spectral_entropy_extractor/main.py`
- `DataProcessor/AudioProcessor/src/core/npz_savers/spectral_entropy.py`
- `DataProcessor/AudioProcessor/src/extractors/spectral_entropy_extractor/render.py`
- `DataProcessor/AudioProcessor/schemas/spectral_entropy_extractor_npz_v2.json`
- `DataProcessor/AudioProcessor/src/extractors/spectral_entropy_extractor/SCHEMA.md`
- `DataProcessor/AudioProcessor/src/extractors/spectral_entropy_extractor/README.md`
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (defaults: basic stats enabled)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (default basic stats enabled)
- `DataProcessor/AudioProcessor/docs/MAIN_INDEX.md`
- `DataProcessor/docs/COMPONENTS_DESC.md`

---

### 7) Follow-ups (required to close audit)

- Валидационный прогон на audio-present наборе + запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`:
  - `audio_duration_sec`, `sample_rate`
  - family `spectral`: N segments + stats длительностей
  - доля `segment_mask=false`
  - sanity по диапазону entropy (finite, не отрицательная)
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/spectral_entropy_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
