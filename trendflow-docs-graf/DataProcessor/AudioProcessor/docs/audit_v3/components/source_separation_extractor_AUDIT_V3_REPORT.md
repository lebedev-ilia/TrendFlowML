## Audit v3 report — `source_separation_extractor` (AudioProcessor)

### 0) TL;DR

Компонент доведён до Audit v3: введён per-extractor контракт `source_separation_extractor_npz_v2` (human+machine schema), убраны object-dict поля в NPZ, введён `segment_mask` для silent/zero-energy окон, short audio теперь `empty(audio_too_short)`, NaN/inf от модели — fail-fast (error), HTML render переведён в offline-only режим (vanilla canvas, без Plotly CDN). Model-facing subset зафиксирован как минимальный и стабильный.

---

### 1) Ownership / Versions

- **component_name**: `source_separation_extractor`
- **producer_version**: `3.0.0`
- **schema_version**: `source_separation_extractor_npz_v2`
- **audit_v3_status**: `passed`

Machine schema:
- `DataProcessor/AudioProcessor/schemas/source_separation_extractor_npz_v2.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/source_separation_extractor/SCHEMA.md`

---

### 2) Inputs

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json` family `source_separation` (no-fallback)

---

### 3) Outputs (NPZ = source-of-truth)

#### 3.1 Model-facing (tabular, frozen)

`feature_names/feature_values` фиксируют минимальный набор:

- mean shares: `share_{vocals,drums,bass,other}_mean`
- `dominant_source_id`, `dominant_source_share`
- `source_balance_score`
- `source_transitions_count`, `source_stability_score`
- `segments_count`, `sample_rate`

#### 3.2 Analytics

- canonical axis: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- `share_mean: float32[4]` всегда
- structured per-source stats (no dicts):
  - `source_distribution_ratio: float32[4]`
  - `source_segments_count: int32[4]`
  - `source_duration_sec: float32[4]`
- optional sequences: `share_sequence[N,4]`, `energy_sequence[N,4]`
- optional scalars (если computed): advanced temporal + quality `quality_*`

#### 3.3 Meta

`meta` включает baseline meta + `model_name`, `weights_digest`, `features_enabled`, `source_separation_contract_version`.

---

### 4) Semantics (empty/error)

- **audio too short (<5s)**: `status="empty"`, `empty_reason="audio_too_short"`
- **audio silent**: `status="empty"`, `empty_reason="audio_silent"`
- **invalid model output (NaN/inf/negative energies)**: `status="error"` (fail-fast)
- **silent windows внутри аудио**: `segment_mask=false`, агрегаты считаются по mask

---

### 5) Renderer

- Offline-only HTML render: без CDN, графики через vanilla `<canvas>`.

---

### 6) Files changed

- `DataProcessor/AudioProcessor/src/extractors/source_separation_extractor/main.py`
- `DataProcessor/AudioProcessor/src/core/npz_savers/source_separation.py`
- `DataProcessor/AudioProcessor/src/extractors/source_separation_extractor/render.py`
- `DataProcessor/AudioProcessor/schemas/source_separation_extractor_npz_v2.json`
- `DataProcessor/AudioProcessor/src/extractors/source_separation_extractor/SCHEMA.md`
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (preset/flag note)
- `DataProcessor/AudioProcessor/src/extractors/source_separation_extractor/README.md`
- `DataProcessor/AudioProcessor/docs/MAIN_INDEX.md`

---

### 7) Follow-ups

- Добавить запись в `DataProcessor/docs/audit_v3/RUN_LOG.md` после прогона на audio-present validation set (video*.mp4).
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/source_separation_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
