## Audit v3 report — `tempo_extractor` (AudioProcessor)

### 0) TL;DR

Компонент доведён до Audit v3 контракта: per-extractor схема `tempo_extractor_npz_v1`, canonical segment axis (`segment_start_sec`/`segment_end_sec`/`segment_center_sec`/`segment_mask`), `bpm_by_segment` вместо `windowed_bpm`. Partial segment failures → `segment_mask=False`, `bpm_by_segment=NaN`. Empty semantics: `tempo_all_segments_failed` при всех failed. Offline HTML render (vanilla canvas, без CDN).

---

### 1) Ownership / Versions

- **component_name**: `tempo_extractor`
- **producer_version**: `2.0.0`
- **schema_version**: `tempo_extractor_npz_v1`
- **audit_v3_status**: `implemented`

Machine schema:

- `DataProcessor/AudioProcessor/schemas/tempo_extractor_npz_v1.json`

Human schema:

- `DataProcessor/AudioProcessor/src/extractors/tempo_extractor/SCHEMA.md`

---

### 2) Inputs / Sampling

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json`:
  - required: `families.tempo.segments`

---

### 3) Outputs (NPZ = source-of-truth)

#### 3.1 Model-facing (tabular, frozen)

`feature_names`/`feature_values` — скаляры:

- `tempo_bpm`, `tempo_bpm_mean`, `tempo_bpm_median`, `tempo_bpm_std`
- `tempo_confidence`, `duration_sec`, `sample_rate`, `segments_count`
- `tempo_bpm_by_segment_mean`, `tempo_bpm_by_segment_median`, `tempo_bpm_by_segment_std`

Missing → **NaN**.

#### 3.2 Analytics

- canonical axis: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- per-segment: `bpm_by_segment` (float32[N], NaN для failed)
- optional: `tempo_estimates`, `warnings`

#### 3.3 Meta

`meta` включает baseline meta + `status`, `empty_reason`, echo параметров.

---

### 4) Semantics (empty/error)

- **all segments failed**: `status="empty"`, `empty_reason="tempo_all_segments_failed"`
- **partial failures**: `segment_mask[i]=False`, `bpm_by_segment[i]=NaN` для failed сегментов
- **missing segments**: `status="error"` (no-fallback, до вызова extractor)

---

### 5) Renderer

- Offline-only HTML render: без CDN, график BPM vs segment_center_sec через vanilla `<canvas>`.
- Fallback: legacy NPZ с `windowed_times_sec`/`windowed_bpm` поддерживается.

---

### 6) Files changed

- `DataProcessor/AudioProcessor/src/extractors/tempo_extractor/__init__.py` (v2.0.0, canonical axis, partial failures, empty semantics)
- `DataProcessor/AudioProcessor/src/core/npz_savers/tempo.py` (segment_*, bpm_by_segment, legacy fallback)
- `DataProcessor/AudioProcessor/src/extractors/tempo_extractor/render.py` (vanilla canvas, no CDN, segment_center_sec/bpm_by_segment)
- `DataProcessor/AudioProcessor/schemas/tempo_extractor_npz_v1.json` (new)
- `DataProcessor/AudioProcessor/src/extractors/tempo_extractor/SCHEMA.md` (new)
- `DataProcessor/AudioProcessor/src/extractors/tempo_extractor/README.md` (updated)
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping: tempo_extractor_npz_v1)

---

### 7) Downstream

- **onset_extractor**: использует `tempo_bpm` из payload — без изменений.
- **main_processor**: использует `tempo_bpm` — без изменений.
- **rhythmic_extractor**: shared family `tempo`, читает свои BPM — без изменений.

---

### 8) Follow-ups (optional)

- Валидационный прогон на audio-present наборе + запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`.
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/tempo_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
