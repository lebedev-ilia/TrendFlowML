## Audit v3 report — `rhythmic_extractor` (AudioProcessor)

### 0) TL;DR

`rhythmic_extractor` доведён до Audit v3 контракта: per-extractor схема `rhythmic_extractor_npz_v2`, canonical segment axis (`segment_start_sec`/`segment_end_sec`/`segment_center_sec` + `segment_mask`), required family = `families.tempo` (shared sampling requirement). “No beats detected” больше **не является ошибкой**: компонент возвращает `status="ok"` с `beats_count=0` и `NaN` для tempo/regularity/variation/consistency. Beat events сделаны token-ready: `beat_times_sec[M]` + `beat_segment_index[M]`, с `.npy` fallback (пути в meta). HTML render переведён в offline-only режим (vanilla canvas, без CDN). NPZ больше не хранит `payload`.

---

### 1) Ownership / Versions

- **component_name**: `rhythmic_extractor`
- **owner_processor**: `AudioProcessor`
- **producer**: `rhythmic_extractor`
- **producer_version**: `2.0.0`
- **schema_version**: `rhythmic_extractor_npz_v2`
- **audit_v3_status**: `passed`

Machine schema:
- `DataProcessor/AudioProcessor/schemas/rhythmic_extractor_npz_v2.json`

Human schema:
- `DataProcessor/AudioProcessor/src/extractors/rhythmic_extractor/SCHEMA.md`

---

### 2) Inputs

- Segmenter outputs:
  - `frames_dir/audio/audio.wav`
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - required family: `tempo`
    - migration note: legacy `families.rhythmic` может приниматься временно; факт фиксируется в `meta.sampling_family_used`.

---

### 3) Outputs (NPZ = source-of-truth)

Файл артефакта:
- `result_store/<platform_id>/<video_id>/<run_id>/rhythmic_extractor/rhythmic_extractor_features.npz`

#### 3.1 Model-facing (tabular)

`feature_names`/`feature_values` содержат frozen subset:
- `rhythm_tempo_bpm`, `rhythm_beats_count`, `rhythm_beat_density`
- `rhythm_regularity`, `rhythm_tempo_variation`, `rhythm_beat_consistency`
- `duration_sec`, `sample_rate`, `segments_count`

Semantics:
- если beats не найдены → `rhythm_beats_count=0`, tempo/regularity/variation/consistency = `NaN`.
- если все сегменты упали → `status="empty"`, `empty_reason="rhythmic_all_segments_failed"`, `segment_mask=false` для всех.

#### 3.2 Analytics (canonical segment axis)

- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`: float32[N]
- `segment_mask`: bool[N] (false для failed сегментов)

#### 3.3 Analytics (beat events, token-ready)

Опционально:
- `beat_times_sec`: float32[M]
- `beat_segment_index`: int32[M]

`.npy` fallback (для больших M):
- `meta.beat_times_sec_npy`, `meta.beat_segment_index_npy`

#### 3.4 Debug-only

- `meta`: baseline meta + `backend`, `hop_length`, `features_enabled`, `sampling_family_used` + пути `.npy`.

---

### 4) Renderer

- Offline-only HTML (без CDN): beat timeline на `<canvas>`, сводка model-facing и interval stats, raw render-context JSON.

---

### 5) Files changed (high-level)

- `DataProcessor/AudioProcessor/src/extractors/rhythmic_extractor/main.py` (preset defaults, no-beats ok, strict alignment + mask, token-ready beats)
- `DataProcessor/AudioProcessor/src/core/npz_savers/rhythmic.py` (Audit v3 NPZ: no payload, new keys)
- `DataProcessor/AudioProcessor/src/extractors/rhythmic_extractor/render.py` (offline render, no payload)
- `DataProcessor/AudioProcessor/schemas/rhythmic_extractor_npz_v2.json` (new machine schema)
- `DataProcessor/AudioProcessor/src/extractors/rhythmic_extractor/SCHEMA.md` (new human schema)
- `DataProcessor/AudioProcessor/run_cli.py` (schema mapping)
- `DataProcessor/AudioProcessor/src/core/segments_loader.py` + `src/core/extractor_runner.py` (family `tempo` requirement + migration)
- docs sync: `AudioProcessor/docs/MAIN_INDEX.md`, `rhythmic_extractor/README.md`

---

### 6) Open items / follow-ups

- Добавить запись в `DataProcessor/docs/audit_v3/RUN_LOG.md` после реального прогона на audio-present validation pack.
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/rhythmic_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
