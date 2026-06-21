## Audit v3 report — `voice_quality_extractor` (AudioProcessor)

### 0) TL;DR

Компонент доведён до Audit v3 контракта: per-extractor схема `voice_quality_extractor_npz_v1`, canonical segment axis (`segment_start_sec`/`segment_end_sec`/`segment_center_sec`/`segment_mask`), per-segment массивы (`jitter_by_segment`, `shimmer_by_segment`, `hnr_by_segment`). Partial segment failures → `segment_mask=False`, NaN в per-segment. Empty semantics: `voice_quality_all_segments_failed` при всех failed. Feature preset: jitter+shimmer+hnr по умолчанию. Offline HTML render (vanilla canvas, без CDN). main_processor: NaN для missing фичей.

---

### 1) Ownership / Versions

- **component_name**: `voice_quality_extractor`
- **producer_version**: `3.0.0`
- **schema_version**: `voice_quality_extractor_npz_v1`
- **audit_v3_status**: `implemented`

Machine schema:

- `DataProcessor/AudioProcessor/schemas/voice_quality_extractor_npz_v1.json`

Human schema:

- `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/SCHEMA.md`

---

### 2) Inputs / Sampling

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json`:
  - required: `families.voice_quality.segments`

---

### 3) Outputs (NPZ = source-of-truth)

#### 3.1 Model-facing (tabular, frozen)

`feature_names`/`feature_values` — скаляры (feature-gated):

- jitter: `vq_jitter`, `vq_jitter_mean`, `vq_jitter_std`, `vq_jitter_min`, `vq_jitter_max`
- shimmer: `vq_shimmer`, ...
- hnr: `vq_hnr_like_db`, ...
- f0_stats: `vq_f0_mean`, `vq_f0_std`, `vq_f0_min`, `vq_f0_max`, `vq_f0_median`, `vq_f0_stability`, `vq_voice_presence_ratio`
- quality: `vq_voice_quality_score`, `vq_breathiness_score` (если jitter+shimmer+hnr)

Missing → **NaN**.

#### 3.2 Analytics

- canonical axis: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`
- per-segment: `jitter_by_segment`, `shimmer_by_segment`, `hnr_by_segment` (float32[N], NaN для failed)

#### 3.3 Meta

`meta` включает baseline meta + `status`, `empty_reason`, `features_enabled`, `f0_method`.

---

### 4) Semantics (empty/error)

- **all segments failed**: `status="empty"`, `empty_reason="voice_quality_all_segments_failed"`
- **partial failures**: `segment_mask[i]=False`, NaN в per-segment для failed сегментов
- **missing segments**: `status="error"` (no-fallback, до вызова extractor)

---

### 5) Renderer

- Offline-only HTML render: без CDN, график jitter vs segment_center_sec через vanilla `<canvas>`.

---

### 6) Files changed

- `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/main.py` (v3.0.0, canonical axis, partial failures, empty semantics, per-segment arrays, feature preset)
- `DataProcessor/AudioProcessor/src/core/npz_savers/voice_quality.py` (segment_*, *_by_segment, flat keys)
- `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/render.py` (vanilla canvas, no CDN, flat keys)
- `DataProcessor/AudioProcessor/schemas/voice_quality_extractor_npz_v1.json` (new)
- `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/SCHEMA.md` (new)
- `DataProcessor/AudioProcessor/src/extractors/voice_quality_extractor/README.md` (updated)
- `DataProcessor/AudioProcessor/run_cli.py` (schema_version mapping: voice_quality_extractor_npz_v1)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (NaN для missing vq_*)
- `DataProcessor/AudioProcessor/src/core/cli_args.py` (enable jitter/shimmer/hnr default True, disable flags)
- `DataProcessor/AudioProcessor/src/core/main_processor.py` (defaults for enable flags)

---

### 7) Downstream

- **pitch_extractor**: опциональная интеграция — voice_quality загружает f0 из pitch при совпадении families.
- **main_processor**: flat_payload с NaN для missing vq_jitter/vq_shimmer/vq_hnr_like_db.

---

### 8) Follow-ups (optional)

- Валидационный прогон на audio-present наборе + запись в `DataProcessor/docs/audit_v3/RUN_LOG.md`.
---

## Навигация

[Audit v3 index](../README.md) · [Extractor README](../../../src/extractors/voice_quality_extractor/docs/README.md) · [AudioProcessor](../../MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
