## `key_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `key_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `key_extractor_npz_v1`
- **source-of-truth**: NPZ (render = dev-only)

### Inputs (contract)

- **Segmenter output**:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` schema `audio_segments_v1`
    - **required family**: `key` (`families.key.segments[]`)
- **Valid empty**: `audio_present=false` → `status="empty"`
- **No-fallback**: missing/empty `families.key.segments` при `audio_present=true` → error

### Model system

- ML модели **не используются** (librosa/Essentia signal processing + Krumhansl-Schmuckler profiles). `meta.models_used[]` пустой.

### Sampling requirements

- required family: `key`
- Audit v3: только `run_segments()`; `run()` отключён.

### Outputs (tiers)

#### model_facing (frozen subset)

Tabular (`feature_names` / `feature_values`, всё **float32** через общий пайп `add`):
- `sample_rate`, `hop_length`, `duration`, **`key_id`** (0–23; в float-таблице), **`key_confidence`**

Строки и категории **не** в tabular (иначе `as_float` дал бы NaN):
- **`key_name`**, **`key_mode`**, **`key_method`** (как в payload `method`), **`key_confidence_category`**, **`key_confidence_reason`**, **`key_low_confidence_warning`** → **`meta`** (ключи `key_name`, `key_mode`, `key_method`, …)
- Дополнительно **`meta.key_id`** дублируется при `key_id >= 0` (как раньше)

Per-segment sequences (strict-aligned):
- `key_id_by_segment`: int32[N] (masked → -1)
- `key_confidence_by_segment`: float32[N] (masked → NaN)

#### analytics

Time axis + mask:
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`

Optional (feature-gated):
- `key_scores`: float32[24]
- `key_names_sequence`, `key_modes_sequence`
- `key_transitions`, `key_stability_score`, `key_distribution`, etc.

#### debug

- `meta` (run identity + versions + status + timings)
- `chroma_reused`: bool в meta.extra (если использован shared chroma)

Опционально (audit v4.2):
- `key_resource_profile` (dict|None): snapshot RSS/VMS/VRAM, включается через `AP_KEY_RESOURCE_PROFILE=1`

### Empty vs Error semantics

- **empty**: upstream `audio_present=false`
- **error**: missing/empty `families.key.segments` при `audio_present=true`, key detection failure

### Strict alignment semantics

- Выходные массивы имеют длину `N = len(segments)` из Segmenter.
- Ошибка сегмента / сегмент < 0.5s → `segment_mask=false`, `key_id_by_segment=-1`, `key_confidence_by_segment=NaN`.
