## `hpss_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `hpss_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `hpss_extractor_npz_v1`
- **source-of-truth**: NPZ (render = dev-only)

### Inputs (contract)

- Segmenter output:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` (`audio_segments_v1`)
    - **required family**: `hpss` (`families.hpss.segments[]`)
- **No-fallback**: если family `hpss` отсутствует/пустая при запуске extractor → **error**

### Model system

- ML модели **не используются** (librosa signal processing only). `meta.models_used[]` пустой.

### Sampling requirements (Audio)

- required family: `hpss`
- Audit v3: только `run_segments()`; `run()` отключён.

### Outputs (tiers)

#### model_facing (frozen subset)

Tabular (набор строк зависит от включённых флагов и режима `run_segments` vs legacy; порядок см. `npz_savers/hpss.py`):
- `hpss_harmonic_share`, `hpss_percussive_share`
- при полном наборе energy metrics: энергии, stability, `hpss_separation_quality`, `hpss_balance_score`, опционально mean/std долей по сегментам
- `hpss_dominance` (**строка**) не входит в `feature_values` — только **`meta.hpss_dominance`** (или debug meta в савере)

Per-segment sequences (strict-aligned):
- `hpss_harmonic_share_by_segment: float32[N]` (masked → NaN)
- `hpss_percussive_share_by_segment: float32[N]` (masked → NaN)

#### analytics

Time axis + mask:
- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`

Optional (feature-gated):
- `hpss_harmonic_share_series`, `hpss_percussive_share_series` — в **legacy full-clip** пути; **`run_segments()`** глобальные ряды **не строит** (HPSS по окнам). В коде Audit v3 `meta.features_enabled` для **`run_segments()`** не включает `waveforms` и `time_series` (даже если флаги CLI включены).
- `hpss_energy_*`, `hpss_*_stability`, `hpss_separation_quality`
- spectral features (centroid, bandwidth, rolloff)

#### debug

- `meta` (run identity + versions + status + timings)
- waveform paths в `meta.extra` (не массивы в NPZ)

Опционально (audit v4.2):
- `hpss_resource_profile` (dict|None): snapshot RSS/VMS/VRAM, включается через `AP_HPSS_RESOURCE_PROFILE=1`

### Empty vs Error semantics

- **empty**: upstream `audio_present=false` → AudioProcessor пишет `status="empty"`
- **error**: missing/empty `families.hpss.segments` при `audio_present=true`, HPSS/STFT failure

### Strict alignment semantics

- Выходные массивы имеют длину `N = len(segments)` из Segmenter.
- Ошибка сегмента → `segment_mask=false`, `hpss_*_by_segment` = NaN.
- Агрегаты считают только валидные сегменты.
