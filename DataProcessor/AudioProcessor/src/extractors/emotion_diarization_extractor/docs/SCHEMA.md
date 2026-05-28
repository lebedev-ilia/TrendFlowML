## `emotion_diarization_extractor` — SCHEMA (Audit v3)

### Artifact

- **producer**: `emotion_diarization_extractor`
- **artifact_kind**: NPZ
- **schema_version**: `emotion_diarization_extractor_npz_v1`
- **source-of-truth**: NPZ (рендер = dev-only)

### Inputs (contract)

- Segmenter output:
  - `frames_dir/audio/audio.wav` (если `audio_present=true`)
  - `frames_dir/audio/segments.json` (`audio_segments_v1`)
    - **required family**: `emotion` (`families.emotion.segments[]`)
- **No-fallback**:
  - если family `emotion` отсутствует/пустая при запуске extractor → **error**

### Model system (offline / reproducibility)

- Emotion diarization weights загружаются **строго локально** через `dp_models` (ModelManager-only, no-network).
- `meta.models_used[]` + `meta.model_signature` отражают фактически использованную модель.

### Sampling requirements (Audio)

- required family: `emotion`
- extractor **не** поддерживает full-audio fallback в audited режиме: только `run_segments()`.

### Outputs (tiers)

#### model_facing

Tabular (frozen subset в пределах `schema_version`):
- `segments_count` (число валидных сегментов, `sum(segment_mask)`)
- `emotion_entropy`
- `dominant_emotion_id`
- `dominant_emotion_prob`
- `emotion_transitions_count`
- `emotion_stability_score`
- `emotion_diversity_score`

Per-segment sequences (strict-aligned):
- `emotion_id: int32[N]` (для masked сегментов `-1`)
- `emotion_confidence: float32[N]` (для masked сегментов `NaN`)

#### analytics

Time axis + mask (strict alignment):
- `segment_start_sec: float32[N]`
- `segment_end_sec: float32[N]`
- `segment_center_sec: float32[N]`
- `segment_mask: bool[N]`

Labels:
- `emotion_labels: object[C]`

Optional (feature-gated):
- `emotion_probs: float32[N,C]`
- `emotion_mean_probs: float32[C]`
- `emotion_distribution` / `emotion_segments_per_emotion` / `emotion_duration_per_emotion` — в NPZ сохраняются как **scalar `numpy.ndarray` dtype=`object`**, доступ к словарю: `arr.item()` (см. `npz_savers/emotion_diarization.py`).
- `emotion_quality_metrics` — то же (один dict на артефакт).

#### debug

- `meta` (run identity + versions + status + models_used/model_signature + timings)

Опционально (audit v4.2):
- `emotion_diarization_resource_profile` (dict|None): snapshot RSS/VMS/VRAM, включается через `AP_EMOTION_DIARIZATION_RESOURCE_PROFILE=1`

### Empty vs Error semantics

- **empty**:
  - upstream `audio_present=false` → AudioProcessor пишет `status="empty"`
  - `audio_silent` (если включена silence detection)
  - `audio_too_short` (аудио/сегменты < 5 секунд) — valid empty
- **error**:
  - missing/empty `families.emotion.segments` при `audio_present=true`
  - отсутствуют локальные веса модели в `dp_models`
  - неконсистентные shapes/dtypes, невалидные вероятности (NaN/Inf/out-of-range)

### Strict alignment semantics

- Выходные массивы имеют длину `N = len(segments)` из Segmenter.
- Ошибка/пустота сегмента → `segment_mask=false`, значения sequences = `NaN`/`-1`.
- Агрегаты считают только валидные сегменты (`segment_mask=true`).

