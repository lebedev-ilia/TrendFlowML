## `speaker_diarization_extractor` (Speaker diarization) — Audit v3

### Назначение

Диаризация спикеров: определить **кто говорит когда** на полном аудио и опубликовать:

- **token-ready speaker turns** (плоские массивы `turn_*`)
- компактные стабильные агрегаты (через `feature_names/feature_values`)
- структурные per-speaker массивы (`speaker_*`)

### Audit v3 policy (фиксировано)

- **No-network runtime**: модели грузятся **только** через `dp_models` (ModelManager). Любые HuggingFace/whisperx downloads запрещены.
- **Privacy**: raw transcript/words **не извлекаются** и **не сохраняются**.
- **Sampling**: Segmenter-owned family `diarization` с **ровно одним** full-audio окном.
- **NPZ source-of-truth**: per-extractor schema `speaker_diarization_extractor_npz_v2`, `allow_extra_keys=false`.

### Версии

- **producer_version**: `3.1.1`
- **schema_version**: `speaker_diarization_extractor_npz_v2`
- **human schema**: `SCHEMA.md`

### Inputs

- `frames_dir/audio/audio.wav`
- `frames_dir/audio/segments.json`:
  - required: `families.diarization.segments`
  - required: ровно 1 сегмент (`start_sec`, `end_sec`) покрывает полный аудио-интервал

### Outputs (NPZ)

#### Audit v4 — заметки по NPZ

- **Tabular** на reference **A**: **F=10**, NaN **0**; строки (**`device_used`**, **`model_name`**, **`weights_digest`**) — в **`meta`**, не в tabular.
- Порядок скаляров и состав — как в `npz_savers/speaker_diarization.py` (включает **`sample_rate`**, **`rms`**, **`peak`**).
- Observability (audit v4.2): `meta.stage_timings_ms`, `meta.speaker_diarization_resource_profile` (env: `AP_SPEAKER_DIARIZATION_RESOURCE_PROFILE=1`)

#### 1) Model-facing (tabular)

`feature_names: object[F]`, `feature_values: float32[F]` — замороженный набор (**F=10**):

- `speaker_count`, `duration_sec`, `sample_rate`, `rms`, `peak`
- `speaker_balance_score`, `dominant_speaker_id`
- `speaker_turns_count`, `speaker_turns_density`, `speaker_transitions_count`

#### 2) Analytics arrays

- **canonical axis (Segmenter window)**: `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask` (ожидаемо \(N=1\))
- **turns (token-ready)**: `turn_start_sec`, `turn_end_sec`, `turn_speaker_id`, `turn_mask` (длина \(K\))
- **per-speaker**: `speaker_ids`, `speaker_duration_sec`, `speaker_time_ratio`, `speaker_turns_count_by_speaker` (длина \(S=speaker_count\))

### Empty / Error semantics

- **`empty(audio_silent)`**: silence detection triggers
- **`empty(audio_missing_or_extract_failed)`**: upstream reports no audio
- **`error`**: ModelManager cannot load `pyannote_speaker_diarization` or runtime failure (fail-fast)

### Render (dev-only)

Offline-only HTML render (без CDN): vanilla `<canvas>` для timeline turns и bar-chart time ratios.

### Performance (GPU placement, 2026-04)

- После загрузки пайплайна из ModelManager (`pyannote_speaker_diarization`, inprocess) экстрактор **переносит модуль на `cuda`**, если `device`/`auto` выбрали GPU. Раньше из-за раннего `return` после `get()` блок `.to(cuda)` не выполнялся, и инференс часто оставался на CPU при доступной видеокарте (сильный регресс по wall time на длинном аудио).
- При **CUDA OOM** сохраняется прежняя политика: перенос пайплайна и волны на **CPU** и повторный прогон (**тот же** чекпоинт и контракт выхода). После успешного fallback поле **`device_used`** в payload отражает **`cpu`**.
- Детали в контексте E2E: [`backend/docs/E2E_WORKLOG_VISUAL_SEMANTICS_2026-04.md`](../../../../../../backend/docs/E2E_WORKLOG_VISUAL_SEMANTICS_2026-04.md) (§ 3.7).

