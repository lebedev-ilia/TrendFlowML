# `speaker_diarization_extractor` — engineering log (Audit v4.2)

**Компонент:** `speaker_diarization_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/speaker_diarization_extractor_audit_v4.md`](../audio_processor/speaker_diarization_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/speaker_diarization_extractor_l2/speaker_diarization_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2

### 1) L2 stats tooling (скрипт статистики)

- Добавлен скрипт: `AudioProcessor/src/extractors/speaker_diarization_extractor/scripts/audit_v4_npz_stats.py`
- Артефакты:
  - `storage/audit_v4/speaker_diarization_extractor_l2/speaker_diarization_extractor_audit_v4_stats.json`
  - `storage/audit_v4/speaker_diarization_extractor_l2/figures/` (hist + corr heatmap)

### 2) Тайминги этапов (meta.stage_timings_ms)

**Изменения:**
- `SpeakerDiarizationExtractor.run()` теперь пишет реальные замеры (ms) на базе `time.perf_counter()`:
  - `load_models_ms`, `load_audio_ms`, `to_numpy_ms`, `silence_detection_ms`, `diarize_ms`, `build_payload_ms`, `total_ms`

### 3) Профиль ресурсов (meta.speaker_diarization_resource_profile) — env-gated

**Включение:** `AP_SPEAKER_DIARIZATION_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/speaker_diarization_extractor/utils/resource_profile.py`
  - `is_speaker_diarization_resource_profile_enabled()`
  - `capture_speaker_diarization_resource_profile(stage=...)` (RSS/VMS + CUDA best-effort)
- `speaker_diarization_extractor/main.py`: запись `payload["speaker_diarization_resource_profile"]` со снапшотами `at_start` / `at_end`.

### 4) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/speaker_diarization.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.speaker_diarization_resource_profile` ← `payload["speaker_diarization_resource_profile"]`

### 5) Документация

- `AudioProcessor/src/extractors/speaker_diarization_extractor/docs/README.md`: версия и описание observability полей.
- `AudioProcessor/src/extractors/speaker_diarization_extractor/docs/SCHEMA.md`: добавлены поля observability.

## Версии

- `speaker_diarization_extractor` версия: **3.1.0 → 3.1.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли длинные прогоны (серии/батчи) для сравнения wall-time/RSS.
- Не делали оптимизации алгоритмов/моделей (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.

