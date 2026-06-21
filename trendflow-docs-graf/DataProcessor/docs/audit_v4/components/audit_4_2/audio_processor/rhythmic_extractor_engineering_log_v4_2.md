# `rhythmic_extractor` — engineering log (Audit v4.2)

**Компонент:** `rhythmic_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/rhythmic_extractor_audit_v4.md`](../audio_processor/rhythmic_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/rhythmic_extractor_l2/rhythmic_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2

### 1) L2 stats tooling (скрипт статистики)

- Добавлен скрипт: `AudioProcessor/src/extractors/rhythmic_extractor/scripts/audit_v4_npz_stats.py`
- Артефакты:
  - `storage/audit_v4/rhythmic_extractor_l2/rhythmic_extractor_audit_v4_stats.json`
  - `storage/audit_v4/rhythmic_extractor_l2/figures/` (hist + corr heatmap)

### 2) Тайминги этапов (meta.stage_timings_ms)

**Изменения:**
- `RhythmicExtractor.run()` теперь пишет реальные замеры (ms) на базе `time.perf_counter()`:
  - `load_audio_ms`, `normalize_audio_ms`, `beat_track_ms`, `compute_metrics_ms`
  - `save_artifacts_ms`, `validate_output_ms`, `total_ms`
- `RhythmicExtractor.run_segments()` теперь пишет:
  - `process_segments_ms`, `aggregate_metrics_ms`
  - `save_artifacts_ms`, `validate_output_ms`, `total_ms`

### 3) Профиль ресурсов (meta.rhythmic_resource_profile) — env-gated

**Включение:** `AP_RHYTHMIC_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/rhythmic_extractor/utils/resource_profile.py`
  - `is_rhythmic_resource_profile_enabled()`
  - `capture_rhythmic_resource_profile(stage=...)` (RSS/VMS + CUDA best-effort)
- `rhythmic_extractor/main.py`: запись `payload["rhythmic_resource_profile"]` со снапшотами `at_start` / `at_end`.

### 4) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/rhythmic.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.rhythmic_resource_profile` ← `payload["rhythmic_resource_profile"]`

### 5) Документация

- `AudioProcessor/src/extractors/rhythmic_extractor/docs/README.md`: обновлена версия и добавлено описание `stage_timings_ms` + `rhythmic_resource_profile`.
- `AudioProcessor/src/extractors/rhythmic_extractor/docs/SCHEMA.md`: добавлены поля observability.

## Версии

- `rhythmic_extractor` версия: **2.0.0 → 2.0.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли длинные прогоны (серии/батчи) для сравнения wall-time/RSS.
- Не делали оптимизации алгоритмов/библиотек (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
