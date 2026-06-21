# `pitch_extractor` — engineering log (Audit v4.2)

**Компонент:** `pitch_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/pitch_extractor_audit_v4.md`](../audio_processor/pitch_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/pitch_extractor_l2/pitch_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2 (наблюдаемость на уровне компонента)

### 1) Тайминги этапов (meta.stage_timings_ms)

**Изменения:**
- `PitchExtractor.run()` теперь пишет реальные замеры (ms) на базе `time.perf_counter()`:
  - `load_audio_ms`, `normalize_audio_ms`, `extract_pitch_ms`
  - `save_artifacts_ms`, `validate_output_ms`, `total_ms`
- `PitchExtractor.run_segments()` теперь пишет:
  - `load_segments_ms` (=0.0, сегменты приходят извне)
  - `process_segments_ms`, `aggregate_results_ms`, `validate_output_ms`, `total_ms`

### 2) Профиль ресурсов (meta.pitch_resource_profile) — env-gated

**Включение:** `AP_PITCH_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/pitch_extractor/utils/resource_profile.py`
  - `resource_profile_enabled()`
  - `snapshot_process_resources()` (RSS/VMS + GPU allocated/reserved если доступно)
  - `prefix_snapshot()`
- `pitch_extractor/main.py`: запись `payload["pitch_resource_profile"]` (best-effort) с полями `*_at_start`, `*_at_end`.

### 3) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/pitch.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.pitch_resource_profile` ← `payload["pitch_resource_profile"]`

### 4) Документация

- `AudioProcessor/src/extractors/pitch_extractor/docs/README.md`: обновлена версия и добавлено описание `meta.stage_timings_ms` + `meta.pitch_resource_profile`.
- `AudioProcessor/src/extractors/pitch_extractor/docs/SCHEMA.md`: добавлены поля observability (meta.extra.*).

## Версии

- `pitch_extractor` версия: **2.0.0 → 2.0.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли длинные прогоны (серии/батчи) для сравнения wall-time/RAM/VRAM.
- Не делали оптимизации алгоритмов/библиотек (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
