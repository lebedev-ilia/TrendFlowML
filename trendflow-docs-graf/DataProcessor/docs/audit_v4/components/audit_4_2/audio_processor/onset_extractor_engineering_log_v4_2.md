# `onset_extractor` — engineering log (Audit v4.2)

**Компонент:** `onset_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/onset_extractor_audit_v4.md`](../audio_processor/onset_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/onset_extractor_l2/onset_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2 (наблюдаемость на уровне компонента)

### 1) Тайминги этапов (meta.stage_timings_ms)

**Изменения:**
- `OnsetExtractor.run()` теперь пишет реальные замеры (ms) на базе `time.perf_counter()`:
  - `load_audio_ms`, `normalize_audio_ms`, `extract_onsets_ms`
  - `compute_metrics_ms`, `save_artifacts_ms`, `validate_output_ms`, `total_ms`

### 2) Профиль ресурсов (meta.onset_resource_profile) — env-gated

**Включение:** `AP_ONSET_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/onset_extractor/utils/resource_profile.py`
  - `resource_profile_enabled()`
  - `snapshot_process_resources()` (RSS/VMS best-effort)
  - `prefix_snapshot()`
- `onset_extractor/main.py`: запись `payload["onset_resource_profile"]` (best-effort) с полями `*_at_start`, `*_at_end`.

### 3) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/onset.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.onset_resource_profile` ← `payload["onset_resource_profile"]`

### 4) Документация

- `AudioProcessor/src/extractors/onset_extractor/docs/README.md`: добавлено описание `meta.stage_timings_ms` + `meta.onset_resource_profile`, версия обновлена.
- `AudioProcessor/src/extractors/onset_extractor/docs/SCHEMA.md`: добавлены поля observability (meta.extra.*).

## Версии

- `onset_extractor` версия: **2.0.0 → 2.0.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не делали длинные прогоны (серии/батчи) для сравнения wall-time/RSS.
- Не делали оптимизации алгоритмов/библиотек (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
