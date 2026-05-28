# `source_separation_extractor` — engineering log (Audit v4.2)

**Компонент:** `source_separation_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/source_separation_extractor_audit_v4.md`](../audio_processor/source_separation_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/source_separation_extractor_l2/source_separation_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Контекст набора A+B (важно для интерпретации статистики)

- В наборе A+B часть прогонов возвращает `status=empty` с `empty_reason=audio_silent`. В этих случаях доли/доминирование по контракту кодируются как `NaN` (а агрегаты по tabular в L2 нужно интерпретировать с учётом `meta.status`).

## Что сделано в v4.2

### 1) L2 stats tooling (скрипт статистики)

- Добавлен скрипт: `AudioProcessor/src/extractors/source_separation_extractor/scripts/audit_v4_npz_stats.py`
- Артефакты:
  - `storage/audit_v4/source_separation_extractor_l2/source_separation_extractor_audit_v4_stats.json`
  - `storage/audit_v4/source_separation_extractor_l2/figures/` (hist + corr heatmap)

### 2) Тайминги этапов (meta.stage_timings_ms)

**Изменения:**
- `SourceSeparationExtractor.run_segments()` теперь пишет `stage_timings_ms` (ms), конвертируя внутренние секции (`*_sec`) в:
  - `load_audio_ms`, `silence_detection_ms`, `padding_ms`, `inference_ms`
  - `postprocess_ms`, `aggregates_ms`, `total_ms`

### 3) Профиль ресурсов (meta.source_separation_resource_profile) — env-gated

**Включение:** `AP_SOURCE_SEPARATION_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/source_separation_extractor/utils/resource_profile.py`
  - `is_source_separation_resource_profile_enabled()`
  - `capture_source_separation_resource_profile(stage=...)` (RSS/VMS + CUDA best-effort)
- `source_separation_extractor/main.py`: запись `payload["source_separation_resource_profile"]` со снапшотами `at_start` / `at_end`.

### 4) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/source_separation.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.source_separation_resource_profile` ← `payload["source_separation_resource_profile"]`

### 5) Документация

- `AudioProcessor/src/extractors/source_separation_extractor/docs/README.md`: обновлена версия и добавлено описание `stage_timings_ms` + `source_separation_resource_profile`.
- `AudioProcessor/src/extractors/source_separation_extractor/docs/SCHEMA.md`: добавлены поля observability.

## Версии

- `source_separation_extractor` версия: **3.0.0 → 3.0.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли длинные прогоны (серии/батчи) для сравнения wall-time/RSS.
- Не делали оптимизации модели/препроцесса (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.

