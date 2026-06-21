# `spectral_extractor` — engineering log (Audit v4.2)

**Компонент:** `spectral_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/spectral_extractor_audit_v4.md`](../audio_processor/spectral_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/spectral_extractor_l2/spectral_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2

### 1) L2 stats tooling (скрипт статистики)

- Добавлен скрипт: `AudioProcessor/src/extractors/spectral_extractor/scripts/audit_v4_npz_stats.py`
- Артефакты:
  - `storage/audit_v4/spectral_extractor_l2/spectral_extractor_audit_v4_stats.json`
  - `storage/audit_v4/spectral_extractor_l2/figures/` (hist + corr heatmap)

### 2) Тайминги этапов (meta.stage_timings_ms)

**Изменения:**
- `SpectralExtractor.run()` пишет реальные замеры (ms) на базе `time.perf_counter()`:
  - `load_audio_ms`, `normalize_audio_ms`, `extract_features_ms`
  - `save_artifacts_ms`, `validate_output_ms`, `total_ms`
- `SpectralExtractor.run_segments()` пишет:
  - `process_segments_ms`, `aggregate_results_ms`, `validate_output_ms`, `total_ms`

### 3) Профиль ресурсов (meta.spectral_resource_profile) — env-gated

**Включение:** `AP_SPECTRAL_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/spectral_extractor/utils/resource_profile.py`
  - `is_spectral_resource_profile_enabled()`
  - `capture_spectral_resource_profile(stage=...)` (RSS/VMS + CUDA best-effort)
- `spectral_extractor/main.py`: запись `payload["spectral_resource_profile"]` со снапшотами `at_start` / `at_end`.

### 4) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/spectral.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.spectral_resource_profile` ← `payload["spectral_resource_profile"]`

### 5) Документация

- `AudioProcessor/src/extractors/spectral_extractor/docs/README.md`: обновлена версия и добавлено описание observability полей.
- `AudioProcessor/src/extractors/spectral_extractor/docs/SCHEMA.md`: добавлены поля observability.

## Версии

- `spectral_extractor` версия: **2.0.0 → 2.0.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли длинные прогоны (серии/батчи) для сравнения wall-time/RSS.
- Не делали оптимизации алгоритмов (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
