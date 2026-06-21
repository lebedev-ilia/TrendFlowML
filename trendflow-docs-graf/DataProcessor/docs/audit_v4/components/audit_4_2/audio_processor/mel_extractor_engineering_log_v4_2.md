# `mel_extractor` — engineering log (Audit v4.2)

**Компонент:** `mel_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/mel_extractor_audit_v4.md`](../audio_processor/mel_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/mel_extractor_l2/mel_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2 (наблюдаемость на уровне компонента)

### 1) Тайминги этапов (meta.stage_timings_ms)

**Зачем:** сделать измеримыми ключевые стадии выполнения, чтобы сравнивать “до/после” и ловить регрессии.

**Изменения:**
- `MelExtractor.run()` теперь заполняет `payload["stage_timings_ms"]` реальными замерами (ms) на базе `time.perf_counter()`:
  - `load_audio_ms`, `normalize_audio_ms`, `to_device_ms`
  - `extract_mel_ms`, `compute_statistics_ms`, `compute_spectral_features_ms`, `compute_additional_metrics_ms`
  - `build_payload_ms`, `save_artifacts_ms`, `validate_output_ms`, `total_ms`
- `MelExtractor.run_segments()` заполняет:
  - `load_segments_ms` (=0.0, сегменты приходят извне)
  - `process_segments_ms` (цикл по сегментам: load → mel → метрики)
  - `aggregate_results_ms` (агрегация по валидным сегментам)
  - `save_artifacts_ms`, `validate_output_ms`, `total_ms`

### 2) Профиль ресурсов (meta.mel_resource_profile) — env-gated

**Зачем:** быстрые, дешёвые “снимки” RAM/VRAM в начале/конце выполнения.

**Включение:** `AP_MEL_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/mel_extractor/utils/resource_profile.py`
  - `resource_profile_enabled()`
  - `snapshot_process_resources()` (RSS/VMS + GPU allocated/reserved если доступно)
  - `prefix_snapshot()`
- `mel_extractor/main.py`: запись `payload["mel_resource_profile"]` с полями `*_at_start`, `*_at_end` (best-effort).

### 3) Протяжка в NPZ meta

**Зачем:** чтобы тайминги/ресурсы были доступны из артефакта `result_store` без внешних логов.

**Изменения:**
- `AudioProcessor/src/core/npz_savers/mel.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.mel_resource_profile` ← `payload["mel_resource_profile"]`

### 4) Документация

- `AudioProcessor/src/extractors/mel_extractor/docs/README.md`: обновлена версия и добавлено описание `meta.stage_timings_ms` + `meta.mel_resource_profile`.
- `AudioProcessor/src/extractors/mel_extractor/docs/SCHEMA.md`: добавлены поля observability (meta.extra.*).

## Версии

- `mel_extractor` версия: **2.1.0 → 2.1.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли “долгий прогон” (серии/батчи) для сравнения wall-time/RAM/VRAM.
- Не делали оптимизации алгоритмов/библиотек (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — это отдельный шаг §12.1–12.2.

## Следующие шаги (если закрываем Audit v4.2 полностью)

- Прогон A (и/или набор из 2+ видео) с `AP_MEL_RESOURCE_PROFILE=1` и сравнением `stage_timings_ms`/RSS/VRAM.
- Golden (§4.8) по свежему NPZ (после фиксов/версионирования), если нужно “закрывать” регрессионный след.
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
