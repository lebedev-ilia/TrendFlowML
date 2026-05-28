# `mfcc_extractor` — engineering log (Audit v4.2)

**Компонент:** `mfcc_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/mfcc_extractor_audit_v4.md`](../audio_processor/mfcc_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/mfcc_extractor_l2/mfcc_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2 (наблюдаемость на уровне компонента)

### 1) Тайминги этапов (meta.stage_timings_ms)

**Зачем:** измеримость стадий и регресс‑контроль.

**Изменения:**
- `MFCCExtractor.run()` теперь заполняет `payload["stage_timings_ms"]` реальными замерами (ms) на базе `time.perf_counter()`:
  - `load_audio_ms`, `normalize_audio_ms`, `to_device_ms`
  - `extract_mfcc_ms`, `compute_statistics_ms`, `compute_additional_metrics_ms`
  - `build_payload_ms`, `save_artifacts_ms`, `validate_output_ms`, `total_ms`
- `MFCCExtractor.run_segments()` теперь заполняет:
  - `load_segments_ms` (=0.0, сегменты приходят извне)
  - `process_segments_ms`, `aggregate_results_ms`, `save_artifacts_ms`, `validate_output_ms`, `total_ms`

### 2) Профиль ресурсов (meta.mfcc_resource_profile) — env-gated

**Включение:** `AP_MFCC_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/mfcc_extractor/utils/resource_profile.py`
  - `resource_profile_enabled()`
  - `snapshot_process_resources()` (RSS/VMS + GPU allocated/reserved если доступно)
  - `prefix_snapshot()`
- `mfcc_extractor/main.py`: запись `payload["mfcc_resource_profile"]` (best-effort) с полями `*_at_start`, `*_at_end`.

### 3) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/mfcc.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.mfcc_resource_profile` ← `payload["mfcc_resource_profile"]`

### 4) Документация

- `AudioProcessor/src/extractors/mfcc_extractor/docs/README.md`: обновлена версия и добавлено описание `meta.stage_timings_ms` + `meta.mfcc_resource_profile`.
- `AudioProcessor/src/extractors/mfcc_extractor/docs/SCHEMA.md`: добавлены поля observability (meta.extra.*).

## Версии

- `mfcc_extractor` версия: **2.1.0 → 2.1.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли длинные прогоны (серии/батчи) для сравнения wall-time/RAM/VRAM.
- Не делали оптимизации алгоритмов/библиотек (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.

## Следующие шаги (если закрываем Audit v4.2 полностью)

- Прогон A/серии с `AP_MFCC_RESOURCE_PROFILE=1` и сравнением `stage_timings_ms`/RSS/VRAM.
- Golden (§4.8) по свежему NPZ, если нужно регрессионное закрепление после фиксов.

