# `tempo_extractor` — engineering log (Audit v4.2)

**Компонент:** `tempo_extractor`  
**Контекст:** Audit v4.2 “engineering bridge” — профилирование/наблюдаемость/оптимизации **после** закрытия эмпирики уровня **L2** (A+B).  

## Связанные документы

- **L2 отчёт (эмпирика A+B):** [`../audio_processor/tempo_extractor_audit_v4.md`](../audio_processor/tempo_extractor_audit_v4.md)
- **L2 статистика (JSON + figures):** `storage/audit_v4/tempo_extractor_l2/tempo_extractor_audit_v4_stats.json` (+ `figures/`)
- **RUN_LOG:** [`../../RUN_LOG.md`](../../RUN_LOG.md)
- **План/критерии:** [`../../AUDIT_4_CRITERIA_AND_PLAN.md`](../../AUDIT_4_CRITERIA_AND_PLAN.md) (см. §12.4 / §12.4.6)

## Что сделано в v4.2

### 1) L2 stats tooling (скрипт статистики)

- Добавлен скрипт: `AudioProcessor/src/extractors/tempo_extractor/scripts/audit_v4_npz_stats.py`
- Артефакты:
  - `storage/audit_v4/tempo_extractor_l2/tempo_extractor_audit_v4_stats.json`
  - `storage/audit_v4/tempo_extractor_l2/figures/` (hist + corr heatmap)

### 2) Тайминги этапов (meta.stage_timings_ms)

**Изменения:**
- `TempoExtractor.run()` пишет замеры (ms) на базе `time.perf_counter()`:
  - `load_audio_ms`, `to_numpy_ms`, `estimate_ms`, `windowed_bpm_ms` (если включено), `build_payload_ms`, `total_ms`
- `TempoExtractor.run_segments()` пишет замеры (ms):
  - `process_segments_ms`, `full_track_ms`, `build_payload_ms`, `total_ms`
- `stage_timings_ms` прокинут в **NPZ meta** (см. пункт 4).

### 3) Профиль ресурсов (meta.tempo_resource_profile) — env-gated

**Включение:** `AP_TEMPO_RESOURCE_PROFILE=1`

**Добавлено:**
- `AudioProcessor/src/extractors/tempo_extractor/utils/resource_profile.py`
  - `is_tempo_resource_profile_enabled()`
  - `capture_tempo_resource_profile(stage=...)` (RSS/VMS + CUDA best-effort)
- `tempo_extractor/__init__.py`: запись `payload["tempo_resource_profile"]` со снапшотами `at_start` / `at_end` (включая `empty` ранние возвраты).

### 4) Протяжка в NPZ meta

**Изменения:**
- `AudioProcessor/src/core/npz_savers/tempo.py`
  - `meta.extra.stage_timings_ms` ← `payload["stage_timings_ms"]`
  - `meta.extra.tempo_resource_profile` ← `payload["tempo_resource_profile"]`

### 5) Документация

- `AudioProcessor/src/extractors/tempo_extractor/docs/README.md`: версия + observability поля.
- `AudioProcessor/src/extractors/tempo_extractor/docs/SCHEMA.md`: добавлены поля observability.

## Версии

- `tempo_extractor` версия: **2.0.0 → 2.0.1** (наблюдаемость Audit v4.2)

## Что НЕ делали (явно)

- Не выполняли длинные прогоны (серии/батчи) для сравнения wall-time/RSS.
- Не делали оптимизации алгоритмов (кроме добавления телеметрии).
- Не делали интеграцию с оркестратором (`scheduler_runtime_report.json`) — отдельный шаг §12.1–12.2.

