# `clap_extractor` — инженерный журнал и связь с Audit v4 / **4.2**

**Назначение:** зафиксировать изменения **наблюдаемости/ресурсов** компонента `clap_extractor` после эмпирического отчёта Audit v4 (L2), не дублируя таблицы статистики.

**Версия компонента в коде (после правок):** `1.1.1` (`DataProcessor/AudioProcessor/src/extractors/clap_extractor/__init__.py`)

---

## 1. Канонические документы (source-of-truth)

| Документ | Роль |
|----------|------|
| [Отчёт Audit v4 — `clap_extractor` (L2)](../audio_processor/clap_extractor_audit_v4.md) | Статистика выхода на **A+B**, вердикт |
| [Критерии и план v4](../../AUDIT_4_CRITERIA_AND_PLAN.md) | §3.1 уровни отчёта, §4.* метрики, §12.x ресурсы/скорость |
| [`clap_extractor` README](../../../../AudioProcessor/src/extractors/clap_extractor/docs/README.md) | Контракт, поля, observability, env‑флаги |
| [`clap_extractor` SCHEMA](../../../../AudioProcessor/src/extractors/clap_extractor/docs/SCHEMA.md) | NPZ‑ключи и meta‑поля (в т.ч. `stage_timings_ms`) |
| [Журнал прогонов v4](../../RUN_LOG.md) | Ссылки на `result_store`, tooling, A/B/C |

**Артефакты L2 (статистика выхода):**

- `storage/audit_v4/clap_extractor_l2/clap_extractor_audit_v4_stats.json`
- `storage/audit_v4/clap_extractor_l2/figures/`
- Скрипт статистики: `DataProcessor/AudioProcessor/src/extractors/clap_extractor/scripts/audit_v4_npz_stats.py`

---

## 2. Что изменено после L2 (Audit 4.2: profiling/observability, без смены контракта)

### 2.1 Наблюдаемость: `meta.stage_timings_ms`

В `run_segments()` добавлены реальные тайминги стадий (мс) в `payload.stage_timings_ms`, а `npz_savers/clap.py` прокидывает их в `meta.stage_timings_ms`.

Примеры ключей:

- `preprocess_segments_ms`
- `inference_ms`
- `build_outputs_ms`
- `robust_aggregate_ms`
- `total_ms`

### 2.2 Профиль ресурсов (опционально, env‑gated)

Добавлен `meta.clap_resource_profile` (best‑effort снимки RSS/VMS и (если доступно) GPU через `torch`):

- включение: `AP_CLAP_RESOURCE_PROFILE=1`
- снимки: `*_at_start`, `*_at_end`

Файл утилиты: `DataProcessor/AudioProcessor/src/extractors/clap_extractor/utils/resource_profile.py`.

---

## 3. Влияние на аудитные статистики

- Алгоритм извлечения эмбеддингов и табличные фичи **не менялись**; изменения касаются **meta** и диагностики.
- Для полного закрытия Audit 4.2 по скорости/ресурсам нужны реальные прогоны “до/после” с включённым env‑profiling (см. план §12.x).
---

## Навигация

[Audit v4 hub](../README.md) · [DataProcessor](../../../../MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
