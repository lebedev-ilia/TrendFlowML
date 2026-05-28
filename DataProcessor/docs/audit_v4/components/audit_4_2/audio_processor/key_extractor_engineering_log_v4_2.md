# `key_extractor` — инженерный журнал и связь с Audit v4 / **4.2**

**Назначение:** зафиксировать изменения **наблюдаемости/ресурсов** компонента `key_extractor` после эмпирического отчёта Audit v4 (L2), не дублируя таблицы статистики.

**Версия компонента в коде (после правок):** `2.1.1` (`DataProcessor/AudioProcessor/src/extractors/key_extractor/main.py`)

---

## 1. Канонические документы (source-of-truth)

| Документ | Роль |
|----------|------|
| [Отчёт Audit v4 — `key_extractor` (L2)](../audio_processor/key_extractor_audit_v4.md) | Статистика выхода на **A+B**, вердикт |
| [Критерии и план v4](../../AUDIT_4_CRITERIA_AND_PLAN.md) | §3.1 уровни отчёта, §4.* метрики, §12.x ресурсы/скорость |
| [`key_extractor` README](../../../../AudioProcessor/src/extractors/key_extractor/docs/README.md) | Контракт, поля, observability, env‑флаги |
| [`key_extractor` SCHEMA](../../../../AudioProcessor/src/extractors/key_extractor/docs/SCHEMA.md) | NPZ‑ключи и meta‑поля |
| [Журнал прогонов v4](../../RUN_LOG.md) | Ссылки на `result_store`, tooling, A/B/C |

**Артефакты L2 (статистика выхода):**

- `storage/audit_v4/key_extractor_l2/key_extractor_audit_v4_stats.json`
- `storage/audit_v4/key_extractor_l2/figures/`
- Скрипт статистики: `DataProcessor/AudioProcessor/src/extractors/key_extractor/scripts/audit_v4_npz_stats.py`

---

## 2. Что изменено после L2 (Audit 4.2: profiling/observability, без смены контракта)

### 2.1 Наблюдаемость: `meta.stage_timings_ms`

В `run_segments()` добавлены тайминги стадий (мс) в `payload.stage_timings_ms`, а `npz_savers/key.py` прокидывает их в `meta.stage_timings_ms`.

Примеры ключей:

- `process_segments_ms`
- `load_audio_ms_total`
- `detect_key_ms_total`
- `postprocess_ms`
- `total_ms`

### 2.2 Профиль ресурсов (опционально, env‑gated)

Добавлен `meta.key_resource_profile` (best‑effort снимки RSS/VMS и (если доступно) GPU через `torch`):

- включение: `AP_KEY_RESOURCE_PROFILE=1`
- снимки: `*_at_start`, `*_at_end`

Файл утилиты: `DataProcessor/AudioProcessor/src/extractors/key_extractor/utils/resource_profile.py`.

---

## 3. Влияние на аудитные статистики

- Алгоритм key‑детекта **не менялся**; изменения касаются **meta** и диагностики.
- Для полного закрытия Audit 4.2 по скорости/ресурсам нужны реальные прогоны “до/после” с включённым env‑profiling (см. план §12.x).

