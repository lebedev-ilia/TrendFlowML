# `chroma_extractor` — инженерный журнал и связь с Audit v4 / **4.2**

**Назначение:** зафиксировать изменения **наблюдаемости/ресурсов** компонента `chroma_extractor` после эмпирического отчёта Audit v4 (L2), не дублируя таблицы статистики.

**Версия компонента в коде (после правок):** `2.1.1` (`DataProcessor/AudioProcessor/src/extractors/chroma_extractor/main.py`)

---

## 1. Канонические документы (source-of-truth)

| Документ | Роль |
|----------|------|
| [Отчёт Audit v4 — `chroma_extractor` (L2)](../audio_processor/chroma_extractor_audit_v4.md) | Статистика выхода на **A+B**, корреляции tabular, вердикт |
| [Критерии и план v4](../../AUDIT_4_CRITERIA_AND_PLAN.md) | §3.1 уровни отчёта, §4.* метрики, §12.x ресурсы/скорость |
| [`chroma_extractor` README](../../../../AudioProcessor/src/extractors/chroma_extractor/docs/README.md) | Контракт, флаги, поля, мета |
| [`chroma_extractor` SCHEMA](../../../../AudioProcessor/src/extractors/chroma_extractor/docs/SCHEMA.md) | NPZ‑ключи и обязательные meta поля |
| [Журнал прогонов v4](../../RUN_LOG.md) | Ссылки на `result_store`, tooling, A/B/C |

**Артефакты L2 (статистика выхода):**

- `storage/audit_v4/chroma_extractor_l2/chroma_extractor_audit_v4_stats.json`
- `storage/audit_v4/chroma_extractor_l2/figures/`
- Скрипт статистики: `DataProcessor/AudioProcessor/src/extractors/chroma_extractor/scripts/audit_v4_npz_stats.py`

---

## 2. Что изменено после L2 (Audit 4.2: profiling, без смены контракта)

### 2.1 Наблюдаемость: `meta.stage_timings_ms`

В `payload.stage_timings_ms` теперь записываются реальные тайминги стадий (мс), а `npz_savers/chroma.py` прокидывает их в `meta.stage_timings_ms`.

Примеры ключей:

- `run()`: `load_audio_ms`, `normalize_audio_ms` (если включено), `tuning_ms`, `extract_chroma_ms`, `normalize_chroma_ms`, `compute_minimal_ms`, `save_artifacts_ms`, `validate_output_ms`, `total_ms`
- `run_segments()`: `load_segments_ms`, `load_full_audio_ms`, `tuning_ms`, `process_segments_ms`, `aggregate_results_ms`, `save_artifacts_ms`, `validate_output_ms`, `total_ms`

### 2.2 Профиль ресурсов (опционально, env‑gated)

Добавлен `meta.chroma_resource_profile` (best‑effort снимки RSS/VMS и (если доступно) GPU через `torch`):

- включение: `AP_CHROMA_RESOURCE_PROFILE=1`
- снимки: `*_at_start`, `*_at_end`

### 2.3 Микро‑оптимизации без изменения контракта

- Убраны лишние `astype(np.float32)`/копии:
  - `features["_shared_chroma"]` и `features["chroma"]` (debug) больше не дублируют один и тот же массив.
- В сегментном цикле `run_segments()` среднее по кадрам теперь считается с `dtype=np.float32` (меньше аллокаций и конверсий).

---

## 3. Влияние на аудитные статистики

- Алгоритм хромы и набор фич **не менялись**; изменения касаются **meta** и диагностики.
- После внедрения 2.1.1 рекомендуется зафиксировать **golden (§4.8)** на reference A и сверить, что новые `meta.stage_timings_ms`/`meta.chroma_resource_profile` появились в NPZ при включённом env.

