# `emotion_diarization_extractor` — инженерный журнал и связь с Audit v4 / **4.2**

**Назначение:** зафиксировать изменения **наблюдаемости/ресурсов** компонента `emotion_diarization_extractor` после эмпирического отчёта Audit v4 (L2), не дублируя таблицы статистики.

**Версия компонента в коде (после правок):** `3.1.2` (`DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/main.py`)

---

## 1. Канонические документы (source-of-truth)

| Документ | Роль |
|----------|------|
| [Отчёт Audit v4 — `emotion_diarization_extractor` (L2)](../audio_processor/emotion_diarization_extractor_audit_v4.md) | Статистика выхода на **A+B**, вердикт |
| [Критерии и план v4](../../AUDIT_4_CRITERIA_AND_PLAN.md) | §3.1 уровни отчёта, §4.* метрики, §12.x ресурсы/скорость |
| [`emotion_diarization_extractor` README](../../../../AudioProcessor/src/extractors/emotion_diarization_extractor/docs/README.md) | Контракт, поля, observability, env‑флаги |
| [`emotion_diarization_extractor` SCHEMA](../../../../AudioProcessor/src/extractors/emotion_diarization_extractor/docs/SCHEMA.md) | NPZ‑ключи и meta‑поля |
| [Журнал прогонов v4](../../RUN_LOG.md) | Ссылки на `result_store`, tooling, A/B/C |

**Артефакты L2 (статистика выхода):**

- `storage/audit_v4/emotion_diarization_extractor_l2/emotion_diarization_extractor_audit_v4_stats.json`
- `storage/audit_v4/emotion_diarization_extractor_l2/figures/`
- Скрипт статистики: `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/scripts/audit_v4_npz_stats.py`

---

## 2. Что изменено после L2 (Audit 4.2: profiling/observability, без смены контракта)

### 2.1 Наблюдаемость: `meta.stage_timings_ms`

В `run_segments()` уже было профилирование стадий в секундах (`timings[*_sec]`). В рамках Audit 4.2 добавлено:

- конвертация в `payload.stage_timings_ms` (миллисекунды)
- прокидывание в NPZ meta через `npz_savers/emotion_diarization.py` → `meta.stage_timings_ms`

### 2.2 Профиль ресурсов (опционально, env‑gated)

Добавлен `meta.emotion_diarization_resource_profile` (best‑effort снимки RSS/VMS и (если доступно) GPU через `torch`):

- включение: `AP_EMOTION_DIARIZATION_RESOURCE_PROFILE=1`
- снимки: `*_at_start`, `*_at_end`

Файл утилиты: `DataProcessor/AudioProcessor/src/extractors/emotion_diarization_extractor/utils/resource_profile.py`.

### 2.3 Оптимизации без изменения контракта

- **Silence detection**: убрана сборка большого `concat = np.concatenate(...)` по всем сегментам; RMS/peak теперь считаются **стримингом** по `waves_valid` (меньше RAM и быстрее на длинных клипах).
- **Inference prep**: `wav_lens` теперь клипается **in-place** (`np.clip(..., out=...)`), меньше временных массивов.
- **Scatter back**: заполнение `emotion_id_full`/`emotion_conf_full` и `emotion_probs` (feature‑gated) теперь делается **векторно** по `valid_indices` вместо Python‑цикла.

---

## 3. Влияние на аудитные статистики

- Выходные поля и вычисления эмоций **не менялись**; изменения касаются **meta** и диагностики.
- Для полного закрытия Audit 4.2 по скорости/ресурсам нужны реальные прогоны “до/после” с включённым env‑profiling (см. план §12.x).

