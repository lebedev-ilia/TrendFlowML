# `band_energy_extractor` — инженерный журнал и связь с Audit v4 / **4.2**

**Назначение:** зафиксировать изменения **наблюдаемости/производительности** в `band_energy_extractor` после эмпирического отчёта Audit v4 (L2), не переписывая сам L2‑отчёт и не дублируя таблицы статистики.

**Версия компонента в коде (после правок):** `2.1.1` (`DataProcessor/AudioProcessor/src/extractors/band_energy_extractor/main.py`)

---

## 1. Канонические документы (source-of-truth)

| Документ | Роль |
|----------|------|
| [Отчёт Audit v4 — `band_energy_extractor` (L2)](../audio_processor/band_energy_extractor_audit_v4.md) | Статистика выхода на **A+B**, корреляции tabular, вердикт |
| [Критерии и план v4](../../AUDIT_4_CRITERIA_AND_PLAN.md) | §3.1 уровни отчёта, §4.* метрики, §12.x ресурсы/скорость |
| [`band_energy_extractor` README](../../../../AudioProcessor/src/extractors/band_energy_extractor/docs/README.md) | Контракт, флаги, поля, мета |
| [`band_energy_extractor` SCHEMA](../../../../AudioProcessor/src/extractors/band_energy_extractor/docs/SCHEMA.md) | NPZ‑ключи и обязательные meta поля |
| [Журнал прогонов v4](../../RUN_LOG.md) | Ссылки на `result_store`, tooling, A/B/C |

**Артефакты L2 (статистика выхода):**

- `storage/audit_v4/band_energy_extractor_l2/band_energy_extractor_audit_v4_stats.json`
- `storage/audit_v4/band_energy_extractor_l2/figures/`
- Скрипт статистики: `DataProcessor/AudioProcessor/src/extractors/band_energy_extractor/scripts/audit_v4_npz_stats.py`

---

## 2. Что изменено после L2 (Audit 4.2: profiling + perf, без смены контракта)

### 2.1 Наблюдаемость: `meta.stage_timings_ms` (обязательное поле схемы)

Добавлено заполнение `payload.stage_timings_ms` в обоих режимах:

- `run()`:
  - `load_audio_ms`, `normalize_audio_ms` (если включено), `compute_bands_ms`, `compute_shares_ms`, `balance_metrics_ms`, `validate_output_ms`, `total_ms`
- `run_segments()`:
  - `load_segments_ms`, `process_segments_ms`, `aggregate_results_ms`, `validate_output_ms`, `total_ms`
  - счётчики: `segments_count`, `segments_valid`, `segments_masked_short`

Итог: `stage_timings_ms` теперь проходит в `meta` через `npz_savers/band_energy.py`.

### 2.2 Профиль ресурсов (опционально, env‑gated)

Добавлен best‑effort профиль `meta.band_energy_resource_profile` с RSS/VMS и (если доступно) GPU метриками через `torch`:

- включение: `AP_BAND_ENERGY_RESOURCE_PROFILE=1`
- снимки: `*_at_start`, `*_at_end`

### 2.3 `meta.duration` в `run_segments()`

Раньше в сегментном пути `payload.duration` не задавался ⇒ в `meta.duration` попадал `null`. Теперь в `run_segments()` выставляется span по сегментам: \( \max(end\_sec) - \min(start\_sec) \).

### 2.4 Производительность: кэш масок полос и более лёгкая STFT power

- **Кэш масок частотных полос** по `(sr, n_fft)` внутри экземпляра экстрактора — экономит время на длинных сериях сегментов.
- STFT power считается через in‑place умножение (`mag *= mag`) вместо `** 2` (меньше временных аллокаций).

### 2.5 Безопасность `shared_features` (best-effort guard)

Добавлен guard, который игнорирует `shared_features['stft_magnitude']`, если его размерность/число кадров **не похоже** на текущий `y` (защита от случайного «не того окна»).

---

## 3. Влияние на аудитные статистики

- **Tabular доли / `band_energy_shares`**: алгоритм не менялся; изменения касаются **meta** и оптимизаций вычисления.
- **Нужно обновить reference A / golden (§4.8)**: после правок 2.1.1 следует зафиксировать golden‑сигнатуру на **A** (см. `RUN_LOG.md`).

