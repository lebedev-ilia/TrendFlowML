# `onset_extractor` — описание фич и трассировка артефактов

Реализация: [`../main.py`](../main.py) (`OnsetExtractor`), NPZ: [`npz_savers/onset.py`](../../../core/npz_savers/onset.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `onset_extractor/onset_extractor_features.npz`.

---

## 1. Код (Audit v3)

- Бэкенд: **librosa** и/или **essentia** (поле `backend` в **meta**).
- **`run()`**: онсеты по полному треку, метрики, опционально сохранение `onset_times` в **debug `.npy`**, путь в `meta.onset_times_npy`.
- **`run_segments()`**: онсеты по сегментам → объединение, дедуп, сортировка; **метрики на агрегированной** временной оси. Канонические **`segment_*`** (в типичном прогоне `segment_mask` все `true` — см. SCHEMA).

Флаги `_features_enabled`: `basic_features` (счётчик, плотность, `insufficient_onsets`), `interval_stats` (интервалы), `rhythmic_metrics` (tempo estimate, regularity, syncopation, strength, …), `time_series` (только артефакты/указатели, не плотный ряд в NPZ).

| Слой | Содержимое |
|------|------------|
| Tabular | `sample_rate`, `hop_length`, `duration`, `segments_count` + feature-gated скаляры (см. савер) |
| Meta | `backend`, `onset_contract_version` (`onset_contract_v1`), `onset_times_npy` (opt), `stage_timings_ms`, `onset_resource_profile` (opt) |
| **Не в meta савера** | `device_used` есть в **payload** в коде, но **не** добавляется в `meta_extra` в `save_onset_npz` — в плоском meta NPZ **нет** `device_used`; колонка `meta_device_used` в wide-CSV может заполняться **глобальным** merge пайплайна. |

`schema_version`: **`onset_extractor_npz_v2`**.

---

## 2. NPZ (референс `…/onset_extractor_features.npz`)

**7** членов: `feature_names`, `feature_values`, четыре `segment_*`, `meta` (без массива `onset_times` внутри `.npz` — только путь в meta при сохранении артефакта).

---

## 3. Batch CSV / melt

`view_csv_melt_interesting.json` → `onset_extractor` (`meta_backend`, `meta_onset_times_npy`, `meta_device_used` при наличии в строке, …).

---

## 4. QA / CLI

`view_csv_feature_qa.json` (`onset_extractor`); [`../utils/validate_onset.py`](../utils/validate_onset.py) — `<npz> [--struct] [--qa]`. **`--qa`** проверяет и **плоский meta**, и **tabular** (`feature_names` / `feature_values`), как в wide CSV.

**Ориентиры:** `onset_tempo_estimate` (если есть) 0…400 BPM в QA; `onset_density_per_sec` ≥ 0; доли/regularity 0…1, если определены как такие.

---

## 5. Чеклист

1. Четыре сегментных оси **одинаковой длины N**.  
2. `feature_names` / `feature_values` согласованы по длине.  
3. `meta.schema_version` содержит `onset_extractor`.  
4. Список tabular-имён соответствует включённым **feature-gated** группам.
