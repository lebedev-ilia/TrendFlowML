# `band_energy_extractor` — описание фич и трассировка артефактов

Связка **код** (`main.py` + `npz_savers/band_energy.py`) → **NPZ** → **batch CSV (meta-only)** → **melt-HTML** и **QA** (`view_csv_feature_qa.json`).

Пользовательская схема: `docs/SCHEMA.md`, обзор: `docs/README.md`. Идентичность: компонент в run-каталоге `band_energy_extractor` (см. `BaseExtractor` / `run_cli`: имя `band_energy`, артефакт `band_energy_extractor_features.npz`).

---

## 1. Код: `payload` и флаги

`BandEnergyExtractor` в `main.py` (аудит v3: только `librosa`, три фиксированные полосы Hz, без basic/extended/dynamics). Режимы: **`run()`** (целый файл) и **`run_segments()`** (сегменты `spectral` family).

| Группа | Поля / смысл |
|--------|----------------|
| Канон | `band_edges` (3 пары Hz), `band_energy_shares` (3 float, **сумма ≈ 1**), `sample_rate`, `n_fft`, `hop_length`, `duration` (с: целый файл = длина аудио; **сегменты** = span по `start_sec`/`end_sec` **или** `None` при сбое) |
| Баланс (опц.) | `enable_balance_metrics` → `band_balance_score`, `band_dominance` (0..2), `band_contrast` — попадают в `payload` и в tabular NPZ/ meta при включённом флаге |
| Сегменты (опц.) | `enable_time_series` → `segment_centers_sec`, `segment_durations`, `segment_mask`, `band_shares_by_segment` |
| Контракт | `band_energy_contract_version` (`band_energy_contract_v1`), `_features_enabled` (напр. `time_series`, `balance_metrics`) |
| Тайминги | `stage_timings_ms` (разный набор для `run` vs `run_segments`: load/compute/validate, для сегментов — `process_segments_ms`, `aggregate_results_ms`, `segments_*`, и т.д.) |
| Профилирование | `band_energy_resource_profile` (env) |

`device_used`, `method` = `"librosa"`.

---

## 2. NPZ (`save_band_energy_npz`)

**Файл:** `band_energy_extractor/band_energy_extractor_features.npz` (`npz_saver`).

| Ключ | Содержимое |
|------|------------|
| `feature_names` / `feature_values` | Tabular: `band_share_{low,mid,high}`; при balance — `band_balance_score`, `band_contrast`, `band_dominant_band`; `NaN` если доли не из 3 элементов |
| `band_edges_hz` | `float32[3,2]` границы полос |
| `band_energy_shares` | `float32[3]` |
| `segment_*` , `band_shares_by_segment` | Только если `time_series` в `features_enabled` в этом прогоне |
| `meta` | `build_meta` + `extra`: контракт, `features_enabled`, `stage_timings_ms`, `method`, `sample_rate`, `n_fft`, `hop_length`, `duration`, balance-поля при флаге, `band_energy_resource_profile` |

`schema_version` в проде: **`band_energy_extractor_npz_v1`**.

**Пример без time_series** (5 ключей): `feature_names`, `feature_values`, `band_edges_hz`, `band_energy_shares`, `meta`.

---

## 3. Batch CSV

Как и для остальных audio-модулей: в строку попадает только **`_flatten_meta(meta)`** (строки &lt; 200 символов, `stage_timings_ms` → `meta_timing_*`) + манифест/merge.

**Не попадает в CSV:** `feature_names`/`feature_values` (доли `band_share_*` в wide-отчёте **нет**), массивы `band_edges_hz` / `band_energy_shares` — смотреть NPZ. Правила `flatten_meta` по `meta` — как в `DataProcessor/qa/component_feature_qa` / batch-отчёт (вложенные dict/list, длинные строки не плоские).

Плоские `meta_method`, `meta_sample_rate`, `meta_n_fft`, `meta_hop_length`, `meta_duration` — **есть** в CSV (как пишет савер в `meta`).

---

## 4. Melt-HTML

`view_csv_melt_interesting.json` → `band_energy_extractor`: `include` (device, duration, hop, method, n_fft, sample_rate, status) + `add_all_meta_timing` → `meta_timing_*`. Часть `meta_*` по умолчанию скрыта в melt (`MELT_SUPPRESS_REPEATING` в `view_csv.py` — `meta_schema_version`, `meta_producer_version`, …). Чтобы увидеть их, **`--melt-show-repeating-meta`**.

Пояснения: `view_csv_feature_descriptions_ru.json` + эвристики `view_csv_melt_captions_ru.py`.

---

## 5. QA

- Правила: `storage/result_store/view_csv_feature_qa.json` → `components.band_energy_extractor` (+ `any_component`, напр. `duration_ms`).
- Подсветка: `view_csv.py --melt … --melt-qa`.
- CLI: `utils/validate_band_energy.py <npz> [--struct] [--qa]` — **`--struct`**: доли (сумма ~1), формы `band_edges_hz` / tabular; **`--qa`**: плоский meta, те же column names, что в wide CSV.

---

## 6. Быстрый чеклист

1. Код: `enable_*` в `__init__`, ветки `run` / `run_segments` в `main.py`.  
2. NPZ: `unzip -l` / `numpy.load` — ожидаемые ключи; при `time_series` — доп. массивы.  
3. CSV: строка `component=band_energy_extractor` — `meta_*`, `meta_timing_*`.  
4. Доли: из NPZ `band_energy_shares` / tabular; сумма **≈ 1** (см. `--struct`).
