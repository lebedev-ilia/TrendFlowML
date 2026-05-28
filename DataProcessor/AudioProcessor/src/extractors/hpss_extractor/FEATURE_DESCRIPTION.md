# `hpss_extractor` — описание фич и трассировка артефактов

Реализация: [`main.py`](main.py) (`HPSSEtractor` / класс в модуле), NPZ: [`npz_savers/hpss.py`](../../core/npz_savers/hpss.py). Схема: `docs/SCHEMA.md`. Артефакт: `hpss_extractor/hpss_extractor_features.npz`.

---

## 1. Код (Audit v3)

- **Только `run_segments()`** с family **`hpss`**. `run()` для полного файла в audit-режиме отключён (см. SCHEMA).
- **librosa** HPSS (без отдельной ML-модели); `meta.models_used` в типичном прогоне пустой.
- Флаги: `enable_energy_metrics`, `enable_spectral_features`; в **`run_segments`** не выставляются **`time_series`** / **`waveforms`** (глобальные ряды и волны по полному клипу не строятся — только пооконные метрики).

| Группа | Содержимое |
|--------|------------|
| Ось | `segment_*_sec`, `segment_mask`, `hpss_harmonic_share_by_segment`, `hpss_percussive_share_by_segment` (NaN при mask=false) |
| Tabular | Зависит от флагов: доли, энергии, stability, spectral means, `sample_rate`, `n_fft`, `hop_length`, `duration`, `hpss_*` гиперпараметры, `segments_count` (см. `npz_savers/hpss.py`) |
| Доминанта | `hpss_dominance` — строка **`harmonic` / `percussive` / `mixed`**; в NPZ **только в `meta`**, не в `feature_values` |
| Контракт | `hpss_contract_version` в payload → `meta` |
| Тайминги | `stage_timings_ms` (`process_segments_ms`, `aggregate_ms`, `validate_ms`, `total_ms`, …) |

---

## 2. NPZ

- Базово: `feature_names` / `feature_values`, сегментные массивы, `meta`.
- Опционально: ряды `hpss_*_share_series` при legacy/флагах (см. савер).
- `schema_version`: **`hpss_extractor_npz_v1`**.

Пример прогона (только сегменты + tabular, без time series / waveforms), участники `.npz`:

`feature_names`, `feature_values`, `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`, `hpss_harmonic_share_by_segment`, `hpss_percussive_share_by_segment`, `meta` — **9** файлов в архиве.

---

## 3. Batch CSV

Плоский **`meta`**: `device_used`, `hpss_contract_version`, `hpss_dominance` (если считалась), тайминги. Скаляры из tabular (`sample_rate`, `n_fft`, …) в **meta савер не кладёт** — они в **`feature_names`/`feature_values`** в NPZ; в wide CSV их **нет** (как у других extractors).

---

## 4. Melt / QA / CLI

`view_csv_melt_interesting.json`, `view_csv_feature_qa.json`, `view_csv_feature_descriptions_ru.json`; `validate_hpss.py <npz> [--struct] [--qa]`.

---

## 5. Чеклист

1. Family `hpss`, непустые сегменты.  
2. Одинаковая длина N по сегментным массивам.  
3. Доли/энергии — из NPZ tabular, доминанта — из `meta`.
