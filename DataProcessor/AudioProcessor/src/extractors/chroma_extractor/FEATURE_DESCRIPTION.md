# `chroma_extractor` — описание фич и трассировка артефактов

Связка **код** (`main.py` + `npz_savers/chroma.py`) → **NPZ** → **batch CSV (meta-only)** → **melt** и **QA**.

Схема: `docs/SCHEMA.md`, обзор: `docs/README.md`. Каталог артефакта: `chroma_extractor` → `chroma_extractor_features.npz`. Внутренняя метка: `name = "chroma"`.

---

## 1. Код: `payload` / флаги (Audit v3)

- **librosa**; `n_chroma=12`, `normalize="l1"`, `chroma_type` ∈ `cqt` | `stft` (без silent fallback).
- **Отключено в audit:** `enable_basic_stats`, `enable_extended_stats`, `enable_stats_vector` (RuntimeError).
- **Режимы:** `run()` (полный файл), `run_segments()` (family `chroma` в `segments.json`).

| Группа | Содержимое |
|--------|------------|
| Вектор / скаляры | `chroma_mean` (12), `chroma_entropy`, `chroma_harmonic_stability`, `chroma_contrast`, `chroma_dominant_class` (0..11), `chroma_dominant_energy`, `tuning_estimate`, `tuning_failed` |
| Tuning | один раз на полном аудио; при сбое оценки → `0.0` и `tuning_failed=true` |
| Служебное для downstream | `_shared_chroma` (2D для `run`, 12×1 proxy для `run_segments`) — **не** в NPZ как поле, инъекция в раннере |
| Time series (опц.) | `enable_time_series` → `chroma` (2D) только в **`run()`**; в `run_segments` — `segment_*` + `chroma_mean_by_segment`, **без** полного `chroma` в контракте |
| Мета-поля в payload | `sample_rate`, `hop_length`, `n_fft`, `duration`, `device_used`, `chroma_type`, `normalize`, `n_chroma`, `chroma_frames`, `segments_count` (сегментный режим), `chroma_contract_version`, `chroma_time_series_omitted` (при усечённом debug chroma) |
| Тайминги | `stage_timings_ms` (разные ключи в `run` vs `run_segments`: tuning, extract_chroma, process_segments, aggregate, …) |
| Профилирование | `chroma_resource_profile` (env) |

---

## 2. NPZ (`save_chroma_npz`)

- **Tabular:** `chroma_mean_{C,…,B}` + скаляры (см. `SCHEMA.md`); `feature_names` / `feature_values`.
- **Массивы:** `chroma_mean`, `chroma_entropy`, `chroma_harmonic_stability`, `chroma_contrast`, `chroma_dominant_class`, `chroma_dominant_energy`, `tuning_estimate`.
- **Опц.:** `chroma` (спектрограмма 12×T) при time_series в `run()`; `segment_*`, `chroma_mean_by_segment` при time_series + сегменты.
- **meta:** `chroma_contract_version`, `chroma_type`, `normalize`, `tuning_failed` (bool), `chroma_time_series_omitted`, `sample_rate`, `hop_length`, `n_fft`, `duration_sec` ← из `payload["duration"]`, `segments_count`, `stage_timings_ms`, `device_used` (и др. из `extra`).

`schema_version`: **`chroma_extractor_npz_v1`**.

**Пример прогона с сегментами + time series:** 14 файлов в `.npz` (tabular + 7 базовых массивов + meta + 4 segment + `chroma_mean_by_segment`).

---

## 3. Batch CSV

Плоский вывод **только** `meta` (как `flatten_meta` в `DataProcessor/qa`). Tabular `chroma_mean_*` / `feature_values` **не** в wide-строке. Числа по хрому смотреть в NPZ.

---

## 4. Melt-HTML

`view_csv_melt_interesting.json` → `chroma_extractor`: `meta_*` (type, duration_sec, segments, tuning_failed, time_series_omitted, …) + `add_all_meta_timing`. Дубликаты `meta_schema_version` / `meta_producer` скрываются `MELT_SUPPRESS_REPEATING` (см. `view_csv.py`); **показать** — `--melt-show-repeating-meta`.

---

## 5. QA

- `storage/result_store/view_csv_feature_qa.json` → `components.chroma_extractor`.
- `validate_chroma.py <npz> --qa` / `--struct`.
- Melt: `--melt-qa`.

---

## 6. Чеклист

1. Код: `run` / `run_segments`, `enable_time_series`, tuning policy.  
2. NPZ: набор ключей; при сегментах — согласованные длины `segment_*` и `chroma_mean_by_segment`.  
3. CSV: `meta_chroma_*`, `meta_timing_*` для сравнения ранов.  
4. Табличные веса — из NPZ `feature_values`, не из CSV.
