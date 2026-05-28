# `clap_extractor` — описание фич и трассировка артефактов

Реализация: [`__init__.py`](__init__.py) (`CLAPExtractor`), NPZ: [`src/core/npz_savers/clap.py`](../../core/npz_savers/clap.py) (от каталога экстрактора). Схема: `docs/SCHEMA.md`. Артефакт: `clap_extractor/clap_extractor_features.npz`.

---

## 1. Код и режимы (Audit v3)

- **Только `run_segments()`** с family `clap` в `segments.json`. `run()` отключён (ошибка «use run_segments»).
- Модель **только локально** через `dp_models` / `laion_clap` (без сетевых загрузок).
- **Предобработка:** ресемплинг к `sample_rate` (по умолчанию 48 kHz), нормализация амплитуды, **обрезка** фрагментов длиннее `max_audio_length` (10 с) → флаги `trimmed_*` в payload / meta.
- **Агрегация:** robust mean по сегментам (`_robust_aggregate`, квантили по нормам сегментных эмбеддингов).
- Параллель: препроцесс сегментов (ThreadPool), инференс микробатчами (`model_batch_size`).

| Группа | Содержимое |
|--------|------------|
| Эмбеддинги | `embedding` [D], `embedding_sequence` [N,D] (NaN для невалидных), `segment_*_sec`, `segment_mask`, `segment_embedding_norm` |
| Tabular (add) | `embedding_dim`, `clap_norm`, `clap_magnitude_mean`, `clap_magnitude_std`, `segments_count` |
| Счётчики / trim | `trimmed_segments_count`, `trimmed_ratio`, `max_audio_length_sec`, `sample_rate`, `device_used`, `stage_timings_ms`, `clap_resource_profile` (env) |
| Планировщик | `scheduler_knobs` в payload (не все попадают в NPZ meta — см. савер) |

`embedding_dim` по умолчанию **512**.

---

## 2. NPZ

| Ключ | Смысл |
|------|--------|
| `feature_names` / `feature_values` | Tabular: `embedding_dim`, `clap_norm`, `clap_magnitude_*`, `segments_count` |
| `embedding`, `embedding_present`, `embedding_sequence` | Канон |
| `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`, `segment_embedding_norm` | Ось времени |
| `meta` | + `sample_rate`, `device_used`, `embedding_dim`, `max_audio_length_sec`, `trimmed_segments_count`, `trimmed_ratio`, `stage_timings_ms`, … |

`schema_version`: **`clap_extractor_npz_v1`**.

---

## 3. Batch CSV

Только плоский **`meta`** (`flatten_meta`). Табличные `clap_*` из `feature_values` **не** дублируются в wide-строке — смотреть NPZ.

---

## 4. Melt / пояснения

`view_csv_melt_interesting.json` → `clap_extractor`; QA: `view_csv_feature_qa.json`; описания: `view_csv_feature_descriptions_ru.json`.

---

## 5. CLI

`utils/validate_clap.py <npz> [--struct] [--qa]`

---

## 6. Чеклист

1. Сегменты family `clap`, непустой список.  
2. NPZ: согласованность N по `segment_*` и `embedding_sequence`.  
3. `trimmed_ratio` ∈ [0,1] при валидном прогоне.  
4. Табличные скаляры — из NPZ, не из CSV-only.
