# `key_extractor` — описание фич и трассировка артефактов

Реализация: [`../main.py`](../main.py) (`KeyExtractor`), NPZ: [`npz_savers/key.py`](../../../core/npz_savers/key.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `key_extractor/key_extractor_features.npz`.

---

## 1. Код (Audit v3)

- Только **`run_segments()`** с family **`key`** (`families.key.segments[]`); **`run()`** отключён.
- Метод: **librosa** (Krumhansl–Schmuckler) или **Essentia** (`key_method`); ML-моделей нет, `meta.models_used` пустой.
- Опционально: shared **chroma** из `shared_features` → `chroma_reused` в meta.
- **`key_id`**: 0–23 (12 тонов × major/minor), **-1** в per-segment массивах при `segment_mask=false`.
- Флаги `_features_enabled`: `detailed_scores`, `top_k`, `time_series`, `key_changes`, `stability_metrics` — влияют на tabular, `key_scores`, последовательности и доп. поля в meta.

| Слой | Содержимое |
|------|------------|
| Tabular | `sample_rate`, `hop_length`, `duration`, `key_id`, `key_confidence` (float32) |
| Meta (строки/категории) | `key_name`, `key_mode`, `key_method`, `key_confidence_category`, `key_confidence_reason`, `key_low_confidence_warning`, при `key_id ≥ 0` дубль `key_id` в meta |
| Ось сегментов | `segment_*`, `key_id_by_segment`, `key_confidence_by_segment` |
| Всегда в NPZ | `key_scores`: **(24,)** — нули, если `detailed_scores` выключен |
| Опционально | последовательности `key_*_sequence` при `time_series`; top-k / transitions / stability в meta по флагам |

Контракт: `key_contract_version` → **`key_contract_v1`**.

---

## 2. NPZ

- `schema_version`: **`key_extractor_npz_v1`**.
- Пример прогона (без time series): **10** членов архива:

`feature_names`, `feature_values`, `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`, `key_id_by_segment`, `key_confidence_by_segment`, `key_scores`, `meta`.

---

## 3. Batch CSV

Плоский **meta** (как в `flatten_meta`): имена колонок вида `meta_key_*`, `meta_chroma_reused`, `meta_status`, тайминги `meta_timing_*`. Скаляры tabular (`sample_rate`, `hop_length`, …) в wide-строке приходят из слияния с feature row, не обязаны дублировать все поля в одном `meta` dict.

---

## 4. Melt / QA / CLI

`view_csv_melt_interesting.json`, `view_csv_feature_qa.json`, `view_csv_feature_descriptions_ru.json`; [`../utils/validate_key.py`](../utils/validate_key.py) `<npz> [--struct] [--qa]`.

---

## 5. Чеклист

1. Сегменты `families.key` непусты при `audio_present=true`.  
2. Одинаковая длина **N** по `segment_*`, `key_id_by_segment`, `key_confidence_by_segment`.  
3. `key_scores` форма **(24,)**.  
4. При включённом `time_series` — согласованность длины последовательностей с **N**.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
