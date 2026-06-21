# `quality_extractor` — описание фич и трассировка артефактов

Реализация: [`../main.py`](../main.py) (`QualityExtractor`), NPZ: [`npz_savers/quality.py`](../../../core/npz_savers/quality.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `quality_extractor/quality_extractor_features.npz`.

---

## 1. Код (Audit v3)

- Family **primary**; `run()` (полный файл) и `run_segments()` (окна Segmenter).
- Флаги `_features_enabled`: `basic_metrics` (DC, clipping, crest, `quality_score`, …), `dynamic_metrics` (dynamic range), `frame_analysis` (распределение уровней кадров).
- Ряды (DC, clipping, crest, frame RMS, …) **не** кладутся в NPZ — только **пути** в `meta` (`*_npy`).

| Слой | Содержимое |
|------|------------|
| Tabular | Всегда: `sample_rate`, `average_channels`, `frame_len_ms`, `hop_ms`, `clip_threshold`, `duration`, `segments_count` + feature-gated блоки (см. савер) |
| Сегменты | `segment_*` (строгое N) |
| Meta | `device_used`, `quality_contract_version` (`quality_contract_v1`), `features_enabled`, `stage_timings_ms`, `quality_resource_profile` (opt), пути `*_series_npy` / `clipping_segments_series_npy` при сохранении артефактов |

`schema_version`: **`quality_extractor_npz_v2`**.

---

## 2. NPZ (референс `…/quality_extractor_features.npz`)

**7** членов: `feature_names`, `feature_values`, четыре `segment_*`, `meta` (ряды вне `.npz` — см. `meta`).

---

## 3. Batch CSV / melt

`view_csv_melt_interesting.json` → `quality_extractor`.

---

## 4. QA / CLI

`view_csv_feature_qa.json` (`quality_extractor`); [`../utils/validate_quality.py`](../utils/validate_quality.py) — `<npz> [--struct] [--qa]`.

**Ориентиры:** `clipping_ratio` 0…1; `quality_score` часто 0…1; `clip_threshold` < 1; динамический диапазон (dB) в разумных пределах.

---

## 5. Чеклист

1. Сегментные оси **одной длины N**.  
2. `feature_names` / `feature_values` согласованы; набор имён = включённые feature-группы.  
3. `meta.schema_version` содержит `quality_extractor`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
