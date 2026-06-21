# `rhythmic_extractor` — описание фич и трассировка артефактов

Реализация: [`../main.py`](../main.py) (`RhythmicExtractor`), NPZ: [`npz_savers/rhythmic.py`](../../../core/npz_savers/rhythmic.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `rhythmic_extractor/rhythmic_extractor_features.npz`.

---

## 1. Код (Audit v3)

- **Бэкенд** beat tracking: **librosa** или **essentia** (явный выбор, no-fallback).
- Сэмплинг: family **`tempo`** (окна, разделяемые с `tempo_extractor`); `sampling_family_used` в meta при миграциях.
- **`run()`** / **`run_segments()`**: агрегированные биты, `rhythm_*` model-facing в tabular; доп. скаляры (интервалы, IBI-tempo, syncopation, …) — при наличии в `payload` как **отдельные float32-ключи** в корне NPZ.
- `beat_times_sec` / `beat_segment_index` в NPZ **или** offloaded в **`.npy`**, пути `beat_times_sec_npy` / `beat_segment_index_npy` в meta (тогда массивы в `.npz` пустые/отсутствуют).

| Слой | Содержимое |
|------|------------|
| Tabular (фикс. порядок) | `rhythm_tempo_bpm`, `rhythm_beats_count`, `rhythm_beat_density`, `rhythm_regularity`, `rhythm_tempo_variation`, `rhythm_beat_consistency`, `duration_sec`, `sample_rate`, `segments_count` |
| Сегменты | `segment_*` |
| Beats (opt) | `beat_times_sec` **[M]**, `beat_segment_index` **[M]** |
| Extra scalars (opt) | `rhythm_avg_period_sec`, `rhythm_median_bpm`, … (см. савер) |
| Meta | `backend`, `hop_length`, `rhythmic_contract_version` (`rhythmic_contract_v1`), `sampling_family_used`, пути к `.npy`, `stage_timings_ms`, `features_enabled` |
| **device_used** | В **payload** есть, в `meta_extra` савера **не** добавляется; `meta_device_used` в melt/CSV — при глобальном merge, как у onset/pitch. |

`schema_version`: **`rhythmic_extractor_npz_v2`**.

---

## 2. NPZ (референс `…/rhythmic_extractor_features.npz`)

Пример: **23** члена: tabular, четыре `segment_*`, набор `rhythm_*_*.npy` scalar keys (по одному 0d float в архиве), `meta` — без `beat_times` в теле (в этом прогоне биты, вероятно, в `.npy` или 0 длина).

---

## 3. Batch CSV / melt

`view_csv_melt_interesting.json` → `rhythmic_extractor` (`meta_backend`, `meta_hop_length`, `meta_sampling_family_used`, …).

---

## 4. QA / CLI

`view_csv_feature_qa.json` (`rhythmic_extractor`); [`../utils/validate_rhythmic.py`](../utils/validate_rhythmic.py) — `<npz> [--struct] [--qa]`.

**Ориентиры:** `rhythm_tempo_bpm` 0…400; `rhythm_regularity` / `rhythm_beat_consistency` часто 0…1; плотность ≥ 0.

---

## 5. Чеклист

1. Четыре сегментных массива **одинаковой длины N**.  
2. Если есть `beat_times_sec` и `beat_segment_index` — **одинаковая M**.  
3. `feature_names` / `feature_values` длина согласована, ~9 tabular-имён (плюс NaN-заглушки).  
4. `meta.schema_version` содержит `rhythmic_extractor`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
