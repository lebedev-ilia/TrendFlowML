# `loudness_extractor` — описание фич и трассировка артефактов

Реализация: [`../__init__.py`](../__init__.py) (`LoudnessExtractor`), NPZ: [`npz_savers/loudness.py`](../../../core/npz_savers/loudness.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `loudness_extractor/loudness_extractor_features.npz`.

---

## 1. Код

- **`run()`**: полный файл — RMS/peak/dBFS, опционально **LUFS** (`pyloudnorm`), frame-wise RMS статистики (mean/std/median/p10/p90), `frames_count`.
- **`run_segments()`**: family **`primary`** — per-segment `segment_rms` / `peak` / `dbfs` / `segment_lufs` (NaN, если LUFS не посчитан), агрегаты по валидным сегментам `segment_rms_*`, плюс **полнотрековый** прогон `_compute_from_np` на целом файле для глобальных метрик (как в `payload`: сначала сегменты, затем full-track).

| Слой | Содержимое |
|------|------------|
| Tabular | `loudness_rms`, `loudness_peak`, `loudness_dbfs`, `loudness_lufs` (NaN, если LUFS нет), `duration_sec`, `sample_rate`, `frame_rms_*`, `frames_count`, `segments_count`, `segment_rms_mean` … `segment_rms_p90` |
| Массивы | `lufs_present` (bool, скаляр-флаг: есть ли конечный LUFS хотя бы по full или сегментам) |
| Ось N | `segment_*_sec`, `segment_mask`, `segment_rms`, `segment_peak`, `segment_dbfs`, `segment_lufs` (mask=false → NaN в метриках) |
| Meta | `device_used`, `stage_timings_ms`, `loudness_resource_profile` (opt), `status`, … |

`schema_version`: **`loudness_extractor_npz_v2`**.

---

## 2. NPZ (референсный прогон)

Пример: **12** членов архива: `feature_names`, `feature_values`, `lufs_present`, 8 сегментных массивов (`segment_*` + per-segment метрики), `meta`.

---

## 3. Batch CSV / melt

Широкая таблица раскрывает **tabular** + плоский **meta** (`meta_device_used`, `meta_status`, `meta_timing_*`, …). Флаг `lufs_present` в CSV обычно приходит отдельной колонкой, если пайплайн выгружает top-level поля NPZ, а не только meta.

В melt «interesting» для loudness: `meta_device_used`, `meta_producer_version`, `meta_schema_version`, `meta_status` и все `meta_timing_*` (`add_all_meta_timing`: true). Скаляры громкости в wide-таблице — без префикса `meta_` (см. RU-описания в `view_csv_feature_descriptions_ru.json`).

---

## 4. QA / CLI

`view_csv_feature_qa.json` (компонент `loudness_extractor`); валидатор: [`../utils/validate_loudness.py`](../utils/validate_loudness.py) — `<npz> [--struct] [--qa]`. **`--qa`**: плоский **meta** + **tabular** (`feature_values`) + скаляр **`lufs_present`** из NPZ (для правил `loudness_*`, `lufs_present`, `sample_rate`, …).

**Ориентиры** (не жёсткие границы в NPZ): RMS/peak ≥ 0 (с eps в коде); dBFS обычно < 0; LUFS в типичном диапазоне **−30…0 LUFS** для громкого контента (широкий QA в CSV — при необходимости сузить на проде).

---

## 5. Чеклист

1. `run_segments`: все сегментные массивы одной длины **N**.  
2. `lufs_present` — один булев элемент.  
3. Tabular: **~18** скаляров (см. SCHEMA); пропуски — **NaN**, не нули-заглушки.  
4. `meta.schema_version` содержит `loudness_extractor`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
