# `mel_extractor` — описание фич и трассировка артефактов

Реализация: [`../main.py`](../main.py) (`MelExtractor`), NPZ: [`npz_savers/mel.py`](../../../core/npz_savers/mel.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `mel_extractor/mel_extractor_features.npz`.

---

## 1. Код (Audit v3)

- **Torch / torchaudio**: mel-спектрограмма, опционально GPU.
- Режимы: **`run()`** (полный файл) и **`run_segments()`** (family **primary** / сегменты из Segmenter).
- Флаги `_features_enabled`: `basic_features` (shape, elements), `statistics` (per-bin `mel_mean`/`std`/`min`/`max` длины **M = n_mels**), `spectral_features` (вклад в агрегаты), `time_series` (сегментные ряды), `stats_vector` (склейка mean/std/min/max → один вектор, обычно **4M** float).

| Слой | Содержимое |
|------|------------|
| Tabular | `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`, `power`, `duration_sec`, `segments_count` (если в payload), при `basic_features` — `mel_shape_0/1`, `mel_elements` |
| Скаляры | `mel_energy`, `mel_centroid_*`, `mel_bandwidth_*`, `mel_spectrogram_entropy`, `mel_spectrogram_contrast`, `mel_rolloff`, `mel_flatness`, `mel_stability` (косинусная схожесть соседних `mel_mean` по валидным сегментам) |
| Ось N | `segment_*_sec`, `segment_mask` |
| Опц. | `mel_mean`…`mel_max` **[M]**, `mel_stats_vector` **[4M]**, `mel_mean_by_segment` **[N,M]**, `mel_energy_by_segment`, `mel_centroid_mean_by_segment`, `mel_bandwidth_mean_by_segment` |
| Meta | `device_used`, `mel_contract_version` (`mel_contract_v1`), `stage_timings_ms`, `mel_resource_profile` (opt), пути `mel_spectrogram_npy` / `mel_series_npy` (debug) |

`schema_version`: **`mel_extractor_npz_v2`**.

---

## 2. NPZ (референс `…/mel_extractor/mel_extractor_features.npz`)

**16** членов (пример): `feature_names`, `feature_values`, четыре `segment_*`, `mel_mean`–`mel_max` (длина **M = n_mels** из tabular), `mel_stats_vector` (4M при stats_vector+statistics), `mel_mean_by_segment` **(N×M)** + три сегментных ряда, `meta`. Набор ключей зависит от **feature flags** (если флаги выключены — соответствующие массивы отсутствуют в `.npz`).

---

## 3. Batch CSV / melt

Широкая таблица: tabular-скаляры + плоский **meta** (`meta_device_used`, `meta_mel_contract_version`, `meta_timing_*`, …). См. `view_csv_melt_interesting.json` → `mel_extractor`.

---

## 4. QA / CLI

`view_csv_feature_qa.json` (`mel_extractor`); [`../utils/validate_mel.py`](../utils/validate_mel.py) — `<npz> [--struct] [--qa]`.

**Ориентиры:** `fmax` ≤ Nyquist (`sample_rate/2`); `n_mels` обычно 16…512; dB-спектр в коде клипуется [−120, 0].

---

## 5. Чеклист

1. Четыре сегментных оси **одинаковой длины N**.  
2. Если есть `mel_mean` — длина **M**; `mel_stats_vector` длина **4M** (при включённых statistics + stats_vector).  
3. Если есть `mel_mean_by_segment` — форма **(N, M)**; сегментные скаляры — длина **N**.  
4. `meta.schema_version` содержит `mel_extractor`.
