# `mfcc_extractor` — описание фич и трассировка артефактов

Реализация: [`../main.py`](../main.py) (`MFCCExtractor`), NPZ: [`npz_savers/mfcc.py`](../../../core/npz_savers/mfcc.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `mfcc_extractor/mfcc_extractor_features.npz`.

---

## 1. Код (Audit v3)

- **Torch / torchaudio** MFCC; опционально GPU (эвристика по длительности/размеру).
- Режимы: **`run()`** (весь файл) и **`run_segments()`** (сегменты Segmenter, family **primary**).
- `_features_enabled`: `basic_features` (глоб. статистики `mfcc_mean`/`std`/`min`/`max` длины **M = n_mfcc**), `deltas` (delta / delta-delta mean/std в NPZ, плюс `delta_mean_by_segment` вместе с time_series), `time_series` (`mfcc_mean_by_segment` **[N,M]**, `mfcc_energy_by_segment` **[N]**, опционально `delta_mean_by_segment`).

| Слой | Содержимое |
|------|------------|
| Tabular | `sample_rate`, `n_mfcc`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax`, `duration_sec`, `segments_count` (если есть), `mfcc_energy`, `mfcc_centroid`, `mfcc_bandwidth`, `mfcc_stability` |
| Ось N | `segment_*_sec`, `segment_mask` |
| Опц. | `mfcc_mean`…`mfcc_max` **[M]**, `delta_*` **[M]**, `mfcc_mean_by_segment` **(N,M)**, `mfcc_energy_by_segment` **[N]**, `delta_mean_by_segment` **(N,M)** |
| Meta | `device_used`, `mfcc_contract_version` (`mfcc_contract_v1`), `stage_timings_ms`, `mfcc_resource_profile` (opt), `mfcc_npy` (путь к тяжёлому .npy) |

`schema_version`: **`mfcc_extractor_npz_v2`**.

---

## 2. NPZ (референс `…/mfcc_extractor_features.npz`)

**14** членов: `feature_names`, `feature_values`, четыре `segment_*`, `mfcc_mean`–`mfcc_max` (M коэф.), `mfcc_mean_by_segment`, `mfcc_energy_by_segment`, `delta_mean_by_segment`, `meta`. Набор зависит от **feature flags** (без basic/time_series/deltas — часть массивов отсутствует).

---

## 3. Batch CSV / melt

Плоский **meta** + tabular-скаляры в wide-строке. См. `view_csv_melt_interesting.json` → `mfcc_extractor`.

---

## 4. QA / CLI

`view_csv_feature_qa.json` (`mfcc_extractor`); [`../utils/validate_mfcc.py`](../utils/validate_mfcc.py) — `<npz> [--struct] [--qa]`.

**Ориентиры:** `n_mfcc` обычно 13–40; `fmax` ≤ Nyquist; `mfcc_stability` ∈ (0, 1] по определению `1/(1+std)`.

---

## 5. Чеклист

1. Сегментные оси **одинаковой длины N**.  
2. `mfcc_mean`… длина **M = n_mfcc** (из tabular).  
3. `mfcc_mean_by_segment` — **(N, M)**; `mfcc_energy_by_segment` — **N**; при deltas — `delta_mean_by_segment` **(N, M)**.  
4. `meta.schema_version` содержит `mfcc_extractor`.
