# `spectral_entropy_extractor` — описание фич и трассировка артефектов

**Компонент (NPZ / CSV):** `spectral_entropy_extractor`  
**producer_version (код):** `2.0.1` (см. `main.py`)  
**schema_version NPZ:** `spectral_entropy_extractor_npz_v2`  
**Артефакт:** `spectral_entropy_extractor/spectral_entropy_extractor_features.npz`

Реализация: [`../main.py`](../main.py) (`SpectralEntropyExtractor`, внутренний ключ экстрактора `spectral_entropy`), савер: [`npz_savers/spectral_entropy.py`](../../../core/npz_savers/spectral_entropy.py). Схема: [`SCHEMA.md`](SCHEMA.md).

---

## 1. Код (Audit v3)

- Family **spectral** в Segmenter (`families.spectral.segments[]`).
- **`run()`** / **`run_segments()`**; нет `payload` в NPZ — только строгие ключи.
- Tabular: **`spectral_entropy_mean`**, **`spectral_entropy_std`** (агрегаты).
- **Ось N**: `segment_*` + `entropy_mean_by_segment`, `entropy_std_by_segment` (обязательны); опционально min/max/flatness/spread по сегментам (см. савер + флаги в payload).

| Meta (extra) | `spectral_entropy_contract_version` (`spectral_entropy_contract_v1`), `device_used`, echo параметров STFT/mel, `stage_timings_ms`, `spectral_entropy_resource_profile` (opt) |

`schema_version`: **`spectral_entropy_extractor_npz_v2`**.

---

## 2. NPZ (референс `…/spectral_entropy_extractor_features.npz`)

**9** членов: `feature_names`, `feature_values`, четыре `segment_*`, `entropy_mean_by_segment`, `entropy_std_by_segment`, `meta`.

---

## 3. Batch CSV / melt

`view_csv_melt_interesting.json` — `spectral_entropy_extractor` (уже богатый список meta echo + device).

---

## 4. QA / CLI

`view_csv_feature_qa.json` — секция `spectral_entropy_extractor`.

**Валидатор:**

```bash
python utils/validate_spectral_entropy.py <path/to/spectral_entropy_extractor_features.npz> [--struct] [--qa]
```

**Ориентиры:** энтропия спектра в натуральных/норм. ед. (часто 0…`log n_bins`); per-segment консистентны с N.

---

## 5. Чеклист

1. `segment_*` и обязательные per-segment массивы **одинаковой длины N**.  
2. Все присутствующие optional per-segment ряды — длина **N**.  
3. `meta.schema_version` содержит `spectral_entropy_extractor`.  
4. Tabular: 2+ скаляра (замороженный минимум в SCHEMA).
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
