# `pitch_extractor` — описание фич и трассировка артефактов

Реализация: [`../main.py`](../main.py) (`PitchExtractor`), NPZ: [`npz_savers/pitch.py`](../../../core/npz_savers/pitch.py). Схема: [`SCHEMA.md`](SCHEMA.md). Артефакт: `pitch_extractor/pitch_extractor_features.npz`.

---

## 1. Код (Audit v3)

- **Backend** `classic` (PYIN/YIN) или **torchcrepe** (fail-fast, no-fallback).
- **`run()`**: полный файл; **`run_segments()`** — f0 по сегментам, агрегаты, строгая ось `segment_*` (при пустом pitch возможен `status=empty`).

Флаги `_features_enabled`: `basic_stats` (f0 mean/std/…, контур-метрики, `pitch_octave_distribution` в NPZ как **object**-scalar), `stability_metrics`, `delta_features`, `method_stats` (pyin/yin/torchcrepe столбцы в tabular).

| Слой | Содержимое |
|------|------------|
| Tabular | `sample_rate`, `hop_length`, `frame_length`, `fmin`, `fmax`, `duration`, `segments_count` + feature-gated f0/ритм-метрики (см. савер) |
| Meta | `backend` (`classic` / `torchcrepe`), `f0_method` (строка; в сегментах может быть `aggregated` или `pyin`/`yin`/…), `pitch_contract_version` (`pitch_contract_v1`), `f0_series_npy` (opt), `stage_timings_ms`, `pitch_resource_profile` (opt) |
| Массивы | `segment_*`; опционально **`pitch_octave_distribution`**: 0d `object` с dict (не flatten в CSV как число) |
| **Примечание** | `device_used` в **payload** есть, в `save_pitch_npz` **в `meta` не пишется**; `meta_device_used` в wide-строке может появляться от **глобального** merge, как у других экстракторов. |

`schema_version`: **`pitch_extractor_npz_v2`**.

---

## 2. NPZ (референс `…/pitch_extractor_features.npz`)

**8** членов: `feature_names`, `feature_values`, четыре `segment_*`, `pitch_octave_distribution` (object), `meta`. Полоса f0 и ряды f0 **не** в бинарнике NPZ — путь в `meta.f0_series_npy` (debug .npy).

---

## 3. Batch CSV / melt

`view_csv_melt_interesting.json` → `pitch_extractor`. Для сабграфа meta см. `meta_backend`, `meta_f0_method`, договоры, тайминги.

---

## 4. QA / CLI

`view_csv_feature_qa.json` (`pitch_extractor`); [`../utils/validate_pitch.py`](../utils/validate_pitch.py) — `<npz> [--struct] [--qa]`.

**Ориентиры:** f0 в Гц: `fmin`–`fmax` (часто 50–2000); `f0_std` и пр. зависят от материала; `voiced_fraction_pyin` ∈ [0, 1] если есть.

---

## 5. Чеклист

1. Сегментные оси **одной длины N**.  
2. `feature_names` / `feature_values` согласованы.  
3. `meta.schema_version` содержит `pitch_extractor`.  
4. Список tabular соответствует **включённым** группам фич.
