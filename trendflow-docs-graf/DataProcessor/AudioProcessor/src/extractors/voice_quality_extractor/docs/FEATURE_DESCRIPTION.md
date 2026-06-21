# voice_quality_extractor — описание фич (Audit v3/v4)

**Компонент:** `voice_quality_extractor`  
**schema_version NPZ:** `voice_quality_extractor_npz_v1`  
**Контракт:** `voice_quality_contract_v1` (`VOICE_QUALITY_CONTRACT_VERSION` / `meta.voice_quality_contract_version`)

## Назначение

Метрики **качества голоса** по окнам Segmenter (family `voice_quality`): **jitter**, **shimmer**, **HNR-подобный** показатель, статистики **f0**, при необходимости — длинные ряды `f0` / `amps` / `hnr_vals` (в NPZ как 1D; крупные массивы могут дублироваться в `_artifacts/*.npy`, см. README).

## Tabular (`feature_names` / `feature_values`)

**Всегда (числа):** `sample_rate`, `duration`, `f0_fmin`, `f0_fmax`, `segments_count`.

**По флагам `_features_enabled`:** см. `npz_savers/voice_quality.py` — `vq_jitter*`, `vq_shimmer*`, `vq_hnr*`, `vq_f0_*`, `vq_voice_presence_ratio`, при всех трёх базовых метриках — `vq_voice_quality_score`, `vq_breathiness_score`.

**Не в tabular:** `f0_method` (`yin` / `pyin` / `torchcrepe`) — только **meta** (`meta_f0_method` в плоском виде).

## Ось сегмента и per-segment

Длина **N:** `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`.  
При включённых группах: `jitter_by_segment`, `shimmer_by_segment`, `hnr_by_segment` — float32 **[N]**, NaN при `segment_mask=false`.

## Доп. массивы в NPZ

`f0`, `amps`, `hnr_vals` — кадровые/подряд ряды (могут быть пустыми); длина **не** обязана совпадать с N.

## Meta

`voice_quality_contract_version`, `features_enabled`, `device_used`, `f0_method`, `stage_timings_ms`, опц. `voice_quality_resource_profile` (env `AP_VOICE_QUALITY_RESOURCE_PROFILE=1`).

## Пусто

`empty_reason=voice_quality_all_segments_failed` — все сегменты с маской false / неуспех.

## Схема

Детали: `docs/SCHEMA.md`, machine schema: `schemas/voice_quality_extractor_npz_v1.json`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
