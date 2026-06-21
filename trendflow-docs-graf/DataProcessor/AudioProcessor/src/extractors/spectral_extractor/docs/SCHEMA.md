# spectral_extractor — NPZ Schema (Audit v3)

**schema_version**: `spectral_extractor_npz_v2`  
**producer**: `spectral_extractor`

## Overview

NPZ артефакт spectral_extractor содержит спектральные признаки (centroid, bandwidth, flatness, rolloff, ZCR, contrast, slope) по сегментам Segmenter (family `spectral`). Без ключа `payload`; все данные в плоских ключах.

## Tiers

- **model_facing**: `feature_names`, `feature_values` (tabular path)
- **analytics**: canonical axis + per-segment arrays
- **debug**: `meta`

## Required keys

| Key | Tier | dtype | shape | Description |
|-----|------|-------|-------|-------------|
| `feature_names` | model_facing | object | [F] | Имена фичей (tabular) |
| `feature_values` | model_facing | float32 | [F] | Значения фичей (tabular) |
| `segment_start_sec` | analytics | float32 | [N] | Начало сегментов (сек) |
| `segment_end_sec` | analytics | float32 | [N] | Конец сегментов (сек) |
| `segment_center_sec` | analytics | float32 | [N] | Центр сегментов (сек) |
| `segment_mask` | analytics | bool | [N] | Маска валидных сегментов |
| `meta` | debug | object | — | Метаданные |

## Optional keys (feature-gated)

| Key | Tier | dtype | shape | When present |
|-----|------|-------|-------|--------------|
| `centroid_mean_by_segment` | analytics | float32 | [N] | `basic_features` |
| `bandwidth_mean_by_segment` | analytics | float32 | [N] | `basic_features` |
| `flatness_mean_by_segment` | analytics | float32 | [N] | `basic_features` |
| `rolloff_mean_by_segment` | analytics | float32 | [N] | `basic_features` |
| `zcr_mean_by_segment` | analytics | float32 | [N] | `basic_features` |
| `contrast_mean_by_segment` | analytics | float32 | [N] | `contrast` |
| `slope_mean_by_segment` | analytics | float32 | [N] | `advanced_features` |
| `spectral_contrast_bands` | analytics | float32 | [B,T] | `contrast` + `keep_contrast_bands` |

## Empty semantics

- `status="empty"`, `empty_reason="audio_too_short"`: аудио < 1 сек
- `status="empty"`, `empty_reason="spectral_all_segments_failed"`: все сегменты failed

## NaN policy

Missing значения кодируются как **NaN** (не нулевые заглушки).

## Sampling

- Required family: `families.spectral` в `audio/segments.json`
- No-fallback: отсутствие family → error

## Tabular / meta (Audit v4)

- **`device_used`**: строка — только в **`meta`** (baseline), не в `feature_values` (раньше ошибочно попадала в tabular и давала NaN).
- При **`run_segments()`** поле **`duration`** в tabular — **охват оси сегментов** \(\max(segment\_end\_sec)-\min(segment\_start\_sec)\), не обязательно полная длительность файла; **`hop_length`**, **`n_fft`** берутся из конфигурации экстрактора и дублируются в **`meta`** при наличии в payload.
- Observability (Audit v4.2, optional): `meta.stage_timings_ms`, `meta.spectral_resource_profile` (env: `AP_SPECTRAL_RESOURCE_PROFILE=1`).
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
