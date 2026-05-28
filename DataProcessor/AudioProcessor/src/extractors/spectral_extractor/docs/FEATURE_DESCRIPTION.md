# spectral_extractor — описание фич (Audit v3/v4)

**Компонент (NPZ / CSV):** `spectral_extractor`  
**producer_version (код):** `2.0.1` (см. `main.py`; внутренний ключ экстрактора — `spectral`)  
**schema_version NPZ:** `spectral_extractor_npz_v2`  
**Файл:** `spectral_extractor/spectral_extractor_features.npz`  
**Контракт:** `spectral_contract_v1` (`SPECTRAL_CONTRACT_VERSION` в payload / `meta.spectral_contract_version`)

## Назначение

Базовые спектральные признаки librosa на сегментах Segmenter (family `spectral`): центроид, ширина, плоскостность, rolloff, ZCR; опционально — контраст и наклон спектра; агрегаты по валидным сегментам + per-segment средние.

## Gating (включает группы)

| Флаг | Что пишется |
|------|-------------|
| `enable_basic_features` | `spectral_*_stats`, per-seg `*_mean_by_segment`, доп. метрики (ratio, entropy, correlation) |
| `enable_contrast` | `spectral_contrast_*`, `contrast_mean_by_segment`, опц. `spectral_contrast_bands` [B,T] |
| `enable_advanced_features` | `spectral_slope_*`, `slope_mean_by_segment`, опц. `spectral_flatness_db_stats` |
| `enable_time_series` | кадровые series (и/или пути `*_npy` для длинных рядов) |

## Табular (`feature_names` / `feature_values`)

Скаляры, выровненные с именами (см. `npz_savers/spectral.py`):

- **Параметры/объём:** `sample_rate`, `hop_length`, `n_fft`, `duration`, `segments_count`
- **База:** `spectral_centroid_*`, `spectral_bandwidth_*`, `spectral_flatness_*`, `spectral_rolloff_*`, `zcr_*` (mean/std/min/max/median где есть)
- **Доп. basic:** `spectral_centroid_median_metric`, `spectral_bandwidth_ratio`, `spectral_rolloff_ratio`, `spectral_flatness_entropy`
- **Contrast:** `spectral_contrast_*`, `spectral_contrast_variance`
- **Advanced:** `spectral_slope_*`, `spectral_slope_stability`, при наличии — `spectral_flatness_db_*`

Единицы: частоты в **Гц**; flatness и ZCR в **[0, 1]**; contrast/slope в единицах librosa/агрегата.

## Ось сегментов (аналитика)

Обязательные массивы длины **N** (`[N]` float32/bool):

- `segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask`

Per-segment (при включённых группах), те же **N**, NaN при маске/ошибке:

- `centroid_mean_by_segment`, `bandwidth_mean_by_segment`, `flatness_mean_by_segment`, `rolloff_mean_by_segment`, `zcr_mean_by_segment`
- `contrast_mean_by_segment`, `slope_mean_by_segment` — по флагам

Опционально: `spectral_contrast_bands` — **float32** `[B, T]` (если contrast + `keep_contrast_bands`).

## Meta (`meta` object)

Помимо базовых полей saver (`producer`, `producer_version`, `schema_version`, `status`, `created_at`):

- `spectral_contract_version` — контракт
- `device_used`, `sample_rate`, `hop_length`, `n_fft`, `average_channels`, `keep_contrast_bands`
- `features_enabled` — список строк (в плоский CSV обычно не разворачивается)
- `stage_timings_ms` — этапы (в плоский вид: `meta_timing_*`)
- опц. `spectral_resource_profile` (env `AP_SPECTRAL_RESOURCE_PROFILE=1`)
- опц. `spectral_features_correlation` (dict, не плоский)
- `empty_reason` при `status=empty`

## Семантика пусто

- `empty` + `audio_too_short` — длина короче 1 с (run)
- `empty` + `spectral_all_segments_failed` — все сегменты невалидны (run_segments)

## Валидатор

```bash
python utils/validate_spectral.py <path/to/spectral_extractor_features.npz> [--struct] [--qa]
```

## Ссылка на схему

Подробные ключи и tier: `docs/SCHEMA.md`.
