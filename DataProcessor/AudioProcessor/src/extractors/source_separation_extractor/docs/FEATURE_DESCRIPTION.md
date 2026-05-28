# source_separation_extractor — описание фич (Audit v3/v4)

**Компонент:** `source_separation_extractor`  
**producer_version (код):** `3.0.1` (см. `main.py`)  
**schema_version NPZ:** `source_separation_extractor_npz_v2`  
**Файл:** `source_separation_extractor/source_separation_extractor_features.npz`  
**Контракт:** `source_separation_contract_v1` (`SOURCE_SEPARATION_CONTRACT_VERSION` / `meta.source_separation_contract_version`)

## Назначение

Разделение микса на 4 источника **vocals, drums, bass, other** (порядок фиксирован в `source_order` и `runtime_params` модели). Сте́мы **не** сохраняются в NPZ; выход — **доли энергии (shares)** по окнам Segmenter (family `source_separation`), агрегаты и опциональные ряды/качество.

**Пороги:** длина аудио **короче 5 с** → empty `audio_too_short`; тишина по окнам → empty `audio_silent`; нет **fallback** при сбое модели.

## Tabular (`feature_names` / `feature_values`) — фиксированный набор (v2)

| Поле | Смысл |
|------|--------|
| `share_vocals_mean` … `share_other_mean` | Средние доли 4 классов (по `share_mean`, агрегат по **немаскированным** сегментам) |
| `dominant_source_id` | Индекс доминирующего источника (0…3) |
| `dominant_source_share` | Доля выбранного класса (0…1) |
| `source_balance_score` | Скаляр баланса (см. main; выше — равномернее) |
| `source_transitions_count` | Число смены доминанты вдоль N |
| `source_stability_score` | Стабильность распределения (см. main) |
| `segments_count` | N (число сегментов) |
| `sample_rate` | Гц (из препроцесса/модели) |

`device_used`, `model_name`, `weights_digest` — **только в `meta`**, не в `feature_values`.

## Векторы и сегменты (NPZ)

| Ключ | shape | Примечание |
|------|-------|------------|
| `segment_start_sec` / `end` / `center` | N | float32 |
| `segment_mask` | N | `false` — тихое/нулевое окно, исключено из агрегатов |
| `share_mean` | 4 | float32, средние доли |
| `share_std` | 4 | опц., если флаг + вычислено |
| `source_distribution_ratio` / `source_segments_count` / `source_duration_sec` | 4 | структурированная статистика по источникам |
| `source_order` | 4 (object) | `["vocals","drums","bass","other"]` |
| `share_sequence` | (N,4) | опц., квант в токенайзер |
| `energy_sequence` | (N,4) | опц. |

## Опц. скаляры (если в payload)

`source_entropy_mean`, `energy_balance_mean` и др., per-source `*_delta_*`, `*_stability`, `*_dominance_ratio`, `quality_*` — см. `npz_savers/source_separation.py` и machine schema.

## Meta

Базовый meta + `model_name`, `weights_digest`, `features_enabled` (как list — не плоский flatten), `stage_timings_ms`, опц. `source_separation_resource_profile`.

## Валидатор

```bash
python utils/validate_source_separation.py <path/to/source_separation_extractor_features.npz> [--struct] [--qa]
```

## Ссылка

Детальная схема: `docs/SCHEMA.md`.
