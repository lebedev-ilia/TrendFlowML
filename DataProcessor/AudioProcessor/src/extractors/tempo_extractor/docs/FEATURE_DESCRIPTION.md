# tempo_extractor — описание фич (Audit v3/v4)

**Компонент:** `tempo_extractor`  
**schema_version NPZ:** `tempo_extractor_npz_v1`  
**Контракт:** `tempo_contract_v1` (`TEMPO_CONTRACT_VERSION` / `meta.tempo_contract_version` в сейвере)

## Назначение

Оценка **BPM** на базе librosa: onset-энергия → `tempo` / `beat.tempo` (в зависимости от версии librosa), агрегат по кандидатам. В режиме **`run_segments()`** (family `tempo`) — BPM **на сегмент** + глобальные метрики с **полного трека** для совместимости; при полном отказе сегментов — `empty` / `tempo_all_segments_failed`.

## Tabular (`feature_names` / `feature_values`)

| Имя | Смысл |
|-----|--------|
| `tempo_bpm` | Основной BPM (median/mean по `aggregate` на full-track; в сегментном пайпе — из full) |
| `tempo_bpm_mean`, `tempo_bpm_median`, `tempo_bpm_std` | Статистика по вектору кандидатов темпа (full-track) |
| `tempo_confidence` | `1 / (1 + std / (mean + ε))` по кандидатам темпа, типично **0…1** |
| `duration_sec` | `duration` из payload (сек) |
| `sample_rate` | Гц |
| `tempo_bpm_by_segment_mean` / `median` / `std` | Агрегаты по **валидным** `bpm_by_segment` (finite) |
| `segments_count` | Число сегментов N |

`device_used` **не** в tabular (только **meta**), см. `npz_savers/tempo.py` / audit v4.

## Массивы в NPZ (вне tabular)

| Ключ | Описание |
|------|----------|
| `tempo_estimates` | float32 **[K]** — вектор кандидатов темпа (full-track) |
| `segment_start_sec`, `segment_end_sec`, `segment_center_sec` | float32 **[N]** |
| `segment_mask` | bool **[N]** |
| `bpm_by_segment` | float32 **[N]** — BPM на сегмент; **NaN** при fail |
| `warnings` | object: строки предупреждений (`tempo_out_of_range`, `low_confidence`, `signal_too_quiet`, `full_track_failed`, …) |

## Meta

Базовый meta saver + `extra`: `empty_reason`, `tempo_contract_version`, `stage_timings_ms` → плоский **`meta_timing_*`**, опц. `tempo_resource_profile` (env `AP_TEMPO_RESOURCE_PROFILE=1`).

## Пусто / ошибки

- `status=empty`, `empty_reason=tempo_all_segments_failed` — ни один сегмент не дал валидный BPM.

## Схема в репозитории

`DataProcessor/AudioProcessor/schemas/tempo_extractor_npz_v1.json`  
Детали ключей: `docs/SCHEMA.md`.
