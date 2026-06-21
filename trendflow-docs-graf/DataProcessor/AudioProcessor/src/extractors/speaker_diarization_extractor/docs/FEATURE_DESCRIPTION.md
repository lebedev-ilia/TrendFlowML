# speaker_diarization_extractor — описание фич (Audit v3/v4)

**Компонент:** `speaker_diarization_extractor`  
**producer_version (код):** `3.1.1` (см. `main.py`)  
**schema_version NPZ:** `speaker_diarization_extractor_npz_v2`  
**Файл:** `speaker_diarization_extractor/speaker_diarization_extractor_features.npz`  
**Контракт:** `diarization_contract_v1` (`DIARIZATION_CONTRACT_VERSION` / `meta.diarization_contract_version`)

## Назначение

Диаризация «кто говорит и когда»: **турны** по полному аудио (одно окно Segmenter family `diarization`), PyTorch/pyannote через **ModelManager**, без сетевых загрузок в рантайме. **Транскрипт/слова** в NPZ **не** хранятся.

## Tabular (F=10, фикс. порядок в сейвере)

| Имя | Смысл |
|-----|--------|
| `speaker_count` | Число говорящих S |
| `duration_sec` | Длительность (с) |
| `sample_rate` | Гц |
| `rms` / `peak` | Амплитудные агрегаты входного сигнала (после загрузки) |
| `speaker_balance_score` | Баланс говорения (0…1) |
| `dominant_speaker_id` | Индекс доминантного спикера (0…S-1) или NaN |
| `speaker_turns_count` | Число турнов K (длина оси turn) |
| `speaker_turns_density` | K / duration_sec |
| `speaker_transitions_count` | Сколько раз менялся говорящий подряд по турнам |

`model_name` / `weights_digest` / `device_used` — только **meta**, не tabular.

## Ось сегмента (ожидается N=1)

`segment_start_sec`, `segment_end_sec`, `segment_center_sec`, `segment_mask` — `float32`/`bool` длины **N**.

## Турны (длина K)

`turn_start_sec`, `turn_end_sec`, `turn_speaker_id` (int32), `turn_mask` (bool) — согласованы по **K**.

## Per-speaker (длина S)

`speaker_ids`, `speaker_duration_sec`, `speaker_time_ratio`, `speaker_turns_count_by_speaker` — согласованы по **S**; `speaker_time_ratio` суммируется ожидаемо к ~1 по речи (зависит от определения «речи» в пайпе).

## Meta

`diarization_contract_version`, `features_enabled`, `stage_timings_ms` (типично: `load_models_ms`, `load_audio_ms`, `to_numpy_ms`, `silence_detection_ms`, `diarize_ms`, `build_payload_ms`, `total_ms`), опц. `speaker_diarization_resource_profile`.

## Пусто

`audio_missing_or_extract_failed`, `audio_silent` — см. `docs/SCHEMA.md`.

## Валидатор

```bash
python utils/validate_speaker_diarization.py <path/to/speaker_diarization_extractor_features.npz> [--struct] [--qa]
```

## Ссылка

Полная схема: `docs/SCHEMA.md`, реализация ключей: `npz_savers/speaker_diarization.py`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
