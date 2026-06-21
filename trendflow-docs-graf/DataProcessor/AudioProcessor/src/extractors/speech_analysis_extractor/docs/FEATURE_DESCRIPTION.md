# speech_analysis_extractor — описание фич (Audit v3/v4)

**Компонент:** `speech_analysis_extractor`  
**schema_version NPZ:** `speech_analysis_extractor_npz_v1`  
**Контракт:** `speech_analysis_contract_v1` (`SPEECH_ANALYSIS_CONTRACT_VERSION` / `meta.speech_analysis_contract_version`)

## Назначение

Run-level **bundle** без сырого текста: агрегирует результаты **ASR** (токены/id языка), **диаризации** (спикеры, доли) и опционально **pitch** из уже выполненных экстракторов (`extractor_results`). Выравнивание токен ↔ спикер **не** делается.

## Tabular (`feature_names` / `feature_values`)

**Всегда (при успешном сохранении):** `duration_sec`, `sample_rate`.

**Если в `meta.features_enabled` есть `asr_metrics`:**  
`asr_segments_count`, `asr_token_total`, `asr_token_mean`, `asr_token_std`, `asr_token_density_per_sec`, `asr_speech_rate_wpm`.

**Если есть `diarization_metrics`:**  
`speaker_count`, `dominant_speaker_share`, `speaker_balance_score`, `speaker_transitions_count`, `diar_segments_count`.

**Если есть `pitch_metrics`:**  
`pitch_enabled` и скаляры `pitch_f0_*`, `pitch_stability` (см. сейвер). Поля pitch **не** пишутся в tabular, если pitch не смержен (флаг `pitch_metrics` не попадает в enabled).

`device_used` — только **meta**, не tabular.

## Analytics (помимо tabular)

| Ключ | Описание |
|------|----------|
| `asr_lang_id_by_segment` | int32 **[N_asr]** — id языка по ASR-окнам |
| `speaker_ids` | int32 **[N_spk]** — уникальные/упорядоченные id спикеров из diar |
| `asr_lang_distribution` | object (dict): доли по языкам |
| `pitch_distribution` | object (dict): доли по октавным бинам и т.п. |

Пустой массив и пустой dict допустимы, если соответствующий блок отключён (сейвер кладёт нули/пустое).

## Meta

`speech_analysis_contract_version`, `features_enabled`, `stage_timings_ms` (→ `meta_timing_*`), опц. `speech_analysis_resource_profile` (env `AP_SPEECH_ANALYSIS_RESOURCE_PROFILE=1`).

Типичные тайминги: `silence_detection_ms`, `asr_ms`, `diarization_ms`, `pitch_ms`, `aggregates_ms`, `total_ms`.

## Пусто

- `audio_too_short` — длительность по сегментам **короче 5 с** (эвристика)  
- `audio_missing_or_extract_failed` — тихий сегмент (silence probe)

## Схема

Подробнее: `docs/SCHEMA.md`.
---

## Навигация

[README](README.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
