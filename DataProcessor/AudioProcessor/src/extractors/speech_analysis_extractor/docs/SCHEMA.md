# speech_analysis_extractor — NPZ Schema (Audit v3)

**schema_version**: `speech_analysis_extractor_npz_v1`  
**producer**: `speech_analysis_extractor`

## Overview

NPZ артефакт speech_analysis_extractor — bundle-агрегатор: объединяет ASR, diarization и опционально pitch в run-level скаляры. Без canonical segment axis; все данные — run-level или per-ASR-segment / per-speaker индексы.

## Tiers

- **model_facing**: `feature_names`, `feature_values` (tabular path)
- **analytics**: `asr_lang_id_by_segment`, `speaker_ids`, `asr_lang_distribution`, `pitch_distribution`
- **debug**: `meta`

## Required keys

| Key | Tier | dtype | shape | Description |
|-----|------|-------|-------|-------------|
| `feature_names` | model_facing | object | [F] | Имена фичей (tabular) |
| `feature_values` | model_facing | float32 | [F] | Значения фичей (tabular) |
| `meta` | debug | object | — | Метаданные |

## Optional keys (feature-gated)

| Key | Tier | dtype | shape | When present |
|-----|------|-------|-------|--------------|
| `asr_lang_id_by_segment` | analytics | int32 | [N_asr] | `asr_metrics` enabled |
| `speaker_ids` | analytics | int32 | [N_spk] | `diarization_metrics` enabled |
| `asr_lang_distribution` | analytics | object | — | `asr_metrics` enabled |
| `pitch_distribution` | analytics | object | — | `pitch_metrics` enabled |

## feature_names / feature_values (tabular)

Базовые (всегда):

- `duration_sec`, `sample_rate`

Feature-gated (при включении):

- **asr_metrics**: `asr_segments_count`, `asr_token_total`, `asr_token_mean`, `asr_token_std`, `asr_token_density_per_sec`, `asr_speech_rate_wpm`
- **diarization_metrics**: `speaker_count`, `dominant_speaker_share`, `speaker_balance_score`, `speaker_transitions_count`, `diar_segments_count`
- **pitch_metrics**: `pitch_enabled`, `pitch_f0_mean`, `pitch_f0_std`, `pitch_f0_min`, `pitch_f0_max`, `pitch_f0_range`, `pitch_stability` — эти имена попадают в NPZ **только если** pitch-зависимость реально смержена в payload (в **`meta.features_enabled`** нет `pitch_metrics`, если `pitch_enabled=false` или нет `pitch_result`)

## Audit v4

- **`device_used`**: строка — только в **`meta`** (baseline), не в tabular (в савере не используется).
- Observability (audit v4.2, optional): `meta.stage_timings_ms`, `meta.speech_analysis_resource_profile` (env: `AP_SPEECH_ANALYSIS_RESOURCE_PROFILE=1`)

## Empty semantics

- `status="empty"`, `empty_reason="audio_too_short"`: аудио < 5 сек
- `status="empty"`, `empty_reason="audio_missing_or_extract_failed"`: тихое аудио (silence detection)

## NaN policy

Missing значения кодируются как **NaN** (не нулевые заглушки).

## Sampling

- Required families: `families.asr`, `families.diarization` в `audio/segments.json`
- No-fallback: пустые сегменты → error
