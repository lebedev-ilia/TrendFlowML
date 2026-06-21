# tempo_extractor NPZ schema (Audit v3)

**schema_version**: `tempo_extractor_npz_v1`

## Overview

NPZ артефакт tempo_extractor содержит:
- **Model-facing**: `feature_names`, `feature_values` (скаляры)
- **Analytics**: canonical segment axis + `bpm_by_segment`
- **Meta**: baseline meta + `status`, `empty_reason`

## Canonical axis

- `segment_start_sec` (float32[N]) — начало каждого сегмента (сек)
- `segment_end_sec` (float32[N]) — конец каждого сегмента (сек)
- `segment_center_sec` (float32[N]) — центр сегмента (сек)
- `segment_mask` (bool[N]) — маска валидных сегментов (False = сегмент failed)

## Per-segment arrays

- `bpm_by_segment` (float32[N]) — BPM для каждого сегмента. NaN для failed сегментов.

## Scalar features (feature_names / feature_values)

- `tempo_bpm` — основной BPM (median/mean по валидным сегментам)
- `tempo_bpm_mean`, `tempo_bpm_median`, `tempo_bpm_std`
- `tempo_confidence` — уверенность (0.0–1.0)
- `duration_sec`, `sample_rate`, `segments_count`
- `tempo_bpm_by_segment_mean`, `tempo_bpm_by_segment_median`, `tempo_bpm_by_segment_std` (по валидным сегментам)

## Empty semantics

- `status="empty"`, `empty_reason="tempo_all_segments_failed"` — все сегменты failed
- При partial failures: `segment_mask` содержит False для failed сегментов, `bpm_by_segment` — NaN для них

## NaN policy

- Missing/failed segment BPM → NaN в `bpm_by_segment`
- Агрегаты (mean/median/std) считаются только по валидным (finite) значениям

## Machine schema

`DataProcessor/AudioProcessor/schemas/tempo_extractor_npz_v1.json`

## Audit v4

- **Tabular:** только числа; **`device_used`** — в **`meta`** (baseline), не в `feature_values` (савер их не добавляет).
- **`duration_sec`** в tabular — **`duration`** из payload (в `run_segments` это оценка на аудио, загруженном как полный трек для глобальных `tempo_bpm_*`). На reference **A** значение совпало с \(\max(segment\_end\_sec)\); при расхождениях с другими экстракторами проверяйте загрузку WAV и окна family **tempo**.
- **`meta`**: поле **`tempo_contract_version`** (строка, по умолчанию `tempo_contract_v1`), задаётся савером из payload экстрактора. **`features_enabled`** по-прежнему не пишется (у tempo нет feature-gates в NPZ).
- Observability (audit v4.2, optional): `meta.stage_timings_ms`, `meta.tempo_resource_profile` (env: `AP_TEMPO_RESOURCE_PROFILE=1`)
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
