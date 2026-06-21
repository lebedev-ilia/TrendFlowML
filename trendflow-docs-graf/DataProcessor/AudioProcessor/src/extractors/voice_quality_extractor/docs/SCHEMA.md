# voice_quality_extractor NPZ schema (Audit v3)

**schema_version**: `voice_quality_extractor_npz_v1`

## Overview

NPZ артефакт voice_quality_extractor содержит:
- **Model-facing**: `feature_names`, `feature_values` (скаляры)
- **Analytics**: canonical segment axis + per-segment arrays
- **Meta**: baseline meta + `status`, `empty_reason`, `features_enabled`

## Canonical axis

- `segment_start_sec` (float32[N]) — начало каждого сегмента (сек)
- `segment_end_sec` (float32[N]) — конец каждого сегмента (сек)
- `segment_center_sec` (float32[N]) — центр сегмента (сек)
- `segment_mask` (bool[N]) — маска валидных сегментов (False = сегмент failed)

## Per-segment arrays (feature-gated)

- `jitter_by_segment` (float32[N]) — jitter для каждого сегмента. NaN для failed.
- `shimmer_by_segment` (float32[N]) — shimmer для каждого сегмента. NaN для failed.
- `hnr_by_segment` (float32[N]) — HNR (dB) для каждого сегмента. NaN для failed.

## Scalar features (feature_names / feature_values)

Feature-gated по `_features_enabled`:
- **jitter**: `vq_jitter`, `vq_jitter_mean`, `vq_jitter_std`, `vq_jitter_min`, `vq_jitter_max`
- **shimmer**: `vq_shimmer`, `vq_shimmer_mean`, ...
- **hnr**: `vq_hnr_like_db`, `vq_hnr_mean`, ...
- **f0_stats**: `vq_f0_mean`, `vq_f0_std`, `vq_f0_min`, `vq_f0_max`, `vq_f0_median`, `vq_f0_stability`, `vq_voice_presence_ratio`
- **quality** (если jitter+shimmer+hnr): `vq_voice_quality_score`, `vq_breathiness_score`

Всегда (tabular — числа): `sample_rate`, `duration`, `f0_fmin`, `f0_fmax`, `segments_count`.

**`f0_method`** (`"yin"` \| `"pyin"` \| `"torchcrepe"`): только в **`meta`**, не в `feature_values`.

## Empty semantics

- `status="empty"`, `empty_reason="voice_quality_all_segments_failed"` — все сегменты failed
- При partial failures: `segment_mask` содержит False для failed сегментов

## NaN policy

- Missing/failed segment → NaN в per-segment массивах
- Агрегаты (mean/median/std) считаются только по валидным (finite) значениям
- Model-facing скаляры при отсутствии фичи → NaN (не 0)

## Pitch integration

Опциональная интеграция с `pitch_extractor`: при совпадении families и наличии f0 в pitch_payload — переиспользуется. Иначе — своя оценка f0. Документировано в README.

## Machine schema

`DataProcessor/AudioProcessor/schemas/voice_quality_extractor_npz_v1.json`

## Audit v4

- Строковый **`f0_method`** не должен проходить через float tabular (раньше давал **NaN** через `as_float`) — **исправлено в `npz_savers/voice_quality.py`**; значение в **`meta.f0_method`**.
- Observability (audit v4.2, optional): `meta.stage_timings_ms`, `meta.voice_quality_resource_profile` (env: `AP_VOICE_QUALITY_RESOURCE_PROFILE=1`)
---

## Навигация

[README](README.md) · [FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
