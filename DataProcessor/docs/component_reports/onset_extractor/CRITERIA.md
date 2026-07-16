# Критерии приёмки: onset_extractor

Версия: v2.1.1 | Дата: 2026-07-16

## Универсальные гейты (U1–U6)

| # | Критерий | Порог |
|---|----------|-------|
| U1 | validate_onset.py --schema --struct --qa rc=0 для всех NPZ | rc=0 всегда |
| U2 | segment_start/end/center/mask одинаковой длины N; feature_names/values длины совпадают | exact |
| U3 | nan_rate = 0 (кроме onset_tempo_consistency — NaN by design при tempo_payload=None) | nan=0 исключая 1 поле |
| U4 | empty NPZ: rc=0, feature_values NaN×4; ok NPZ разных N без падений | rc=0 |
| U5 | Golden: max\|Δ\| ≈ 0.0 для finite значений (6e-08, float32 точность); librosa детерминирован | <1e-6 |
| U6 | Дискриминативность: onset_count CV=67%, onset_density_per_sec CV=34%, onset_strength_mean CV=39% | ≥10% |

## Специфичные критерии (C1–C4)

| # | Критерий | Порог |
|---|----------|-------|
| C1 | onset_tempo_consistency: NaN допустим при tempo_payload=None (опциональный dep tempo_extractor); конечен при наличии dep | NaN OK при absent dep |
| C2 | onset_count ≥ 0, целое; onset_density_per_sec > 0 для ненулевого аудио | ≥0 |
| C3 | onset_regularity_score ∈ [0, 1]; onset_syncopation_score ∈ [0, 1] | [0,1] |
| C4 | segment axis (start/end/center/mask) N-согласованы; нет onset_by_segment (только scalar) | exact |

## Примечания по дизайну

- **onset_tempo_consistency = NaN**: по design когда `tempo_payload is None` (tempo_extractor не был включён в run config). Из 10 runs: 5 с tempo_payload (0.737), 5 без (NaN).
- **empty NPZ**: 4 feature_values (не 19!) — либо сокращённая схема, либо только config params; всё NaN.
- **Нет onset_by_segment**: компонент не разбивает onsets по сегментам — только scalar aggregates + segment axis (для temporal контекста).
