# Критерии приёмки: tempo_extractor

Версия: v2.1.1 | Дата: 2026-07-16

## Универсальные гейты (U1–U6)

| # | Критерий | Порог |
|---|----------|-------|
| U1 | validate_tempo.py --schema --struct --qa rc=0 для всех NPZ | rc=0 всегда |
| U2 | segment_start/end/center/mask одинаковой длины N; bpm_by_segment.size=N | exact |
| U3 | nan_rate = 0 в feature_values (ok NPZ); NaN by design в empty NPZ (все 11 feature NaN) | nan=0 для ok |
| U4 | empty NPZ: rc=0, все feature_values NaN; ok NPZ разных длин N без падений | rc=0 |
| U5 | Golden: max\|Δ\| = 0.0 (librosa beat tracking детерминирован при одном входном файле) | 0.0 |
| U6 | Дискриминативность: tempo_bpm_mean CV ≥ 10% или tempo_bpm_std CV ≥ 10% | ≥10% |

## Специфичные критерии (C1–C4)

| # | Критерий | Порог |
|---|----------|-------|
| C1 | bpm_by_segment: size=N, NaN допустим при mask=False (сегмент не прошёл BPM оценку) | size=N |
| C2 | tempo_bpm > 0, finite; типичный диапазон 60..240 BPM | (0, 300] |
| C3 | tempo_confidence ∈ [0, 1], finite | [0,1] |
| C4 | tempo_estimates: 1D array (debug vector), finite или absent | 1D |

## Примечания по дизайну

- **bpm_by_segment NaN при mask=False**: сегмент слишком короткий или без ритма → BPM не вычисляется, значение NaN. Это by design.
- **tempo_bpm часто = tempo_bpm_median**: librosa beat.beat_track возвращает один BPM → global==median. Различимость лучше через tempo_bpm_mean (per-segment average).
- **tempo_confidence CV=2.1%**: librosa всегда даёт confidence ~0.9 для аудио с ритмом. Не discriminative, но не баг.
- **empty NPZ**: все 11 feature_values = NaN by design при audio_missing_or_extract_failed.
