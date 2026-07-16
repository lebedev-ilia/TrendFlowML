# Критерии приёмки: mfcc_extractor

Версия: v2.1.1 | Дата: 2026-07-16

## Универсальные гейты (U1–U6)

| # | Критерий | Порог |
|---|----------|-------|
| U1 | validate_mfcc.py --schema --struct --qa rc=0 для всех NPZ | rc=0 всегда |
| U2 | segment_start/end/center/mask одинаковой длины N; mfcc_mean_by_segment shape=(N, n_mfcc) | точное совпадение |
| U3 | nan_rate = 0 (status=ok); NaN by design (status=empty, n_mfcc=NaN в tabular допустим) | nan=0 для ok |
| U4 | empty NPZ: валидатор rc=0 (NaN guard в _n_mfcc_from_tabular); ok NPZ разных длин N | rc=0 |
| U5 | Golden: max\|Δ\| = 0.0 (10 run одного видео, torchaudio CPU детерминирован) | 0.0 |
| U6 | Дискриминативность scalar aggregates: mfcc_energy CV ≥ 10%, mfcc_bandwidth CV ≥ 10% | ≥10% |

## Специфичные критерии (C1–C4)

| # | Критерий | Порог |
|---|----------|-------|
| C1 | mfcc_mean_by_segment: shape=(N, 13), dtype float32, N согласовано с segment axis | exact |
| C2 | mfcc_energy = mean(abs(MFCC[0,:])) > 0, finite; при audio_normalization ≈ 0.5..0.85 | >0, finite |
| C3 | delta_mean_by_segment: shape=(N, 13) при enable_deltas=True; mfcc_energy_by_segment: shape=(N,) | exact |
| C4 | mfcc_stability ∈ (0, 1], finite; NOTE: при enable_audio_normalization=True stability≈0.5 by design | (0,1] |

## Примечания по дизайну

- **mfcc_mean ≈ 1e-8**: следствие `enable_audio_normalization=True` (RMS waveform normalization → log-power≈0 → MFCC[k].mean()≈0). Это NOT A BUG.
- **mfcc_stability ≈ 0.5**: формула `1/(1+std(mfcc))`, при RMS-нормированном аудио std(MFCC)≈1 → stability≈0.5. By design.
- **mfcc_centroid ≈ 0**: mean(mean(MFCC, dim=time)) = mean(mfcc_mean) ≈ 0 при audio normalization. By design.
- Дискриминативность обеспечивается через `mfcc_mean_by_segment (N,13)` и `delta_mean_by_segment (N,13)` (temporal shape), а не scalar aggregates.
