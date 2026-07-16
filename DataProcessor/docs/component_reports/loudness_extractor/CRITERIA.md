# Критерии приёмки: loudness_extractor

**Согласовано:** 2026-07-16  
**Версия компонента:** loudness_extractor v2.1.1 (RMS/peak/dBFS + optional LUFS)

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий | Применимость |
|------|----------|-------------|
| U1 | validate_loudness.py rc=0 | ✅ |
| U2 | segment_start_sec монотонны | ✅ |
| U3 | status=ok → feature_values ≥17/18 finite (loudness_lufs=NaN by design) | ✅ |
| U4 | status=empty → fv NaN×18, seg_n=0, reason заполнен | ✅ |
| U5 | Golden: 10 ok-runs одного видео → max\|Δfv\|<1e-5 | ✅ |
| U6 | Разные длины без падений (15 ok-видео) | ✅ |

## Специфические критерии (C1–C3)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | feature_names = 18 штук (frozen set) | 18 |
| C2 | segment_rms∈[0,1], segment_peak≤1.0+ε | физически осмысленные значения |
| C3 | loudness_lufs=NaN ↔ lufs_present=False (консистентность флага) | 100% консистентно |

## Исключения

- **loudness_lufs=NaN by design** — pyloudnorm optional dep; lufs_present=False при недоступности
- **status=empty** (2 NPZ): fv NaN×18, seg_n=0 (audio_missing_or_extract_failed) — by design
- **U5 max|Δ|=6.71e-08** — float32 precision noise, не детерминизм-баг
