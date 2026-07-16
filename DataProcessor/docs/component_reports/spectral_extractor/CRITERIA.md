# Критерии приёмки: spectral_extractor

**Согласовано:** 2026-07-16  
**Версия компонента:** spectral_extractor (schema spectral_extractor_npz_v2)

---

## Универсальные гейты (U1–U6)

| Гейт | Критерий | Применимость |
|------|----------|-------------|
| U1 | validate_spectral.py rc=0 | ✅ |
| U2 | segment_start_sec монотонны | ✅ |
| U3 | status=ok → feature_values 0/46 NaN (все finite) | ✅ |
| U4 | status=empty → fn=5, fv[0-3]=NaN, fv[4]=0.0 (segments_count), empty_reason в meta | ✅ |
| U5 | Golden: 12 ok-runs → max\|Δfv\|=0.0 | ✅ |
| U6 | Разные длины без падений (seg_counts 5–30) | ✅ |

## Специфические критерии (C1–C2)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | feature_names=46 (frozen set) | 46 |
| C2 | spectral_centroid_mean>0, spectral_flatness∈[0,1], zcr≥0 | физически осмысленные |

## Исключения

- **status=empty**: fn=5 базовых параметров (sample_rate/hop_length/n_fft/duration/segments_count), fv[0-3]=NaN (конфиги неизвестны), fv[4]=0.0 — by design
- **seg_n=1 при empty** (placeholder segment arrays) — by design (validate_spectral.py принимает)
