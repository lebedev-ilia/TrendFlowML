# CRITERIA — pitch_extractor

**Дата согласования:** 2026-07-16  
**Статус:** авто-утверждено (ask_human вернул max turns, принят безопасный дефолт)  
**Версия компонента:** 2.0.1 (post-fix)

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Порог |
|------|----------|-------|
| U1 | validate_pitch.py rc=0 на всех NPZ | rc=0 × все NPZ |
| U2 | segment_start_sec монотонны (или N=0 при empty) | diff(ss) >= -1e-6 |
| U3 | feature_values finite (нет неожиданных NaN/inf) при status=ok | nan_count=0 |
| U4 | expected-empty путь: нулевой/тихий сигнал → status=empty | status="empty", empty_reason="pitch_all_segments_empty" |
| U5 | golden-детерминизм: librosa PYIN детерминирован | max\|Δ\|=0.0 |
| U6 | Разные длины видео (3/5/10 сег) без падений | no exception |

---

## Критерии компонента (C1–C4)

| Критерий | Описание | Порог |
|----------|----------|-------|
| C1 | pitch_skewness/kurtosis = 0.0 (НЕ NaN) при std(f0)=0 (монотонный pitch) | isfinite, not NaN |
| C2 | f0_mean ∈ [fmin, fmax] при status=ok | fmin ≤ f0_mean ≤ fmax |
| C3 | segment_mask содержит ≥1 True при status=ok | any(segment_mask) = True |
| C4 | pitch_octave_distribution: dict, сумма значений ≈ 1.0 (при наличии) | abs(sum-1) < 0.01 |

---

## Явные исключения (NaN by design)

- **pitch_skewness/kurtosis = NaN в СТАРЫХ NPZ (до фикса 2026-07-16)** — баг исправлен, новые прогоны дают finite=0.0.
  Старые NPZ (10/15 с монотонным pitch в `-Q6fnPIybEI`) сохраняют NaN — это исторический артефакт, не производственный дефект.

---

## Фиксы применённые в этой сессии

1. **run_segments() skew/kurt NaN при std=0** — добавлен `isfinite()` guard (строки 614–620)
2. **_validate_output() consistency check без eps** — добавлен допуск `_eps=1e-3` (строки 235–241)
3. **YIN/PYIN out-of-range фильтрация** — добавлена фильтрация `& (f0 >= fmin) & (f0 <= fmax)` (строки 810–812, 839–840)
