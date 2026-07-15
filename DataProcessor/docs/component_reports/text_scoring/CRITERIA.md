# Критерии приёмки: text_scoring

**Согласовано:** 2026-07-16  
**VERSION:** 2.0.1 · схема text_scoring_npz_v2 · 35 признаков

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Критерий |
|------|----------|
| U1 | `validate_text_scoring.py --struct --ranges` rc=0 на ≥3 видео (ok + empty пути) |
| U2 | `frame_indices` строго возрастающие; `times_s` соответствуют `union_timestamps_sec[frame_indices]` |
| U3 | Различимость: status=ok при наличии OCR, status=empty при отсутствии; feature_values CV>0 между разными OCR-наборами |
| U4 | Все 3 expected-empty пути → status=empty rc=0: (а) нет ocr.npz, (б) пустой ocr_raw, (в) OCR за пределами frame_indices |
| U5 | Golden детерминизм: повтор с теми же данными → diff=0.0 (CPU-numpy/scipy детерминированы) |
| U6 | N=5, N=30, N=200 кадров отрабатывают без падений |

## Критерии компонента (C1–C4)

| ID | Критерий | Порог |
|----|----------|-------|
| C1 | status=ok → text_present=1.0, text_frames_ratio∈(0,1], core-блок (text_frames_ratio/count/continuity) без NaN | 0 NaN в core-блоке при ok-пути |
| C2 | status=empty → text_present=0.0 финит, остальные 34 NaN **by design** (OK) | 34 NaN ожидаемы |
| C3 | CTA-блок (cta_timestamp…cta_last_position) NaN при cta_presence=0 **by design** (OK) | 9 NaN ожидаемы при нет CTA |
| C4 | text_area_fraction∈[0,1], text_frames_ratio∈[0,1], text_on_screen_continuity_normalized∈[0,1] при status=ok | все ∈ [0,1] |

## NaN by design (явные исключения)

- `status=empty` → 34 из 35 фич NaN (только text_present=0.0). **Это норма.**
- `cta_timestamp`, `cta_first_timestamp`, `cta_mean_timestamp`, `cta_last_timestamp`, `cta_first_position`, `cta_mean_position`, `cta_last_position`, `time_to_first_text_sec`, `time_to_first_text_position` → NaN при отсутствии CTA / текста. **By design.**
- `ocr_language_entropy`, `text_movement_speed`, `text_emphasis_peaks_count` → NaN при `enable_language_entropy/enable_text_movement_speed/enable_text_peaks=False`. **By design (отключённые фичефлаги).**
