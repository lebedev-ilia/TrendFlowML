# Критерии приёмки: spectral_entropy_extractor

**Дата согласования:** 2026-07-16  
**Компонент:** `spectral_entropy_extractor`  
**Schema:** `spectral_entropy_extractor_npz_v2`  
**Тип:** CPU-only (librosa/numpy), нет GPU  

---

## Универсальные хард-гейты (U1–U6)

| Гейт | Описание | Порог | Проверка |
|------|----------|-------|----------|
| U1 | validate_schema + validate_structure rc=0 | 100% | все NPZ в storage |
| U2 | segment_start_sec монотонны, start ≤ end | нет нарушений | проверка diff≥0 |
| U3 | Различимость entropy_mean по видео (CV) | CV ≥ 0.3 | факт: 0.716 |
| U4 | expected-empty: audio_missing → корректная структура | feature_values=[nan,nan], seg=[] | 2 empty NPZ |
| U5 | Golden детерминизм (librosa CPU) | max\|Δ\| ≤ 1 ULP (1.19e-7) | librosa CPU без OMP-пиннинга |
| U6 | Разные длины видео отрабатывают | N=0..30 без ошибок | 17 NPZ |

---

## Специфичные критерии компонента (C1–C3)

| Критерий | Описание | Порог | Комментарий |
|----------|----------|-------|-------------|
| C1 | entropy_mean ∈ [0, log₂(1025)] | [0, ~10.0] | теоретический максимум при n_fft=2048 |
| C2 | NaN в feature_values **только** при status=empty | 0 NaN при ok, NaN by design при empty | audio_missing → [nan, nan] |
| C3 | Различимость CV(entropy_mean) по видео-корпусу | ≥ 0.3 | см. U3 |

---

## Примечания

- **Golden max\|Δ\|=1.19e-7 (1 ULP float32)**: норма для librosa CPU без `OMP_NUM_THREADS=1`. Аналогично color_light (раздел 7). Не требует пиннинга для штампа.
- **Empty-path**: два сценария: (1) `audio_missing_or_extract_failed` (оркестратор) → N=0, seg=[]; (2) `audio_too_short` (main.py) → N=1, mask=[False]. Оба корректны.
- **Flatness/spread по умолчанию отключены** (`enable_flatness=False`, `enable_spread=False`) — в production только basic_stats. Схема поддерживает опциональные поля.
- **GPU не нужен** — полная валидация возможна из 17 NPZ в storage.
