# Критерии приёмки: detalize_face

**Согласовано:** 2026-07-16  
**Schema:** `detalize_face_npz_v3`  
**Производитель:** `DetalizeFaceModule` v2.0.2 (numpy/CPU, без GPU)  
**Deps:** `core_face_landmarks` (обязательная)

---

## Универсальные хард-гейты (U1–U6)

| # | Гейт | Критерий | Ожидание |
|---|------|----------|----------|
| U1 | Валидатор выхода | `validate_detalize_face_npz.py --struct` rc=0 | PASS на всех тестовых NPZ |
| U2 | Ось времени | `times_s` монотонна, `frame_indices` совпадает с Segmenter-осью | PASS |
| U3 | Finite/health | `primary_compact_features` finite; не константа на корпусе | nan_rate=0%, std>0 |
| U4 | Expected-empty | Видео без лиц → status=empty, валидный NPZ (все ключи, shape корректна) | PASS |
| U5 | Golden-детерминизм | Два прогона на синтетических landmarks → diff=0 | max\|Δ\|=0.0 |
| U6 | Разные длины | N=43/65/69/119 кадров — все отрабатывают без ошибок | PASS |

---

## Компонентные критерии (C1–C5)

| # | Критерий | Порог | Исключение |
|---|----------|-------|------------|
| C1 | `nan_rate(primary_compact_features[face_present])` | = 0% | `primary_compact_features[~primary_valid]` заполняется нулями (не NaN) по дизайну — ОК |
| C2 | `cf_std(primary_compact_features[primary_valid])` по корпусу | > 0.5 | Видео с единственным кадром с лицом — std=0 допустимо |
| C3 | Golden-детерминизм (чистый numpy) | diff = 0.0 | — |
| C4 | `face_present_ratio` на ok-видео (с лицами) | > 0 | Видео без лиц → status=empty, ratio=0.0 — ОК |
| C5 | Schema v3: обязательные ключи, dtype/shape, time_axis согласована с `union_timestamps_sec` | Все поля в норме | — |

---

## Примечания

- `primary_compact_features (N,40)`: нули при `primary_valid=False` — **by design, не NaN**.
- `faces_agg`: dict по tracking_id, пустой при status=empty — ОК.
- `aggregated.compact_l2_mean` порядка ~2000–3000 (крупная шкала); Encoder должен нормализовать или использовать маску.
