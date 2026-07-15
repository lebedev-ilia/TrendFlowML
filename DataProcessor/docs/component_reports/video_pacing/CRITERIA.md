# CRITERIA — video_pacing (согласовано с владельцем 2026-07-13)

Компонент: `video_pacing` (VisualProcessor). Темп монтажа: shot/склейки, motion/semantic/color кривые.
Выход: NPZ `video_pacing/video_pacing_features.npz`, schema `video_pacing_npz_v3`, producer_version 2.0.1.
Model-facing: 3 кривые длины N + `shot_boundary_frame_indices (S)` + 57 табличных скаляров `_FEATURE_NAMES_V1`.
Hard deps (no-fallback): `cut_detection`, `core_clip`, `core_optical_flow` — union-domain, согласованы по индексам.

## Решение по фиче-гейтингу (вариант B)
Дефолт: **enable_entropy_features=true, enable_histograms=true**; **pace_curve_peaks / periodicity / bursts = off**.
- Оживает 8 фич: shot_duration_entropy, shot_length_gini, tempo_entropy, 5×shot_length_histogram_5bins.
- Остаются NaN by-design (гейтнуты, 5 фич): pace_curve_peaks_mean_prominence, pace_curve_dominant_period_sec,
  pace_curve_power_at_period, semantic_change_burst_count, color_change_bursts.
- Итог: **52 активных фичи / 5 NaN-by-design** из 57.

## Универсальные хард-гейты (pass/fail)
- U1: валидатор `validate_video_pacing.py --struct --ranges --qa` → rc=0.
- U2: ось времени: `frame_indices` строго ↑, `times_s = union_timestamps_sec[frame_indices]`, монотонна.
- U3: активные фичи finite и не-константа по матрице роликов; кривые длины N, finite (кроме 0 в первом элементе где by-design).
- U4: expected-empty путь работает (сдвиг индексов деп → нет пересечения → штатный no-fallback/empty).
- U5: golden-детерминизм (см. C4).
- U6: разные длины видео отрабатывают (короткие/длинные из матрицы).

## Критерии под компонент
- **C1 (различимость):** CV = std/mean по роликам > 0.30 хотя бы у 3 из 4 метрик:
  `cuts_per_10s`, `shot_duration_mean`, `mean_motion_speed_per_shot`, `frame_embedding_diff_mean`.
- **C2 (согласованность деп):** `shots_count` согласован с числом shot-boundaries из cut_detection;
  3 кривые длины N; `motion_norm_per_sec_mean ≥ 0` для finite; доли ∈ [0,1].
- **C3 (NaN-политика):** в активной части вектора (52 фичи) NaN допустим ТОЛЬКО у structural
  (climax_speed/pacing_symmetry) на малошотовых роликах (shots_count<4) — иначе fail.
  5 гейтнутых фич = NaN by-design (не считаются нарушением).
- **C4 (golden):** детерминированные numpy/opencv-фичи (SSIM/LAB/HSV/Canny) → diff=0 при пиннинге потоков
  (OMP_NUM_THREADS=1). Кривые motion/semantic/color совпадают побайтово между прогонами.

## Сырые числа — в REPORT_YYYY-MM-DD.md
