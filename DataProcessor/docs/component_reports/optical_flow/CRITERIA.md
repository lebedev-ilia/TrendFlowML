# Критерии приёмки: optical_flow (module)

Согласованы с владельцем 2026-07-14. Версия компонента 2.0.2, схема optical_flow_npz_v3.

## Универсальные хард-гейты (U1–U6)

| Гейт | Критерий | Порог |
|---|---|---|
| U1 | validate_optical_flow_npz.py --struct --ranges rc=0 на всех видео | rc=0 (0 ошибок) |
| U2 | times_s монотонный ↑, значения из union_timestamps_sec | нарушений = 0 |
| U3 | frame_feature_values finite ≥ 98% (NaN только при missing frames), feature_values finite (кроме empty-пути) | ≥ 0.98 |
| U4 | expected-empty: core.status=empty → optical_flow status=empty, motion=all-NaN, missing_frame_ratio=1.0, rc=0 | PASS |
| U5 | golden побайтово идентичен (CPU-only pure numpy) | diff=[] |
| U6 | Работает на видео ≥5 разных длин (23–300 кадров) | PASS на всех |

## Критерии под компонент (C1–C4)

| Критерий | Порог |
|---|---|
| C1: различимость — std motion_curve_mean по видео | CV ≥ 0.20 |
| C2: per-frame матрица (N,16) — ни один столбец не константа | std > 1e-4 по каждому столбцу на корпусе |
| C3: диапазоны агрегатов | missing_frame_ratio ∈ [0,1]; motion_mean/median/p90 ≥ 0; flow_consistency_mean ∈ [0,1] |
| C4: выравнивание — при совпадении frame_indices с core | missing_count = 0 (NaN=0 в frame_feature_values) |

## Заметки
- bg_ratio ≈ 0.40 by design (from core_optical_flow — известная особенность, не баг)
- NaN в motion_norm_per_sec_mean[0] by design (первый кадр = нет предыдущего)
- Компонент CPU-only (RAFT только в core_optical_flow)
