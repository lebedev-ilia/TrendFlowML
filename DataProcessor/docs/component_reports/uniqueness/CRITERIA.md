# Критерии приёмки: uniqueness

Дата согласования: 2026-07-16  
Компонент: uniqueness (VisualProcessor)  
Схема: uniqueness_npz_v4  
Статус: авто-штамп (все гейты PASS)

---

## Универсальные хард-гейты

| Гейт | Критерий |
|------|----------|
| U1 | validate_uniqueness.py батч-режим → rc=0 на всех 23 реальных NPZ |
| U2 | frame_indices строго возрастают; times_s неубывает; размер cos_dist_next = N-1 |
| U3 | diversity_score варьируется между видео (std=0.085, CV=1.156); max_sim_to_other ∈ [0,1] finite; cos_dist_next ∈ [0,2] finite |
| U4 | Empty-path: если frame_indices пуст → run() бросает RuntimeError (no-fallback по дизайну); process_batch BaseModule перехватывает → status=empty без NPZ. Не создаёт empty NPZ — это по дизайну. |
| U5 | Golden детерминизм: чистый numpy, max\|Δ\|=0.0 на 23 парах core_clip→uniqueness (N от 12 до 119) |
| U6 | Разные длины: N=12,43,65,69,119 — все 23 OK без падений |

## Критерии компонента

| Критерий | Порог |
|----------|-------|
| C1 | health: feature_values nan_rate = 0.0 на всех ok-NPZ (0/23) |
| C2 | diversity_score ∈ [0,1]: min=0.033, max=0.314 — ненулевой диапазон (различимость подтверждена) |
| C3 | Degenerate path (all-identical frames): _otsu_threshold_and_quality_0_1 не крашится (all-NaN sigma_b2 guard) — ИСПРАВЛЕНО (v1.0.2 → guard before nanargmax) |
| C4 | cos_dist_next size = N-1 при N≥2; пустой массив при N=1 (без краша) |

## Известные исключения (NaN by design)

- NaN в feature_values: НЕТ при status=ok (нет NaN by design для scalars)
- Empty NPZ: отсутствует — uniqueness не создаёт empty NPZ (у него нет пустого выходного пути; при отсутствии кадров оркестратор возвращает status=empty в памяти)
- N=1: cos_dist_next=[]; pairwise_sim=NaN; temporal_change_mean=NaN — корректно (нет пар)
