# Критерии приёмки: frames_composition

> Согласовано 2026-07-13. Владелец: "можно сильнее залезть в логику и попытаться ее улучшить".

## Универсальные хард-гейты

| ID | Критерий | Порог | Метод |
|----|----------|-------|-------|
| U1 | Валидатор rc=0 (--struct --ranges) на batch 24 NPZ | 100% | validate_frames_composition.py batch |
| U2 | Ось времени: frame_indices строго ↑, times_s неубывает | 100% | из validator struct |
| U3 | health: нет Inf, нет `__all_non_finite` в feature_values, nan_fv=0 при status=ok | 100% | numpy check |
| U4 | expected-empty path: status=empty, D=1, F=1, rc=0 | path работает | прогон с нет-лица видео |
| U5 | Golden детерминизм: max\|Δ\|=0.0 (чистый numpy/opencv, нет стохастики) | 0.0 | 2 прогона, побайтовый diff |
| U6 | Разные длины видео: N варьируется ≥3× | N от min до max ≥3× | из прогона |

## Компонентные критерии

| ID | Критерий | Порог | Обоснование |
|----|----------|-------|-------------|
| C1 | face-NaN% by design: nan%(face_center_x/y, face_area_ratio, anchor_*, thirds_alignment) ≈ (1 - face_present_ratio)×N / N с |Δ|≤0.01 | ≤0.01 | Плавающие NaN только у face-зависимых столбцов, остальные finite |
| C2 | Различимость ≥3/5 ключевых фич с CV≥0.15 на разных видео (edge_density__mean, line_strength__mean, face_present__mean, negative_space_ratio__mean, symmetry_score__mean) | ≥3/5 CV≥0.15 | На 4 OK видео из result_store: 5/5 PASS (0.70, 0.91, 0.70, 0.64, 0.38) |
| C3 | frame_feature_present_ratio[j] = mean(isfinite(ffv[:,j])) с \|Δ\|≤2e-6 | ≤2e-6 | Алгебраическое свойство, проверяется на 4 OK видео |
| C4 | style_dominant_id варьируется (не константа) между разными видео | ≠ константа | После фикса depth CV-нормализации |

## Исключения (NaN by design)

- `face_center_x/y`, `face_area_ratio`, `anchor_distance`, `anchor_type_id`, `thirds_alignment` — NaN при face_present=0 на кадре. Штатное поведение.
- `neg_space_balance_lr` = 1.0 при нет объектов (нет объектов = идеальный баланс). После фикса 2026-07-13 (было 0.0).

## Логические исправления в этой сессии

1. **`neg_space_balance_lr = 0.0` при no_obj → 1.0**: семантический баг, `_bbox_stats_for_frame` early-exit возвращал 0.0 вместо правильного 1.0.
2. **Depth нормализация в style_probs**: `_clip01(depth_std)` при depth_std≈250 всегда давал 1.0 → `ds = clip01(depth_std/depth_mean)` (CV). Аналогично для bokeh_proxy: `(p95-p05)/1024.0` вместо сырого clip.
