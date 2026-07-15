# Критерии приёмки: story_structure

Согласованы: 2026-07-16 (брифинг → одобрение владельца).

## Универсальные хард-гейты (U1–U6)

| Гейт | Критерий |
|---|---|
| U1 | Валидатор `validate_story_structure.py --struct` → rc=0 |
| U2 | `frame_indices` строго возрастает; `times_s` неубывает |
| U3 | Нет неожиданных NaN/Inf: story_energy_curve finite ≥ 0.99; topic_shift_curve NaN by design при topic_shift_curve_present=False — OK |
| U4 | expected-empty: N < min_frames (30) → RuntimeError (нет empty-пути у компонента); тест с N≥30 и нет лиц → any_face=False, status=ok |
| U5 | Golden-детерминизм: 2 прогона → max\|Δ\|=0.0 |
| U6 | 3 видео разных длин (N=43/65/119) → schema+struct OK, rc=0 |

## Компонентные критерии

| Критерий | Порог |
|---|---|
| C1 | `story_energy_curve` finite rate ≥ 0.99; topic_shift_curve — NaN by design при отсутствии текста — OK |
| C2 | `feature_values` (22 элемента) finite; `climax_position_norm` ∈ [0,1]; `number_of_peaks` ≥ 0 |
| C3 | `hook_to_avg_energy_ratio` после фикса (знаменатель std+eps): finite и CV>0 на корпусе (разумный диапазон ~[-50,50]) |
| C4 | `climax_frame_index` — union-domain frame id (не позиция в seq), задекларировано в SCHEMA.md |

## Примечания
- **Фикс (согласован):** знаменатель `hook_to_avg_energy_ratio` заменён с `mean(combined_s)+eps` (mean≈0 у z-score → ±862k) на `std(combined_s)+eps` (корректно для z-score, аналог Sharpe ratio, даёт разумные ~±50).
- U4: у story_structure нет status=empty пути — при N<30 бросает RuntimeError (no-fallback). Это задокументировано. Тест U4 = видео без лиц (any_face=False) → status=ok, main_character_screen_time=0.
- topic_shift_curve_present=False на всём корпусе (нет OCR/text deps) — это нормально.
