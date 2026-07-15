# CRITERIA — color_light (согласовано 2026-07-12)

Компонент CPU-only (numpy/opencv/sklearn/scipy). Hard dep: scene_classification (ось Segmenter).
Model-facing: `frame_compact_features (M,16)` + `FRAME_COMPACT_KEYS` + `aggregated.frame_compact`.
Analytics: `video_features` (~543 ключа), `scenes`/`frames` (debug).

## Универсальные хард-гейты (pass/fail)
- U1 `validate_color_light.py --struct --ranges` → rc=0 на всех прогонах.
- U2 Ось времени: `times_s` и `sequence_times_s` неубывающие; len согласованы с индексами.
- U3 Health: `frame_compact_features` finite, не-константа, shape (M,16), dtype float32,
  имена == FRAME_COMPACT_KEYS.
- U4 Expected-empty: after_filt_empty → status=empty, все ключи присутствуют, нулевые длины,
  compact (0,16), rc валидатора=0.
- U5 Golden-детерминизм: реран на тех же входах → diff=0 (CPU-детерминизм).
- U6 Разные длины видео (≥2 ролика разной длительности) отрабатывают, status=ok.

## Критерии под компонент
- C1 (health compact): `frame_compact_features` NaN=0%, Inf=0% на непустых прогонах.
- C2 (различимость): ≥8 из 16 компакт-dim имеют внутривидео std>0 И присутствует межвидовая
  вариация (компакт не константа между роликами). Провал → разбор «мёртвых» компонентов.
- C3 (golden CPU): max|Δ| = 0 по `frame_compact_features` + `aggregated.frame_compact` при реране.
- C4 (NaN-политика video_features): NaN ограничены документированным набором.
  6 aesthetic (nima_mean/std, laion_mean/std, cinematic_lighting_score, professional_look_score) =
  NaN by design (модели не подключены; есть маски `*_present`).
  `color_distribution_gini`: разобрать эмпирически на 5 видео (вкл. empty/edge) —
  если NaN только при неопределённом/const hue_mean → by-design; если hue_mean валиден, а gini=NaN →
  БАГ, чинить перед сдачей.

## Error-семантика (проверить)
- нет scene_classification → status=error.
- нет union_timestamps_sec → status=error.
