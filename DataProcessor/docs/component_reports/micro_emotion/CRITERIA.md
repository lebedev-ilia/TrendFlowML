# CRITERIA — micro_emotion (согласовано с владельцем 2026-07-13)

## Универсальные хард-гейты (pass/fail)
- U1: валидатор выхода `validate_micro_emotion.py` rc=0 (0 error-issues).
- U2: ось времени `times_s = union_timestamps_sec[frame_indices]`, строго возрастает / non-decreasing;
  `frame_indices` строго возрастают, ≥0, int32.
- U3: health — `frame_features`/`compact22`/`feature_values` finite там, где не NaN-by-design; 0 Inf;
  верные shape/dtype/range; `compact22 (N,22)` float32.
- U4: expected-empty — видео без лиц → `status=empty`, `empty_reason=no_faces_in_video`,
  compact22 (N,22) all-NaN, все ключи схемы, validator rc=0.
- U5: golden-детерминизм постобработки — при фиксированном OpenFace CSV повторный прогон даёт
  побайтово идентичный compact22/feature_values (max|Δ|=0).
- U6: разные длины видео отрабатывают (N варьируется; уже есть N=12/43/65/119).

## Критерии под компонент
- C1 (ФИКС strip): на синтетическом OpenFace CSV с ведущими пробелами и варьируемыми
  AU12_r/AU06_r/pose_Rx/gaze_angle_x/landmarks — после фикса в compact22 соответствующие колонки
  НЕ константа: std(AU12_delta_norm), std(pose_Rx_norm|Ry), std(gaze_x_norm) > 0. ДО фикса они = 0 (регресс-контроль).
- C2 (различимость): PCA обучается — `au_pca_var_explained_1..3` finite (не NaN); хотя бы одна AU/pose/gaze
  колонка compact22 имеет CV или std заметно > 0 (per-column variation).
- C3 (micro-expr): при заложенных AU-спайках `microexpr_count > 0` и `event_times_s` непусты;
  на плоском сигнале — 0 (без ложных срабатываний).
- C4 (missing-policy, оставить как есть): вне-лицевые кадры → NaN by design; видео без лиц → all-NaN compact22;
  `au_pca_var_explained_4/5`, `landmarks_pca_*` могут быть NaN когда PCA-компонент недоступен (< n_components) — штатно.

## Явные исключения (NaN by design)
- Кадры без лица: строки compact22 = NaN. Видео без лиц: весь compact22 NaN + status=empty.
- feature_values PCA-хвост (var_explained_4/5, landmarks_pca при малом n) — NaN когда компонент не вычислим.
- frame_features F=2 (time_norm, face_present_any) — оставить (расширение = отдельный PR).

## Инфра
- OpenFace — только Docker `--gpus all`; логика постобработки валидируется синтетикой без docker.
  Реальный docker-OpenFace (детерминизм/версия image) — отдельный прод-этап.
