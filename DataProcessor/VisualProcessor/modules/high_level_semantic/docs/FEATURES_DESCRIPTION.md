# `high_level_semantic` — features (v1)

Каноническое описание для melt / wide CSV / QA: **`docs/FEATURE_DESCRIPTION.md`**.

Файл описывает **фичи/тензоры**, которые пишет `high_level_semantic` в NPZ (schema `high_level_semantic_npz_v2`).

## 0) Source-of-truth и единица обработки

- **Unit**: `frame` (union-domain)
- **Time axis**: `times_s[i] == union_timestamps_sec[frame_indices[i]]`
- **Embeddings source-of-truth**: `core_clip.frame_embeddings` (не копируется целиком в этот артефакт; encoder читает `core_clip` напрямую)
- **Scene source**: `cut_detection` (shot boundaries + scene grouping)

## 1) Dense time-series (per-frame)

### `frame_features (N, F) float32`

Матрица per-frame фичей (NaN = missing optional modality). Имена колонок в:

### `frame_feature_names (F,) object[str]`

Текущий состав (v1):
- `clip_sim_prev`: cosine similarity между соседними `core_clip` embeddings (NaN на первом кадре)
- `clip_novelty_prev`: \(1 - clip\_sim\_prev\)
- `scene_pos_norm`: позиция кадра внутри сцены (0..1)
- `loudness_dbfs`: интерполированный `dbfs` из `loudness_extractor` на `times_s` (NaN если модальность недоступна)
- `tempo_bpm`: интерполированный `bpm` из `tempo_extractor` на `times_s` (NaN если модальность недоступна)
- `emo_valence`: интерполированный `valence` из `emotion_face` на `times_s` (NaN если недоступно)
- `emo_arousal`: интерполированный `arousal` из `emotion_face`
- `emo_intensity`: \(sqrt(valence^2 + arousal^2)\)

Дополнительно (v2):
- `frame_feature_present_ratio (F,) float32`: доля finite по каждой колонке `frame_features` (модели могут использовать как quality mask).

## 2) Scenes (for ML + UI)

### `scene_id (N,) int32`

Идентификатор сцены для каждого sampled кадра.

### `scene_embeddings (S, D) float32`

Scene embedding = mean по кадрам сцены от `core_clip.frame_embeddings`, затем L2-нормализация.

Scene metadata arrays:
- `scene_start_frame_idx (S,) int32`
- `scene_end_frame_idx (S,) int32` (end-exclusive proxy из `cut_detection`)
- `scene_start_time_s (S,) float32`
- `scene_end_time_s (S,) float32`
- `scene_duration_s (S,) float32`
- `scene_representative_frame_idx (S,) int32`

## 3) Sparse events stream (for encoder/UI)

Unified events:
- `event_times_s (E,) float32`
- `event_type_id (E,) int16`
- `event_strength (E,) float32`
- `event_frame_pos (E,) int32`

Taxonomy v1 (stored in `ui.event_type_map`):
- `1`: hard_cut (from cut_detection)
- `200`: semantic_jump (top-k peaks by `clip_novelty_prev`)
v2 refinement:
- semantic_jump детектируется по **локальным пикам** `clip_novelty_prev` с `min_distance_frames` suppression (меньше “спама” событий).
- `210`: emotion_keyframe (from emotion_face keyframes; best-effort mapping)

## 4) Text snapshot features (optional copy)

Если включена группа `text`, модуль копирует privacy-safe табличные фичи TextProcessor:
- `text_feature_names`
- `text_feature_values`

Это удобно для downstream “single read” сценариев, но source-of-truth остаётся `text_processor/text_features.npz`.
