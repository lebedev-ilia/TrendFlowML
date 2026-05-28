# Описание фичей модуля similarity_metrics (Audit v3)

`similarity_metrics` — модуль сравнения видео с reference set (конкуренты/референсы ниши) и выдачи UI‑объяснимых метрик.

## Output artifact

- **File**: `similarity_metrics/results.npz`
- **Schema**: `similarity_metrics_npz_v3`

## 1) Intra-video coherence (per-frame)

- `frame_indices (N,) int32`
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `centroid_sims (N,) float32` — cosine(frame_embedding, video_centroid)
- `temporal_sim_next (N-1,) float32` — cosine(frame_t, frame_{t+1})

## 2) Reference similarity (per-run aggregates)

Если задан `reference_set_id`, модуль грузит reference pack из `dp_models` и считает cosine similarity по модальностям:
- `clip` (visual semantic) — **обязателен**
- `audio_clap` — **optional** (если нет аудио → `NaN`, отсутствие допустимо)
- `text` — optional (нет текста допустимо)
- `pacing` — optional
- `quality` — optional
- `emotion` — optional (нет лиц допустимо)

Также сохраняются агрегаты `reference_similarity_*` в `feature_names/feature_values` (mean_topn/max/p10 по CLIP).

## 3) UI payload

`meta.ui_payload` (schema `similarity_metrics_ui_v1`) содержит:
- `topk_refs[]` — top‑K reference videos (`reference_video_id`) + `scores_by_modality`
- флаги `text_present` и др.

## Library-only metrics

Старый большой набор “topic/audio/emotion/…” метрик (scipy/sklearn) вынесен в:
- `similarity_metrics_library.py`
и **не используется** baseline пайплайном.
Схожесть паттернов аудио энергии. Вычисляется через корреляцию Пирсона между кривыми энергии.

## 6. Emotion & Behavior Similarity

### emotion_curve_similarity
Схожесть кривых эмоций. Вычисляется через корреляцию Пирсона между кривыми эмоций.

### pose_motion_similarity
Схожесть движения поз. Вычисляется через косинусную схожесть фичей движения поз.

### behavior_pattern_similarity
Схожесть паттернов поведения. Вычисляется через корреляцию Пирсона между кривыми поведения.

## 7. Temporal / Pacing Similarity

### pacing_curve_similarity
Схожесть кривых pacing. Вычисляется через корреляцию Пирсона между кривыми pacing.

### shot_duration_distribution_similarity
Схожесть распределения длительностей кадров. Вычисляется через Earth Mover's Distance между распределениями.

### scene_length_similarity
Схожесть длительностей сцен. Вычисляется через сравнение средних и стандартных отклонений длительностей сцен (mean/std) с нормализацией по масштабу.

### temporal_pattern_novelty
Новизна временного паттерна: 1.0 - mean_pacing_similarity. Показывает уникальность временного ритма относительно референсов (выше = более уникальный pacing).

## 8. High-level Comparative Scores

### overall_similarity_score
Общая эвристическая оценка схожести, вычисляемая как взвешенная сумма метрик по категориям:
- semantic: 25%
- topics: 15%
- visual: 15%
- text: 10%
- audio: 15%
- emotion: 10%
- temporal: 10%

### uniqueness_score
Оценка уникальности: 1.0 - overall_similarity_score. Рекомендуется использовать как дополнительный агрегат, а не как единственный источник правды (в production желательно обучать отдельный агрегатор).

### trend_alignment_score
Оценка соответствия трендам. По умолчанию совпадает с `overall_similarity_score`; в production рекомендуется обучать отдельную метрику с учётом recency и популярности референсов.

### viral_pattern_score
Оценка схожести с вирусными видео. По умолчанию совпадает с `overall_similarity_score`; может быть уточнена через отдельный supervised‑классификатор/siamese‑модель над метриками схожести и метаданными референсных видео.

## 9. Group / Batch Metrics

### cluster_similarity_mean
Средняя схожесть между всеми парами видео в батче. Показывает кластерную схожесть набора видео.

### inter_video_variance_topics
Межвидео дисперсия по темам. Показывает разнообразие тем в батче.

### inter_video_variance_emotions
Межвидео дисперсия по эмоциям. Показывает разнообразие эмоций в батче.

### inter_video_variance_editing
Межвидео дисперсия по монтажу (частота склеек). Показывает разнообразие стилей монтажа в батче.

### inter_video_variance_audio
Межвидео дисперсия по аудио (темп). Показывает разнообразие аудио характеристик в батче.

## Методы вычисления

1. **Embedding Comparison**: Используется косинусная схожесть для сравнения embeddings.
2. **Distribution Comparison**: Используется Earth Mover's Distance для сравнения распределений.
3. **Curve Comparison**: Используется корреляция Пирсона для сравнения временных кривых.
4. **Set Comparison**: Используется Jaccard similarity для сравнения множеств тем/концептов.
5. **Top-N Averaging**: Метрики усредняются по топ-N наиболее похожим видео для устойчивости.

## Зависимости

- **numpy**: Для численных вычислений
- **scipy**: Для статистических метрик (correlation, distance)
- **sklearn**: Для вычисления схожести

## Использование

Модуль требует:
- **video_embedding**: Embedding текущего видео (может быть получен из high_level_semantic модуля)
- **reference_embeddings**: Список embeddings референсных видео (из файла JSON)
- Опционально: фичи из других модулей для более детального сравнения

Модуль может использовать данные из других модулей через флаги:
- `--use-high-level-semantic`: Использовать результаты high_level_semantic для video embedding
- `--use-text-scoring`: Использовать результаты text_scoring для text features
- `--use-visual-features`: Использовать визуальные фичи из других модулей

