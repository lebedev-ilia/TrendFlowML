# Описание фичей модуля action_recognition

## Общее описание

`action_recognition` извлекает временные эмбеддинги действий людей в видео на базе **SlowFast R50** и агрегирует компактные метрики для ML/аналитики.

Компонент работает **строго от `frame_indices` Segmenter** и треков из `core_object_detections`.

## Используемые модели

- **SlowFast R50** (torch, in‑process)  
  ModelManager spec: `slowfast_r50_action_recognition`  
  Веса: `dp_models/bundled_models/visual/action_recognition/model.safetensors`

## Структура выходных данных

Выход — NPZ (per‑track). Для каждого трека сохраняются:

### 1) Sequence features

#### 1.1 `embedding_normed_256d`
L2‑нормализованные эмбеддинги для каждого клипа.  
Формат: `[num_clips, 256]`.

### 2) Aggregate features

#### 2.1 Core dynamics
- `mean_embedding_norm_raw` — средняя норма **raw** эмбеддингов (до проекции)
- `std_embedding_norm_raw` — std норм raw эмбеддингов
- `max_temporal_jump` — максимальный скачок между соседними клипами (L2 по normed)
- `mean_temporal_jump` — средний скачок между соседними клипами
- `stability` — стабильность действий (PCA + KMeans)
- `num_switches` — число переключений между кластерами
- `num_clips` — количество клипов в треке
- `track_frame_count` — количество кадров в треке

### 3) UI/Diagnostics (для визуализации)

- `clip_center_frame_indices` — индексы центров клипов (union‑domain)
- `clip_center_times_s` — времена центров клипов (если доступен `union_timestamps_sec`)
- `temporal_jumps` — per‑clip динамика изменений

## Важно

- Отсутствие треков трактуется как валидная пустота (`status="empty"`, `empty_reason="no_faces_in_video"`).
- Ошибки модели/весов приводят к исключению (no‑fallback).

