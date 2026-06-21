# Описание фичей модуля action_recognition

Сводка артефакта, путей, meta → CSV, melt/QA: **`docs/FEATURE_DESCRIPTION.md`**

## ⚠️ Статус качества

**ВНИМАНИЕ**: Модуль требует **доработки качества** и **более тщательной проверки**. Все фичи генерируются корректно с точки зрения схемы и контрактов, но требуют валидации на репрезентативных датасетах для подтверждения качества алгоритмов.

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
- `max_temporal_jump` — максимальный скачок между соседними клипами (L2 по normed)
- `mean_temporal_jump` — средний скачок между соседними клипами
- `stability` — стабильность действий (PCA + KMeans, доля самой длинной непрерывной серии одинаковых кластеров, 0-1)
- `stability_centroid_dist` — альтернативная метрика стабильности: среднее расстояние до центроида кластера (меньше = более стабильные действия)
- `num_switches` — число переключений между кластерами (из KMeans labels)
- `num_clips` — количество клипов в треке
- `track_frame_count` — количество кадров в треке

### 3) UI/Diagnostics (для визуализации)

- `clip_center_frame_indices` — индексы центров клипов (union‑domain)
- `clip_center_times_s` — времена центров клипов (если доступен `union_timestamps_sec`)
- `temporal_jumps` — per‑clip динамика изменений

## Важно

- Отсутствие треков трактуется как валидная пустота (`status="empty"`, `empty_reason="no_person_detections"`).
- Ошибки модели/весов приводят к исключению (no‑fallback).

## Области для валидации качества

### Метрики стабильности

- **`stability`** (PCA+KMeans): Требуется проверка корректности выбора числа кластеров K и валидация на различных типах действий
- **`stability_centroid_dist`**: Альтернативная метрика требует проверки на edge cases (очень короткие/длинные треки)
- **`num_switches`**: Валидация корректности подсчета переключений между кластерами

### Временные метрики

- **`max_temporal_jump`** / **`mean_temporal_jump`**: Требуется проверка корректности вычисления L2 расстояний между соседними клипами
- **`temporal_jumps`**: Должен содержать `num_clips-1` элементов (jumps между клипами), требуется валидация

### Сегментация

- **Группировка person детекций**: Логика создания сегментов из последовательных кадров требует проверки на edge cases
- **Параметры `segment_gap_sec` и `min_person_confidence`**: Требуется валидация оптимальных значений на различных видео

### Эмбеддинги

- **`embedding_normed_256d`**: Требуется проверка корректности L2 нормализации и проекции 2048d → 256d
- **Временная согласованность**: Проверка корректности `clip_center_times_s` и `clip_center_frame_indices` относительно реальных кадров
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [Module README](../README.md) · [VisualProcessor](../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../docs/MAIN_INDEX.md)
