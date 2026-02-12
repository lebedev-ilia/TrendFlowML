## `speaker_diarization_extractor` (Speaker diarization)

### Назначение

Выполняет **диаризацию спикеров** (определение "кто говорит когда") на основе сегментов аудио. Экстрактор вычисляет эмбеддинги спикеров для каждого сегмента через in-process модель (ECAPA-TDNN), затем кластеризует их для идентификации отдельных спикеров. Результат включает временные метки сегментов с назначенными ID спикеров и средние эмбеддинги для каждого спикера.

**Версия**: 2.2.0  
**Категория**: speech  
**GPU**: preferred (модель работает in-process, GPU предпочтительна)

### Входы

- **`audio/audio.wav`** (любой аудио файл, поддерживаемый AudioUtils)
- **`audio/segments.json`** (сегменты от Segmenter, family: `diarization`)

**Требования**:
- Минимальная длительность аудио: **5 секунд** (иначе ошибка)
- Сегменты должны быть предоставлены через `run_segments()` (метод `run()` не поддерживается)

### Выходы

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Обязательные поля (всегда присутствуют)

- `speaker_count`: количество обнаруженных спикеров (int)
- `speaker_ids`: список ID спикеров `[0, 1, 2, ...]` (list[int])
- `segments_count`: количество обработанных сегментов (int)
- `duration`: общая длительность аудио (float, секунды)
- `segment_start_sec`: список времен начала сегментов (list[float])
- `segment_end_sec`: список времен окончания сегментов (list[float])
- `segment_center_sec`: список времен центров сегментов (list[float])
- `sample_rate`: частота дискретизации аудио (int, по умолчанию 16000 Hz)
- `model_name`: имя модели из ModelManager (str)
- `device_used`: устройство обработки (str, обычно `"cuda"`)
- `rms`: RMS значение аудио (float)
- `peak`: пиковое значение аудио (float)
- `diarization_contract_version`: версия контракта для валидации совместимости (str, `"diarization_contract_v1"`)

#### Feature-gated поля (включаются через флаги)

**`--diar-enable-speaker-segments`**:
- `speaker_segments`: список словарей с информацией о сегментах спикеров:
  ```python
  [
      {
          "start": float,           # Время начала сегмента (секунды)
          "end": float,             # Время окончания сегмента (секунды)
          "duration": float,        # Длительность сегмента (секунды)
          "speaker_id": int,        # ID спикера (0, 1, 2, ...)
          "segment_index": int      # Индекс сегмента в исходном списке
      },
      ...
  ]
  ```

**`--diar-enable-speaker-embeddings`**:
- `speaker_embeddings_mean`: средние эмбеддинги для каждого спикера, shape `[N_speakers, embedding_dim]` float32
  - Каждая строка соответствует одному спикеру
  - Эмбеддинг вычисляется как среднее всех сегментов спикера

**`--diar-enable-speaker-stats`**:
- `speaker_stats`: словарь статистики по каждому спикеру:
  ```python
  {
      speaker_id: {
          "segments_count": int,      # Количество сегментов спикера
          "total_duration": float      # Общая длительность сегментов (секунды)
      },
      ...
  }
  ```

**`--diar-enable-speaker-durations`**:
- `speaker_time_ratios`: доли времени по спикерам (dict[speaker_id, ratio])
- `speaker_balance_score`: метрика баланса времени между спикерами (0 = один спикер, 1 = равномерное распределение)
- `dominant_speaker_id`: ID спикера с максимальной долей времени
- `speaker_segments_density`: среднее количество сегментов спикера в секунду
- `speaker_transitions_count`: количество переходов между спикерами

**`--diar-enable-clustering-metrics`**:
- `clustering_metrics`: метрики качества кластеризации:
  ```python
  {
      "silhouette_score": float,              # Silhouette score (чем выше, тем лучше)
      "davies_bouldin_score": float,           # Davies-Bouldin index (чем ниже, тем лучше)
      "calinski_harabasz_score": float,       # Calinski-Harabasz index (чем выше, тем лучше)
      "mean_intra_cluster_distance": float,    # Среднее внутрикластерное расстояние
      "mean_inter_cluster_distance": float,    # Среднее межкластерное расстояние
  }
  ```

**`--diar-enable-segment-embeddings`**:
- `segment_embeddings`: все индивидуальные эмбеддинги для каждого сегмента (list[list[float]], shape `[N_segments, embedding_dim]`)

#### Специальные случаи

**Пустое аудио** (status="empty"):
- `status`: `"empty"`
- `empty_reason`: `"audio_silent"`
- `speaker_segments`: пустой список (если включено)
- `speaker_count`: 0
- `speaker_embeddings_mean`: пустой массив (если включено)
- Остальные поля присутствуют

### Feature Dependencies

**Зависимости между фичами**:
- `speaker_durations` зависит от `speaker_stats` (использует `total_duration` для вычисления `speaker_time_ratios`)
- `clustering_metrics` зависит от результатов кластеризации (требует `speaker_segments` или `speaker_ids`)

**Зависимости от других extractors**:
- Нет явных зависимостей от других extractors
- Используется в `speech_analysis_extractor` как зависимость (требует `speaker_segments` и `speaker_embeddings_mean`)

**Contract version для совместимости**:
- `diarization_contract_version="diarization_contract_v1"` используется для валидации совместимости с downstream extractors (например, `speech_analysis_extractor`)

### Алгоритм

#### 1. Предобработка аудио

1. Загрузка сегментов из `segments.json` (family: `diarization`)
2. Для каждого сегмента:
   - Загрузка аудио через `AudioUtils.load_audio_segment()`
   - Ресемплирование до `sample_rate` (по умолчанию 16000 Hz)
   - Преобразование в моно канал
3. Паддинг сегментов до максимальной длины для батчинга
4. Проверка на тишину (если `--diar-disable-silence-detection` не установлен):
   - Если конкатенированное аудио тихое (peak < `silence_peak_threshold`, RMS < `silence_rms_threshold`), возвращается пустой результат

#### 2. Извлечение эмбеддингов через in-process модель

- **Вход**: паддингнутые аудио сегменты, shape `[B, max_length]` float32
- **Выход**: эмбеддинги спикеров, shape `[B, embedding_dim]` float32
- **Модель**: ECAPA-TDNN (speechbrain) через ModelManager
  - Spec name: `speaker_diarization_{size}_inprocess`
  - Runtime: `inprocess` (PyTorch)
  - Engine: `torch`
  - Precision: `fp16` (на CUDA) или `fp32` (на CPU)
- **Batching**: если `batch_size` не задан и сегментов >100, автоматически разбивается на батчи по 100 сегментов

#### 3. Валидация эмбеддингов

- Проверка dtype (float32)
- Проверка shape (2D [N, D])
- Проверка на NaN/inf
- Проверка диапазонов значений
- Проверка размерности embedding (8-2048)

#### 4. Оценка количества спикеров

**Методы оценки** (выбирается через `--diarization-speaker-count-method`):

- **`heuristic`** (default): эвристика на основе mean cosine distance:
  - Вычисляется среднее косинусное расстояние между всеми парами эмбеддингов
  - Эвристика на основе порогов:
    - `mean_distance > 0.40` → 4 спикера (или max_speakers, если меньше)
    - `mean_distance > 0.32` → 3 спикера (или max_speakers, если меньше)
    - `mean_distance > 0.24` → 2 спикера (или max_speakers, если меньше)
    - `mean_distance > 0.16` → 2 спикера (или max_speakers, если меньше)
    - Иначе → min_speakers
  - Финальное количество кластеризуется в диапазоне `[min_speakers, max_speakers]`

- **`silhouette`**: оптимальный k на основе silhouette score:
  - Перебираются значения k от `min_speakers` до `max_speakers`
  - Для каждого k выполняется кластеризация и вычисляется silhouette score
  - Выбирается k с максимальным silhouette score

- **`fixed`**: используется `min_speakers` как фиксированное значение

**Default для обучения**: `silhouette` (оптимальный k для качества)

#### 5. Кластеризация эмбеддингов

**Методы кластеризации** (выбирается через `--diarization-clustering-method`):

- **`agglomerative`** (default для обучения): Agglomerative Clustering (иерархическая кластеризация)
  - Метрика: косинусное расстояние
  - Linkage: average
  - Точнее, но медленнее для больших наборов данных

- **`kmeans`**: KMeans кластеризация
  - Быстрее для больших наборов данных (>100 сегментов)
  - Менее точная, чем Agglomerative

- **`auto`**: автоматический выбор метода:
  - Если сегментов >100 → используется KMeans
  - Иначе → используется Agglomerative

**Default для обучения**: `agglomerative` (лучшее качество)

#### 6. Валидация кластеризации

- Проверка согласованности меток с количеством сегментов
- Проверка согласованности уникальных меток с количеством спикеров
- Проверка диапазона меток [0, n_speakers-1]

#### 7. Постобработка

1. Группировка сегментов по speaker_id
2. Вычисление средних эмбеддингов для каждого спикера (если включено)
3. Вычисление статистики (количество сегментов, общая длительность)
4. Формирование `speaker_segments` с временными метками (если включено)
5. Вычисление агрегатов (time ratios, balance score, transitions) (если включено)
6. Вычисление метрик качества кластеризации (если включено)

### Конфигурация

#### Параметры модели

**CLI аргументы** (через `--diarization-*`):
- `--diarization-model-size`: размер модели (`"small"` | `"large"`, default: `"small"`)
- `--diarization-triton-batch-size`: размер батча для Triton (int | None, default: None = auto, >100 segments → split)
- `--diarization-clustering-method`: метод кластеризации (`"agglomerative"` | `"kmeans"` | `"auto"`, default: `"agglomerative"`)
- `--diarization-speaker-count-method`: метод оценки количества спикеров (`"heuristic"` | `"silhouette"` | `"fixed"`, default: `"heuristic"`)
- `--diarization-silence-peak-threshold`: порог peak для детекции тишины (float, default: 1e-3)
- `--diarization-silence-rms-threshold`: порог RMS для детекции тишины (float, default: 1e-4)

**Hardcoded параметры** (не настраиваются через CLI/config):
- `min_speakers`: 1 (минимальное количество спикеров)
- `max_speakers`: 6 (максимальное количество спикеров)
- `sample_rate`: 16000 Hz (частота дискретизации)
- `device`: определяется автоматически (`"auto"` | `"cuda"` | `"cpu"`)

**Примечание**: `min_speakers` и `max_speakers` не настраиваются через CLI или `global_config.yaml` и остаются с дефолтными значениями (1 и 6). Это сделано для упрощения конфигурации, так как эти параметры редко требуют изменения.

**Параметры модели** (из ModelManager spec `speaker_diarization_{model_size}_inprocess`):
- Модель загружается локально через `dp_models.ModelManager`
- Используется ECAPA-TDNN архитектура из speechbrain
- Checkpoint хранится в `audio/speaker_diarization/{size}.pt`

#### Feature Gating (персональные флаги)

Все фичи по умолчанию **отключены** (opt-in):

- `--diar-enable-speaker-segments`: включить `speaker_segments` (timeline с speaker IDs)
- `--diar-enable-speaker-embeddings`: включить `speaker_embeddings_mean` (mean embeddings per speaker)
- `--diar-enable-speaker-stats`: включить `speaker_stats` (статистики по спикерам)
- `--diar-enable-speaker-durations`: включить `speaker_time_ratios` и агрегаты (доли времени, balance score, transitions)
- `--diar-enable-clustering-metrics`: включить `clustering_metrics` (метрики качества кластеризации)
- `--diar-enable-segment-embeddings`: включить `segment_embeddings` (все индивидуальные эмбеддинги)
- `--diar-disable-silence-detection`: отключить проверку на тишину (по умолчанию включена)

**Примечание**: `--diar-disable-silence-detection` обрабатывается через `feature_flags.disable_silence_detection` в `global_config.yaml`. Если установлено в `true`, то silence detection отключается.

**Рекомендации для обучения моделей**:
- Включить все фичи для максимального качества и полноты данных
- Использовать `clustering_method="agglomerative"` и `speaker_count_method="silhouette"` для лучшего качества

### Архитектура

1. **Инициализация**:
   - Получение модели через `ModelManager.get_spec()` → `speaker_diarization_{model_size}_inprocess`
   - Проверка runtime = "inprocess"
   - Загрузка модели через `ModelManager.get()` (ECAPA-TDNN)
   - Инициализация `AudioUtils`

2. **Обработка сегментов** (`run_segments()`):
   - Валидация входных данных (длительность ≥ 5 сек)
   - Загрузка и предобработка аудио сегментов
   - Паддинг до максимальной длины
   - Проверка на тишину (если включено)
   - Инференс через in-process модель для получения эмбеддингов (с батчингом, если нужно)
   - Валидация эмбеддингов
   - Оценка количества спикеров (выбранным методом)
   - Кластеризация эмбеддингов (выбранным методом)
   - Валидация кластеризации
   - Постобработка и формирование payload (feature-gated)

3. **Вспомогательные методы**:
   - `_estimate_speaker_count_heuristic()`: эвристическая оценка количества спикеров
   - `_estimate_speaker_count_silhouette()`: оценка на основе silhouette score
   - `_cluster_speakers_agglomerative()`: кластеризация через Agglomerative Clustering
   - `_cluster_speakers_kmeans()`: кластеризация через KMeans
   - `_process_from_labels()`: формирование финального результата из меток кластеров
   - `_validate_embeddings()`: полная валидация эмбеддингов
   - `_validate_clustering_labels()`: валидация меток кластеризации
   - `_compute_clustering_metrics()`: вычисление метрик качества кластеризации
   - `_classify_triton_error()`: классификация ошибок Triton с детальными error codes

4. **Обработка ошибок**:
   - Модель не найдена → `RuntimeError`
   - Невалидный вход → `RuntimeError`
   - Ошибка inference → `RuntimeError`
   - Аудио < 5 сек → `RuntimeError`
   - Пустые сегменты → `ValueError`
   - Валидация эмбеддингов/кластеризации → `ValueError`

### Обработка ошибок

**Политика NO FALLBACK**:
- Отсутствие модели → ошибка
- Модель не загружена → ошибка
- Аудио < 5 секунд → ошибка
- Пустые сегменты → ошибка
- Валидация эмбеддингов/кластеризации → ошибка

**Специальные случаи**:
- **Тихое аудио**: возвращается `status="empty"`, `empty_reason="audio_silent"` (если silence detection включен)
- **Несоответствие sample rate**: ошибка с описанием
- **Неожиданная форма эмбеддингов**: ошибка с описанием

### Особенности

- **In-process**: модель работает локально через PyTorch (не через Triton)
- **Сегментная обработка**: работает только с сегментами от Segmenter
- **Автоматическая оценка количества спикеров**: эвристика или silhouette score
- **Выбор метода кластеризации**: Agglomerative (default для обучения), KMeans (быстрее), Auto (автоматический выбор)
- **Средние эмбеддинги**: сохраняются средние эмбеддинги для каждого спикера (не все сегменты, если не включено)
- **Нет fallback**: строгая политика - ошибка при отсутствии модели
- **Паддинг сегментов**: сегменты разной длины падятся до максимальной для батчинга
- **Batching**: автоматическое разбиение на батчи при большом количестве сегментов (>100)
- **Progress reporting**: обновление прогресса каждые 10% сегментов (если сегментов ≥10)
- **Feature gating**: все фичи opt-in через персональные флаги
- **Contract versioning**: версия контракта для валидации совместимости с downstream extractors

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (кластеризация, предобработка)
- **GPU**: предпочтительна (модель работает in-process на GPU)
- **Memory**: ~1.0 GB GPU memory для модели
- **Network**: не требуется (модель загружается локально)
- **Estimated duration**: ~6.0 секунд для типичного аудио файла

**Параметры производительности**:
- `model_size`: `small` быстрее, `large` точнее
- `max_speakers`: больше спикеров → больше времени на кластеризацию
- `clustering_method`: `kmeans` быстрее, `agglomerative` точнее
- `speaker_count_method`: `silhouette` медленнее (перебирает k), `heuristic` быстрее
- Длина сегментов: влияет на размер входных данных для Triton
- `triton_batch_size`: контроль размера батча для Triton (None = auto)

### Visualization

**Рекомендации для UI/сайта**:

1. **Timeline визуализация**:
   - Горизонтальная временная шкала с цветовой кодировкой спикеров
   - Каждый сегмент отображается как полоса с цветом, соответствующим speaker_id
   - Интерактивные tooltips с информацией о сегменте (start, end, duration, speaker_id)
   - Zoom и pan для навигации по длинным видео

2. **Статистика спикеров**:
   - Pie chart или bar chart с долями времени по спикерам (`speaker_time_ratios`)
   - Таблица со статистикой по каждому спикеру (segments_count, total_duration, time_ratio)
   - Highlight доминирующего спикера (`dominant_speaker_id`)

3. **Метрики качества кластеризации**:
   - Отображение silhouette score, Davies-Bouldin index, Calinski-Harabasz index
   - Визуализация внутрикластерных и межкластерных расстояний
   - Индикаторы качества (good/fair/poor) на основе метрик

4. **Агрегаты**:
   - Отображение `speaker_balance_score` (0-1, где 1 = идеальный баланс)
   - Количество переходов между спикерами (`speaker_transitions_count`)
   - Плотность сегментов (`speaker_segments_density`)

5. **HTML renderer для дебага**:
   - Локальный HTML renderer доступен через `render_speaker_diarization_extractor_html()`
   - Включает timeline plot, статистику, метрики качества
   - Только для локального дебага, не в production артефактах

**Пример использования HTML renderer**:
```python
from src.core.renderer import render_speaker_diarization_extractor_html

html_path = render_speaker_diarization_extractor_html(
    npz_path="result_store/.../speaker_diarization_extractor/speaker_diarization_extractor_features.npz",
    output_path="debug_diarization.html"
)
```

### Связанные компоненты

- **ModelManager** (`dp_models`): управление моделями и спецификациями
- **speechbrain**: библиотека для ECAPA-TDNN модели speaker embeddings
- **torch**: PyTorch для inference
- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **Segmenter**: генерация сегментов (family: `diarization`)
- **scikit-learn**: AgglomerativeClustering, KMeans, метрики качества кластеризации
- **speech_analysis_extractor**: использует результаты diarization как зависимость

### Примечания

1. **Требования**: экстрактор требует:
   - Модель эмбеддингов спикеров (ECAPA-TDNN) в `dp_models/bundled_models/audio/speaker_diarization/{size}.pt`
   - Спецификация модели в ModelManager (`speaker_diarization_{size}_inprocess`)
   - Установленный `speechbrain` и `torch`

2. **Сегментация**: экстрактор работает только с сегментами от Segmenter. Метод `run()` не поддерживается.

3. **Оценка количества спикеров**: 
   - `heuristic` (default) - быстрая эвристика на основе косинусных расстояний
   - `silhouette` - оптимальный k на основе silhouette score (рекомендуется для обучения)
   - `fixed` - фиксированное значение min_speakers

4. **Кластеризация**: 
   - `agglomerative` (default для обучения) - точнее, но медленнее
   - `kmeans` - быстрее для больших наборов данных
   - `auto` - автоматический выбор на основе количества сегментов

5. **Эмбеддинги**: 
   - По умолчанию сохраняются только средние эмбеддинги для каждого спикера (экономия места)
   - Все индивидуальные эмбеддинги доступны через `--diar-enable-segment-embeddings`

6. **Тишина**: если аудио тихое и silence detection включен, возвращается пустой результат (не ошибка).

7. **Паддинг**: сегменты разной длины падятся до максимальной для эффективного батчинга в Triton.

8. **Speaker IDs**: ID спикеров начинаются с 0 и идут последовательно. Количество спикеров определяется автоматически в диапазоне `[min_speakers, max_speakers]` (hardcoded: min=1, max=6).

9. **Feature gating**: все фичи opt-in через персональные флаги. Для обучения рекомендуется включить все фичи для максимального качества.

10. **Contract versioning**: версия контракта `diarization_contract_v1` используется для валидации совместимости с downstream extractors (например, `speech_analysis_extractor`).
