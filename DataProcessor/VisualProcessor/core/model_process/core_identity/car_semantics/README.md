## Component: `car_semantics` (semantic head, v1)

### Назначение

`car_semantics` распознает автомобили в видео:
- определяет make (производитель), model (модель), segment (класс)
- использует CLIP через Embedding Service для извлечения embeddings автомобилей
- сравнивает с базой известных автомобилей через Embedding Service
- возвращает per-track и per-frame top‑K идентификаций с make/model/segment

Компонент следует контракту semantic head:
- Требует `core_object_detections.frame_indices` (shared sampling group)
- Выравнивает выходные данные по тем же `frame_indices`
- Выводит результаты в NPZ формате с иерархической информацией (make → model → segment)

### Пайплайн для наполнения базы автомобилей

Для работы `car_semantics` нужна база известных автомобилей. В этом компоненте есть полноценный offline‑пайплайн:

- `add_car.py` — полуавтоматическое наполнение `known_cars/` из видео и фото
- `sync_known_cars_to_embedding_service.py` — синхронизация готовой базы в Embedding Service (категория `car`)

#### 1. Структура `known_cars/`

После разметки `known_cars/` выглядит так:

```
car_semantics/known_cars/
├── tesla_model_3/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── bmw_x5/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── toyota_camry/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
```

- Имя папки = label автомобиля (используется как `name` в Embedding Service)
- Внутри — несколько кропов машины с padding (размеры могут варьироваться)

#### 2. Сбор автомобилей: `add_car.py`

Скрипт `add_car.py`:
- Проходит по видео в `example/example_videos` и по фото в `example/example_photo`
- Детектирует машины через YOLO (`yolo11x_41_best.pt`, класс `car`)
- Извлекает кропы с padding через `crop_utils.crop_with_padding`
- Показывает каждый кроп и просит ввести **label** (строку: `tesla_model_3`, `bmw_x5`, `toyota_camry`, ...)
- Сохраняет кроп в `known_cars/<label>/<N>.jpg`

Запуск:

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/car_semantics/add_car.py
```

Особенности:
- Кадры из видео выбираются **равномерно** (по 20 кадров на видео)
- Используется YOLO для детекции (класс `car`, ID=2)
- Поддерживается GUI OpenCV и fallback на `matplotlib` или сохранение превью в файл
- Кропы извлекаются с padding (15% по умолчанию) для лучшего контекста

Требования:
- Модель YOLO: `dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt`
- Переменная окружения `DP_MODELS_ROOT` (опционально) для указания пути к моделям

#### 3. Синхронизация с Embedding Service

Когда `known_cars/` заполнен, нужно записать автомобили в Embedding Service:

Скрипт: `sync_known_cars_to_embedding_service.py`

Алгоритм для каждой машины:
1. Считает все фото из `known_cars/<car_name>/`
2. Для каждого фото:
   - CLIP `clip_image_336` через Triton → embedding → L2-нормализация
3. Усредняет эмбеддинги:
   - `avg_emb = mean(embeddings, axis=0)`
   - финальная L2-нормализация `avg_emb`
4. Записывает в Embedding Service (категория `car`) через `EmbeddingManager.add_from_embedding`:
   - `category = "car"`
   - `name = <car_name>`
   - `embedding = avg_emb`
   - `metadata = {"source": "known_cars", "num_images": N}`

Запуск:

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/car_semantics/sync_known_cars_to_embedding_service.py
```

Требования:
- Запущен Embedding Service (см. `embedding_service/README.md`)
- Запущен Triton Inference Server с моделью `clip_image_336`
- Настроены переменные окружения PostgreSQL (`POSTGRES_*`) или `.env`

#### 4. Проверка, что автомобили попали в Embedding Service

Проверка категорий и количества:

```bash
curl http://localhost:8001/categories
curl "http://localhost:8001/categories/car/count"
```

Поиск по новому фото:

```bash
curl -X POST "http://localhost:8001/search" \
  -F "category=car" \
  -F "top_k=5" \
  -F "similarity_threshold=0.5" \
  -F "image=@path/to/car.jpg"
```

Ожидаемый результат:
- В выдаче будут `name` из `known_cars` (например, `tesla_model_3`, `bmw_x5`, `toyota_camry`)
- В `metadata.num_images` можно увидеть, сколько фото использовалось для усреднения.

### Структура компонента

```
car_semantics/
├── main.py                      # Основной модуль обработки
├── embedding_service_client.py  # HTTP клиент для Embedding Service
├── crop_utils.py                # Утилиты для кропов и предобработки
├── render.py                    # Render system (JSON и HTML визуализация)
├── __init__.py                  # Инициализация модуля
└── README.md                    # Эта документация
```

### Интеграция с Embedding Service

Компонент **использует Embedding Service** для хранения и поиска эмбеддингов автомобилей:

1. **Категория**: `car` или `car_semantic`
2. **Модель**: `clip_336` (CLIP 336x336 для баланса качества и скорости)
3. **Использование**:
   - Хранение эмбеддингов известных автомобилей в Embedding Service
   - Поиск похожих автомобилей через Embedding Service (`POST /search`)
   - Добавление новых автомобилей через API (`POST /objects/add`)
   - Обновление информации об автомобилях (`PATCH /objects/{id}`)
   - Batch добавление автомобилей (`POST /objects/batch_add`)

**Преимущества использования Embedding Service**:
- Единая база данных для всех автомобилей
- Быстрый поиск через FAISS индексы
- Удобное управление базой (добавление/удаление/обновление)
- Хранение метаданных (make, model, segment, body_type, price_bucket и т.д.)
- Иерархическая структура (make → model → segment)
- Горячее обновление (новые автомобили доступны сразу)

**Пример использования**:
```python
# Добавить автомобиль в базу
POST http://localhost:8001/objects/add
{
    "category": "car",
    "name": "Toyota Camry 2023",
    "image": <car_image>,
    "metadata": {
        "make": "Toyota",
        "model": "Camry",
        "year": 2023,
        "segment": "mid-size sedan",
        "body_type": "sedan",
        "price_bucket": "mid-range"
    }
}

# Распознать автомобиль на кадре (из bbox)
POST http://localhost:8001/search
{
    "category": "car",
    "image": <crop_from_bbox>,
    "top_k": 3,
    "similarity_threshold": 0.65
}

# Batch добавление автомобилей
POST http://localhost:8001/objects/batch_add
{
    "category": "car",
    "images": [<car1>, <car2>, ...],
    "names": ["Car 1", "Car 2", ...],
    "metadata_list": [
        {"make": "Toyota", "model": "Camry", ...},
        {"make": "Honda", "model": "Accord", ...}
    ]
}
```

**Алгоритм работы**:
1. Компонент получает bbox proposals из `core_object_detections` (класс `car`)
2. Извлекает crop из кадра по bbox
3. Отправляет crop в Embedding Service для поиска похожих автомобилей
4. Возвращает top‑K результатов с make, model, segment и similarity scores

**Таксономия автомобилей**:
- **Make** (производитель): Toyota, Honda, BMW, Mercedes-Benz и т.д.
- **Model** (модель): Camry, Accord, 3 Series, C-Class и т.д.
- **Segment** (класс): compact, mid-size, luxury и т.д.
- **Body Type**: sedan, SUV, coupe, hatchback и т.д.
- **Price Bucket**: entry-level, mid-range, luxury и т.д.

### Входы (required)

- `frames_dir/metadata.json`:
  - `core_object_detections.frame_indices` (shared sampling group)
  - `union_timestamps_sec`
- `rs_path/core_object_detections/detections.npz` (для bbox proposals и tracks, класс `car`)
- Embedding Service доступен (для поиска в базе автомобилей)

### Output (NPZ)

Путь: `rs_path/car_semantics/car_semantics.npz`

**Artifact filename**: `car_semantics.npz` (фиксированное имя, `ARTIFACT_FILENAME`)

**Schema version**: `car_semantics_npz_v1`

Ключи (v1):
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `track_ids (T,) int32` - ID треков
- `track_topk_car_ids (T, 3) int32` - Top‑3 автомобилей на трек
- `track_topk_scores (T, 3) float32` - Similarity scores
- `track_topk_makes (T, 3) str` - Make для top‑3
- `track_topk_models (T, 3) str` - Model для top‑3
- `track_topk_segments (T, 3) str` - Segment для top‑3
- `frame_topk_car_ids (N, 3) int32` - Top‑3 автомобилей на кадр
- `frame_topk_scores (N, 3) float32` - Similarity scores
- `meta` (object dict): статус + информация о базе автомобилей + models_used

**Meta обязательные поля** (baseline contract):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `dataprocessor_version` (может быть "unknown" в baseline)
- `status` (`ok`/`empty`/`error`), `empty_reason` (если `status="empty"`)
- `models_used[]` (если используются модели)
- `stage_timings_ms` (dict): тайминги стадий выполнения в миллисекундах:
  - `initialization`: инициализация компонента
  - `load_deps`: загрузка зависимостей (`core_object_detections`)
  - `process_frames`: обработка треков и поиск автомобилей
  - `saving`: сохранение артефакта
  - `total`: общее время выполнения

### Early validation (Embedding Service)

Компонент выполняет **раннюю проверку доступности Embedding Service**:
- Проверка health endpoint через `embedding_client._ensure_url()`
- Тестовый запрос с первым треком (dummy image) для проверки работоспособности search endpoint
- Если тест не проходит (например, 500 ошибка):
  - Выдается одно предупреждение вместо множества ошибок
  - Пропускается обработка всех треков
  - Заполняются пустые результаты вместо обработки с ошибками
- Если тест проходит — продолжается обычная обработка

**Преимущества**:
- Fail-fast: проблемы обнаруживаются до начала обработки всех треков
- Улучшенный UX: одно предупреждение вместо сотен ошибок
- Экономия ресурсов: не тратится время на обработку, которая все равно завершится ошибкой

### No-fallback

- Если Embedding Service недоступен при инициализации → **error** (fail-fast)
- Если Embedding Service недоступен во время обработки → компонент пропускает все треки с предупреждением
- Если нет `core_object_detections` → **error**
- Если автомобиль не найден → возвращает пустой результат (не error)

### Использование

#### Запуск через CLI:

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    --embedding-service-url http://localhost:8001 \
    --topk 3 \
    --similarity-threshold 0.65 \
    --max-tracks 50 \
    --max-dets-per-frame 3 \
    --pad-ratio 0.15 \
    --use-sharpness
```

#### Параметры:

- `--frames-dir` (required): Директория с кадрами и `metadata.json`
- `--rs-path` (required): Путь к result store (например, `result_store/platform/video/run`)
- `--embedding-service-url` (optional): URL Embedding Service (по умолчанию: `http://localhost:8001` или из `EMBEDDING_SERVICE_URL`)
- `--topk` (optional): Количество топ результатов (по умолчанию: `3`)
- `--similarity-threshold` (optional): Минимальный порог similarity (по умолчанию: `0.0`, диапазон: `0.0-1.0`)
- `--max-tracks` (optional): Максимальное количество треков для обработки (cost control)
- `--max-dets-per-frame` (optional): Максимальное количество детекций на кадр (cost control)
- `--pad-ratio` (optional): Коэффициент паддинга для кропов (по умолчанию: `0.15` = 15% с каждой стороны)
- `--use-sharpness` (optional): Использовать метрику резкости для выбора лучшего кропа

#### Примеры использования:

**Базовое использование:**
```bash
python main.py --frames-dir frames/ --rs-path results/
```

**С cost controls:**
```bash
python main.py \
    --frames-dir frames/ \
    --rs-path results/ \
    --max-tracks 30 \
    --max-dets-per-frame 3
```

**С настройками качества:**
```bash
python main.py \
    --frames-dir frames/ \
    --rs-path results/ \
    --pad-ratio 0.20 \
    --use-sharpness \
    --similarity-threshold 0.70
```

### Детали реализации

#### Алгоритм обработки:

1. **Загрузка детекций**:
   - Загружает `core_object_detections/detections.npz`
   - Проверяет наличие `frame_indices` в метаданных
   - Фильтрует детекции по классу `car` / `vehicle`

2. **Группировка по трекам**:
   - Группирует детекции по `track_id`
   - Предупреждает, если треки отсутствуют (генерирует per-detection track IDs)

3. **Выбор лучшего кропа**:
   - Для каждого трека извлекает все кропы с паддингом
   - Выбирает лучший кроп по формуле: `score × area × (optional sharpness)`
   - Использует Laplacian variance для оценки резкости

4. **Поиск через Embedding Service**:
   - Отправляет лучший кроп в Embedding Service
   - Использует retry механизм (3 попытки с exponential backoff)
   - Обрабатывает пустые результаты (warning, не error)

5. **Извлечение метаданных**:
   - Извлекает `make`, `model`, `segment` из метаданных результата
   - Сохраняет иерархическую информацию в выходных массивах

6. **Агрегация результатов**:
   - Track-level: top-K результатов для каждого трека с make/model/segment
   - Frame-level: дедупликация по `car_name`, выбор лучшей similarity для каждого автомобиля

#### Обработка ошибок:

- **Fail-fast политика**: 
  - Если Embedding Service недоступен при инициализации → **error** (fail-fast)
  - Если все треки упали с ошибками → **error** (fail-fast)
  - Если автомобиль не найден (пустые результаты) → **warning** (не error, это нормально)
- **Retry механизм**: Все запросы к Embedding Service автоматически повторяются при ошибках (3 попытки с exponential backoff)
- **Валидация данных**: Проверка размеров массивов, соответствия frame_indices
- **Graceful degradation**: При ошибке отдельного трека компонент продолжает работу, но если все треки упали - падает с ошибкой
- **Логирование**: Все ошибки и предупреждения логируются

#### Оптимизации:

- **Sharpness pre-computation**: Резкость вычисляется один раз для всех кропов трека
- **Batch API**: Использует `search_batch` метод Embedding Service для batch-запросов (если доступен)
- **Cost control**: Ограничение количества треков и детекций для контроля производительности

### Progress / state events

Компонент публикует прогресс выполнения в `state_events.jsonl` (baseline contract):

**Стадии выполнения**:
- `start` → `load_deps` → `process_frames` → `save` → `done`

**Гранулярный прогресс**:
- Во время стадии `process_frames` компонент публикует прогресс обработки треков (≥10 обновлений)
- Формат события: `{"progress": 0.0-1.0, "done": int, "total": int, "stage": "process_frames"}`

**Использование**: Backend сайта может читать `state_events.jsonl` для отображения прогресса анализа в реальном времени.

### Cost control

- Только proposals класса `car` / `vehicle` (по умолчанию)
- 1 crop на track (выбирается по `score × area × (optional sharpness)`)
- Лимиты: `--max-tracks`, `--max-dets-per-frame`
- Top‑K фиксирован: **3** (настраивается через `--topk`)

### Troubleshooting

#### Embedding Service недоступен:

```
WARNING: car_semantics | Embedding Service test request failed: ...
Skipping all tracks to avoid repeated errors.
```

**Решение**: 
- Убедитесь, что Embedding Service запущен:
```bash
cd DataProcessor/embedding_service
python run_server.py
```
- Проверьте, что категория `car` настроена в Embedding Service
- Проверьте логи Embedding Service для детальной информации об ошибках
- Компонент автоматически пропустит все треки и заполнит пустые результаты при недоступности сервиса

#### Треки не найдены:

```
WARNING: tracks not found in detections.npz. Generating per-detection track IDs
```

**Решение**: Убедитесь, что `core_object_detections` предоставляет треки. Если треки отсутствуют, компонент продолжит работу, но результаты могут быть менее точными.

#### Пустые результаты от Embedding Service:

```
WARNING: Embedding Service returned empty results for track X
```

**Решение**: 
- Проверьте, что в базе Embedding Service есть автомобили категории `car`
- Попробуйте уменьшить `--similarity-threshold`
- Убедитесь, что качество изображения достаточное

#### Ошибка валидации размеров:

```
RuntimeError: Mismatched detection array shapes
```

**Решение**: Проверьте, что `core_object_detections/detections.npz` создан корректно и все массивы имеют совместимые размеры.

#### Отсутствие метаданных make/model/segment:

Если в результатах Embedding Service нет метаданных `make`, `model`, `segment`, соответствующие поля будут заполнены пустыми строками. Убедитесь, что при добавлении автомобилей в Embedding Service вы предоставляете эти метаданные.

#### Все треки упали с ошибками:

```
RuntimeError: car_semantics | All X tracks failed with Embedding Service errors. Service may be misconfigured or unavailable.
```

**Решение**: 
- Проверьте логи Embedding Service для детальной информации об ошибках
- Убедитесь, что Embedding Service запущен и доступен
- Проверьте, что категория `car` настроена в Embedding Service
- Убедитесь, что Triton доступен и модель `clip_336` загружена

### Parallelization

- **Внутренний**: компонент обрабатывает треки последовательно, используя batch поиск через Embedding Service (если доступен)
- **Внешний**: компонент безопасно параллелить по разным видео/`run_id` (per-run storage)

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор детекций из всех видео → группировка по трекам → выбор лучших кропов → batch поиск через Embedding Service → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - **Batch поиск**: группировка кропов из всех видео в батчи для уменьшения HTTP запросов
  - **Переиспользование клиента**: клиент Embedding Service создается один раз и используется для всех батчей
  - **Параллельная обработка**: обработка треков из разных видео параллельно

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **2-5x** (за счет batch поиска и лучшего использования ресурсов)
- Для single video: **1.1-1.2x** (за счет оптимизации обработки)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Human-friendly визуализация (Render System)

`car_semantics` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/car_semantics/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по распознаванию автомобилей (frames_count, tracks_count, unique_cars_count, top1_score_mean/std/min/max/median, confident_predictions_count/ratio)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, top1_car_id, top1_car_name, top1_score, unique_cars_count, topk_scores)
- **Distributions**: распределения top1_scores и topk_scores (min, max, mean, std, median, percentiles)
- **Top cars**: топ автомобили по количеству кадров и среднему score с make/model/segment

Render-context может быть использован:
- **LLM** для генерации текстовых описаний распознанных автомобилей в видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions, car pie charts)
- **Debugging**: быстрая проверка качества распознавания без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../car_semantics/_render/render.html`
- Содержит интерактивные графики (Chart.js):
  - Timeline: top-1 car scores по времени с цветовой кодировкой автомобилей
  - Distributions: статистики по top1_scores и topk_scores
  - Top cars: таблица с топ автомобилями и их метриками (make, model, segment)
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
car_semantics:
  embedding_service_url: "http://localhost:8005"
  topk: 5
  similarity_threshold: 0
  max_tracks: ""
  max_dets_per_frame: ""
  pad_ratio: 0.15
  use_sharpness: false
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

