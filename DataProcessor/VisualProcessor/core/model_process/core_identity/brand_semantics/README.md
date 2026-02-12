## Component: `brand_semantics` (semantic head, v1)

### Назначение

`brand_semantics` распознает бренды и логотипы в видео:
- использует CLIP через Embedding Service для извлечения embeddings логотипов
- сравнивает с базой известных брендов через Embedding Service
- возвращает per-track и per-frame top‑K идентификаций брендов

Компонент следует контракту semantic head:
- Требует `core_object_detections.frame_indices` (shared sampling group)
- Выравнивает выходные данные по тем же `frame_indices`
- Выводит результаты в NPZ формате

### Структура компонента

```
brand_semantics/
├── main.py                      # Основной модуль обработки
├── embedding_service_client.py  # HTTP клиент для Embedding Service
├── crop_utils.py                # Утилиты для кропов и предобработки
├── render.py                    # Render system (JSON и HTML визуализация)
├── __init__.py                  # Инициализация модуля
└── README.md                    # Эта документация
```

### Интеграция с Embedding Service

Компонент **использует Embedding Service** для хранения и поиска эмбеддингов брендов:

1. **Категория**: `brand` или `brand_semantic`
2. **Модель**: `clip_336` (CLIP 336x336 для баланса качества и скорости)
3. **Использование**:
   - Хранение эмбеддингов известных брендов/логотипов в Embedding Service
   - Поиск похожих брендов через Embedding Service (`POST /search`)
   - Добавление новых брендов через API (`POST /objects/add`)
   - Обновление информации о брендах (`PATCH /objects/{id}`)
   - Batch добавление брендов (`POST /objects/batch_add`)

**Преимущества использования Embedding Service**:
- Единая база данных для всех брендов
- Быстрый поиск через FAISS индексы
- Удобное управление базой (добавление/удаление/обновление)
- Хранение метаданных (название, алиасы, промпты, категории и т.д.)
- Поддержка мультиязычности (алиасы на разных языках)
- Горячее обновление (новые бренды доступны сразу)

**Пример использования**:
```python
# Добавить бренд в базу
POST http://localhost:8001/objects/add
{
    "category": "brand",
    "name": "Coca-Cola",
    "image": <logo_image>,
    "metadata": {
        "aliases_en": ["Coke", "Coca Cola"],
        "aliases_ru": ["Кока-Кола", "Кока Кола"],
        "prompts_en": ["Coca-Cola logo", "red Coca-Cola bottle"],
        "prompts_ru": ["логотип Кока-Колы", "красная бутылка"],
        "category": "beverage"
    }
}

# Распознать бренд на кадре (из bbox)
POST http://localhost:8001/search
{
    "category": "brand",
    "image": <crop_from_bbox>,
    "top_k": 5,
    "similarity_threshold": 0.7
}

# Batch добавление брендов
POST http://localhost:8001/objects/batch_add
{
    "category": "brand",
    "images": [<logo1>, <logo2>, ...],
    "names": ["Brand 1", "Brand 2", ...],
    "metadata_list": [{...}, {...}, ...]
}
```

**Алгоритм работы**:
1. Компонент получает bbox proposals из `core_object_detections` (особенно `logo_region`)
2. Извлекает crop из кадра по bbox
3. Отправляет crop в Embedding Service для поиска похожих брендов
4. Возвращает top‑K результатов с similarity scores

### Входы (required)

- `frames_dir/metadata.json`:
  - `core_object_detections.frame_indices` (shared sampling group)
  - `union_timestamps_sec`
- `rs_path/core_object_detections/detections.npz` (для bbox proposals и tracks)
- Embedding Service доступен (для поиска в базе брендов)

### Output (NPZ)

Путь: `rs_path/brand_semantics/brand_semantics.npz`

**Artifact filename**: `brand_semantics.npz` (фиксированное имя, `ARTIFACT_FILENAME`)

**Schema version**: `brand_semantics_npz_v1`

Ключи (v1):
- `frame_indices (N,) int32`
- `times_s (N,) float32`
- `track_ids (T,) int32` - ID треков
- `track_topk_brand_ids (T, 5) int32` - Top‑5 брендов на трек
- `track_topk_scores (T, 5) float32` - Similarity scores
- `frame_topk_brand_ids (N, 5) int32` - Top‑5 брендов на кадр
- `frame_topk_scores (N, 5) float32` - Similarity scores
- `meta` (object dict): статус + информация о базе брендов + models_used

**Meta обязательные поля** (baseline contract):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `dataprocessor_version` (может быть "unknown" в baseline)
- `status` (`ok`/`empty`/`error`), `empty_reason` (если `status="empty"`)
- `models_used[]` (если используются модели)
- `stage_timings_ms` (dict): тайминги стадий выполнения в миллисекундах:
  - `initialization`: инициализация компонента
  - `load_deps`: загрузка зависимостей (`core_object_detections`)
  - `process_frames`: обработка треков и поиск брендов
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
- Если бренд не найден → возвращает пустой результат (не error)

### Использование

#### Запуск через CLI:

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    --embedding-service-url http://localhost:8001 \
    --topk 5 \
    --similarity-threshold 0.7 \
    --max-tracks 100 \
    --max-dets-per-frame 5 \
    --pad-ratio 0.15 \
    --use-sharpness
```

#### Параметры:

- `--frames-dir` (required): Директория с кадрами и `metadata.json`
- `--rs-path` (required): Путь к result store (например, `result_store/platform/video/run`)
- `--embedding-service-url` (optional): URL Embedding Service (по умолчанию: `http://localhost:8001` или из `EMBEDDING_SERVICE_URL`)
- `--topk` (optional): Количество топ результатов (по умолчанию: `5`)
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
    --max-tracks 50 \
    --max-dets-per-frame 3
```

**С настройками качества:**
```bash
python main.py \
    --frames-dir frames/ \
    --rs-path results/ \
    --pad-ratio 0.20 \
    --use-sharpness \
    --similarity-threshold 0.75
```

### Детали реализации

#### Алгоритм обработки:

1. **Загрузка детекций**:
   - Загружает `core_object_detections/detections.npz`
   - Проверяет наличие `frame_indices` в метаданных
   - Фильтрует детекции по классу `logo_region` / `text_region`

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

5. **Агрегация результатов**:
   - Track-level: top-K результатов для каждого трека
   - Frame-level: дедупликация по `brand_name`, выбор лучшей similarity для каждого бренда

#### Обработка ошибок:

- **Fail-fast политика**: 
  - Если Embedding Service недоступен при инициализации → **error** (fail-fast)
  - Если все треки упали с ошибками → **error** (fail-fast)
  - Если бренд не найден (пустые результаты) → **warning** (не error, это нормально)
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

- Только proposals класса `logo_region` / `text_region` (по умолчанию)
- 1 crop на track (выбирается по `score × area × (optional sharpness)`)
- Лимиты: `--max-tracks`, `--max-dets-per-frame`
- Top‑K фиксирован: **5** (настраивается через `--topk`)

### Troubleshooting

#### Embedding Service недоступен:

```
WARNING: brand_semantics | Embedding Service test request failed: ...
Skipping all tracks to avoid repeated errors.
```

**Решение**: 
- Убедитесь, что Embedding Service запущен:
```bash
cd DataProcessor/embedding_service
python run_server.py
```
- Проверьте, что категория `brand` настроена в Embedding Service
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
- Проверьте, что в базе Embedding Service есть бренды категории `brand`
- Попробуйте уменьшить `--similarity-threshold`
- Убедитесь, что качество изображения достаточное

#### Ошибка валидации размеров:

```
RuntimeError: Mismatched detection array shapes
```

**Решение**: Проверьте, что `core_object_detections/detections.npz` создан корректно и все массивы имеют совместимые размеры.

#### Все треки упали с ошибками:

```
RuntimeError: brand_semantics | All X tracks failed with Embedding Service errors. Service may be misconfigured or unavailable.
```

**Решение**: 
- Проверьте логи Embedding Service для детальной информации об ошибках
- Убедитесь, что Embedding Service запущен и доступен
- Проверьте, что категория `brand` настроена в Embedding Service
- Убедитесь, что Triton доступен и модель `clip_336` загружена

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

`brand_semantics` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/brand_semantics/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по распознаванию брендов (frames_count, tracks_count, unique_brands_count, top1_score_mean/std/min/max/median, confident_predictions_count/ratio)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, top1_brand_id, top1_brand_name, top1_score, unique_brands_count, topk_scores)
- **Distributions**: распределения top1_scores и topk_scores (min, max, mean, std, median, percentiles)
- **Top brands**: топ бренды по количеству кадров и среднему score

Render-context может быть использован:
- **LLM** для генерации текстовых описаний распознанных брендов в видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions, brand pie charts)
- **Debugging**: быстрая проверка качества распознавания без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../brand_semantics/_render/render.html`
- Содержит интерактивные графики (Chart.js):
  - Timeline: top-1 brand scores по времени с цветовой кодировкой брендов
  - Distributions: статистики по top1_scores и topk_scores
  - Top brands: таблица с топ брендами и их метриками
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
brand_semantics:
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
