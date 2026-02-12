## Component: `core_face_identity` (semantic head, v1)

### Назначение

`core_face_identity` идентифицирует известных людей (celebrity retrieval) в видео:
- извлекает face crops из `core_face_landmarks` (bbox из landmarks)
- использует Embedding Service для поиска похожих лиц
- возвращает per-frame top‑K идентификаций с similarity scores

Компонент следует контракту semantic head:
- Требует `core_face_landmarks` (для извлечения face bbox и определения кадров с лицами)
- Читает `frame_indices` из `core_face_landmarks/landmarks.npz` и фильтрует по `face_present`
- Выравнивает выходные данные по отфильтрованным `frame_indices` (только кадры с лицами)
- Выводит результаты в NPZ формате

### Структура компонента

```
face_identity/
├── main.py                      # Основной модуль обработки
├── embedding_service_client.py  # HTTP клиент для Embedding Service
├── render.py                    # Render system (JSON и HTML визуализация)
├── add_person.py                # Полуавтоматическое наполнение known_people/
├── sync_known_people_to_embedding_service.py  # Синхронизация базы в Embedding Service
├── face_aligment.py             # Утилиты для выравнивания лиц
├── faiss_index.py               # Legacy FAISS индекс (deprecated, используйте Embedding Service)
├── known_people/                # Локальная база лиц (для подготовки)
└── README.md                    # Эта документация
```

### Интеграция с Embedding Service

Компонент **использует Embedding Service** для хранения и поиска эмбеддингов лиц:

1. **Категория**: `face`
2. **Модель**: `arcface` (ArcFace для извлечения embeddings лиц, размерность 512)
3. **Использование**:
   - Хранение эмбеддингов известных людей в Embedding Service
   - Поиск похожих лиц через Embedding Service (`POST /search`)
   - Добавление новых людей в базу через API (`POST /objects/add`)
   - Обновление информации о людях (`PATCH /objects/{id}`)
   - Batch добавление лиц (`POST /objects/batch_add`)

**Преимущества использования Embedding Service**:
- Единая база данных для всех известных людей
- Быстрый поиск через FAISS индексы
- Удобное управление базой (добавление/удаление/обновление)
- Хранение метаданных (имя, роли, источники и т.д.)
- Автоматическая нормализация embeddings (L2-нормализация)
- Горячее обновление (новые лица доступны сразу после добавления)

**Пример использования API**:
```python
# Добавить известного человека в базу
POST http://localhost:8001/objects/add
{
    "category": "face",
    "name": "ASATAchannel",
    "image": <face_image>,
    "metadata": {
        "type": "celebrity",
        "platform": "youtube",
        "aliases": ["ASATA", "ASATA channel"]
    }
}

# Идентифицировать лицо на кадре
POST http://localhost:8001/search
{
    "category": "face",
    "image": <face_crop>,
    "top_k": 5,
    "similarity_threshold": 0.7
}
```

### Входы (required)

- `frames_dir/metadata.json`:
  - `union_timestamps_sec` (required by contract)
  - `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version` (required run identity)
- `rs_path/core_face_landmarks/landmarks.npz` (для извлечения face bbox из landmarks и определения кадров с лицами)
- Embedding Service доступен (для поиска в базе лиц)

**Важно**: Компонент **не использует** `core_object_detections.frame_indices` из metadata. Вместо этого:
- Читает `frame_indices` из `core_face_landmarks/landmarks.npz`
- Фильтрует их по `face_present` - оставляет только кадры, где были найдены лица
- Обрабатывает только отфильтрованные кадры

**Обработка ошибок**:

- **Fail-fast политика**: 
  - Если Embedding Service недоступен при инициализации → **error** (fail-fast)
  - Если все лица упали с ошибками → **error** (fail-fast)
  - Если лицо не найдено (пустые результаты) → **warning** (не error, это нормально)
- **Retry механизм**: Все запросы к Embedding Service автоматически повторяются при ошибках (3 попытки с exponential backoff)
- **Валидация данных**: Проверка размеров массивов, соответствия frame_indices
- **Graceful degradation**: При ошибке отдельного лица компонент продолжает работу, но если все лица упали - падает с ошибкой
- **Логирование**: Все ошибки и предупреждения логируются

### Early validation (Embedding Service)

Компонент выполняет **раннюю проверку доступности Embedding Service**:
- Проверка health endpoint через `embedding_client._ensure_url()`
- Тестовый запрос с первым кадром, где есть лицо (dummy image), для проверки работоспособности search endpoint
- Если тест не проходит (например, 500 ошибка):
  - Выдается одно предупреждение вместо множества ошибок
  - Пропускается обработка всех кадров
  - Заполняются пустые результаты вместо обработки с ошибками
- Если тест проходит — продолжается обычная обработка

**Преимущества**:
- Fail-fast: проблемы обнаруживаются до начала обработки всех кадров
- Улучшенный UX: одно предупреждение вместо сотен ошибок
- Экономия ресурсов: не тратится время на обработку, которая все равно завершится ошибкой

**No-fallback policy**:
- Если Embedding Service недоступен при инициализации → **error** (fail-fast)
- Если Embedding Service недоступен во время обработки → компонент пропускает все кадры с предупреждением
- Если нет `core_face_landmarks` → **error**
- Если нет `frame_indices` или `face_present` в landmarks.npz → **error**
- Если лиц в видео нет → возвращает valid empty (`status="empty"`, `empty_reason="no_faces_in_video"`)

### Output (NPZ)

Путь: `rs_path/core_face_identity/face_identity.npz`

**Artifact filename**: `face_identity.npz` (фиксированное имя, `ARTIFACT_FILENAME`)

**Schema version**: `core_face_identity_npz_v1`

**Ключи (v1)**:
- `frame_indices (N,) int32` - Индексы кадров (из metadata)
- `times_s (N,) float32` - Временные метки из `union_timestamps_sec[frame_indices]`
- `face_ids (N, K) int32` - ID известных людей на каждом кадре (-1 если нет результата)
- `face_names (N, K) str` - Имена известных людей на каждом кадре (пустая строка если нет)
- `face_similarities (N, K) float32` - Similarity scores (0.0 если нет результата)
- `meta` (object dict): статус + информация о базе лиц + models_used

**Meta обязательные поля** (baseline contract):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `dataprocessor_version` (может быть "unknown" в baseline)
- `status` (`ok`/`empty`/`error`), `empty_reason` (если `status="empty"`)
- `models_used[]` (если используются модели)
- `stage_timings_ms` (dict): тайминги стадий выполнения в миллисекундах:
  - `initialization`: инициализация компонента
  - `load_deps`: загрузка зависимостей (`core_face_landmarks`)
  - `process_frames`: обработка кадров и поиск лиц
  - `saving`: сохранение артефакта
  - `total`: общее время выполнения

### Sampling / units-of-processing requirements

Компонент **работает только с кадрами, где были найдены лица**:
- Читает `frame_indices` из `core_face_landmarks/landmarks.npz` (не из metadata)
- **Фильтрует** `frame_indices` по `face_present` - оставляет только кадры, где `core_face_landmarks` нашел хотя бы одно лицо
- **Не генерирует семплинг сам** (no-fallback)
- Выравнивает выходные данные по отфильтрованным `frame_indices` (только кадры с лицами)

**Важно**: Компонент обрабатывает **только те кадры, на которых `core_face_landmarks` нашел лица**. Это означает:
- Если на кадре нет лиц → кадр **не обрабатывается** (не попадает в выходной NPZ)
- Выходной NPZ содержит только кадры с лицами
- Если лиц в видео нет → компонент возвращает valid empty (`status="empty"`, `empty_reason="no_faces_in_video"`)

**Требования к выборке**:
- `core_face_landmarks` должен быть выполнен перед `core_face_identity`
- `core_face_landmarks` определяет, какие кадры будут обработаны (только те, где найдены лица)
- Количество обрабатываемых кадров зависит от количества кадров с лицами в `core_face_landmarks`

**Зависимости от других компонентов**:
- `core_face_landmarks` **обязателен** и должен быть выполнен перед `core_face_identity` (no-fallback)
- `core_face_landmarks` является источником истины для `frame_indices` (не `core_object_detections`)

### Models

Компонент использует **Embedding Service** для поиска лиц:

#### HTTP Service (Embedding Service)

1. **Embedding Service** (face recognition)
   - **Triton**: ❌ Нет (HTTP service)
   - **Runtime**: `http`
   - **Engine**: `http`
   - **Precision**: `fp32`
   - **Device**: `cpu` (сервер Embedding Service)
   - **Model**: `arcface` (ArcFace для извлечения embeddings, размерность 512)
   - **Base URL**: настраивается через `--embedding-service-url` или `EMBEDDING_SERVICE_URL` env (default: `http://localhost:8001`)

**Примечание**: Embedding Service использует ArcFace модель для извлечения embeddings лиц. Модель работает на сервере Embedding Service, компонент только отправляет HTTP запросы.

### Parallelization

- **Внутренний**: компонент обрабатывает кадры последовательно, используя batch поиск через Embedding Service (если доступен)
- **Внешний**: компонент безопасно параллелить по разным видео/`run_id` (per-run storage)

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор лиц из всех видео → группировка в батчи → batch поиск через Embedding Service → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - **Batch поиск**: группировка лиц из всех видео в батчи для уменьшения HTTP запросов
  - **Переиспользование клиента**: клиент Embedding Service создается один раз и используется для всех батчей
  - **Параллельная обработка**: обработка лиц из разных видео параллельно

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **2-5x** (за счет batch поиска и лучшего использования ресурсов)
- Для single video: **1.1-1.2x** (за счет оптимизации обработки)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Progress / state events

Компонент публикует прогресс выполнения в `state_events.jsonl` (baseline contract):

**Стадии выполнения**:
- `start` → `load_deps` → `process_frames` → `save` → `done`

**Гранулярный прогресс**:
- Во время стадии `process_frames` компонент публикует прогресс обработки кадров (≥10 обновлений)
- Формат события: `{"progress": 0.0-1.0, "done": int, "total": int, "stage": "process_frames"}`

**Использование**: Backend сайта может читать `state_events.jsonl` для отображения прогресса анализа в реальном времени.

**Рекомендации**:
- Для ускорения обработки можно запускать несколько экземпляров на разных видео параллельно
- Batch API в Embedding Service (если будет реализован) позволит обрабатывать несколько лиц за один запрос

### Performance characteristics

**Единица обработки**: `frame` (один кадр с возможными несколькими лицами)

**Типичные значения** (зависит от количества лиц на кадре):
- Latency per frame: ~100-500 ms (зависит от количества лиц и задержки Embedding Service)
- CPU RAM peak: ~200-300 MB
- GPU VRAM: не используется (компонент работает через HTTP)

**Факторы производительности**:
- Количество лиц на кадре (каждое лицо требует отдельный HTTP запрос)
- Задержка Embedding Service (network latency)
- Размер face crops (влияет на время передачи по сети)

**Полные данные**: см. `docs/models_docs/resource_costs/core_face_identity_costs_v1.json` (если доступно)

### Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Similarity sanity**: similarity scores в диапазоне [0, 1], top-1 обычно > 0.3 для корректных распознаваний
- **Консистентность**: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- **Стабильность**: одинаковые люди должны иметь похожие similarity scores на соседних кадрах
- **Coverage**: проверка, что распознавание покрывает разные части видео (начало/середина/конец)

### Human-friendly визуализация (Render System)

`face_identity` генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_face_identity/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по распознаванию лиц (frames_count, unique_faces_count, total_identifications, confident_predictions_count/ratio, top1_score_mean/std/min/max/median)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, top1_face_id, top1_face_name, top1_score, is_confident, unique_faces_count, topk_scores)
- **Distributions**: распределения top1_scores и all_scores (min, max, mean, std, median, percentiles)
- **Top faces**: топ лица по количеству кадров и среднему score

Render-context может быть использован:
- **LLM** для генерации текстовых описаний распознанных людей в видео
- **Frontend** для построения графиков и визуализаций (timeline charts, distributions, face pie charts)
- **Debugging**: быстрая проверка качества распознавания без загрузки NPZ

**HTML debug страница** (опционально):
- Путь: `result_store/.../core_face_identity/_render/render.html`
- Содержит интерактивные графики (Chart.js):
  - Timeline: top-1 face scores по времени с цветовой кодировкой лиц
  - Distributions: статистики по top1_scores и all_scores
  - Top faces: таблица с топ лицами и их метриками
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
face_identity:
  embedding_service_url: "http://localhost:8005"
  topk: 5
  similarity_threshold: 0
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

### Использование

#### Базовый запуск

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    --embedding-service-url http://localhost:8001
```

#### С параметрами

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    --embedding-service-url http://localhost:8001 \
    --topk 10 \
    --similarity-threshold 0.7
```

#### Параметры командной строки

- `--frames-dir` (required): Директория с кадрами и `metadata.json`
- `--rs-path` (required): Путь к result_store (например, `result_store/platform/video/run`)
- `--embedding-service-url` (optional): URL Embedding Service (default: из `EMBEDDING_SERVICE_URL` env или `http://localhost:8001`)
- `--topk` (optional): Количество top результатов на кадр (default: 5)
- `--similarity-threshold` (optional): Минимальный порог similarity (default: 0.0, range: 0.0-1.0)

### Подготовка базы лиц (`known_people/`)

Для работы `core_face_identity` нужна база известных людей. В этом компоненте есть полноценный offline‑пайплайн:

- `add_person.py` — полуавтоматическое наполнение `known_people/` из видео и фото
- `sync_known_people_to_embedding_service.py` — синхронизация готовой базы в Embedding Service (категория `face`)

#### 1. Структура `known_people/`

После разметки `known_people/` выглядит так:

```
face_identity/known_people/
├── mrbeast/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── taylorswift/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── uriydud/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── ...
```

- Имя папки = label человека (используется как `name` в Embedding Service)
- Внутри — несколько нормализованных и выровненных фото лица (224x224 / 256x256)

#### 2. Сбор лиц: `add_person.py`

Скрипт `add_person.py`:
- Проходит по видео в `example/example_videos` и по фото в `example/example_photo`
- Детектирует лица через `InsightFace (buffalo_l)`
- Выравнивает лицо через `face_aligment.align_face`
- Показывает каждое лицо и просит ввести **label** (строку: `mrbeast`, `taylorswift`, `uriydud`, ...)
- Сохраняет выровненное лицо в `known_people/<label>/<N>.jpg`

Запуск:

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/face_identity/add_person.py
```

Особенности:
- Кадры из видео выбираются **равномерно** (по 20 кадров на видео)
- Поддерживается GUI OpenCV и fallback на `matplotlib` или сохранение превью в файл

#### 3. Синхронизация с Embedding Service

Когда `known_people/` заполнен, нужно записать людей в Embedding Service:

Скрипт: `sync_known_people_to_embedding_service.py`

Алгоритм для каждого человека:
1. Считает все фото из `known_people/<person_name>/`
2. Для каждого фото:
   - `InsightFace (buffalo_l)` → `face.embedding` → L2-нормализация
3. Усредняет эмбеддинги:
   - `avg_emb = mean(embeddings, axis=0)`
   - финальная L2-нормализация `avg_emb`
4. Записывает в Embedding Service (категория `face`) через `EmbeddingManager.add_from_embedding`:
   - `category = "face"`
   - `name = <person_name>`
   - `embedding = avg_emb`
   - `metadata = {"source": "known_people", "num_images": N}`

Запуск:

```bash
cd DataProcessor
python VisualProcessor/core/model_process/core_identity/face_identity/sync_known_people_to_embedding_service.py
```

Требования:
- Запущен Embedding Service (см. `embedding_service/README.md`)
- Настроены переменные окружения PostgreSQL (`POSTGRES_*`) или `.env`

#### 4. Проверка, что лица попали в Embedding Service

Проверка категорий и количества:

```bash
curl http://localhost:8001/categories
curl "http://localhost:8001/categories/face/count"
```

Поиск по новому фото:

```bash
curl -X POST "http://localhost:8001/search" \
  -F "category=face" \
  -F "top_k=5" \
  -F "similarity_threshold=0.5" \
  -F "image=@path/to/face.jpg"
```

Ожидаемый результат:
- В выдаче будут `name` из `known_people` (например, `mrbeast`, `taylorswift`, `uriydud`)
- В `metadata.num_images` можно увидеть, сколько фото использовалось для усреднения.

### Алгоритм работы

1. Компонент загружает `core_face_landmarks/landmarks.npz` (no-fallback)
2. Инициализирует Embedding Service клиент и проверяет доступность (fail-fast)
3. Извлекает `frame_indices` из landmarks.npz и фильтрует их по `face_present`:
   - Оставляет только кадры, где `face_present[i]` содержит хотя бы одно `True`
   - Если таких кадров нет → возвращает valid empty artifact
4. Для каждого отфильтрованного кадра (только кадры с лицами):
   - Загружает кадр через `FrameManager`
   - Для каждого лица в кадре (из `face_landmarks`):
     - Извлекает bbox из landmarks (min/max координаты с padding 5%)
     - Кропает лицо из кадра
     - Отправляет crop в Embedding Service для поиска (`POST /search`)
       - Использует retry механизм (3 попытки с exponential backoff)
       - При ошибке логирует warning и увеличивает счетчик failed_faces
     - Получает top-K результатов с similarity scores
   - Дедуплицирует результаты по имени (берет лучший similarity для каждого имени)
   - Заполняет выходные массивы (`face_ids`, `face_names`, `face_similarities`)
5. Проверяет fail-fast условие: если все лица упали с ошибками → **error**
6. Сохраняет результаты в NPZ с полным meta (все обязательные поля baseline contract)

### Valid empty outputs

Если лиц в видео нет или Embedding Service не нашел совпадений:
- Компонент возвращает NPZ со `status="empty"`
- `empty_reason="no_faces_in_video"` (если лиц нет) или пустые результаты (если совпадений нет)
- Выходные массивы заполнены значениями по умолчанию:
  - `face_ids`: -1
  - `face_names`: пустые строки
  - `face_similarities`: 0.0

Это **валидный empty**, не ошибка.

### Troubleshooting

#### Embedding Service недоступен:

```
WARNING: core_face_identity | Embedding Service test request failed: ...
Skipping all frames to avoid repeated errors.
```

**Решение**: 
- Убедитесь, что Embedding Service запущен:
```bash
cd DataProcessor/embedding_service
python run_server.py
```
- Проверьте, что категория `face` настроена в Embedding Service
- Проверьте логи Embedding Service для детальной информации об ошибках
- Компонент автоматически пропустит все кадры и заполнит пустые результаты при недоступности сервиса

#### Пустые результаты от Embedding Service:

```
WARNING: Embedding Service returned empty results for frame X, face Y
```

**Решение**: 
- Проверьте, что в базе Embedding Service есть лица категории `face`
- Попробуйте уменьшить `--similarity-threshold`
- Убедитесь, что качество изображения достаточное

#### Ошибка валидации размеров:

```
RuntimeError: Mismatched array shapes
```

**Решение**: Проверьте, что `core_face_landmarks/landmarks.npz` создан корректно и все массивы имеют совместимые размеры.

#### Все лица упали с ошибками:

```
RuntimeError: core_face_identity | All X faces failed with Embedding Service errors. Service may be misconfigured or unavailable.
```

**Решение**: 
- Проверьте логи Embedding Service для детальной информации об ошибках
- Убедитесь, что Embedding Service запущен и доступен
- Проверьте, что категория `face` настроена в Embedding Service
- Убедитесь, что Triton доступен и модель для категории `face` загружена

### Ссылки

- **Baseline component audit criteria**: `docs/baseline/BASELINE_COMPONENT_AUDIT_CRITERIA.md`
- **Artifacts and schemas**: `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- **Segmenter contract**: `docs/contracts/SEGMENTER_CONTRACT.md`
- **Embedding Service**: `embedding_service/README.md`
