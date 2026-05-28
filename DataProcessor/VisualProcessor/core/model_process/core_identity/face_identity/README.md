## Component: `core_face_identity` (semantic head, Audit v3)

**Version**: `0.2` (Audit v3)  
**Schema**: `core_face_identity_npz_v2`

### Changelog

#### v0.2 (Audit v3, schema v2) — 2026-02-XX

**Критические изменения**:
- **Schema bump**: `core_face_identity_npz_v1` → `core_face_identity_npz_v2`
- **Детерминированный label-space**: добавлены `semantic_label_names` и `semantic_object_ids` для стабильного отображения UUID → int32
- **db_digest**: добавлен SHA256 digest базы лиц для reproducibility
- **meta_json**: добавлен cross-venv safe JSON representation meta
- **face_bbox_xyxy**: добавлено поле для сохранения bbox top-1 лица на каждом кадре (для render assets)

**Улучшения**:
- **Fail-fast политика**: строгая проверка доступности Embedding Service (включая empty case)
- **Строгая валидация UUID**: RuntimeError если UUID не найден в label-space (консистентность базы)
- **Atomic save**: двухпроходная запись NPZ для избежания частично записанных артефактов
- **Render system**: полноценный mini-dashboard с privacy banner, top/anti-top примерами, assets
- **Документация**: добавлен SCHEMA.md, обновлен README с разделом Render

**Breaking changes**:
- `face_ids` теперь содержит int32 индексы из `semantic_label_names` вместо UUID (детерминированное отображение)
- Структура NPZ изменена: добавлены обязательные поля `semantic_label_names`, `semantic_object_ids`, `face_bbox_xyxy`, `meta_json`

#### v0.1 (initial) — до Audit v3

- Базовая реализация face identity recognition
- Schema v1 без детерминированного label-space
- Без db_digest и meta_json
- Без render assets

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

### Early validation (Embedding Service, Audit v3)

Компонент выполняет **раннюю проверку доступности Embedding Service**:
- Проверка health endpoint через `embedding_client._ensure_url()` (fail-fast)
- Загрузка label-space через `GET /categories/face/labels` (fail-fast если база пустая)
- Тестовый запрос с первым кадром, где есть лицо, для проверки работоспособности search endpoint
- Если тест не проходит (например, 500 ошибка):
  - Выдается одно предупреждение вместо множества ошибок
  - Пропускается обработка всех кадров (но если все лица упали → error)
- Если тест проходит — продолжается обычная обработка

**Преимущества**:
- Fail-fast: проблемы обнаруживаются до начала обработки всех кадров
- Улучшенный UX: одно предупреждение вместо сотен ошибок
- Экономия ресурсов: не тратится время на обработку, которая все равно завершится ошибкой
- **Audit v3**: строгая проверка консистентности базы (UUID должен быть в label-space)

**No-fallback policy**:
- Если Embedding Service недоступен при инициализации → **error** (fail-fast)
- Если Embedding Service недоступен во время обработки → компонент пропускает все кадры с предупреждением
- Если нет `core_face_landmarks` → **error**
- Если нет `frame_indices` или `face_present` в landmarks.npz → **error**
- Если лиц в видео нет → возвращает valid empty (`status="empty"`, `empty_reason="no_faces_in_video"`)

### Output (NPZ)

Путь: `rs_path/core_face_identity/face_identity.npz`

**Artifact filename**: `face_identity.npz` (фиксированное имя, `ARTIFACT_FILENAME`)

**Schema version**: `core_face_identity_npz_v2`

**Ключи (v2, semantic-head contract v1)**:
- **Time-axis**:
  - `frame_indices (N,) int32` - Индексы кадров (только кадры с лицами)
  - `times_s (N,) float32` - Временные метки из `union_timestamps_sec[frame_indices]`
- **Label space (deterministic, derived from Embedding Service)**:
  - `semantic_label_names (A,) str`: `"int_id:name"` (детерминированное отображение UUID → int32)
  - `semantic_object_ids (A,) str`: UUID из Embedding Service (aligned с `semantic_label_names`)
- **Per-frame top-K results**:
  - `face_ids (N, K) int32` - Face identity indices (int32 из semantic_label_names), -1 где нет результата
  - `face_names (N, K) str` - Имена известных людей (human-readable), "" где нет результата
  - `face_similarities (N, K) float32` - Similarity scores (0.0-1.0), 0.0 где нет результата
- **Render assets** (Audit v3):
  - `face_bbox_xyxy (N, 4) float32` - Bbox для top-1 лица на каждом кадре (x1, y1, x2, y2), NaN где нет лица
- **Meta**:
  - `meta` (object dict): статус + информация о базе лиц + models_used + db_digest
  - `meta_json` (str): meta как JSON строка (cross-venv safe)

**K=5**: Фиксированный top-K для semantic-head v1 contract.

**См. также**: `SCHEMA.md` для полного описания схемы v2.

**Meta обязательные поля** (baseline contract):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `dataprocessor_version` (может быть "unknown" в baseline)
- `status` (`ok`/`empty`/`error`), `empty_reason` (если `status="empty"`)
- `models_used[]` (если используются модели)
- `model_signature` (детерминированная подпись моделей)
- `stage_timings_ms` (dict): тайминги стадий выполнения в миллисекундах:
  - `initialization`: инициализация компонента
  - `load_deps`: загрузка зависимостей (`core_face_landmarks`)
  - `process_frames`: обработка кадров и поиск лиц
  - `saving`: сохранение артефакта
  - `total`: общее время выполнения
- **DB provenance (reproducibility)**:
  - `db_name`: `"embedding_service"`
  - `db_version`: `"v1"`
  - `db_digest`: SHA256 от канонического списка labels (детерминированная подпись базы)

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

### Human-friendly визуализация (Render System, Audit v3)

`face_identity` генерирует **render-context JSON** и **offline HTML mini-dashboard** для каждого запуска:

#### Render files

- **Render-context JSON**: `result_store/<platform_id>/<video_id>/<run_id>/core_face_identity/_render/render_context.json`
- **HTML mini-dashboard**: `result_store/.../core_face_identity/_render/render.html` (offline, без CDN)
- **Assets** (опционально): `result_store/.../core_face_identity/_render/assets/*.jpg` (face crops для примеров)

#### Render-context JSON структура

- **summary**: ключевые статистики (frames, unique_faces_count, total_identifications, confident_predictions_count/ratio, status)
- **key_facts**: `schema_version`, `producer_version`, `db_digest`, `embedding_model`, `stage_timings_ms`
- **config_highlights**: важные параметры конфига (`top_k`, `similarity_threshold`, `category`)
- **qa_hints**: рекомендации по интерпретации результатов
- **distributions**: статистики по `top1_scores` и `all_scores` (min, max, mean, std, median, percentiles)
- **top_faces**: топ лица по количеству кадров и среднему score
- **examples**: top/anti-top примеры с assets (face crops)
- **timeline**: данные по каждому кадру (frame_index, time_sec, top1_face_name, top1_score, is_confident)

#### HTML mini-dashboard

**Секции**:
1. **Key facts**: schema_version, producer_version, db_digest, embedding_model, frames, unique_faces
2. **Config highlights**: top_k, similarity_threshold, category
3. **Examples (top / anti-top)**: визуальные примеры с face crops (если assets доступны)
4. **Top faces**: таблица с топ лицами по count/avg_score/max_score/min_score
5. **Timeline**: таблица с поиском и фильтрацией по имени лица (первые 3000 кадров)
6. **How to QA**: рекомендации по проверке качества

**Интерактивность** (offline, vanilla JS):
- Поиск по имени лица в timeline таблице
- Сортировка таблиц (через браузер)
- Кликабельные ссылки на assets

**Assets policy**:
- Face crops сохраняются в `_render/assets/` для top/anti-top примеров
- Имена файлов детерминированы: `face_frame_{frame_index}.jpg`
- Рендер работает offline (открытие `render.html` из файловой системы)

**Privacy**:
- Face crops содержат персональные данные (лица)
- В production можно отключить генерацию assets через конфиг
- Render содержит только метаданные (имена, scores), не raw embeddings

#### Конфигурация (в `global_config.yaml`)

```yaml
face_identity:
  embedding_service_url: "http://localhost:8005"
  topk: 5
  similarity_threshold: 0
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML mini-dashboard
    # assets_dir будет автоматически установлен VisualProcessor renderer
```

#### Как читать выход

**Типовые распределения**:
- **top1_scores**: обычно > 0.3 для корректных распознаваний, > 0.7 для confident
- **all_scores**: широкое распределение (0.0-1.0), пик около 0.5-0.7 для известных людей

**Аномалии для поиска**:
- Много `confident=false` при явно известных людях → возможно `similarity_threshold` слишком высокий или база маленькая
- Много `confident=true` при явном мусоре → возможно `similarity_threshold` слишком низкий или качество face crops плохое
- Нестабильность: одинаковые люди имеют сильно различающиеся scores на соседних кадрах → возможно проблемы с качеством face crops или Embedding Service

**Связь с NPZ**:
- NPZ остается source-of-truth
- Render — это "человеческая интерпретация" NPZ, не отдельный контракт данных
- Все данные в render можно восстановить из NPZ

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

### Алгоритм работы (Audit v3, schema v2)

1. Компонент загружает `core_face_landmarks/landmarks.npz` (no-fallback)
2. Инициализирует Embedding Service клиент и проверяет доступность (fail-fast)
3. **Загружает label-space** из Embedding Service (`GET /categories/face/labels`):
   - Получает список всех известных людей с UUID, именами, embedding_model
   - **Fail-fast**: если база пустая (0 labels) → error
   - Создает детерминированное отображение UUID → int32 (сортировка по UUID)
   - Вычисляет `db_digest` (SHA256 от канонического списка labels) для reproducibility
   - Создает `semantic_label_names` и `semantic_object_ids` для детерминированного label-space
4. Извлекает `frame_indices` из landmarks.npz и фильтрует их по `face_present`:
   - Оставляет только кадры, где `face_present[i]` содержит хотя бы одно `True`
   - Если таких кадров нет → возвращает valid empty artifact **только если Embedding Service доступен и база не пустая** (fail-fast в empty case)
5. Для каждого отфильтрованного кадра (только кадры с лицами):
   - Загружает кадр через `FrameManager`
   - Для каждого лица в кадре (из `face_landmarks`):
     - Извлекает bbox из landmarks (min/max координаты с padding 5%)
     - Кропает лицо из кадра
     - Отправляет crop в Embedding Service для поиска (`POST /search`)
       - Использует retry механизм (3 попытки с exponential backoff)
       - При ошибке логирует warning и увеличивает счетчик failed_faces
     - Получает top-K результатов с similarity scores и UUID
     - **Преобразует UUID в int32 индексы** через детерминированное отображение
     - **Fail-fast**: если UUID не найден в label-space → RuntimeError (консистентность базы нарушена)
   - Дедуплицирует результаты по имени (берет лучший similarity для каждого имени)
   - Заполняет выходные массивы (`face_ids` как int32 индексы, `face_names`, `face_similarities`)
   - **Сохраняет bbox top-1 лица** в `face_bbox_xyxy` для render assets
6. Проверяет fail-fast условие: если все лица упали с ошибками → **error**
7. Сохраняет результаты в NPZ с полным meta (все обязательные поля baseline contract):
   - Добавляет `db_digest`, `db_name`, `db_version` для reproducibility
   - Добавляет `meta_json` (cross-venv safe)
   - Использует **atomic save** (двухпроходная запись) для избежания частично записанных артефактов

### Valid empty outputs (Audit v3)

**Valid empty** создается только если:
- Лиц в видео нет (`empty_reason="no_faces_in_video"`)
- **И** Embedding Service доступен
- **И** база не пустая (есть хотя бы один label)

В этом случае:
- Компонент возвращает NPZ со `status="empty"`
- `empty_reason="no_faces_in_video"`
- Выходные массивы заполнены значениями по умолчанию:
  - `frame_indices`: пустой массив
  - `times_s`: пустой массив
  - `face_ids`: пустой массив (0, K)
  - `face_names`: пустой массив (0, K)
  - `face_similarities`: пустой массив (0, K)
  - `face_bbox_xyxy`: пустой массив (0, 4)
- Но `semantic_label_names` и `semantic_object_ids` присутствуют (с db_digest)

**Fail-fast**: Если Embedding Service недоступен или база пустая → компонент падает с ошибкой, не создает empty artifact.

### Troubleshooting

#### Embedding Service недоступен (Audit v3, fail-fast):

```
RuntimeError: core_face_identity | Embedding Service unavailable at http://localhost:8001: ... (fail-fast)
```

**Решение**: 
- Убедитесь, что Embedding Service запущен:
```bash
cd DataProcessor/embedding_service
python run_server.py
```
- Проверьте, что категория `face` настроена в Embedding Service
- Проверьте логи Embedding Service для детальной информации об ошибках
- **Важно (Audit v3)**: Компонент теперь использует fail-fast политику — при недоступности сервиса компонент падает с ошибкой, не создает empty artifact

#### База пустая (0 labels):

```
RuntimeError: core_face_identity | Embedding Service category 'face' has 0 labels (fail-fast)
```

**Решение**:
- Засейте категорию `face` хотя бы несколькими labels через `sync_known_people_to_embedding_service.py`
- Проверьте: `GET /categories/face/labels` должен вернуть `count > 0`

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

#### UUID не найден в label-space (Audit v3):

```
RuntimeError: core_face_identity | Face UUID {uuid} not found in label-space. This indicates database inconsistency...
```

**Решение**:
- Это указывает на неконсистентность базы: label-space был загружен с одним набором labels, но search вернул UUID, которого нет в этом наборе
- Возможные причины: база изменилась между загрузкой label-space и поиском, или race condition
- Перезапустите компонент (label-space будет перезагружен)
- Если проблема повторяется — проверьте логи Embedding Service на предмет изменений базы во время работы

### Ссылки

- **Audit v3 decisions**: `DataProcessor/docs/audit_v3/DECISIONS_AND_RULES.md`
- **Schema**: `SCHEMA.md` (в этой директории)
- **Artifacts and schemas**: `DataProcessor/docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- **Segmenter contract**: `DataProcessor/docs/contracts/SEGMENTER_CONTRACT.md`
- **Embedding Service**: `DataProcessor/embedding_service/README.md`
- **VisualProcessor main index**: `DataProcessor/VisualProcessor/docs/MAIN_INDEX.md`
