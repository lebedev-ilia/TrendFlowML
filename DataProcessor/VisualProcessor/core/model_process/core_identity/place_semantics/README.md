## Component: `place_semantics` (semantic head, v1)

### Назначение

`place_semantics` распознает места и лэндмарки в видео:
- использует CLIP через Embedding Service для извлечения embeddings кадров
- сравнивает с базой известных мест через Embedding Service
- группирует кадры по местам в tracks (временная сегментация)
- возвращает per-track и per-frame top‑K идентификаций мест

Компонент следует контракту semantic head:
- Требует `core_object_detections.frame_indices` (shared sampling group)
- Выравнивает выходные данные по тем же `frame_indices`
- Выводит результаты в NPZ формате

### Структура компонента

```
place_semantics/
├── main.py                      # Основной модуль обработки
├── embedding_service_client.py  # HTTP клиент для Embedding Service
├── quality_report/              # Скрипты для проверки качества
│   └── demo_place_semantics_quality.py
└── README.md                    # Эта документация
```

### Интеграция с Embedding Service

Компонент **использует Embedding Service** для хранения и поиска эмбеддингов мест:

1. **Категория**: `place` или `place_semantic`
2. **Модель**: `clip_448` (CLIP 448x448 для высокого качества распознавания мест)
3. **Использование**:
   - Хранение эмбеддингов известных мест/лэндмарков в Embedding Service
   - Поиск похожих мест через Embedding Service (`POST /search`)
   - Добавление новых мест через API (`POST /objects/add`)
   - Обновление информации о местах (`PATCH /objects/{id}`)

**Преимущества использования Embedding Service**:
- Единая база данных для всех мест
- Быстрый поиск через FAISS индексы
- Удобное управление базой (добавление/удаление/обновление)
- Хранение метаданных (название, страна, город, координаты и т.д.)
- Горячее обновление (новые места доступны сразу)

**Пример использования**:
```python
# Добавить место в базу
POST http://localhost:8001/objects/add
{
    "category": "place",
    "name": "Red Square",
    "image": <place_image>,
    "metadata": {
        "country": "Russia",
        "city": "Moscow",
        "latitude": 55.7539,
        "longitude": 37.6208
    }
}

# Распознать место на кадре
POST http://localhost:8001/search
{
    "category": "place",
    "image": <frame_image>,
    "top_k": 5,
    "similarity_threshold": 0.7
}
```

**Алгоритм работы**:
1. Компонент получает `frame_indices` из `core_object_detections.frame_indices` (shared sampling group)
2. Загружает кадры через `FrameManager`
3. Отправляет каждый кадр в Embedding Service для поиска похожих мест
4. Группирует кадры по местам в tracks (временная сегментация)
5. Возвращает top‑K результатов с similarity scores

### Входы (required)

- `frames_dir/metadata.json`:
  - `core_object_detections.frame_indices` (shared sampling group)
  - `union_timestamps_sec`
- Embedding Service доступен (для поиска в базе мест)

**Early validation**: Компонент выполняет раннюю проверку доступности Embedding Service:
- Проверка health endpoint через `embedding_client._ensure_url()`
- Тестовый запрос с первым кадром для проверки работоспособности search endpoint
- Если тест не проходит (например, 500 ошибка):
  - Выдается одно предупреждение вместо множества ошибок
  - Пропускается обработка всех кадров
  - Заполняются пустые результаты вместо обработки с ошибками
- Если тест проходит — продолжается обычная обработка

**No-fallback**: Если Embedding Service недоступен при инициализации → **RuntimeError** (fail-fast). Если сервис недоступен во время обработки → компонент пропускает все кадры с предупреждением.

### Output (NPZ)

Путь: `rs_path/place_semantics/place_semantics.npz`

**Artifact filename**: `place_semantics.npz` (фиксированное имя, `ARTIFACT_FILENAME`)

**Schema version**: `place_semantics_npz_v1`

Ключи (v1):
- `frame_indices (N,) int32` — shared sampling group
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `track_ids (T,) int32` — ID треков (отдельные tracks для разных мест)
- `track_topk_ids (T, K) int32` — Top‑K мест на трек
- `track_topk_scores (T, K) float32` — Similarity scores для треков
- `track_present_mask (T,) bool` — Маска присутствия треков
- `track_is_confident_top1 (T,) bool` — Флаг уверенности для top-1 места на трек
- `frame_topk_ids (N, K) int32` — Top‑K мест на кадр
- `frame_topk_scores (N, K) float32` — Similarity scores для кадров
- `frame_is_confident_top1 (N,) bool` — Флаг уверенности для top-1 места на кадр
- `semantic_label_names (A,) str` — Массив строк "id:name" для маппинга label_id → place_name
- `threshold_per_label_arr (A,) float32` — Пороги для каждого места (NaN если нет)
- `meta` (object dict): статус + информация о базе мест + models_used

**Meta обязательные поля** (baseline contract):
- `producer`, `producer_version`, `schema_version`, `created_at`
- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `dataprocessor_version` (может быть "unknown" в baseline)
- `status` (`ok`/`empty`/`error`), `empty_reason` (если `status="empty"`)
- `models_used[]` (если используются модели)
- `stage_timings_ms` (dict): тайминги стадий выполнения в миллисекундах:
  - `initialization`: инициализация компонента
  - `load_deps`: загрузка зависимостей (FrameManager, Embedding Service)
  - `process_frames`: обработка кадров и поиск мест
  - `saving`: сохранение артефакта
  - `total`: общее время выполнения

### Sampling requirements (фиксируем требования компонента)

`place_semantics` использует shared sampling group (`core_object_detections.frame_indices`), поэтому требования к выборке совпадают с требованиями `core_object_detections`.

**Требования к выборке**:
- **coverage**: обязательно покрывать начало/середину/конец и быть равномерной по времени
- **cap**: для длинных видео иметь ограничение по числу кадров (чтобы не взрывать стоимость)
- **стабильность**: индексы должны быть отсортированы, уникальны, валидны для union-domain

**Непрерывная кривая выборки** (Segmenter-owned):
- `target_gap_sec = f(duration_s)` — непрерывная монотонная кривая, построенная через log‑log интерполяцию по anchor‑точкам
- `budget_n = round(duration_s / target_gap_sec)` (и затем `N = min(requested_max, budget_n)`)

**Ориентиры по кривой** (приблизительно):
- **≈ 5 минут**: `target_gap_sec ≈ 1s`
- **≈ 10 минут**: `target_gap_sec ≈ 2s`
- **≈ 20 минут**: `target_gap_sec ≈ 3–4s` (целимся около **3.5s**)

**Минимальные/максимальные значения**:
- `min_frames`: 50 (минимум для покрытия видео)
- `max_frames`: 2000 (максимум для контроля стоимости)

Важно:
- Segmenter — единственный владелец sampling
- **DEFERRED** только синтез глобальной `SamplingPolicy` в Segmenter по всем требованиям компонентов
- Но сами требования выше считаются обязательной частью контракта `place_semantics`

### Models

Компонент использует Embedding Service для поиска мест:

**Embedding Service**:
- **Runtime**: `http` (HTTP API)
- **Engine**: `http`
- **Precision**: `fp32`
- **Device**: `cpu` (Embedding Service runs on server)
- **Model**: CLIP (448x448 для высокого качества распознавания мест)

**Зависимости**:
- Компонент не вызывает модели напрямую, использует Embedding Service API
- Embedding Service использует CLIP для извлечения embeddings
- Модели фиксируются в `meta.models_used[]` для воспроизводимости

### Parallelization

- **Внутренний**: компонент обрабатывает кадры последовательно (HTTP запросы к Embedding Service)
  - Каждый кадр отправляется в Embedding Service отдельным запросом
  - Retry механизм (3 попытки с exponential backoff) для надежности
  - Batch API Embedding Service может быть использован в будущем для оптимизации
- **Внешний**: компонент безопасно параллелить по разным видео/`run_id` (per-run storage)
  - Разные экземпляры компонента могут работать параллельно на разных видео
  - Требования к изоляции: разные `run_id`, разные `result_store` пути

**Ограничения**:
- Thread-safety: компонент не thread-safe (каждый экземпляр работает в отдельном процессе)
- Требования к Embedding Service: сервис должен поддерживать параллельные запросы
- Требования к памяти: peak memory зависит от количества кадров и размера изображений

### Performance characteristics

**Единица обработки**: `frame` (один кадр)

**Типичные значения** (зависят от Embedding Service и сетевых условий):

| Resolution | Latency per frame | CPU RAM peak | Notes |
|------------|-------------------|--------------|-------|
| 1920x1080 | ~200-500 ms | ~100-200 MB | HTTP latency + Embedding Service processing |

**Для видео с N кадрами**: Total latency ≈ N × latency_per_frame

**Факторы производительности**:
- Сетевая задержка до Embedding Service
- Нагрузка на Embedding Service
- Размер изображений (качество JPEG)
- Количество параллельных запросов

**Оптимизации**:
- Batch API Embedding Service (если доступен) может ускорить обработку
- Кэширование результатов для повторных прогонов
- Параллельная обработка кадров (если Embedding Service поддерживает)

**Полные данные**: см. `docs/models_docs/resource_costs/place_semantics_costs_v1.json` (планируется)

### Использование

#### Запуск через CLI:

```bash
python main.py \
    --frames-dir /path/to/frames \
    --rs-path /path/to/result_store \
    --embedding-service-url http://localhost:8001 \
    --topk 5 \
    --similarity-threshold 0.0 \
    --min-track-length 3 \
    --max-gap-sec 5.0
```

#### Параметры:

- `--frames-dir` (required): Директория с кадрами и `metadata.json`
- `--rs-path` (required): Путь к result store (например, `result_store/platform/video/run`)
- `--embedding-service-url` (optional): URL Embedding Service (по умолчанию: `http://localhost:8001` или из `EMBEDDING_SERVICE_URL`)
- `--topk` (optional): Количество топ результатов (по умолчанию: `5`)
- `--similarity-threshold` (optional): Минимальный порог similarity (по умолчанию: `0.0`, диапазон: `0.0-1.0`)
- `--min-track-length` (optional): Минимальное количество кадров в треке (по умолчанию: `3`)
- `--max-gap-sec` (optional): Максимальный разрыв между кадрами для объединения треков (по умолчанию: `5.0` секунд)

#### Примеры использования:

**Базовое использование:**
```bash
python main.py --frames-dir frames/ --rs-path results/
```

**С настройками качества:**
```bash
python main.py \
    --frames-dir frames/ \
    --rs-path results/ \
    --similarity-threshold 0.7 \
    --min-track-length 5
```

### Детали реализации

#### Алгоритм обработки:

1. **Загрузка кадров**:
   - Загружает `frame_indices` из `metadata.json[core_object_detections.frame_indices]`
   - Создает `FrameManager` для доступа к кадрам
   - Проверяет наличие `union_timestamps_sec`

2. **Поиск мест на кадрах**:
   - Для каждого кадра отправляет запрос в Embedding Service
   - Использует retry механизм (3 попытки с exponential backoff)
   - Обрабатывает пустые результаты (warning, не error)

3. **Группировка в tracks**:
   - Группирует кадры с одинаковым top-1 местом в tracks
   - Объединяет треки, если разрыв между кадрами ≤ `max_gap_sec`
   - Фильтрует треки короче `min_track_length`

4. **Агрегация результатов**:
   - Track-level: top-K результатов для каждого трека (дедупликация по place_name)
   - Frame-level: top-K результатов для каждого кадра (дедупликация по place_name)
   - Confidence flags: `track_is_confident_top1` и `frame_is_confident_top1` на основе `similarity_threshold`

#### Обработка ошибок:

- **Retry механизм**: Все запросы к Embedding Service автоматически повторяются при ошибках (3 попытки)
- **Валидация данных**: Проверка размеров массивов, соответствия frame_indices
- **Graceful degradation**: При ошибке отдельного кадра компонент продолжает работу (не падает весь процесс)
- **Логирование**: Все ошибки и предупреждения логируются

#### Оптимизации:

- **Batch API (заготовка)**: Подготовлен метод для batch-запросов (пока использует fallback)
- **Temporal segmentation**: Группировка кадров в tracks для уменьшения шума
- **Cost control**: Параметры `min_track_length` и `max_gap_sec` для контроля качества

### Progress / state events

Компонент публикует прогресс выполнения в `state_events.jsonl` (baseline contract):

**Стадии выполнения**:
- `start` → `load_deps` → `process_frames` → `save` → `done`

**Гранулярный прогресс**:
- Во время стадии `process_frames` компонент публикует прогресс обработки кадров (≥10 обновлений)
- Формат события: `{"progress": 0.0-1.0, "done": int, "total": int, "stage": "process_frames"}`

**Использование**: Backend сайта может читать `state_events.jsonl` для отображения прогресса анализа в реальном времени.

### Features

Компонент выдает следующие фичи:

**Per-frame фичи**:
- `frame_topk_ids (N, K) int32`: Top-K идентификаций мест на кадр
- `frame_topk_scores (N, K) float32`: Similarity scores для кадров
- `frame_is_confident_top1 (N,) bool`: Флаг уверенности для top-1 места

**Per-track фичи**:
- `track_topk_ids (T, K) int32`: Top-K идентификаций мест на трек
- `track_topk_scores (T, K) float32`: Similarity scores для треков
- `track_is_confident_top1 (T,) bool`: Флаг уверенности для top-1 места

**Метаданные**:
- `semantic_label_names (A,) str`: Массив строк "id:name" для маппинга label_id → place_name
- `threshold_per_label_arr (A,) float32`: Пороги для каждого места (NaN если нет)

**Влияние на стоимость**:
- Каждый кадр требует HTTP запрос к Embedding Service (~200-500 ms)
- Для видео с N кадрами: Total cost ≈ N × cost_per_frame
- Batch API может снизить стоимость (если доступен)

### Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Консистентность**: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- **Валидация NPZ**: Артефакт проходит валидацию через `artifact_validator.validate_npz()`
- **Temporal segmentation**: Проверка группировки кадров в tracks (логичность временных сегментов)

Human-friendly demo (HTML):
- `quality_report/demo_place_semantics_quality.py` — генерирует HTML с timeline, thumbnails, consecutive similarity scores, и статистикой по местам

### Troubleshooting

#### Embedding Service недоступен:

```
WARNING: place_semantics | Embedding Service test request failed: ...
Skipping all frames to avoid repeated errors.
```

**Решение**: 
- Убедитесь, что Embedding Service запущен:
```bash
cd DataProcessor/embedding_service
python run_server.py
```
- Проверьте, что категория `place` настроена в Embedding Service
- Проверьте логи Embedding Service для детальной информации об ошибках
- Компонент автоматически пропустит все кадры и заполнит пустые результаты при недоступности сервиса

#### Пустые результаты от Embedding Service:

```
WARNING: Embedding Service returned empty results for frame X
```

**Решение**: 
- Проверьте, что в базе Embedding Service есть места категории `place`
- Попробуйте уменьшить `--similarity-threshold`
- Убедитесь, что качество изображения достаточное

#### Нет tracks:

Если все кадры не имеют мест или tracks слишком короткие, компонент создаст пустые массивы для tracks, но продолжит работу.

**Решение**: 
- Проверьте, что в базе Embedding Service есть места
- Уменьшите `--min-track-length` для более чувствительной группировки
- Увеличьте `--max-gap-sec` для более агрессивного объединения треков
