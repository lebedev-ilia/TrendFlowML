## Component: `place_semantics` (semantic head, Audit v3)

**Version**: `0.2` (Audit v3)  
**Schema**: `place_semantics_npz_v2`  
**Audit v3 Status**: `passed`

### Changelog

#### v0.2 (Audit v3, schema v2) — 2026-02-XX

**Критические изменения**:
- **Schema bump**: `place_semantics_npz_v1` → `place_semantics_npz_v2`
- **Использование core_clip/embeddings.npz**: компонент теперь использует frame embeddings из `core_clip/embeddings.npz` вместо отправки кадров в Embedding Service (соответствие схеме SCHEMA_SEMANTIC_HEADS_NPZ.md)
  - Прямое сравнение embeddings через cosine similarity (10-50x быстрее, чем HTTP запросы на кадр)
  - Получение embeddings мест из Embedding Service один раз через `get_all_embeddings()`
  - Fallback на image-based search, если embeddings недоступны
- **Детерминированный label-space**: добавлены `semantic_label_names` и `semantic_object_ids` для стабильного отображения UUID → int32
  - Компонент получает полный список labels из Embedding Service заранее (через `get_labels()`)
  - Используется канонический label space для обеспечения стабильности `label_id` между запусками
  - Результаты поиска маппятся к каноническому label space через UUID
- **DB provenance**: добавлены `db_name`, `db_version`, `db_digest` в meta для reproducibility
  - `db_digest`: SHA256 хеш от канонического списка labels (стабильный в пределах версии базы)
  - `db_name`: "embedding_service"
  - `db_version`: "v1"
- **meta_json**: добавлен cross-venv safe JSON representation meta
- **Embedding Service client**: добавлены методы `get_labels()` и `get_all_embeddings()` для получения канонического списка labels и embeddings
- **track_topk_evidence_frame_indices**: добавлено поле для указания union frame indices, где similarity максимальна для каждого top-K места в каждом треке (для debug и консистентности с другими semantic heads)
- **threshold_global**: добавлен в meta для консистентности с другими semantic heads

**Улучшения**:
- **Производительность**: прямое сравнение embeddings (10-50x быстрее, чем HTTP запросы на кадр)
- **Стабильность label space**: использование канонического label space из Embedding Service вместо динамического построения из результатов поиска
- **Соответствие схеме**: использование `core_clip/embeddings.npz` согласно SCHEMA_SEMANTIC_HEADS_NPZ.md
- **Render system**: полностью переписан HTML рендер на offline vanilla JS + SVG
  - Offline render (без CDN), графики — SVG
  - Добавлен SVG timeline график (offline)
  - Mini-dashboard с секциями: Overview, QA (top/anti-top), Timeline, Tables, Meta
  - Интерактивность на vanilla JS: поиск, фильтры, сортировка таблиц
  - Навигация с якорями между секциями
- **Документация**: 
  - Добавлен `SCHEMA.md` (human schema) с полным описанием всех полей, tiers, required/optional
  - Создана JSON схема `place_semantics_npz_v2.json` в `VisualProcessor/schemas/`
  - Обновлен README с разделом "Render (dev-only)" согласно требованиям аудита
- **Расширенный render context**: добавлены `key_facts`, `config_highlights`, `qa_hints`, `top_examples`, `anti_top_examples`

**Breaking changes**:
- **Upstream dependency**: Теперь требует `core_clip/embeddings.npz` (fail-fast, no-fallback) — соответствие схеме SCHEMA_SEMANTIC_HEADS_NPZ.md
- `frame_topk_ids` и `track_topk_ids` теперь содержат int32 индексы из `semantic_label_names` вместо динамических индексов (детерминированное отображение)
- Структура NPZ изменена: добавлены обязательные поля `semantic_object_ids`, `meta_json`, `track_topk_evidence_frame_indices`
- Обязательные meta поля: добавлены `db_name`, `db_version`, `db_digest`, `embedding_service_url`, `place_category`, `topk`, `similarity_threshold`, `threshold_global`, `min_track_length`, `max_gap_sec`, `num_tracks`, `num_places`, `num_frames`
- **NaN-policy**: `track_topk_ids` и `track_topk_scores` теперь инициализируются как `-1` и `NaN` вместо `0` (соответствие NaN-policy из SCHEMA_SEMANTIC_HEADS_NPZ.md)
- **is_confident flags**: Теперь используют `threshold_global` вместо `similarity_threshold` (консистентность с другими semantic heads)

**Технические детали**:
- Канонический label space строится из `labels_canon` (полученного через `get_labels()`), отсортированного по UUID
- `uuid_to_int` маппинг обеспечивает стабильность `label_id` между запусками (при одинаковом `db_digest`)
- Результаты поиска маппятся к каноническому label space через UUID из результатов Embedding Service
- Frame embeddings из `core_clip/embeddings.npz` нормализуются (L2 norm) перед сравнением
- Place embeddings из Embedding Service также нормализуются (L2 norm) перед сравнением
- Cosine similarity вычисляется как `dot(frame_emb_normalized, place_emb_normalized.T)`
- Upstream models из `core_clip` включаются в `models_used` для полной provenance chain
- `core_clip_model_signature` сохраняется в meta для отслеживания версий upstream моделей

#### v0.1 (initial) — до Audit v3

- Базовая реализация place recognition
- Schema v1 без детерминированного label-space
- Динамическое построение label space из результатов поиска
- HTML render — полностью offline (без CDN)

### Назначение

`place_semantics` распознает места и лэндмарки в видео:
- использует frame embeddings из `core_clip/embeddings.npz` (required by schema)
- получает embeddings мест из Embedding Service один раз
- делает прямое сравнение через cosine similarity (10-50x быстрее, чем HTTP запросы на кадр)
- группирует кадры по местам в tracks (временная сегментация)
- возвращает per-track и per-frame top‑K идентификаций мест

Компонент следует контракту semantic head:
- Требует `core_clip/embeddings.npz` (frame embeddings, must cover all required frame_indices) — **required by schema SCHEMA_SEMANTIC_HEADS_NPZ.md**
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

1. **Категория**: `place` (может быть расширена)
2. **Модель**: CLIP (определяется Embedding Service, обычно `clip_224` или `clip_336`)
3. **Канонический label space** (v2, Audit v3):
   - Компонент получает полный список labels из Embedding Service заранее через `GET /categories/place/labels`
   - Строится канонический label space (отсортированный по UUID) для обеспечения стабильности `label_id` между запусками
   - Вычисляется `db_digest` (SHA256 хеш от канонического списка labels) для reproducibility
   - Результаты поиска маппятся к каноническому label space через UUID
4. **Использование** (v0.2, Audit v3):
   - Получение канонического label space через `GET /categories/place/labels` (для `db_digest` и стабильности)
   - Получение embeddings мест через `GET /categories/place/embeddings` (для прямого сравнения, оптимизация)
   - Fallback: поиск похожих мест через Embedding Service (`POST /search`) если embeddings недоступны
   - Добавление новых мест через API (`POST /objects/add`)
   - Обновление информации о местах (`PATCH /objects/{id}`)

**Преимущества использования Embedding Service**:
- Единая база данных для всех мест
- Быстрый поиск через FAISS индексы
- Удобное управление базой (добавление/удаление/обновление)
- Хранение метаданных (название, страна, город, координаты и т.д.)
- Горячее обновление (новые места доступны сразу)

**Оптимизация производительности** (v0.2, Audit v3):
- Прямое сравнение embeddings (10-50x быстрее, чем HTTP запросы на кадр)
- Получение embeddings мест один раз вместо запросов на каждый кадр
- Использование готовых embeddings из `core_clip/embeddings.npz` (переиспользование вычислений)

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

**Алгоритм работы** (v0.2, Audit v3):
1. Компонент получает `frame_indices` из `core_object_detections.frame_indices` (shared sampling group)
2. Загружает frame embeddings из `core_clip/embeddings.npz` (required by schema)
3. Получает embeddings мест из Embedding Service один раз через `get_all_embeddings()` (оптимизация)
4. Делает прямое сравнение embeddings через cosine similarity (10-50x быстрее, чем HTTP запросы на кадр)
5. Fallback: если embeddings недоступны, использует image-based search через Embedding Service
6. Группирует кадры по местам в tracks (временная сегментация)
7. Возвращает top‑K результатов с similarity scores

### Входы (required)

- `frames_dir/metadata.json`:
  - `core_object_detections.frame_indices` (shared sampling group)
  - `union_timestamps_sec`
- `rs_path/core_clip/embeddings.npz`:
  - `frame_indices` (must cover all required frame_indices from core_object_detections)
  - `frame_embeddings` (normalized CLIP embeddings, shape `(N, D)`)
  - `meta` (для provenance chaining: `models_used`, `model_signature`)
- Embedding Service доступен (для получения embeddings мест и label space)

**Early validation** (v0.2, Audit v3):
- Проверка доступности `core_clip/embeddings.npz` (fail-fast, no-fallback)
- Проверка coverage: все required `frame_indices` должны быть в `core_clip.frame_indices` (fail-fast)
- Проверка health endpoint Embedding Service через `embedding_client._ensure_url()` (fail-fast)
- Получение label space через `get_labels()` (fail-fast, если база пустая)
- Попытка получить embeddings мест через `get_all_embeddings()` (fallback на image search, если недоступны)

**No-fallback policy**:
- Отсутствие `core_clip/embeddings.npz` → **RuntimeError** (fail-fast)
- Отсутствие required `frame_indices` в `core_clip` → **RuntimeError** (fail-fast)
- Embedding Service недоступен при инициализации → **RuntimeError** (fail-fast)
- База мест пустая (0 labels) → **RuntimeError** (fail-fast)
- Если embeddings мест недоступны → fallback на image-based search (медленнее, но работает)

### Output (NPZ)

Путь: `rs_path/place_semantics/place_semantics.npz`

**Artifact filename**: `place_semantics.npz` (фиксированное имя, `ARTIFACT_FILENAME`)

**Schema version**: `place_semantics_npz_v2`

**Human schema**: `SCHEMA.md` (рядом с компонентом)  
**Machine schema**: `VisualProcessor/schemas/place_semantics_npz_v2.json`

Ключи (v2):
- `frame_indices (N,) int32` — shared sampling group
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `track_ids (T,) int32` — ID треков (отдельные tracks для разных мест)
- `track_topk_ids (T, K) int32` — Top‑K мест на трек
- `track_topk_scores (T, K) float32` — Similarity scores для треков
- `track_present_mask (T,) bool` — Маска присутствия треков
- `track_is_confident_top1 (T,) bool` — Флаг уверенности для top-1 места на трек
- `track_topk_evidence_frame_indices (T, K) int32` — Union frame indices, где similarity максимальна для каждого top-K места в каждом треке
- `frame_topk_ids (N, K) int32` — Top‑K мест на кадр
- `frame_topk_scores (N, K) float32` — Similarity scores для кадров
- `frame_is_confident_top1 (N,) bool` — Флаг уверенности для top-1 места на кадр
- `semantic_label_names (A,) str` (`"id:name"`) — канонический label space из Embedding Service (стабильный в пределах `db_digest`)
- `semantic_object_ids (A,) str` — UUID из Embedding Service, aligned с `semantic_label_names` (для reproducibility)
- `threshold_per_label_arr (A,) float32` — Пороги для каждого места (NaN если нет)
- `meta` (object dict): статус + информация о базе мест + models_used + DB provenance + `threshold_global` + `core_clip_model_signature` (provenance chaining)
- `meta_json` (str): meta как JSON string (cross-venv safe)

**Важно (v2)**: `semantic_label_names` и `semantic_object_ids` строятся из канонического списка labels, полученного из Embedding Service заранее. Это обеспечивает стабильность `label_id` между запусками при одинаковом `db_digest`.

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

- **Внутренний** (v0.2, Audit v3): компонент обрабатывает кадры векторизованно (прямое сравнение embeddings)
  - Все кадры обрабатываются одновременно через матричное умножение
  - Fallback: если embeddings недоступны, кадры обрабатываются последовательно через HTTP запросы
  - Retry механизм (3 попытки с exponential backoff) для надежности (только в fallback режиме)
- **Внешний**: компонент безопасно параллелить по разным видео/`run_id` (per-run storage)
  - Разные экземпляры компонента могут работать параллельно на разных видео
  - Требования к изоляции: разные `run_id`, разные `result_store` пути

**Ограничения**:
- Thread-safety: компонент не thread-safe (каждый экземпляр работает в отдельном процессе)
- Требования к памяти: peak memory зависит от количества кадров, размера embeddings и количества мест в базе
- Размер матрицы сравнения: `(N, D) @ (D, M)` где N=кадры, D=размерность embeddings, M=количество мест

### Performance characteristics

**Единица обработки**: `frame` (один кадр)

**Типичные значения** (v0.2, Audit v3 — прямое сравнение embeddings):

| Resolution | Latency per frame | CPU RAM peak | Notes |
|------------|-------------------|--------------|-------|
| 1920x1080 | ~1-5 ms | ~100-200 MB | Прямое сравнение embeddings (10-50x быстрее, чем HTTP запросы) |

**Для видео с N кадрами**: Total latency ≈ initialization + N × latency_per_frame
- **Initialization**: ~100-500 ms (загрузка `core_clip/embeddings.npz`, получение embeddings мест из Embedding Service)
- **Per-frame**: ~1-5 ms (прямое сравнение через cosine similarity)

**Факторы производительности**:
- Количество мест в базе (влияет на размер матрицы сравнения)
- Размерность embeddings (обычно 512 для CLIP)
- Количество кадров (линейная зависимость)
- Fallback на image search: ~200-500 ms на кадр (если embeddings недоступны)

**Оптимизации** (v0.2, Audit v3):
- Прямое сравнение embeddings (10-50x быстрее, чем HTTP запросы на кадр)
- Получение embeddings мест один раз вместо запросов на каждый кадр
- Использование готовых embeddings из `core_clip/embeddings.npz` (переиспользование вычислений)
- Векторизованные операции NumPy для batch сравнения

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

#### Алгоритм обработки (v0.2, Audit v3):

1. **Загрузка зависимостей**:
   - Загружает `frame_indices` из `metadata.json[core_object_detections.frame_indices]`
   - Загружает frame embeddings из `core_clip/embeddings.npz` (fail-fast, no-fallback)
   - Проверяет coverage: все required `frame_indices` должны быть в `core_clip.frame_indices`
   - Нормализует frame embeddings (L2 norm)

2. **Получение embeddings мест**:
   - Получает канонический label space через `get_labels()` (для `db_digest` и стабильности)
   - Получает embeddings мест через `get_all_embeddings()` (оптимизация)
   - Нормализует place embeddings (L2 norm)
   - Fallback: если embeddings недоступны, использует image-based search

3. **Прямое сравнение embeddings** (основной путь):
   - Вычисляет cosine similarity: `dot(frame_emb_normalized, place_emb_normalized.T)` → `(N, M)`
   - Применяет `similarity_threshold` (фильтрация низких similarity)
   - Получает top-K для каждого кадра через `argsort`
   - Маппит результаты к каноническому label space через UUID

4. **Fallback: image-based search** (если embeddings недоступны):
   - Создает `FrameManager` для доступа к кадрам
   - Для каждого кадра отправляет запрос в Embedding Service
   - Использует retry механизм (3 попытки с exponential backoff)
   - Обрабатывает пустые результаты (warning, не error)

5. **Группировка в tracks**:
   - Группирует кадры с одинаковым top-1 местом в tracks
   - Объединяет треки, если разрыв между кадрами ≤ `max_gap_sec`
   - Фильтрует треки короче `min_track_length`

6. **Агрегация результатов**:
   - Track-level: top-K результатов для каждого трека (max similarity over time per place)
   - Frame-level: top-K результатов для каждого кадра (дедупликация по place_name)
   - Confidence flags: `track_is_confident_top1` и `frame_is_confident_top1` на основе `threshold_global`
   - Evidence frames: `track_topk_evidence_frame_indices` указывает на кадры с максимальной similarity

#### Обработка ошибок:

- **Fail-fast**: Отсутствие `core_clip/embeddings.npz` или required `frame_indices` → RuntimeError
- **Fail-fast**: Embedding Service недоступен при инициализации → RuntimeError
- **Fail-fast**: База мест пустая (0 labels) → RuntimeError
- **Retry механизм**: Все запросы к Embedding Service автоматически повторяются при ошибках (3 попытки, только в fallback режиме)
- **Валидация данных**: Проверка размеров массивов, соответствия frame_indices
- **Graceful degradation**: При ошибке отдельного кадра компонент продолжает работу (не падает весь процесс)
- **Логирование**: Все ошибки и предупреждения логируются

#### Оптимизации (v0.2, Audit v3):

- **Прямое сравнение embeddings**: 10-50x быстрее, чем HTTP запросы на кадр
- **Векторизованные операции**: Batch сравнение через матричное умножение NumPy
- **Переиспользование вычислений**: Использование готовых embeddings из `core_clip/embeddings.npz`
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

**Влияние на стоимость** (v0.2, Audit v3):
- Прямое сравнение embeddings: ~1-5 ms на кадр (10-50x быстрее, чем HTTP запросы)
- Initialization: ~100-500 ms (загрузка embeddings, получение embeddings мест)
- Для видео с N кадрами: Total cost ≈ initialization + N × 1-5 ms
- Fallback на image search: ~200-500 ms на кадр (если embeddings недоступны)

### Empty vs Error semantics (Audit v3)

**Valid empty outputs**:
- `status="empty"` устанавливается, если:
  - `empty_reason="no_places_detected"`: Embedding Service доступен, база не пустая, но места не найдены на кадрах
  - `empty_reason="embedding_service_unavailable_during_processing"`: сервис стал недоступен во время обработки (редкий случай, обычно fail-fast срабатывает раньше)
- Выходные массивы заполняются значениями по умолчанию:
  - `frame_indices`, `times_s`: присутствуют (из upstream)
  - `track_topk_ids`, `frame_topk_ids`: `-1` (NaN-policy)
  - `track_topk_scores`, `frame_topk_scores`: `NaN` (NaN-policy)
  - `semantic_label_names`, `semantic_object_ids`: присутствуют (из канонического label space)
  - `db_digest`: присутствует (для reproducibility)

**Error cases** (fail-fast, RuntimeError):
- Отсутствие `core_clip/embeddings.npz` → **RuntimeError**
- Отсутствие required `frame_indices` в `core_clip` → **RuntimeError**
- Embedding Service недоступен при инициализации → **RuntimeError**
- База мест пустая (0 labels) → **RuntimeError**
- Отсутствие `core_object_detections.frame_indices` в metadata → **RuntimeError**
- Отсутствие `union_timestamps_sec` в metadata → **RuntimeError**

**No-fallback policy**: Компонент не создает "ok empty" из-за отсутствия required зависимостей. Все hard dependencies проверяются fail-fast.

### Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Консистентность**: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- **Валидация NPZ**: Артефакт проходит валидацию через `artifact_validator.validate_npz()`
- **Temporal segmentation**: Проверка группировки кадров в tracks (логичность временных сегментов)

Human-friendly demo (HTML):
- `quality_report/demo_place_semantics_quality.py` — генерирует HTML с timeline, thumbnails, consecutive similarity scores, и статистикой по местам

### Schema и версионирование

**Schema version**: `place_semantics_npz_v2`

**Human schema**: `SCHEMA.md` (рядом с компонентом)  
**Machine schema**: `VisualProcessor/schemas/place_semantics_npz_v2.json`

**Версионирование**:
- `producer_version`: версия компонента (текущая: `0.2`)
- `schema_version`: версия схемы NPZ (текущая: `place_semantics_npz_v2`)
- `db_digest`: SHA256 хеш канонического списка labels (для reproducibility)

**Правила обратной совместимости**:
- Любое существенное изменение ключей/dtype/shape → bump `schema_version`
- Изменения в структуре meta (добавление required полей) → bump `schema_version`
- `allow_extra_keys=false` в схеме (fail-fast при неизвестных ключах)

### Render (dev-only)

Компонент генерирует **offline mini-dashboard** для визуализации результатов распознавания мест.

**Файлы рендера**:
- `_render/render_context.json` — JSON контекст с данными для рендера
- `_render/render.html` — HTML mini-dashboard (offline, без CDN)

**Структура render.html**:

1. **Overview**:
   - Описание компонента и его назначения
   - **Key facts**: `schema_version`, `producer_version`, `place_category`, количество frames/tracks/places, `db_digest`, время выполнения
   - **Config highlights**: `topk`, `similarity_threshold`, `min_track_length`, `max_gap_sec`
   - **How to QA**: типовые распределения и аномалии для проверки качества

2. **QA (top/anti-top)**:
   - **Top examples**: кадры с наивысшими similarity scores (лучшие распознавания)
   - **Anti-top examples**: кадры с наименьшими scores, но с распознанным местом (подозрительные случаи)
   - Таблицы с возможностью сортировки по любому столбцу

3. **Timeline**:
   - SVG график top-1 place score по времени (offline, без CDN)
   - Визуализация изменений similarity scores на протяжении видео

4. **Tables**:
   - **Top places by frequency**: наиболее часто встречающиеся места с количеством кадров и средним score
   - **All frames**: интерактивная таблица всех кадров (первые 3000) с поиском и фильтрами:
     - Поиск по frame_index/time_sec/place_name
     - Фильтр по минимальному top1_score
     - Фильтр "Confident only"
   - **Distribution statistics**: статистика распределения top1_scores и topk_scores (min/max/mean/std/median/p25/p75)

5. **Meta**:
   - Полный JSON метаданных из NPZ

**Интерактивность**:
- Сортировка таблиц по клику на заголовок (vanilla JS, offline)
- Поиск и фильтры в таблице "All frames" (vanilla JS, offline)
- Навигация по секциям через якоря в меню

**Как читать выход**:
- **Top1_score range**: нормальные значения 0.3-0.9 для корректных распознаваний
- **Confident ratio**: должно быть > 0.5 для видео с четкими местами
- **Tracks count**: должно быть > 0, если места обнаружены; проверьте `min_track_length`, если треков слишком мало
- **Anomaly (low scores)**: top1_score < 0.2 может указывать на отсутствие места или плохое качество
- **Anomaly (high variance)**: std > 0.3 может указывать на нестабильное распознавание

**Связь с NPZ**:
- NPZ остаётся source-of-truth
- Render — это "человеческая интерпретация" NPZ для dev/debug
- Все данные в render берутся из NPZ артефакта

**Время выполнения**:
- Смотрите `meta.stage_timings_ms` в секции Meta или Key facts
- Основные стадии: `initialization`, `load_deps`, `process_frames`, `saving`, `total`

**Параметры конфига, влияющие на результат и стоимость**:
- `topk`: количество топ результатов (по умолчанию 5)
- `similarity_threshold`: минимальный порог similarity (влияет на `is_confident` флаги)
- `min_track_length`: минимальная длина трека (влияет на количество треков)
- `max_gap_sec`: максимальный разрыв между кадрами для объединения треков (влияет на группировку)

**Конфигурация**:

В `global_config.yaml`:
```yaml
place_semantics:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML mini-dashboard
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

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
