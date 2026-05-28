## Component: `franchise_recognition` (semantic head, Audit v3)

**Version**: `0.2` (Audit v3)  
**Schema**: `franchise_recognition_npz_v2`  
**Audit v3 Status**: `passed`  

**Док. фич / melt / QA:** `docs/FEATURE_DESCRIPTION.md` · `utils/validate_franchise_recognition.py`

### Changelog

#### v0.2 (Audit v3, schema v2) — 2026-02-XX

**Критические изменения**:
- **Schema bump**: `franchise_recognition_npz_v1` → `franchise_recognition_npz_v2`
- **Детерминированный label-space**: добавлены `semantic_label_names` и `semantic_object_ids` для стабильного отображения UUID → int32
  - Компонент получает полный список labels из Embedding Service заранее (через `get_labels()`)
  - Используется канонический label space для обеспечения стабильности `label_id` между запусками
  - Результаты поиска маппятся к каноническому label space через UUID
- **DB provenance**: добавлены `db_name`, `db_version`, `db_digest` в meta для reproducibility
  - `db_digest`: SHA256 хеш от канонического списка labels (стабильный в пределах версии базы)
  - `db_name`: "embedding_service"
  - `db_version`: "v1"
- **meta_json**: добавлен cross-venv safe JSON representation meta
- **Embedding Service client**: добавлен метод `get_labels()` для получения канонического списка labels

**Улучшения**:
- **Стабильность label space**: использование канонического label space из Embedding Service вместо динамического построения из результатов поиска
- **Оптимизация с каноническим label space**: при использовании embeddings напрямую маппинг к каноническому label space происходит до вычисления similarity
- **Render system**: полностью переписан HTML рендер на offline vanilla JS + SVG
  - Offline render (без CDN), графики — SVG
  - Добавлен SVG timeline график (offline)
  - Mini-dashboard с секциями: Overview, QA (top/anti-top), Timeline, Tables, Meta
  - Интерактивность на vanilla JS: поиск, фильтры, сортировка таблиц
  - Навигация с якорями между секциями
- **Документация**: 
  - Добавлен `SCHEMA.md` (human schema) с полным описанием всех полей, tiers, required/optional
  - Создана JSON схема `franchise_recognition_npz_v2.json` в `VisualProcessor/schemas/`
  - Обновлен README с разделом "Render (dev-only)" согласно требованиям аудита
- **Расширенный render context**: добавлены `key_facts`, `config_highlights`, `qa_hints`, `top_examples`, `anti_top_examples`

**Breaking changes**:
- `frame_topk_ids` и `track_topk_ids` теперь содержат int32 индексы из `semantic_label_names` вместо динамических индексов (детерминированное отображение)
- Структура NPZ изменена: добавлены обязательные поля `semantic_object_ids`, `meta_json`
- Обязательные meta поля: добавлены `db_name`, `db_version`, `db_digest`, `embedding_service_url`, `franchise_category`, `topk`, `similarity_threshold`, `threshold_global`, `num_franchises`, `num_frames`

**Технические детали**:
- Канонический label space строится из `labels_canon` (полученного через `get_labels()`), отсортированного по UUID
- `uuid_to_int` маппинг обеспечивает стабильность `label_id` между запусками (при одинаковом `db_digest`)
- При использовании embeddings напрямую: маппинг к каноническому label space происходит до построения embeddings matrix
- При fallback (image-based search): результаты поиска маппятся к каноническому label space через UUID

#### v0.1 (initial) — до Audit v3

- Базовая реализация franchise recognition
- Schema v1 без детерминированного label-space
- Динамическое построение label space из результатов поиска
- HTML render — полностью offline (без CDN)

### Назначение

`franchise_recognition` распознает конкретные франшизы/тайтлы в видео:
- игры (video games)
- аниме (anime series)
- мультфильмы (cartoons)

Подход v1: **Embedding Service + CLIP frame embeddings**, при этом OCR **не является hard‑dependency**:
если OCR артефакт отсутствует — компонент продолжает работу через полный поиск по базе франшиз.

### Входы (required)

- `frames_dir/metadata.json`:
  - `core_clip.frame_indices` (sampling group)
  - `union_timestamps_sec`
- `rs_path/core_clip/embeddings.npz` — frame embeddings из core_clip
- **Embedding Service** (обязателен, fail-fast при недоступности):
  - Категория: `franchise`
  - Должен быть доступен по URL (env `EMBEDDING_SERVICE_URL` или `--embedding-service-url`)

### Входы (optional)

- `--ocr-npz <path>`: OCR артефакт (union-domain frames). Минимальная поддерживаемая схема:
  - `ocr_raw` или `ocr_data` → object array с `list[dict]`
  - dict поля: `frame` (int), `bbox` (list[4]), `text` или `text_raw` (str), `confidence` (float, optional)
- Используется для фильтрации кандидатов (если `--use-ocr-filtering` включен и база франшиз большая)

### Выходы

Путь: `result_store/<platform_id>/<video_id>/<run_id>/franchise_recognition/franchise_recognition.npz`

Ключи (v2, совместимо с `SCHEMA_SEMANTIC_HEADS_NPZ.md`):
- `frame_indices (N,) int32` (строго = `core_clip.frame_indices`)
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `semantic_label_names (A,) str` (`"id:name"`) — канонический label space из Embedding Service (стабильный в пределах `db_digest`)
- `semantic_object_ids (A,) str` — UUID из Embedding Service, aligned с `semantic_label_names` (для reproducibility)
- `threshold_per_label_arr (A,) float32` (NaN если нет)
- `track_ids (1,) int32` (=0, video-level aggregate)
- `track_present_mask (1,) bool`
- `track_topk_ids (1,5) int32`, `track_topk_scores (1,5) float32` (max over time)
- `track_is_confident_top1 (1,) bool`
- `track_topk_evidence_frame_indices (1,5) int32` — union frame index, где similarity максимальна
- `frame_topk_ids (N,5) int32`, `frame_topk_scores (N,5) float32`
- `frame_is_confident_top1 (N,) bool`
- `meta` (object dict): стандарт + DB provenance (`db_name`, `db_version`, `db_digest`), Embedding Service URL, OCR stats, `models_used`/`model_signature` (+ chaining от `core_clip`)
- `meta_json` (str): meta как JSON string (cross-venv safe)

**Важно (v2)**: `semantic_label_names` и `semantic_object_ids` строятся из канонического списка labels, полученного из Embedding Service заранее. Это обеспечивает стабильность `label_id` между запусками при одинаковом `db_digest`.

### Early validation (Embedding Service)

Компонент выполняет **раннюю проверку доступности Embedding Service**:
- Проверка health endpoint через `embedding_client._ensure_url()`
- Тестовый запрос с первым кадром для проверки работоспособности search endpoint
- Если тест не проходит (например, 500 ошибка):
  - Выдается одно предупреждение вместо множества ошибок
  - Пропускается обработка всех кадров
  - Заполняются пустые результаты вместо обработки с ошибками
- Если тест проходит — продолжается обычная обработка

**Преимущества**:
- Fail-fast: проблемы обнаруживаются до начала обработки всех кадров
- Улучшенный UX: одно предупреждение вместо сотен ошибок
- Экономия ресурсов: не тратится время на обработку, которая все равно завершится ошибкой

### No-fallback / empty semantics

- Нет `core_clip.frame_indices` / `union_timestamps_sec` / `core_clip/embeddings.npz` coverage → **error**.
- Embedding Service недоступен при инициализации → **error** (fail-fast).
- Embedding Service недоступен во время обработки → компонент пропускает все кадры с предупреждением.
- OCR missing → **не error** (компонент работает через полный поиск).

### Features (выход) — группы и оценки

- **`frame_topk_ids/scores`**:
  - **алгоритм**: поиск по frame embeddings через Embedding Service (CLIP-based)
  - **оценка реализации**: 8/10
  - **полезность**: 9/10 (базовый сигнал для распознавания франшиз, используется downstream компонентами)
- **`track_topk_ids/scores`** (video-level aggregate):
  - **алгоритм**: max over time per franchise
  - **оценка реализации**: 8/10
  - **полезность**: 8/10 (агрегированный сигнал для всего видео)
- **`is_confident_top1`**:
  - **алгоритм**: threshold-based confidence flag (не гейтит top-K)
  - **оценка реализации**: 8/10
  - **полезность**: 7/10 (служебно для фильтрации низкокачественных результатов)

### Sampling / units-of-processing requirements

**Требования к выборке кадров**:

Компонент использует `core_clip.frame_indices` (shared sampling group). Требования к выборке:

- **Coverage**: обязательно покрывать начало/середину/конец видео и быть равномерной по времени
- **Непрерывная кривая**: количество кадров должно зависеть от длительности видео через непрерывную монотонную функцию (без скачков)
- **Минимальное значение**: минимум 10 кадров (для коротких видео)
- **Максимальное значение**: максимум 500 кадров (cap для длинных видео)
- **Целевое значение**: зависит от длительности через кривую `target_gap_sec = f(duration_s)`

**Рекомендуемая политика выборки** (Segmenter-owned):

- `target_gap_sec = f(duration_s)` — непрерывная монотонная кривая, построенная через log‑log интерполяцию по anchor‑точкам
- `budget_n = round(duration_s / target_gap_sec)` (и затем `N = min(requested_max, budget_n)`)

Ориентиры по кривой (приблизительно):
- **≈ 5 минут**: `target_gap_sec ≈ 1s`
- **≈ 10 минут**: `target_gap_sec ≈ 2s`
- **≈ 20 минут**: `target_gap_sec ≈ 3–4s` (целимся около **3.5s**)

**Требования к разрешению**:
- Компонент работает с frame embeddings из `core_clip`, которые уже нормализованы
- Разрешение кадров определяется `core_clip` (обычно 224x224 или 336x336 для CLIP)

**No-fallback policy**:
- Если `core_clip.frame_indices` отсутствует или пустой → **RuntimeError** (no-fallback)
- Если `union_timestamps_sec` отсутствует → **RuntimeError** (no-fallback)
- Если `core_clip/embeddings.npz` не покрывает требуемые `frame_indices` → **RuntimeError** (no-fallback)

**Важно**: Segmenter является единственным владельцем sampling (компонент не генерирует семплинг сам).

### Models

#### GPU Models

1. **CLIP Image Encoder** (через core_clip)
   - **Triton**: ✅ Да (`triton/models/clip_image_224/` или `clip_image_336/`)
   - **Spec name**: `clip_image_224_triton` / `clip_image_336_triton` (ModelManager)
   - **Runtime**: `triton`
   - **Engine**: `onnx` или `tensorrt`
   - **Precision**: `fp16` или `fp32`
   - **Device**: `cuda:0`
   - **Использование**: компонент использует готовые embeddings из `core_clip`, не загружает модель напрямую

#### External Services

1. **Embedding Service** (HTTP API)
   - **Triton**: ❌ Нет (внешний HTTP сервис)
   - **Runtime**: `http`
   - **Engine**: `http`
   - **Precision**: `fp32` (embeddings)
   - **Device**: `cpu` (сервер Embedding Service)
   - **Категория**: `franchise`
   - **Использование**: поиск похожих франшиз по frame embeddings
   - **Требования**: сервис должен быть доступен (fail-fast при недоступности)

### Parallelization

- **Внутренний**: компонент использует оптимизированную обработку с несколькими стратегиями:
  - **Оптимизация 1**: Использование embeddings напрямую (10-50x ускорение)
    - Получение всех franchise embeddings одним запросом через `/categories/franchise/embeddings`
    - Локальное сравнение через cosine similarity (без HTTP запросов для каждого кадра)
    - Автоматически активируется, если в базе есть franchise объекты
  - **Оптимизация 2**: Batch search для image-based fallback (3-10x ускорение)
    - Предзагрузка кадров и группировка в батчи
    - Batch API Embedding Service (если доступен) или параллельная обработка
  - **Оптимизация 3**: Параллельная обработка батчей (2-4x ускорение)
    - Использование ThreadPoolExecutor для параллельной обработки нескольких батчей
    - Ограничение параллелизма (max_workers=4) для избежания перегрузки Embedding Service
  - **Оптимизация 4**: Кеширование результатов поиска
    - Кеш по хешу кадра для избежания повторных запросов
  - Batch size контролируется через `--batch-size` (scheduler-controlled)
- **Внешний**: компонент безопасно параллелить по разным видео/`run_id` (per-run storage)
  - Разные `run_id` → разные `result_store` пути → нет конфликтов
  - Embedding Service должен поддерживать параллельные запросы (обычно поддерживает)

**Ограничения**:
- Компонент зависит от Embedding Service (network latency при image-based fallback)
- При использовании оптимизации с embeddings напрямую зависимость от сети минимальна (1 запрос)
- Рекомендуется использовать OCR filtering для больших баз франшиз (>500)

### Performance characteristics

**Единица обработки**: `frame` (один кадр)

**Типичные значения** (после оптимизаций):

| Режим | Latency для 115 кадров | CPU RAM peak | GPU VRAM | Notes |
|-------|------------------------|--------------|----------|-------|
| **Оптимизированный** (embeddings напрямую) | ~0.1-0.6 секунды | ~300-500 MB | Минимальное | Использует локальное сравнение embeddings |
| **Fallback** (image-based search) | ~6-7 секунд | ~200-300 MB | Минимальное | HTTP запросы для каждого кадра |
| **Без оптимизаций** (старая версия) | ~11-23 секунды | ~200 MB | Минимальное | Последовательные HTTP запросы |

**Ожидаемое ускорение**:
- **Оптимизация с embeddings напрямую**: **10-50x** (зависит от размера базы франшиз)
- **Batch search + параллелизм**: **3-10x** (для image-based fallback)
- **Суммарное ускорение**: **50-100x** по сравнению с базовой версией

**Факторы производительности**:
- **Размер базы франшиз**: влияет на время локального сравнения (обычно <100ms для <1000 франшиз)
- **Количество кадров (N)**: линейная зависимость при использовании embeddings напрямую
- **Network latency**: критично только при image-based fallback
- **Использование OCR filtering**: может ускорить на ~20-30% при базах >500 франшиз

**Оптимизации производительности**:
1. **Использование embeddings напрямую**: получение всех franchise embeddings одним запросом, локальное сравнение через cosine similarity
2. **Batch search**: группировка кадров в батчи для уменьшения количества HTTP запросов
3. **Параллельная обработка**: использование ThreadPoolExecutor для параллельной обработки батчей
4. **Кеширование**: кеш результатов поиска по хешу кадра
5. **Векторизованная OCR фильтрация**: оптимизация обработки OCR данных

**Полные данные**: будут добавлены после проведения измерений в `docs/models_docs/resource_costs/franchise_recognition_costs_v1.json`

### Параметры конфигурации компонента

Все параметры принимаются через аргументы командной строки:

| Параметр | Тип | По умолчанию | Описание | Влияние на скорость/стоимость |
|----------|-----|--------------|----------|------------------------------|
| `--embedding-service-url` | str | env или `http://localhost:8005` | URL Embedding Service | Нет прямого влияния |
| `--topk` | int | 5 | Количество top результатов (contract: must be 5) | Нет влияния (фиксировано) |
| `--similarity-threshold` | float | 0.0 | Минимальный порог similarity | Нет влияния (не гейтит top-K) |
| `--threshold-global` | float | 0.23 | Глобальный порог для `is_confident` | Нет влияния (только флаги) |
| `--ocr-npz` | str | None | Путь к OCR NPZ (опционально) | Нет влияния (опционально) |
| `--ocr-min-confidence` | float | 0.4 | Минимальный confidence OCR | Нет влияния |
| `--ocr-max-events` | int | 5000 | Максимум OCR событий | Нет влияния (cost control) |
| `--use-ocr-filtering` | flag | False | Использовать OCR для фильтрации | Может ускорить при больших базах |
| `--max-franchises-for-full-search` | int | 500 | Порог для полного поиска vs OCR filtering | Влияет на скорость при больших базах |
| `--batch-size` | int | 16 | Batch size для Embedding Service (scheduler-controlled) | Влияет на throughput (если batch API доступен) |

**Влияние параметров на скорость**:
- `--use-ocr-filtering`: может ускорить на **~20-30%** при базах >500 франшиз (за счет фильтрации кандидатов)
- `--batch-size`: влияет на throughput при использовании batch API (если доступен)

**Влияние параметров на стоимость**:
- Все параметры не влияют напрямую на стоимость (компонент использует Embedding Service, который оплачивается отдельно)

### Quality validation & human-friendly inspection

Рекомендуемые проверки:
- **Similarity sanity**: similarity scores в диапазоне [0, 1], top-1 обычно > 0.3 для корректных распознаваний
- **Консистентность**: `times_s` соответствует `union_timestamps_sec[frame_indices]`
- **Стабильность**: одинаковые франшизы должны иметь похожие similarity scores на соседних кадрах
- **Coverage**: проверка, что распознавание покрывает разные части видео (начало/середина/конец)

### Render (dev-only)

**Важно**: Render генерируется только в dev-режиме для проверки качества. В production рендеры не используются. NPZ остаётся source-of-truth.

#### Файлы рендера

`franchise_recognition` генерирует следующие файлы рендера:

- **Render-context JSON**: `result_store/<platform_id>/<video_id>/<run_id>/franchise_recognition/_render/render_context.json`
  - Содержит структурированные данные для генерации HTML и анализа
  - Включает: summary, key_facts, config_highlights, qa_hints, timeline, distributions, top_examples, anti_top_examples
- **HTML mini-dashboard**: `result_store/.../franchise_recognition/_render/render.html`
  - Offline страница (без CDN-зависимостей)
  - Интерактивные таблицы с поиском и сортировкой (vanilla JS)
  - SVG графики для timeline (offline)

#### Как читать выход

**Key facts** (блок вверху страницы):
- `schema_version`: версия схемы NPZ (должна быть `franchise_recognition_npz_v2`)
- `producer_version`: версия компонента (должна быть `0.2`)
- `franchise_category`: категория в Embedding Service (должна быть `franchise`)
- `num_frames`: количество обработанных кадров
- `num_franchises`: количество уникальных франшиз, найденных в видео
- `stage_timings_ms`: время выполнения стадий (initialization, load_deps, process_frames, saving, total)

**Config highlights** (важные параметры):
- `topk`: количество top результатов (всегда 5 для semantic-head contract)
- `similarity_threshold`: минимальный порог similarity (не гейтит top-K, только для фильтрации)
- `threshold_global`: глобальный порог для `is_confident` флагов (по умолчанию 0.23)
- `use_ocr_filtering`: используется ли OCR для фильтрации кандидатов

**Timeline график**:
- Показывает top-1 franchise score по времени
- Ось X: время в секундах
- Ось Y: similarity score (0.0-1.0)
- Нормальные значения: 0.3-0.9 для корректных распознаваний
- Аномалии: score < 0.2 может указывать на отсутствие франшизы или плохое качество

**Top franchises таблица**:
- Показывает топ франшизы по частоте появления в кадрах
- Колонки: Franchise Name, Count (количество кадров), Ratio (доля от общего числа кадров)
- Сортировка: по умолчанию по Count (убывание)
- Интерактивность: поиск по имени франшизы, сортировка по клику на заголовок

**Top / Anti-top примеры**:
- **Top examples**: кадры с наивысшими similarity scores (лучшие распознавания)
- **Anti-top examples**: кадры с низкими scores, но с найденной франшизой (подозрительные случаи)
- Каждый пример показывает: frame_index, time_sec, franchise_name, top1_score, is_confident

**Timeline таблица** (все кадры):
- Показывает данные по каждому кадру
- Колонки: frame_index, time_sec, franchise_name, top1_score, is_confident
- Интерактивность: поиск по frame_index/time_sec/franchise_name, фильтр по is_confident, сортировка
- Ограничение: показываются первые 3000 кадров (для производительности)

#### Типовые распределения и аномалии

**Нормальные распределения**:
- `top1_score_mean`: 0.4-0.7 (типично для видео с четкими франшизами)
- `top1_score_std`: < 0.3 (стабильное распознавание)
- `confident_predictions_ratio`: > 0.5 (для видео с четким контентом франшиз)
- `unique_franchises_count`: 1-3 (обычно одна основная франшиза)

**Аномалии для проверки**:
- `top1_score_mean < 0.2`: возможно, в видео нет франшиз или качество очень низкое
- `top1_score_std > 0.3`: нестабильное распознавание, возможно проблемы с качеством кадров
- `confident_predictions_ratio < 0.3`: мало уверенных предсказаний, возможно нужно снизить `threshold_global`
- `unique_franchises_count > 10`: слишком много разных франшиз, возможно ложные срабатывания

#### Связь с NPZ (source-of-truth)

Рендер является **человекочитаемой интерпретацией** NPZ артефакта:
- Все данные в рендере берутся из NPZ (`franchise_recognition.npz`)
- NPZ остаётся единственным source-of-truth для downstream компонентов
- Рендер может иметь упрощения/агрегации для удобства чтения
- При расхождении всегда используйте данные из NPZ

**Ключевые поля NPZ**:
- `frame_topk_ids/scores`: per-frame top-K результаты
- `track_topk_ids/scores`: video-level aggregate (max over time)
- `frame_is_confident_top1`: флаги уверенности для каждого кадра
- `track_is_confident_top1`: флаг уверенности для всего видео
- `semantic_label_names`: каноническое пространство меток (`"id:name"`)

#### Время выполнения

Время выполнения компонента отображается в блоке **Key facts** → `stage_timings_ms`:
- `initialization`: обычно < 100ms (загрузка metadata.json)
- `load_deps`: зависит от размера embeddings (обычно 50-200ms)
- `process_frames`: основное время (зависит от количества кадров и режима):
  - **Оптимизированный режим** (embeddings напрямую): ~0.1-0.6 секунды для 115 кадров
  - **Fallback режим** (image-based search): ~6-7 секунд для 115 кадров
- `saving`: обычно < 50ms (запись NPZ)
- `total`: сумма всех стадий

#### Параметры конфига, влияющие на результат и стоимость

**Параметры, влияющие на качество**:
- `--threshold-global`: влияет на `is_confident` флаги (не влияет на top-K результаты)
- `--similarity-threshold`: минимальный порог для результатов (не гейтит top-K, только фильтрация)

**Параметры, влияющие на стоимость**:
- `--use-ocr-filtering`: может ускорить на ~20-30% при базах >500 франшиз (за счет фильтрации кандидатов)
- `--batch-size`: влияет на throughput при использовании batch API (если доступен)

**Параметры, не влияющие на результат**:
- `--topk`: фиксировано на 5 (semantic-head contract)
- `--ocr-npz`: опционально, используется только для фильтрации

#### Конфигурация

В `global_config.yaml`:
```yaml
franchise_recognition:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML mini-dashboard
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

**Legacy demo** (deprecated):
- `quality_report/demo_franchise_recognition_quality.py` — генерирует HTML с timeline, thumbnails, top-K results per frame, video-level aggregate, и (опционально) OCR evidence.

**Пример запуска legacy demo**:
```bash
python quality_report/demo_franchise_recognition_quality.py \
    --frames-dir /path/to/frames_dir \
    --rs-path /path/to/result_store \
    --out-dir /path/to/output
```

### Интеграция с Embedding Service

Компонент **использует Embedding Service** для хранения и поиска эмбеддингов франшиз:

1. **Категория**: `franchise` (может быть расширена)
2. **Модель**: CLIP (определяется Embedding Service, обычно `clip_224` или `clip_336`)
3. **Канонический label space** (v2, Audit v3):
   - Компонент получает полный список labels из Embedding Service заранее через `GET /categories/franchise/labels`
   - Строится канонический label space (отсортированный по UUID) для обеспечения стабильности `label_id` между запусками
   - Вычисляется `db_digest` (SHA256 хеш от канонического списка labels) для reproducibility
   - Результаты поиска маппятся к каноническому label space через UUID
4. **Использование** (два режима):
   - **Оптимизированный режим** (рекомендуется):
     - Получение всех franchise embeddings через `GET /categories/franchise/embeddings` (один запрос)
     - Маппинг embeddings к каноническому label space по UUID
     - Локальное сравнение через cosine similarity (без HTTP запросов для каждого кадра)
     - Автоматически активируется, если в базе есть franchise объекты
     - **Ускорение: 10-50x**
   - **Fallback режим** (если база пуста):
     - Поиск похожих франшиз через Embedding Service (`POST /search`)
     - Результаты маппятся к каноническому label space через UUID
     - Batch search для уменьшения количества HTTP запросов
     - Параллельная обработка батчей
     - **Ускорение: 3-10x** по сравнению с последовательным поиском

**Преимущества использования Embedding Service**:
- Централизованное хранение базы франшиз
- Быстрый поиск через FAISS индексы
- Удобное управление базой (добавление/удаление/обновление)
- Хранение метаданных (название, жанр, год и т.д.)
- Версионирование embeddings для разных моделей
- **Оптимизированный API для массового получения embeddings** (`GET /categories/{category}/embeddings`)

**Требования к Embedding Service**:
- Должен быть доступен (fail-fast при недоступности)
- Должен поддерживать категорию `franchise`
- Должен поддерживать endpoint `GET /categories/{category}/labels` (для получения канонического label space)
- Должен поддерживать endpoint `GET /categories/{category}/embeddings` (для оптимизированного режима)
- Должен возвращать результаты поиска в формате: `[{"id": str, "name": str, "similarity": float, "metadata": dict}, ...]`
- Должен возвращать labels в формате: `[{"id": str, "name": str, "embedding_model": str, "embedding_dim": int, "updated_at": str}, ...]`

**Добавление franchise объектов**:
```bash
# Через API
curl -X POST "http://localhost:8005/objects/add" \
  -F "category=franchise" \
  -F "name=Super Mario Bros" \
  -F "image=@/path/to/mario_image.jpg" \
  -F 'metadata={"type":"game","platform":"Nintendo"}'
```

Или через Python:
```python
from embedding_service_client import EmbeddingServiceClient
import cv2

client = EmbeddingServiceClient(base_url="http://localhost:8005")
img = cv2.imread("path/to/franchise_logo.jpg")
result = client.add_object(
    category="franchise",
    name="Super Mario Bros",
    image=img,
    metadata={"type": "game", "platform": "Nintendo"}
)
```

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор кадров из всех видео → группировка в батчи → batch inference через Embedding Service → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - **Использование embeddings напрямую**: получение всех franchise embeddings одним запросом для всех видео
  - **Batch search**: группировка кадров из всех видео в батчи для уменьшения HTTP запросов
  - **Параллельная обработка**: параллельная обработка батчей через ThreadPoolExecutor
  - **Кеширование**: кеш результатов поиска для избежания повторных запросов

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **10-50x** (за счет использования embeddings напрямую и лучшего использования ресурсов)
- Для single video: **10-50x** (за счет оптимизации с embeddings напрямую)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

### Stage timings и progress

Компонент измеряет время выполнения ключевых стадий и сохраняет их в `meta.stage_timings_ms`:

- `initialization` — загрузка `metadata.json`, валидация `frame_indices`
- `load_deps` — загрузка `core_clip` embeddings, инициализация Embedding Service клиента, получение канонического label space (через `get_labels()`), получение franchise embeddings (если используется оптимизация)
- `process_frames` — поиск франшиз (локальное сравнение embeddings или через Embedding Service), маппинг результатов к каноническому label space
- `saving` — формирование `meta` (включая `db_digest`), `meta_json`, и атомарная запись NPZ
- `total` — общее время работы компонента

**Логирование таймингов**:
- После завершения обработки компонент логирует тайминги всех стадий в консоль
- Тайминги также сохраняются в `meta.stage_timings_ms` в NPZ артефакте для последующего анализа

Компонент публикует прогресс в `state_events.jsonl`:
- Стадии: `start → load_deps → process_frames → save → done`
- Гранулярный прогресс во время `process_frames` (≥10 обновлений по кадрам)

### Schema и версионирование

**Schema version**: `franchise_recognition_npz_v2`

**Human schema**: `SCHEMA.md` (рядом с компонентом)  
**Machine schema**: `VisualProcessor/schemas/franchise_recognition_npz_v2.json`

**Версионирование**:
- `producer_version`: версия компонента (текущая: `0.2`)
- `schema_version`: версия схемы NPZ (текущая: `franchise_recognition_npz_v2`)
- `db_digest`: SHA256 хеш канонического списка labels (для reproducibility)

**Правила обратной совместимости**:
- Любое существенное изменение ключей/dtype/shape → bump `schema_version`
- Изменения в структуре meta (добавление required полей) → bump `schema_version`
- `allow_extra_keys=false` в схеме (fail-fast при неизвестных ключах)

### Troubleshooting

#### Embedding Service недоступен:

```
WARNING: franchise_recognition | Embedding Service test request failed: ...
Skipping all frames to avoid repeated errors.
```

**Решение**: 
- Убедитесь, что Embedding Service запущен:
  ```bash
  cd DataProcessor/embedding_service
  python run_server.py
  ```
- Проверьте, что категория `franchise` настроена в Embedding Service
- Проверьте логи Embedding Service для детальной информации об ошибках
- Компонент автоматически пропустит все кадры и заполнит пустые результаты при недоступности сервиса

#### Пустые результаты от Embedding Service:

```
WARNING: Embedding Service returned empty results for frame X
```

**Решение**: 
- Проверьте, что в базе Embedding Service есть франшизы категории `franchise`
- Попробуйте уменьшить `--similarity-threshold`
- Убедитесь, что качество изображения достаточное
