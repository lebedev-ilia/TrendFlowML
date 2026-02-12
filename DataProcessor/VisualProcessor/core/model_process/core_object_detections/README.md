## Component: `core_object_detections` (Tier‑0 baseline)

### Назначение

`core_object_detections` вычисляет детекции объектов на primary выборке кадров (union-domain) и пишет их в `detections.npz`.
В baseline используется **только YOLO (ultralytics)** или **YOLO через Triton**.

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_object_detections": { "frame_indices": [0, 10, 20] }
}
```

No-fallback:
- отсутствие/пустота `frame_indices` ⇒ **error**
- компонент не выбирает кадры сам

### Выход

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_object_detections/detections.npz`

Ключи:
- `frame_indices (N,) int32` — union-domain
- `times_s (N,) float32` — `union_timestamps_sec[frame_indices]`
- `boxes (N, MAX, 4) float32` (xyxy)
- `scores (N, MAX) float32`
- `class_ids (N, MAX) int32`
- `valid_mask (N, MAX) bool`
- `class_names (M,) str` — `"id:name"` mapping
- `meta` (dict, object-array) — canonical meta с `stage_timings_ms` (legacy формат для обратной совместимости)
- `meta_json` (str, unicode array) — meta в формате JSON-строки (предпочтительно для совместимости между виртуальными окружениями)

### Tracking (removed)

Трекинг полностью удален из компонента. Массивы `tracks`, `tracks_list`, `tracks_list_ids` больше не сохраняются в выходном NPZ.

**Downstream компоненты** обновлены для работы без трекинга:
- `core_car_semantics`, `core_brand_semantics`: генерируют per-detection track_ids (каждая детекция получает свой уникальный track_id) на основе `valid_mask`
- `action_recognition`: генерирует сегменты из детекций "person" (class_id=0), группируя последовательные кадры с person детекциями в сегменты

### Stage timings и progress

Компонент измеряет время выполнения ключевых стадий и сохраняет их в `meta.stage_timings_ms`:

- `initialization` — загрузка `metadata.json`, валидация `frame_indices`
- `load_deps` — загрузка модели YOLO, инициализация FrameManager
- `process_frames` — детекция объектов на всех кадрах
- `saving` — формирование `meta` и атомарная запись NPZ
- `total` — общее время работы компонента

**Логирование таймингов**:
- После завершения обработки компонент логирует тайминги всех стадий в консоль:
  ```
  core_object_detections | stage timings (ms): initialization=0.5, load_deps=0.6, process_frames=7899.0, saving=101.2, total=9477.8
  ```
- Тайминги также сохраняются в `meta.stage_timings_ms` в NPZ артефакте для последующего анализа

Компонент публикует прогресс в `state_events.jsonl`:
- Стадии: `start → load_deps → process_frames → save → done`
- Гранулярный прогресс во время `process_frames` (≥10 обновлений по кадрам)

### Batch size (scheduler-controlled)

`--batch-size` обязателен и задаётся верхним scheduler/DynamicBatching (пока вручную в конфиге).
Auto-batching внутри компонента запрещён.

Важно (фиксируем на текущем этапе):
- Мы **НЕ меняем Triton batching** и считаем cost строго **на 1 unit (frame)**.
- Для baseline Triton модель `yolo11x_640` имеет fixed shape **batch=1** (см. `triton/models/yolo11x_640/config.pbtxt`).
- Батчинг для throughput будет делаться **на уровне scheduler’а** как **cross-video micro-batching** (несколько видео → несколько RPC),
  а не через изменение `max_batch_size` модели.

### Meta / models_used

- `models_used[].runtime="inprocess"`
- `engine="ultralytics"`
- `weights_digest="unknown"` (baseline)

---

## Models

### GPU Models

1. **YOLO11x** (object detection)
   - **Triton**: ✅ Да (`triton/models/yolo11x_640/` или другие presets)
   - **Spec name**: `yolo11x_640_triton` (ModelManager, если используется)
   - **Runtime**: `triton` или `inprocess` (ultralytics)
   - **Engine**: `triton` (ONNX/TensorRT) или `ultralytics` (PyTorch)
   - **Precision**: `fp32`
   - **Device**: `cuda` (если доступен) или `cpu`
   - **Model path** (ultralytics): `dp_models/bundled_models/visual/yolo/yolo11x_41_best.pt`
   - **Taxonomy**: 41 класс (v1.0, финальная таксономия для baseline)
   - **Weights digest**: `unknown` (baseline) или `provided_by_deploy` (Triton)
   - **Triton presets**: `yolo11x_320`, `yolo11x_640`, `yolo11x_960` (через `--triton-preprocess-preset`)

**Примечание**: 
- В baseline поддерживаются оба runtime: `ultralytics` (in-process) и `triton` (через Triton Inference Server)
- Для Triton используется fixed batch=1 (baseline contract)
- Batch size контролируется scheduler через `--batch-size` (для ultralytics runtime)

### CPU Models

Нет (модель работает на GPU если доступен, иначе на CPU через ultralytics).

---

## Parallelization

### Внутренний параллелизм

- **Батчинг**: Компонент обрабатывает кадры батчами (размер батча задаётся scheduler через `--batch-size`)
- **Потоки**: Ultralytics YOLO может использовать несколько GPU потоков для inference (если доступно несколько GPU)
- **Ограничения**: Batch size контролируется scheduler, auto-batching внутри компонента запрещён

### Внешний параллелизм

- **Можно запускать несколько экземпляров параллельно** на разных видео (разные `run_id`)
- **Требования к изоляции**:
  - Разные `run_id` для каждого видео
  - Разные пути `result_store` (обеспечивается через `platform_id/video_id/run_id`)
  - Изоляция GPU: если несколько компонентов используют один GPU, рекомендуется запускать последовательно или использовать GPU isolation

### Комбинированный подход

- Внутренний батчинг (batch_size задаётся scheduler) + внешний запуск на разных видео/GPU
- Thread-safety: компонент thread-safe для параллельного запуска на разных видео

### Batch Processing (Stage 3)

Компонент поддерживает **batch processing** для одновременной обработки нескольких видео с оптимизацией:

- **Гибридный батчинг**: сбор кадров из всех видео → группировка в батчи → batch inference через Ultralytics или Triton → распределение результатов обратно по видео
- **Оптимизации производительности**:
  - **Загрузка модели один раз**: модель загружается один раз перед всеми батчами (критично для производительности)
  - **Использование уже загруженных кадров**: кадры из `all_frames` используются напрямую, без повторной загрузки
  - **Батчинг всех кадров из всех видео**: все кадры обрабатываются одним большим батчем (лучше использует GPU)
  - **Векторизованная обработка результатов**: batch операции numpy вместо поэлементной обработки
  - **Предварительная конвертация RGB→BGR**: конвертация всех кадров заранее одним проходом

**Использование**:
- Batch processing автоматически активируется при обработке нескольких видео через `--video-input-dir` или `--video-input-list`
- Контролируется через `visual.batch_processing.enable_gpu_batching` в `global_config.yaml`
- Лимит размера батча: `visual.batch_processing.max_frames_per_gpu_batch`

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **5-10x** (за счет переиспользования модели и лучшего использования GPU)
- Для single video: **1.5-2x** (за счет векторизации и оптимизации обработки)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

---

## Performance characteristics

### Resource costs (measured)

**Источник данных**: `docs/models_docs/resource_costs/core_object_detections_costs_v1.json` (планируется)

**Единица обработки**: `frame`

**Типичные значения (preset="default", runtime="ultralytics")**:

| Resolution | Latency per frame | CPU RAM peak | GPU VRAM peak | Notes |
|------------|-------------------|--------------|---------------|-------|
| 320p | TBD ms | TBD MB | TBD MB | measurements pending |
| 640p | TBD ms | TBD MB | TBD MB | measurements pending |
| 960p | TBD ms | TBD MB | TBD MB | measurements pending |

**Для видео с N кадрами**: Total latency ≈ N × latency_per_frame

**Полные данные**: см. `docs/models_docs/resource_costs/core_object_detections_costs_v1.json` (планируется)

### Оптимизации производительности

Компонент включает следующие оптимизации для batch processing:

1. **Загрузка модели один раз**: модель YOLO загружается один раз перед всеми батчами вместо загрузки для каждого батча (критично для производительности)
2. **Использование уже загруженных кадров**: кадры из `all_frames` используются напрямую, без повторной загрузки из FrameManager
3. **Батчинг всех кадров из всех видео**: все кадры обрабатываются одним большим батчем вместо обработки по видео отдельно (лучше использует GPU)
4. **Векторизованная обработка результатов**: batch операции numpy (`res.boxes.data.cpu().numpy()`) вместо поэлементной обработки через `.item()`
5. **Предварительная конвертация RGB→BGR**: конвертация всех кадров заранее одним проходом
6. **Освобождение памяти**: явное освобождение памяти модели и GPU кеша после обработки

**Ожидаемое ускорение**:
- Для batch processing (несколько видео): **5-10x** (за счет переиспользования модели и лучшего использования GPU)
- Для single video: **1.5-2x** (за счет векторизации и оптимизации обработки)

**Важно**: Оптимизации не влияют на качество результатов — все формулы остались идентичными, изменилась только производительность.

---

## Quality validation & human-friendly inspection

### Human-friendly визуализация (Render System)

Компонент автоматически генерирует **render-context JSON** для каждого запуска:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_object_detections/_render/render_context.json`

Этот JSON содержит:
- **Summary**: статистики по детекциям (frames_count, total_detections, detections_per_frame_mean/std/min/max, unique_classes_count, score_mean/std/min/max/median, box_area_mean/std/min/max/median)
- **Timeline**: данные по каждому кадру (frame_index, time_sec, detections_count, average_score)
- **Distributions**: распределения detections_per_frame, scores и box_areas (min, max, mean, std, median, percentiles)
- **Top classes**: топ-10 классов с количеством детекций

**HTML debug страница** (опционально):
- Путь: `result_store/.../core_object_detections/_render/render.html`
- Содержит интерактивные графики (Chart.js):
  - Timeline: количество детекций и средний score по времени
  - Top classes: таблица с топ-классами и количеством детекций
  - Distributions: статистики по detections_per_frame, scores и box_areas
  - Summary metrics: ключевые показатели в удобном формате

**Конфигурация** (в `global_config.yaml`):
```yaml
core_object_detections:
  render:
    enable_render: true  # Генерировать render-context JSON
    enable_html_render: true  # Генерировать HTML debug страницу
```

**Примечание**: Render генерируется автоматически после успешной обработки компонента (best-effort: ошибки render не валят основной процесс).

### Как проверить качество выхода компонента

#### 1. Human-friendly визуализация (Legacy)

Для визуальной проверки качества детекций, трекинга и классов используйте скрипт:

```bash
python3 DataProcessor/VisualProcessor/core/model_process/core_object_detections/quality_report/demo_core_object_detections_quality.py \
  --frames-dir /path/to/frames_dir \
  --rs-path /path/to/result_store \
  --out-dir /path/to/output \
  --max-frames 20
```

Скрипт создаёт HTML отчёт с:
- **Визуализацией детекций**: кадры с нарисованными bounding boxes, классами, scores, track IDs
- **Статистикой**: общее количество детекций, классов, треков, средние значения
- **Распределением классов**: топ-20 классов с количеством детекций
- **Статистикой треков**: количество треков, средняя/максимальная длина треков
- **Таблицей детекций**: для каждого кадра отдельная таблица со всеми детекциями

**Что проверять визуально**:
- ✅ Корректность детекций (боксы соответствуют объектам на изображении)
- ✅ Корректность классов (правильные названия объектов)
- ❌ False positives (ложные детекции)
- ❌ False negatives (пропущенные объекты)

#### 2. Статистическая валидация

**Ожидаемые диапазоны значений** (для типичных видео):

- **Количество детекций на кадр**: 1-20 (зависит от сцены)
- **Score distribution**: >0.5 для валидных детекций (threshold=0.6 по умолчанию)
- **Track length**: 5-50 кадров (зависит от длительности появления объекта)
- **Количество уникальных треков**: зависит от количества объектов в видео

**Проверка разумности**:
- Отсутствие аномальных значений (NaN где не ожидается)
- `frame_indices` отсортированы и уникальны
- `valid_mask` корректно фильтрует детекции по threshold

#### 3. Интеграция с downstream модулями

Компонент используется следующими downstream компонентами:
- `shot_quality`: проверяет выравнивание `frame_indices` и использует `boxes`, `valid_mask`, `class_ids`
- `cut_detection`: использует для jump-cuts heuristics
- `core_car_semantics`, `core_brand_semantics`: используют bbox proposals и tracks
- `core_place_semantics`: использует `frame_indices` для выравнивания

**Проверка**: Убедитесь, что downstream компоненты корректно читают артефакты и `frame_indices` выровнены с shared sampling group.

### Sampling / units-of-processing requirements

**Требования к выборке кадров**:

Компонент входит в "shared sampling group" с другими core providers (`core_clip`, `core_depth_midas`, `core_face_landmarks`). Все компоненты этой группы должны работать на **одном и том же** primary `frame_indices` (иначе downstream падает из-за mismatch).

Требования к выборке:

- **Coverage**: обязательно покрывать начало/середину/конец видео и быть равномерной по времени
- **Непрерывная кривая**: количество кадров должно зависеть от длительности видео через непрерывную монотонную функцию (без скачков)
- **Минимальное значение**: минимум 50 кадров (для коротких видео)
- **Максимальное значение**: максимум 1500 кадров (cap для длинных видео)
- **Целевое значение**: зависит от длительности через кривую `target_gap_sec = f(duration_s)`

**Рекомендуемая политика выборки** (Segmenter-owned):

- `target_gap_sec = f(duration_s)` — непрерывная монотонная кривая, построенная через log‑log интерполяцию по anchor‑точкам
- `budget_n = round(duration_s / target_gap_sec)` (и затем `N = min(requested_max, budget_n)`)

Ориентиры по кривой (приблизительно):
- **≈ 5 минут**: `target_gap_sec ≈ 1s`
- **≈ 10 минут**: `target_gap_sec ≈ 2s`
- **≈ 20 минут**: `target_gap_sec ≈ 3–4s` (целимся около **3.5s**)

**Требования к разрешению**:

- **min shorter side**: 320 px
- **target**: 640 px
- **max useful**: 1080 px
- **апскейл запрещён** (только downscale)

**No-fallback policy**:
- Если `core_object_detections.frame_indices` отсутствует или пустой → **RuntimeError** (no-fallback)
- Если `union_timestamps_sec` отсутствует → **RuntimeError** (no-fallback)

**Важно**: Segmenter является единственным владельцем sampling (компонент не генерирует семплинг сам).

---

### Что не хватает / roadmap (для encoder’а и transformer‑моделей)

> Цель: компонент даёт сильный “sensor” сигнал (объекты/персоны/геометрия/динамика),
> но чтобы это стало максимально полезно для моделей, нужно добавить несколько вещей.
> При этом **baseline правила сохраняем**: tracking required, Triton batching не меняем, no-network.

#### 1) Устойчивое “кто есть кто” (ReID / appearance)

- **Проблема**: ByteTrack даёт `track_id`, но при пропусках/окклюзиях возможны ID-switch; для модели это “прыгающая идентичность”.
- **Как получить (варианты)**:
  - **ReID‑эмбеддинги** для bbox‑кропов (лёгкая модель, через Triton/ModelManager, local-only) → `det_embedding` / `track_embedding`.
  - **CLIP image embeddings** на кропах (если уже есть core_clip/clip_image ветка) как универсальная appearance‑семантика.
  - “дешёвый суррогат” (если хотим совсем без моделей): цветовые hist/edge density внутри bbox как weak appearance‑подпись.

#### 2) Более богатая семантика (таксономия классов)

**Проблема**

COCO‑классы (и похожие) часто недостаточны для продуктовых задач. Типичные missing‑категории:
- **UI / экран / телефон / монитор / “screen-recording”**
- **logo / brand / product / package / bottle / food specifics**
- **weapon / cigarettes / “unsafe content”**
- **text regions** (не OCR, а именно “в кадре есть текстовый блок”)
- **special domains**: games, cartoons, CG, presentation slides, etc.

Если мы ограничиваемся `class_id` из детектора, то encoder/модель видит “person/car/cell phone”,
но не видит “это именно *phone screen with UI*, *logo brand X*, *product shot*”.

**Важное замечание для задачи “популярность” (что реально имеет смысл детектить)**

С учётом твоей цели (предсказать популярность) “нужная семантика” обычно распадается на 3 типа:
- **бренды/логотипы/товары** (в т.ч. бренд авто/одежды) — чаще всего это *не* детекция “объекта целиком”, а распознавание **атрибутов** по кропу (logo/patch/shape).
- **известные люди** — это почти всегда **face pipeline** (детект лица → эмбеддинг → сравнение/классификация), а не YOLO‑класс “person_X”.
- **известные места/здания** — это чаще **landmark recognition / retrieval** по кадру или по region‑proposal, а не bbox‑детекция.

То есть, “дообучить один детектор на 500 брендов + 500 людей + 500 зданий” обычно плохо масштабируется по данным и ошибкам.
Лучший практический вариант — держать детектор как “proposal generator” (где смотреть), а семантику делать отдельными головами/моделями.

**Что стоит дообучать в детекторе (реалистичная таксономия v1, high‑precision)**

Дообучение YOLO имеет высокий ROI для классов, которые:
- имеют **чёткую геометрию/контуры** и часто встречаются,
- важны как **событие/сцена** (а не как редкая fine‑grained категория),
- не требуют “узнать конкретную личность/бренд” по сути.

Примеры классов v1:
- `screen_phone`, `screen_monitor`, `tv_screen` (или единый `screen`)
- `logo_region` / `brand_mark` (как регион‑proposal для следующего распознавания бренда)
- `text_region` (не OCR, а “есть текстовый блок”)
- `product_closeup` / `package` (если реально частое)
- “сценовые” объекты: `car`, `motorcycle`, `food`, `pet`, `sport_ball`, и т.п. (если в твоём домене это влияет)

А вот **бренды авто/одежды** лучше делать как:
- `car` bbox от детектора → отдельная модель “car make/model classifier” на кропе
- `logo_region` bbox → logo classifier / CLIP‑matching

А **известные люди**:
- `core_face_landmarks`/face detector → face embeddings → celebrity classifier / nearest‑neighbor retrieval

А **здания/места**:
- landmark classifier/retrieval по whole‑frame или по region proposals (можно начать с CLIP retrieval).

**Как сюда “вплетается” `scene_classification`**

`scene_classification` даёт **семантику уровня сцены/контекста**, которая сильно влияет на популярность:
- тип сцены (Places365 label + агрегаты уверенности/энтропии/стабильности)
- онтология: indoor/outdoor + nature/urban
- CLIP‑семантика (из `core_clip`): aesthetic/luxury/atmosphere

Это не заменяет детекции объектов (bbox), но добавляет “где мы находимся” и “какая атмосфера”, что часто коррелирует с performance.

**Рекомендация по роли компонентов (чтобы не было дублирования)**
- `core_object_detections`: bbox proposals + tracking (required)
- `core_face_landmarks` (+ будущий `core_face_identity`): известные люди
- `scene_classification`: контекст/сцена/атмосфера (через Places365 + `core_clip`)
- `core_clip`: open-vocab семантика по кадрам/кропам, если нужно “добавить классы” без переобучения детектора

**Taxonomy v1 (стабильные id / кто за что отвечает)**

**Финальный набор классов (v1.0, утверждён для baseline и продакшена)**: 41 класс

См. `yolo_fine_tune/YOLO_CLASSES_V1_FINAL.md` — полный список классов с описанием.

**Файлы с классами**:
- `yolo_fine_tune/labels_v1_40.txt` (английские названия, 41 класс)
- `yolo_fine_tune/labels_v1_40_ru.txt` (русские названия, 41 класс)

**Категории классов**:
- Люди и группы (2): person, crowd
- Транспорт (6): car, motorcycle, bicycle, bus, truck, pet
- Спорт (1): sports_ball
- Электронные устройства (10): phone, laptop, tablet, smartwatch, watch, headphones, camera, microphone, game_controller, tv_device, monitor_device
- Одежда и аксессуары (11): clothing_top, clothing_bottom, outerwear, suit, dress, shoes, bag, hat, glasses, ring, bracelet, earrings, pendant
- Регионы для semantic heads (2): logo_region, text_region
- Продукты (1): cosmetics_product
- Экраны устройств (4): screen_phone, screen_laptop, screen_monitor, tv_screen
- Продукты питания (1): food_item

**Принципы**:
- Объектный уровень (не брендовый): `phone` (не `iphone`), `car` (не `tesla`)
- Region-proposal классы: `logo_region`, `text_region` для downstream semantic heads
- Экраны как отдельные классы: различаем устройство и его экран
- Стабильные ID: классы фиксированы после начала baseline dataset collection

**Как получить (3 стратегии, от MVP → к тяжёлым)**

1) **Open‑vocabulary поверх bbox (рекомендуемый MVP)**: CLIP‑matching по кропам
- **Идея**: детектор остаётся “геометрическим” сенсором, а семантика добавляется отдельным шагом:
  - берём bbox‑кропы (в RGB),
  - считаем CLIP image embedding (через Triton/ModelManager, local-only),
  - сравниваем с заранее подготовленными text embeddings для списка prompts (также через core_clip / ModelManager),
  - сохраняем `topk_label_ids + topk_scores` для каждого bbox (или для track).
- **Почему это хорошо**:
  - не надо переобучать детектор;
  - можно добавлять/править категории *конфигом* (prompts);
  - легко расширять на домены.
- **Как удержать cost** (важно для dynamic batching):
  - **gating по детекциям**: семантику считаем только для `valid_mask==true` и для top‑K боксов по score/area.
  - **gating по кадрам**: например, только каждый N‑й кадр (или по keyframes).
  - **re-use через tracking (очень важно)**: считаем семантику **1 раз на track** (например на первом появлении и потом раз в M секунд),
    а на остальных кадрах просто “протягиваем” значение по `track_id`.
  - **кроп-ресайз фиксированный** (например 224/336) — единый cost bucket.

2) **Специализированные детекторы как отдельные core providers**
- Примеры: `core_logo_detections`, `core_text_regions`, `core_product_detections`.
- **Плюсы**: точнее по конкретной задаче, меньше prompt‑хаков.
- **Минусы**: больше моделей/веток, тяжелее поддержка, нужно фиксировать лицензии/веса/экспорт.

3) **Новый/расширенный head детектора под нужную таксономию**
- Это “правильное” решение, если есть размеченные данные и понятная целевая таксономия.
- **Минус**: это уже полноценный ML‑проект (данные → train → export → Triton → QA).

**Что именно писать в артефакты (предлагаемый формат)**

Чтобы не раздувать `detections.npz` (и не плодить огромные массивы), лучше хранить sparse/top‑K:
- `semantic_label_names (A,) str` — mapping `"id:name"` для промптов/категорий (stable ids!)
- `det_sem_topk_ids (N, MAX, K) int32` и `det_sem_topk_scores (N, MAX, K) float32`
- (опционально) `track_sem_topk_ids (K_tracks, K)`, `track_sem_topk_scores (...)` — если семантика считается по track

**Оценка качества**
- “точной истины” часто нет, поэтому для MVP достаточно:
  - sanity‑проверка на нескольких видео (qualitative),
  - стабильность по времени: семантика по track не должна “прыгать” каждый кадр,
  - влияние на downstream метрики (через encoder/модель).

#### 3) Форма объектов (маски) и точная геометрия

- **Проблема**: bbox плохо описывает форму/окклюзию; для композиции/quality это шумно.
- **Как получить**:
  - instance segmentation (masks) и хранить `mask_rle`/`mask_area_frac`,
  - компромисс без segmentation: “shape proxies” (edge density / fg fraction) внутри bbox.

#### 4) Калибровка confidence и устойчивость порогов

- **Проблема**: `scores` не калиброваны; один и тот же threshold ведёт себя по‑разному в разных доменах/свете/разрешениях.
- **Как получить**:
  - offline calibration (temperature scaling / isotonic) и хранить `calibrated_score`,
  - или хранить больше статистики (например top‑k score distribution per frame) и дать encoder’у научиться “доверять” сам.

#### 5) “Model-facing” агрегаты (curves/events), чтобы encoder не ел 100 боксов × N кадров

- **Проблема**: raw detections полезны, но неудобны как прямой вход для encoder’а/трансформера.
- **Как получить (почти бесплатно по compute)**:
  - добавить поверх raw detections вычисление **curves** (на кадр):
    - `person_count`, `object_count`, `sum_person_area_frac`, `max_person_area_frac`,
    - `dominant_class_id`, `entropy(class_ids)`, `center_of_mass` и т.п.
  - добавить **events**:
    - появление/исчезновение `person`,
    - переход “1→2+ people”,
    - резкий рост/падение объектов, крупный план человека (по area_frac).

#### 6) Метрики качества трекинга (чтобы encoder понимал, где можно доверять)

- **Проблема**: треки могут рваться/переставляться; downstream не видит confidence трекинга.
- **Как получить**:
  - хранить диагностические метрики: `track_age`, `track_gaps`, `mean_iou_to_det`, `num_switch_suspects`,
  - и/или явные `valid_mask`/`confidence` для track‑level сигналов.


