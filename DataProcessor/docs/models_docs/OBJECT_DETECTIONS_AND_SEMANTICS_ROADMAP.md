## Roadmap: Object detections + richer semantics (до production‑готовности)

Документ фиксирует **полный план работ** вокруг `core_object_detections` и смежных компонент, чтобы в задаче
**предсказания популярности видео** получить максимально информативные, воспроизводимые и дешёвые фичи.

### Контекст и цель

COCO‑классы “из коробки” полезны, но **недостаточны** для popularity‑моделей: критичны бренды/логотипы, конкретные люди,
известные здания/места, “смысл сцены”. Поэтому архитектура строится так:

- **`core_object_detections`** остаётся **proposal generator** (геометрия + tracking + базовые классы таксономии v1).
- “Богатая семантика” делается отдельными **semantic heads** поверх proposals/track’ов:
  - **бренды/логотипы/атрибуты**: `core_brand_semantics` (MVP уже есть)
  - **люди**: `core_face_landmarks` + (будущий) `core_face_identity`
  - **места/здания**: `scene_classification` + (будущий) `core_landmark_semantics`
  - **open-vocabulary**: reuse `core_clip` (эмбеддинги) как универсальный слой

### Инварианты / ограничения (фиксируем)

- **No-network**: все модели резолвятся через `dp_models.ModelManager` из локального bundle, inference — Triton или inprocess.
- **Triton batching не меняем** на этом этапе (фиксируем batch=1 для head’ов и/или baseline моделей; дальше отдельно оптимизируем).
- **Tracking required**: трекинг (ByteTrack) обязателен для `core_object_detections` и используется как gating для head’ов.
- **Shared primary sampling group**: все core providers, зависящие от object detections, используют **тот же** `frame_indices`.
  - Источник истины: `metadata.json: core_object_detections.frame_indices`.
  - No-fallback: пусто/нет → **error** (для baseline/bench/production консистентности).

### Артефакты и контракт (что должно быть на выходе)

#### 1) `core_object_detections` (base layer)

**Назначение**: предложения объектов + базовая семантика + track id + стабильная геометрия.

**Выход** (`rs_path/core_object_detections/detections.npz`, упрощённо):
- `frame_indices (N,)`
- `boxes (N, MAX, 4)` (xyxy)
- `scores (N, MAX)`
- `class_ids (N, MAX)` (из taxonomy v1 / COCO fallback)
- `class_names (C,)` вида `"id:name"`
- `tracks (N, MAX)` (track id или -1)
- `valid_mask (N, MAX)` (bool)
- `meta` с models_used и параметрами

**Качество**:
- задача компонента — **не распознавать бренды**, а дать качественные proposals (особенно `logo_region`, `text_region`, `screen_*`).

#### 2) Semantic heads (поверх proposals/track’ов)

Каждый head обязан:
- читать `core_object_detections/detections.npz`
- использовать тот же `frame_indices` (shared sampling group)
- писать свой NPZ + `meta.models_used` + строгую воспроизводимость параметров
- иметь cost‑controls (лимиты, пороги, gating), чтобы не “взорвать” стоимость

### План работ (этапы от идеи до “полной готовности”)

Ниже этапы в правильном порядке. Мы идём по одному этапу за раз, сохраняя совместимость.

---

## Этап A — “Истина таксономии” (taxonomy v1) и роли компонент

**Цель**: зафиксировать, какие сущности где появляются и кто за что отвечает.

- A1. Утвердить `TAXONOMY_V1.yaml` как единую “карту мира”:
  - список YOLO‑классов v1 (proposal/scene classes) со stable ids
  - какие классы proposal‑уровня должны быть “обязательными” (`logo_region`, `text_region`, `screen_phone`, …)
  - отдельные ветки для брендов/людей/ландмарков (semantic heads)
- A2. Зафиксировать mapping для downstream моделей:
  - stable id map не меняется; добавления — только в конец
  - договориться о минимальном наборе классов v1 (не раздувать до 300 без нужды)

Артефакт: `VisualProcessor/core/model_process/object_detections/TAXONOMY_V1.yaml` (уже есть, будем дополнять).

---

## Этап B — Довести `core_object_detections` до “production‑контракта”

**Цель**: стабильный, быстрый, воспроизводимый proposal layer, без лишних зависимостей и с ясными knobs.

- B1. Contract / schema:
  - документировать NPZ поля, shapes, dtype, `valid_mask` semantics
  - строгие ошибки на пустые `frame_indices` (no-fallback)
- B2. Зависимости и конфиги:
  - `dp_models.ModelManager` → Triton spec (offline)
  - `runtime=inprocess|triton` поведение
  - фиксировать что tracking required
- B3. Оптимизации без изменения batching:
  - минимизировать лишние `FrameManager.get()` (уже частично сделано)
  - выделить “shared sampling group” и проверять alignment
- B4. Улучшение информативности без изменения модели:
  - добавить агрегаты/метрики трекинга (простые):
    - track length, switches proxy, avg score, avg size, coverage по времени
  - добавить простые per‑frame curves:
    - count_per_class, total_dets, person_present, logo_present (из `logo_region`)

Артефакты:
- `core_object_detections/detections.npz`
- (опционально) `core_object_detections/aggregates.npz` (если решим разделять)

---

## Этап C — Кастомное дообучение детектора (качество proposals под домен)

**Цель**: повысить качество “важных” proposals (особенно логотипы/экраны/текст/продукты), где COCO слаб.

- C1. Утвердить список классов v1 для дообучения YOLO:
  - high precision, ограниченный набор (80–120)
  - отдельные proposal‑классы‑“регионы” (`logo_region`, `text_region`, `product_closeup`)
- C2. Датасет:
  - разметка bbox для ключевых классов
  - негативные примеры (чтобы уменьшить FP для logos/text)
- C3. Тренинг/экспорт:
  - обучение → ONNX export → Triton model branch
  - фиксируем `weights_digest` и версионирование
- C4. Совместимость:
  - стабильные ids должны соответствовать taxonomy v1
  - fallback на COCO только для debug/совместимости, но основной режим — v1

---

## Этап D — Semantic head #1: `core_brand_semantics` (MVP → v1)

**Цель**: распознавать бренды/логотипы/эмблемы по bbox‑кропам, не перегружая compute.

Статус: **MVP реализован** (`VisualProcessor/core/model_process/brand_semantics/main.py`).

Дальше доводим до v1:
- D1. Schema/contract:
  - fields: `track_topk_*`, `det_topk_*`, `semantic_label_names`, meta
  - строгая воспроизводимость параметров (topk, min_score, gating, модельные спеки)
- D2. Улучшить gating:
  - считать “1 crop на track” (уже есть)
  - добавить выбор crop по “наиболее резкому/крупному” (score×area×sharpness)
  - ограничение по “классам proposals” (`logo_region` по умолчанию)
- D3. Prompt management:
  - `labels_yaml` — stable id list, несколько prompts на label
  - кешировать text embeddings (внутри запуска) и/или вынести в отдельный artifact при необходимости
- D4. Aggregates for encoder (опционально, но желательно):
  - per‑time curves: “brand_score_max”, “brand_id_top1”
  - events: “brand appears”, “brand persists”

---

## Этап E — Люди: `core_face_landmarks` → `core_face_identity` (future)

**Цель**: “популярные люди” — один из самых сильных сигналов для popularity‑модели.

- E1. `core_face_landmarks` как base:
  - face presence curves/events
  - face crops/embeddings (если есть) — подготовка для identity
- E2. `core_face_identity` (новая голова):
  - вход: face crops/embeddings + track‑like grouping (по лицам)
  - выход: top‑K celebrity ids / names / scores + метрики уверенности
  - модели: retrieval/классификатор (закрытая база знаменитостей), no-network
- E3. Интеграция с `core_object_detections`:
  - optional: согласование person tracks ↔ face tracks (не обязательно на первом этапе)

---

## Этап F — Scene/landmarks: `scene_classification` + `core_landmark_semantics` (future)

**Цель**: сцена (контекст) и известные места сильно влияют на популярность.

- F1. `scene_classification`:
  - вход: `core_clip` frame embeddings
  - выход: per‑frame/top‑K сцен + агрегаты по видео
- F2. `core_landmark_semantics`:
  - retrieval по глобальным embeddings или специализированная модель
  - output: landmark ids/names/scores + events “landmark present”

---

## Этап G — Model-facing aggregates (для encoder/transformer)

**Цель**: encoder’у нужны не сырые bbox’ы, а стабильные временные сигналы/ивенты.

Для каждого слоя (detections/brands/faces/scenes) определить:
- curves (per‑time) и их timestamps
- events (spans) с `event_start_time_s`/`event_end_time_s`
- masks (valid_mask) и NaN semantics при cascade/gating
- reproducibility: thresholds/params arrays (если применимо)

Связать с `Models/docs/contracts/ENCODER_CONTRACT.md`.

---

## Этап H — Dynamic batching checklist + resource_costs + bench

**Цель**: зафиксировать стоимость и политики запуска для scheduler’а.

- H1. Checklist:
  - зависимости, knobs, shared sampling group, ограничения по batching
- H2. Бенчи:
  - `run_checklist_components_micro.py` — добавить head’ы (brand/face/scene) и их параметры
  - matrix (если нужно) по размеру кадров/частоте
- H3. resource_costs JSON:
  - обновить таблицы в `docs/models_docs/resource_costs/*`
  - обновить `DynamicBatch/docs/DYNAMIC_BATCHING_CHECKLIST.md`

---

## Этап I — Финальная интеграция “end-to-end” для popularity модели

**Цель**: один стабильный pipeline, который выдаёт всё нужное encoder’у.

- I1. Определить финальный набор компонент в профиле:
  - `core_object_detections` (custom v1 detector)
  - `core_brand_semantics`
  - `core_face_landmarks` (+ identity когда будет)
  - `scene_classification`
- I2. Протоколы данных:
  - единые timestamps, alignment, masks
  - единые именования feature keys (схемы)
- I3. Валидация качества:
  - sanity checks на видео (бренды/люди/ландмарки)
  - быстрые regressions на cost

---

### Текущий статус (обновлено)

**Реализовано:**
- ✅ `TAXONOMY_V1.yaml` — зафиксирована таксономия v1
- ✅ `core_object_detections` — улучшен, интегрирован с ModelManager
- ✅ `core_brand_semantics` — реализован (MVP → v1)
- ✅ `core_car_semantics` — реализован (v1 retrieval)
- ✅ `core_face_identity` — реализован (v1 retrieval)
- ✅ `core_place_semantics` — реализован (v1 retrieval)
- ✅ ModelManager — реализован в `dp_models/manager.py`
- ✅ Базы semantic heads — инфраструктура готова (`dp_models/bundled_models/semantics/`)
- ✅ Dynamic batching — чек-лист и Q&A документированы

**В процессе:**
- 🔄 YOLO fine-tune — план готов, требуется разметка и обучение
- 🔄 Resource costs — частично измерены (cut_detection, некоторые semantic heads)

**Связанные документы:**
- Детальный план реализации: `SEMANTIC_HEADS_IMPLEMENTATION_PLAN.md`
- Контракты и Q&A: `SEMANTIC_HEADS_CONTRACTS_QA.md`
- Схема NPZ: `SCHEMA_SEMANTIC_HEADS_NPZ.md`
- План дообучения YOLO: `YOLO_FINETUNE_PLAN_V1.md`

### Следующие шаги

1. Завершить разметку и дообучение YOLO (taxonomy v1_40)
2. Довести resource costs для всех компонентов
3. Интеграция в encoder pipeline


