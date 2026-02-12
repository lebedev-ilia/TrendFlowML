# Models Documentation Index

Этот каталог содержит документацию по моделям, используемым в DataProcessor.

## Структура документации

### Канонические документы (source-of-truth)

**Важно**: Канонические правила и контракты перенесены в `Models/docs/`:
- **Model system rules**: `Models/docs/contracts/MODEL_SYSTEM_RULES.md`
- **Encoder contract**: `Models/docs/contracts/ENCODER_CONTRACT.md`
- **Model contracts v1**: `Models/docs/contracts/MODEL_CONTRACTS_V1.md`
- **Исторические Q&A**: `Models/docs/source_migrations/MODELS_Q.md`

### Документы в этом каталоге

#### Планы и roadmap

- **`OBJECT_DETECTIONS_AND_SEMANTICS_ROADMAP.md`** — полный план работ по object detections и semantic heads
- **`SEMANTIC_HEADS_IMPLEMENTATION_PLAN.md`** — план реализации semantic heads (v1)
- **`YOLO_FINETUNE_PLAN_V1.md`** — план дообучения YOLO детектора (40 классов)

#### Контракты и спецификации

- **`SCHEMA_SEMANTIC_HEADS_NPZ.md`** — схема NPZ для semantic heads (brands/cars/places/face identity)
- **`SEMANTIC_HEADS_CONTRACTS_QA.md`** — Q&A по контрактам semantic heads (решения Round 1-3)

#### Гайды по сборке

- **`SEMANTIC_BASES_BUILD_GUIDE.md`** — как собирать offline базы для semantic heads (brands/cars/celebs/places)

#### Dynamic Batching

- **`DynamicBatch/docs/DYNAMIC_BATCHING_CHECKLIST.md`** — чек-лист ресурсов для scheduler'а (latency/memory на unit)
- **`DynamicBatch/docs/DynamicBatching_Q_A.md`** — Q&A по dynamic batching (решения Round 1-5)

#### ModelManager

- **`MODEL_MANAGER_Q.md`** — Q&A по ModelManager (Round 0-12, все решения)
- **`MODEL_MANAGER_PLAN.md`** — план реализации единого ModelManager

#### Baseline и инвентаризация

- **`BASELINE_MODELS.md`** — baseline модели (CPU vs GPU, Triton) + **полный список baseline компонентов DataProcessor** (19 компонентов: 7 visual modules + 3 audio extractors + 9 core providers)
- **`BASELINE_GPU_BRANCHES.md`** — baseline GPU ветки (fixed-shape) + Triton план
- **`PRETRITON_BENCH_AND_EXPORT.md`** — pre-Triton бенчмарки и ONNX экспорт
- **`MODEL_INVENTORY.md`** — инвентаризация моделей в кодовой базе (где что загружается)
- **`MODEL_LICENSES.md`** — инвентаризация лицензий моделей (шаблон)

#### Resource costs

- **`resource_costs/`** — JSON файлы с измеренными ресурсами компонентов:
  - `cut_detection_costs_v1.json`
  - `cut_detection_soft_costs_v1.json`
  - `cut_detection_motion_costs_v1.json`
  - `core_brand_semantics_costs_v1.json`
  - `core_car_semantics_costs_v1.json`
  - `core_face_identity_costs_v1.json`
  - `core_place_semantics_costs_v1.json`

## Устаревшие файлы (stubs)

Следующие файлы являются stub'ами, указывающими на канонические документы в `Models/docs/`:
- `MODELS_Q.md` → `Models/docs/source_migrations/MODELS_Q.md`
- `FEATURE_ENCODER_CONTRACT.md` → `Models/docs/contracts/ENCODER_CONTRACT.md`
- `MODEL_SYSTEM_RULES.md` → `Models/docs/contracts/MODEL_SYSTEM_RULES.md`

Эти файлы оставлены для обратной совместимости со старыми ссылками.

## Быстрая навигация

### Нужно понять правила версионирования/кэша?
→ `Models/docs/contracts/MODEL_SYSTEM_RULES.md`

### Нужно понять контракт encoder'а?
→ `Models/docs/contracts/ENCODER_CONTRACT.md`

### Нужно понять как работает dynamic batching?
→ `DynamicBatch/docs/DynamicBatching_Q_A.md` (решения) + `DynamicBatch/docs/DYNAMIC_BATCHING_CHECKLIST.md` (ресурсы)

### Нужно понять как собирать semantic базы?
→ `SEMANTIC_BASES_BUILD_GUIDE.md`

### Нужно понять план реализации semantic heads?
→ `SEMANTIC_HEADS_IMPLEMENTATION_PLAN.md` + `SEMANTIC_HEADS_CONTRACTS_QA.md`

### Нужно понять план дообучения YOLO?
→ `YOLO_FINETUNE_PLAN_V1.md`

### Нужно найти где используется конкретная модель?
→ `MODEL_INVENTORY.md`

### Нужно узнать какие компоненты входят в baseline?
→ `BASELINE_MODELS.md` (раздел "Baseline компоненты DataProcessor")

