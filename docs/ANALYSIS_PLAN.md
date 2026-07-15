# План анализа проекта TrendFlow (для доведения до прод-готовности на 200k / мульти-нода)

> Цель этого документа — пройти **весь** проект (1121 .py, 863 .md, 359 yaml)
> ничего не упустив, по ходу актуализируя и причёсывая документацию, и собрать
> полную фактическую картину. По итогам пишется большой план реализации
> (`docs/IMPLEMENTATION_PLAN.md`) по трём веткам:
> **Логика** (польза фич для моделей и аналитиков), **Масштабируемость/развёртывание**
> (мульти-нода, горизонтальное масштабирование), **Оптимизации** (убрать/заменить
> медленный код без деградации логики, лучшие библиотеки).

Прогресс анализа и список изменённых документов ведётся в
[`ANALYSIS_CHECKLIST.md`](ANALYSIS_CHECKLIST.md).

## Принципы анализа (для каждой фазы)
1. **Назначение и границы** компонента: что делает, входы/выходы, контракт.
2. **Точки входа** (main/CLI/API/celery task) и зависимости (что вызывает, что вызывает его).
3. **Логика-фичи**: какие фичи на выходе, их польза/риски (по 4 осям playbook:
   корректность, стабильность, различимость, предсказательная ценность).
4. **Масштаб**: состояние (Redis/БД vs in-memory), идемпотентность, параллельность,
   узкие места под 200k/мульти-нода.
5. **Оптимизация**: горячие/медленные места, тяжёлые/неоптимальные зависимости,
   дубли, кандидаты на замену (с сохранением логики).
6. **Документация**: актуализировать README/доки компонента; устаревшее — обновить,
   дублирующее — слить, мусор — удалить (зафиксировать в чеклисте).
7. **Находки** записывать в чеклист как `[branch:logic|scale|opt]` пункты — это сырьё
   для плана реализации.

## Фазы

### Фаза 0 — Корень, сквозная архитектура, конфиги
Файлы/папки: `CLAUDE.md`, `docs/` (19 md), `doc.md`, `poradoc.md`, `d.py`,
`configs/`, `example/`, `_profiles_cache/`, `bootstrap.sh`, `docker-compose.prod.yml`, `deploy/`.
Задачи: единая карта архитектуры и потоков; навести порядок в корневых файлах
(`d.py`/`doc.md`/`poradoc.md` — мусор/черновики → удалить или перенести); сверить
`configs/` (старый `hf_artifacts_manifest.json` vs новый `models_manifest.json`).

### Фаза 1 — Fetcher (ingestion + dataset_collector)
Папка: `Fetcher/` (162 py / 84 md). Особое внимание: `fetcher/dataset_collector/`
(сбор 200k), `backpressure.py`, `celery_queues.py`, `proxies.py`, `metrics.py`,
`api.py`, `orchestrator.py`, `tasks.py`, монитор. **Скрипты скачивания фото для
разметки semantic-баз** — найти и наметить доработку (это вход для пользователя).

### Фаза 2 — DataProcessor: оркестрация и контейнер
`DataProcessor/main.py`, `api/` (88 py), `dag/`, `dp_queue/`, `dp_triton/`,
`common/`, `profiles/`, `configs/`, `state/`, `storage/`, `docker/`, `docker-compose.yml`.
Фокус: как запускается обработка, очереди/воркеры, state в Redis/БД, идемпотентность,
manifest/NPZ контракт, S3 vs fs.

### Фаза 3 — Segmenter (единый источник frame_indices)
`DataProcessor/Segmenter/` + контракт sampling_policy. Критично для логики (общий
сэмплинг) и масштаба (стоимость кадров). Версионирование политики.

### Фаза 4 — VisualProcessor (самый объёмный)
`core/model_process/` (11: core_clip, core_object_detections, core_face_landmarks,
core_depth_midas, core_optical_flow, ocr_extractor, **core_identity** = 6 хедов) +
`modules/` (19). По каждому: фичи (FEATURE_DESCRIPTION), валидатор, стоимость,
зависимость от Triton/ES, no-network. Точка интеграции YOLO.

### Фаза 5 — AudioProcessor
`DataProcessor/AudioProcessor/` (222 py / 124 md): экстракторы (ASR/CLAP/diarization/
emotion/source_separation/spectral...), зависимость от Segmenter, Triton, dp_models.

### Фаза 6 — TextProcessor
`DataProcessor/TextProcessor/` (111 py / 99 md): эмбеддинги (e5), spaCy/BERTopic,
контракт входа ASR/token-only, privacy.

### Фаза 7 — Модельный слой инференса
`dp_models/` (ModelManager, spec_catalog, providers), `embedding_service/`, `triton/`,
`dp_triton/`. Фокус: no-network/fail-fast, digests, маршрутизация inprocess/triton,
батчинг, готовность к мульти-нодному инференсу.

### Фаза 8 — Качество фич и тулинг
`DataProcessor/tools/` (feature_quality_audit, qa_pipeline, drift, golden_compare,
build_training_matrix...), `scripts/`, `qa/`, `monitoring/`, `benchmarks`. Связать с
веткой Логика: как доказываем пользу фич перед обучением.

### Фаза 9 — Backend (сайт-оркестратор)
`backend/` (79 py / 36 md): API, `tasks/` (celery), `dbv2/` (схема), `services/`,
миграции, события/WebSocket. Связь с Fetcher и DataProcessor.

### Фаза 10 — Models (предсказание популярности)
`Models/` (30 py / 24 md): контракты, baseline/v1 трансформер, интерпретируемость.
Связь: какие фичи нужны модели (вход для ветки Логика).

### Фаза 11 — DynamicBatch (планировщик масштаба)
`DynamicBatch/` (14 py): cost-модель, level-1 batching, OOM-backoff, Postgres-registry.
Ключ для ветки Масштаб/оптимизации на 200k.

### Фаза 12 — Развёртывание/инфра (сквозная)
`k8s/`, `docker-compose.prod.yml`, `bootstrap.sh`, `*/monitoring/`, `deploy/`,
`docs/*DEPLOY*`, `docs/K8S*`, `docs/CONTAINER*`. Свести в единый actual deployment-гайд.

### Фаза 13 — Синтез
Свести находки всех фаз в актуальную карту проекта; по каждой из 3 веток —
консолидированный список «как есть / что доводить». Это прямой вход для
`docs/IMPLEMENTATION_PLAN.md`.

## Артефакты по итогам анализа
- Обновлённые/слитые/удалённые документы (лог — в чеклисте).
- `docs/PROJECT_MAP.md` — актуальная карта проекта (создаётся в Фазе 13).
- `docs/IMPLEMENTATION_PLAN.md` — большой пошаговый план доведения (после анализа).

## Что вне моей зоны (делает владелец)
1. Ручная разметка semantic-баз (я готовлю скрипты скачивания фото + тулинг баз).
2. Адаптация алгоритмов под особенности YOLO-датасета (опишет позже).
