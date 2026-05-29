# Главный индекс документации TrendFlowML

Этот документ служит единой точкой входа для навигации по всей документации проекта TrendFlowML. Каждый раздел содержит ссылки на главные индексы подсистем и их краткое описание.

---

## О проекте

TrendFlowML — мультимодальная система для предсказания популярности видео на основе анализа визуального, аудио и текстового контента. Система состоит из нескольких основных компонентов, работающих вместе для извлечения признаков, обработки данных и построения моделей машинного обучения.

**Основная документация**: [MAIN_README.md](MAIN_README.md)

**Дорожная карта до продакшена** (фазы E2E → Text Audit v3 → полный выход → Segmenter/фичи → ML-тюнинг → масштаб ~100k): [PRODUCT_ROADMAP_TO_PRODUCTION.md](PRODUCT_ROADMAP_TO_PRODUCTION.md)

---

## Основные компоненты системы

### DataProcessor

**Описание**: Ядро системы обработки данных. Отвечает за извлечение признаков из видео, аудио и текста через множество специализированных экстракторов и модулей. Включает AudioProcessor, TextProcessor и VisualProcessor.

**README (портфолио)**: [DataProcessor/README.md](../DataProcessor/README.md)

**Главный индекс**: [DataProcessor/docs/MAIN_INDEX.md](../DataProcessor/docs/MAIN_INDEX.md)

**Собеседование / демо**: [DataProcessor/docs/PORTFOLIO_INTERVIEW_GUIDE.md](../DataProcessor/docs/PORTFOLIO_INTERVIEW_GUIDE.md)

**Ключевые разделы документации**:
- Архитектура системы и контракты
- Описания компонентов (60+ экстракторов и модулей)
- Модели и ML-система
- Справочная документация
 - Production schemas (strict typing / versioned contracts): human `SCHEMA.md` + machine schemas + runtime validation (Audit v3); TextProcessor: в т.ч. `title_embedding_cluster_entropy_extractor_output_v1` (`tp_titleclent_*`), `title_to_hashtag_cosine_extractor_output_v1` (`tp_titlehashcos_*`), `semantic_cluster_extractor_output_v1` (`tp_semclust_*`), `topk_similar_titles_extractor_output_v1` (`tp_topktitles_*`), `embedding_shift_indicator_extractor_output_v1` (`tp_embshift_*`), `embedding_source_id_extractor_output_v1` (`tp_embid_*`).
 - TextProcessor Audit v3 preflight: [`DataProcessor/docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md`](../DataProcessor/docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)
 - TextProcessor изолированный смок по 22 экстракторам и 20 сценариям (CPU, временный `DP_MODELS_ROOT`): [`DataProcessor/TextProcessor/scripts/smoke_each_extractor_audit_v3.py`](../DataProcessor/TextProcessor/scripts/smoke_each_extractor_audit_v3.py) · описание команд: [`example/text_audit_v3_smoke/scenarios/README.md`](../example/text_audit_v3_smoke/scenarios/README.md)

**Расположение**: `DataProcessor/`

---

### Backend

**Описание**: Backend-сервис на FastAPI, обеспечивающий REST API, оркестрацию обработки через Celery, управление пользователями, профилями анализа, runs и артефактами. Интегрируется с DataProcessor для выполнения задач обработки видео.

**Главный индекс**: [backend/docs/MAIN_INDEX.md](../backend/docs/MAIN_INDEX.md)

**Ключевые разделы документации**:
- Обзор и архитектура
- API и интерфейсы
- База данных и хранилище
- Конфигурация и операции
- Безопасность и события

**Расположение**: `backend/`

---

### DynamicBatch

**Описание**: Система динамического батчинга для оптимизации обработки множества видео. Управляет очередями, планированием задач и распределением ресурсов между компонентами DataProcessor.

**Документация**: `DynamicBatch/docs/`

**Ключевые документы**:
- `IMPLEMENTATION_PLAN.md` — план реализации
- `DynamicBatching_Q_A.md` — вопросы и ответы
- `BENCHMARK_REGISTRY_CONTRACT.md` — контракт реестра бенчмарков
- `DYNAMIC_BATCHING_CHECKLIST.md` — чеклист реализации

**Расположение**: `DynamicBatch/`

---

### Models

**Описание**: Модели машинного обучения для предсказания популярности видео. Включает baseline модели, v1 трансформеры, контракты моделей, планы разработки и roadmap.

**Документация**: `Models/docs/`

**Ключевые разделы**:
- `contracts/` — контракты моделей (baseline, v1, v2)
- `plan_dev/` — планы разработки
- `roadmaps/` — roadmap развития
- `QA/` — вопросы и ответы по контрактам

**Расположение**: `Models/`

---

### Fetcher

**Описание**: Сервис для загрузки видео с платформ (YouTube) и предобработки. Отвечает за скачивание видео, извлечение метаданных, комментариев и данных канала.

**Документация**: `Fetcher/README.md`

**Расположение**: `Fetcher/`

---

## Структура документации

### DataProcessor
- **Архитектура**: контракты, правила оркестрации, обработка ошибок
- **Компоненты**: детальные описания всех экстракторов и модулей
- **Модели**: инвентаризация, планы, контракты semantic heads
- **Справочная**: Q&A, глобальные правила, stage_map, component_graph

### Backend
- **Обзор**: границы сервиса, потоки данных
- **API**: REST и WebSocket endpoints
- **База данных**: схема БД, хранилище
- **Операции**: конфигурация, деплой, зависимости

### DynamicBatch
- **Реализация**: план, чеклист, Q&A
- **Бенчмарки**: реестр, контракты, анализ

### Models
- **Контракты**: baseline, v1, v2 модели
- **Разработка**: планы, roadmap
- **QA**: вопросы и ответы

---

## Быстрая навигация

### Для начала работы
1. [DataProcessor/docs/MAIN_INDEX.md](../DataProcessor/docs/MAIN_INDEX.md) — понимание системы обработки данных
2. [backend/docs/MAIN_INDEX.md](../backend/docs/MAIN_INDEX.md) — понимание API и интеграции
3. [DataProcessor/docs/COMPONENTS_DESC_INDEX.md](../DataProcessor/docs/COMPONENTS_DESC_INDEX.md) — поиск конкретных компонентов

### Для разработки
1. [DataProcessor/docs/contracts/](../DataProcessor/docs/contracts/) — контракты и правила
2. [DataProcessor/docs/models_docs/](../DataProcessor/docs/models_docs/) — документация по моделям
3. [Models/docs/contracts/](../Models/docs/contracts/) — контракты ML-моделей

### Для архитектуры
1. [DataProcessor/docs/architecture/](../DataProcessor/docs/architecture/) — архитектура системы
2. [backend/docs/reference/backend_qna_contracts.md](../backend/docs/reference/backend_qna_contracts.md) — решения по backend
3. [DataProcessor/docs/reference/GLOBAL.md](../DataProcessor/docs/reference/GLOBAL.md) — глобальные Q&A

---

## Статистика документации

- **DataProcessor**: 60+ компонентов, 5000+ строк описаний
- **Backend**: 13 основных документов
- **DynamicBatch**: 7 документов
- **Models**: 18+ документов в contracts, plans, roadmaps

---

## Обновление индексов

При добавлении новой документации:
1. Обновите соответствующий `MAIN_INDEX.md` в подсистеме
2. При необходимости обновите этот главный индекс
3. Следуйте структуре и формату существующих индексов

---

## Связанные индексы

- **[DataProcessor/docs/MAIN_INDEX.md](../DataProcessor/docs/MAIN_INDEX.md)** — детальный индекс документации DataProcessor (архитектура, контракты, компоненты, модели, процессоры)
- **[backend/docs/MAIN_INDEX.md](../backend/docs/MAIN_INDEX.md)** — индекс документации Backend сервиса

---

