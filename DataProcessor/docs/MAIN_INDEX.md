# Главный индекс документации DataProcessor

Этот документ служит единой точкой входа для навигации по всей документации проекта. Каждый раздел содержит краткое описание документов и ссылки на полные версии.

**Старт (портфолио + prod):** [../README.md](../README.md) · [PORTFOLIO_INTERVIEW_GUIDE.md](PORTFOLIO_INTERVIEW_GUIDE.md) · [Итог сессии 2026-05-29](PORTFOLIO_SESSION_SUMMARY_2026-05-29.md)

**Нормализация:** [PORTFOLIO_NORMALIZATION_PLAN.md](PORTFOLIO_NORMALIZATION_PLAN.md) · [PORTFOLIO_PROGRESS_LOG.md](PORTFOLIO_PROGRESS_LOG.md)

**Карта корня:** [TOP_LEVEL_LAYOUT.md](TOP_LEVEL_LAYOUT.md) · **Ops:** [NORMALIZATION_WAVE5.md](NORMALIZATION_WAVE5.md)

**Главный индекс проекта**: [../../docs/MAIN_INDEX.md](../../docs/MAIN_INDEX.md) — навигация по всей документации TrendFlowML (DataProcessor, Backend, DynamicBatch, Models, Fetcher)

---

## Архитектура системы

### PRODUCTION_ARCHITECTURE.md
**Краткое описание**: Описывает продакшн-архитектуру MVP системы обработки видео/аудио. Включает описание сервисов (Backend API, Job Queue, DataProcessor worker, Triton, PostgreSQL, MinIO, LLM gateway), границы ответственности, схему БД, протоколы коммуникации между сервисами, стратегии батчинга и масштабирования, deployment, мониторинг, безопасность и версионирование моделей.

**Полный документ**: [docs/architecture/PRODUCTION_ARCHITECTURE.md](architecture/PRODUCTION_ARCHITECTURE.md)

### DATAPROCESSOR_API_ARCHITECTURE.md
**Краткое описание**: Подробная архитектура HTTP API для DataProcessor как отдельного сервиса. Содержит анализ текущей архитектуры, рекомендации по API (гибридный подход с HTTP API), архитектурные риски и предупреждения, эволюционный путь (MVP → Redis → Production), критически обязательные компоненты (Redis Streams, subprocess isolation, heartbeat, state machine, idempotency, backpressure), детальную спецификацию всех endpoints, технические детали реализации, Docker конфигурацию, интеграцию с backend (webhook + polling), безопасность, мониторинг, failure handling и план реализации по этапам.

**Полный документ**: [docs/DATAPROCESSOR_API_ARCHITECTURE.md](DATAPROCESSOR_API_ARCHITECTURE.md)

### API_DEVELOPMENT_CHECKLIST.md
**Краткое описание**: Подробный чеклист разработки API для DataProcessor с указанием ссылок на конкретные строки документа DATAPROCESSOR_API_ARCHITECTURE.md. Содержит разбивку по этапам (MVP, Redis + Worker, Мониторинг, Production-ready, Failure Handling, Backend Integration), конкретные задачи с ссылками на строки документа, критерии готовности, зависимости между задачами, приоритеты и метрики прогресса. Используется для отслеживания прогресса разработки.

**Полный документ**: [docs/API_DEVELOPMENT_CHECKLIST.md](API_DEVELOPMENT_CHECKLIST.md)

### BILLING_AND_PRICING.md
**Краткое описание**: Определяет правила биллинга, ценообразования и списания кредитов для компонентов обработки. Содержит комбинированную формулу расчета стоимости (base_cost + compute_time + gpu_time + markup), правила частичного списания, структуру прайс-листа компонентов, механизм оценки стоимости до запуска и калибровку цен на основе реальных метрик.

**Полный документ**: [docs/architecture/BILLING_AND_PRICING.md](architecture/BILLING_AND_PRICING.md)

### DATAPROCESSOR_API_ARCHITECTURE.md
**Краткое описание**: Финальные рекомендации по production-ready архитектуре HTTP API для DataProcessor. Содержит анализ текущей архитектуры, source of truth модель (Storage = source of truth, Redis = cache/queue), архитектурные риски и предупреждения, эволюционный путь (MVP → Redis Streams → Production), критически обязательные компоненты (Redis Streams queue, subprocess isolation, heartbeat + recovery, strict state machine, idempotency, backpressure), детальную спецификацию всех endpoints, технические детали реализации, Docker конфигурацию, интеграцию с backend (hybrid: webhook + polling fallback), безопасность, мониторинг, failure handling и recovery, план реализации по этапам. Оценка архитектуры: надёжность 9.5/10, масштабируемость 9/10, production readiness 9.5/10.

**Полный документ**: [docs/DATAPROCESSOR_API_ARCHITECTURE.md](DATAPROCESSOR_API_ARCHITECTURE.md)

### API_DEVELOPMENT_CHECKLIST.md
**Краткое описание**: Подробный чеклист разработки API для DataProcessor с указанием ссылок на конкретные строки документа DATAPROCESSOR_API_ARCHITECTURE.md. Содержит разбивку по этапам (MVP, Redis + Worker, Мониторинг, Production-ready, Failure Handling, Backend Integration), конкретные задачи с ссылками на строки документа, критерии готовности, зависимости между задачами, приоритеты и метрики прогресса. Используется для отслеживания прогресса разработки и гарантирует реализацию всех критически обязательных компонентов.

**Полный документ**: [docs/API_DEVELOPMENT_CHECKLIST.md](API_DEVELOPMENT_CHECKLIST.md)

---

## Контракты

### CONTRACTS_OVERVIEW.md
**Краткое описание**: Обзор всех контрактов системы DataProcessor. Содержит главные правила (NPZ как source-of-truth, no-fallback policy, Segmenter отвечает за sampling, empty outputs валидны, storage per-run, idempotency, targets, reproducibility), терминологию, текущий baseline execution path, структуру ResultStore и политику параллелизма.

**Полный документ**: [docs/contracts/CONTRACTS_OVERVIEW.md](contracts/CONTRACTS_OVERVIEW.md)

### ARTIFACTS_AND_SCHEMAS.md
**Краткое описание**: Определяет структуру артефактов и схем данных. Описывает NPZ как source-of-truth, структуру result_store per-run, формат manifest.json, обязательные meta-секции в NPZ, принятые schema_version для всех компонентов, стандарты для missing/nullable данных, каноничный словарь empty_reason, **общую систему схем** (human `SCHEMA.md` + machine schemas + runtime/CI validation), форматы артефактов для Audio/Text процессоров и контракт frames_dir для мультимодальной синхронизации.

**Полный документ**: [docs/contracts/ARTIFACTS_AND_SCHEMAS.md](contracts/ARTIFACTS_AND_SCHEMAS.md)

### SCHEMAS_SYSTEM.md
**Краткое описание**: Формальная спецификация **общей системы схем** (human+machine) для NPZ контрактов: формат `vp_schema_v1`, правила required/optional (ключи должны отсутствовать, а не быть `None`), политика `allow_extra_keys`, runtime/CI validation и требования Audit v3 (known schema для audited компонентов).

**Полный документ**: [docs/contracts/SCHEMAS_SYSTEM.md](contracts/SCHEMAS_SYSTEM.md)

### SEGMENTER_CONTRACT.md
**Краткое описание**: Контракт Segmenter — единственного источника frame_indices для всех компонентов. Определяет роль Segmenter в sampling, time-domain как source-of-truth для мультимодальной синхронизации, budgets per component, двухпроходность, структуру frames_dir (только union sampled кадры), параметры analysis_fps/resolution, цветовое пространство RGB, извлечение аудио и контракт audio/segments.json с универсальной кривой sampling для разных семейств экстракторов.

**Полный документ**: [docs/contracts/SEGMENTER_CONTRACT.md](contracts/SEGMENTER_CONTRACT.md)

### PRODUCT_CONTRACT.md
**Краткое описание**: Фиксирует продуктовые решения MVP → v1. Определяет поддерживаемые платформы (YouTube), варианты входа (URL/Upload), профили анализа и оценку стоимости, LLM как presentation layer, валидацию входных данных (длительность 5 сек - 20 мин, разрешение до 1080p), правила YouTube download и preprocessing (конвертация, downscale), временное хранилище.

**Полный документ**: [docs/contracts/PRODUCT_CONTRACT.md](contracts/PRODUCT_CONTRACT.md)

### ORCHESTRATION_AND_CACHING.md
**Краткое описание**: Правила оркестрации, DAG и кэширования. Определяет расположение orchestrator на уровне DataProcessor, режимы required vs optional компонентов, правила partial failures, idempotency ключ для кэширования, структуру задач в очереди, artifact index для быстрого поиска, политику кэша "последние 10k видео", наблюдаемость (timings, GPU/CPU mem, status), retention для frames_dir (7 дней).

**Полный документ**: [docs/contracts/ORCHESTRATION_AND_CACHING.md](contracts/ORCHESTRATION_AND_CACHING.md)

### ERROR_HANDLING_AND_EDGE_CASES.md
**Краткое описание**: Правила обработки ошибок, retry политики и edge cases. Определяет retry только для transient errors (network, timeout, OOM, Triton unavailable), fail-fast для missing dependencies и logic errors, специфичные retry политики для YouTube download/Triton/LLM gateway, OOM handling с автоматическим уменьшением batch_size, обработку edge cases (видео > 20 минут, повреждённые файлы через ffmpeg, видео без звука/кадров), таймауты (отложено до baseline).

**Полный документ**: [docs/contracts/ERROR_HANDLING_AND_EDGE_CASES.md](contracts/ERROR_HANDLING_AND_EDGE_CASES.md)

### PRIVACY_AND_RETENTION.md
**Краткое описание**: Правила приватности и хранения raw данных. Определяет общий принцип хранения только derived features, опции retain_raw_ocr_text и retain_raw_comments (по умолчанию false), требование OAuth-верификации владельца канала для raw storage, hard_cap_days=60 для raw OCR/comments, разделение рисков между OCR и комментариями, право на удаление данных по запросу, правила логирования без PII (raw тексты только в dev-режиме с флагом).

**Полный документ**: [docs/contracts/PRIVACY_AND_RETENTION.md](contracts/PRIVACY_AND_RETENTION.md)

### LLM_RENDERING.md
**Краткое описание**: Правила использования LLM как presentation layer. Определяет роль LLM (только текст, не source-of-truth), guardrails (числа/факты из render-context, явное указание недоступных данных), воспроизводимость через версионирование (llm_provider, llm_model, prompt_version, prompt_hash, locale), кэширование LLM-рендера отдельно от heavy compute, язык текста по locale пользователя (RU default).

**Полный документ**: [docs/contracts/LLM_RENDERING.md](contracts/LLM_RENDERING.md)

### PER_COMPONENT.md
**Краткое описание**: Критерии проверки и правила аудита компонентов. Содержит 13 критериев: отсутствие fallback и сомнительных эвристик, поддержка параллелизма/батчинга, интеграция Triton/Celery, обязательное использование ModelManager для всех ML-моделей, документация в README компонента (вход/зависимости/выход/sampling requirements), качество кода, соответствие контрактам, reproducibility/model system, state/manifest/observability, производительность и память, инфраструктурные зависимости, тестируемость, acceptance criteria.

**Полный документ**: [docs/contracts/PER_COMPONENT.md](contracts/PER_COMPONENT.md)

---

## Audit v3 (feature & logic audit)

Audit v3 — это финальный аудит всех компонентов DataProcessor, фокус на **контрактах/фичах/полезности**, а не на оптимизации.
Все решения и изменения в компонентах должны ссылаться на документы ниже.

### docs/audit_v3/AUDIT_3_DESC.md
**Краткое описание**: Короткое описание целей и процесса Audit v3 (scope, шаги, принципы).

**Полный документ**: [docs/audit_v3/AUDIT_3_DESC.md](audit_v3/AUDIT_3_DESC.md)

### docs/audit_v3/DECISIONS_AND_RULES.md
**Краткое описание**: Source-of-truth решений Audit v3 (NPZ/source-of-truth, no-fallback, model_facing vs analytics, sampling requirements, render policy, ModelManager-only, и т.д.).

**Полный документ**: [docs/audit_v3/DECISIONS_AND_RULES.md](audit_v3/DECISIONS_AND_RULES.md)

### docs/audit_v3/TEMPLATES.md
**Краткое описание**: Шаблоны для единообразной документации компонентов (README template, feature decision record).

**Полный документ**: [docs/audit_v3/TEMPLATES.md](audit_v3/TEMPLATES.md)

### docs/audit_v3/RUN_LOG.md
**Краткое описание**: Dev run-log валидационных прогонов (команды, конфиги, ссылки на артефакты/рендеры, заметки по фактическому sampling).

**Полный документ**: [docs/audit_v3/RUN_LOG.md](audit_v3/RUN_LOG.md)

### docs/audit_v3/VISUALPROCESSOR_ASSESSMENT_REPORT.md
**Краткое описание**: Полный отчёт по VisualProcessor (core+modules): соответствие контрактам `Models`, карта сигналов для Encoder (dense/events/embeddings), оценка полезности/корреляций/шума, пробелы текущей логики (tracking, anti-leakage для reference similarity, конфиг vs baseline contract), и рекомендуемые пресеты конфигурации.

**Полный документ**: [docs/audit_v3/VISUALPROCESSOR_ASSESSMENT_REPORT.md](audit_v3/VISUALPROCESSOR_ASSESSMENT_REPORT.md)

### docs/audit_v3/PR_PROD_READY_MODELS_AND_PROCESSORS.md
**Краткое описание**: “PR план” (крупные стадии) доведения **Models+Encoder+DataProcessor** до prod-ready по логике/фичам/контрактам. Фиксирует scope PR: финальные контракты моделей v2, закрывающий блок по VisualProcessor, и критерии предстоящего audit v3 для Audio/Text/Segmenter. Нужен, чтобы дальше дробить на отдельные планы разработки.

**Полный документ**: [docs/audit_v3/PR_PROD_READY_MODELS_AND_PROCESSORS.md](audit_v3/PR_PROD_READY_MODELS_AND_PROCESSORS.md)

### docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md
**Краткое описание**: Чеклист/критерии предстоящего audit v3 для `AudioProcessor`, `TextProcessor`, `Segmenter` под финальные модельные контракты v2 (`MODEL_INTERFACE_V2`). Фиксирует DoD по time-axis, sequences/tokens readiness, privacy и anti-leakage.

**Полный документ**: [docs/audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md](audit_v3/AUDIT_V3_CRITERIA_AUDIO_TEXT_SEGMENTER.md)

### docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md
**Краткое описание**: Preflight Audit v3 для **TextProcessor** (source-of-truth): расширенный smoke с качественными текстами, **обязательный ASR**, все **22** экстрактора и рекомендуемый порядок аудита, **единая модель эмбеддингов** `intfloat/multilingual-e5-large` через `dp_models`, политика **`tags_extractor`**, корпус/FAISS packs по ходу аудита, ссылки на каталог отчётов `TextProcessor/docs/audit_v3/components/`, дисциплина `RUN_LOG.md`. Ориентирован на LLM-исполнителя (навигация одним экраном).

**Полный документ**: [docs/audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md](audit_v3/TEXTPROCESSOR_AUDIT_V3_PREFLIGHT_RULES.md)

---

## Audit v4 (эмпирический аудит выходов NPZ)

Audit v4 дополняет v3: **статистика реальных артефактов** в `result_store`, каталог полей с описанием алгоритма в `extractors/<name>/docs/README.md`, итоговый вердикт о полезности для **Models** (tabular + encoder path) и аналитики.

### docs/audit_v4/AUDIT_4_CRITERIA_AND_PLAN.md
**Краткое описание**: Критерии (NaN/Inf/нули, распределения, временная ось, корреляции), validation set (e2e reference + diversity + edge), шаблон документации полей, engineering verdict, DoD, связь с `MODEL_INTERFACE_V2` / `ENCODER_CONTRACT` / `FEATURE_ENCODER_CONTRACT`.

**Полный документ**: [docs/audit_v4/AUDIT_4_CRITERIA_AND_PLAN.md](audit_v4/AUDIT_4_CRITERIA_AND_PLAN.md)

### docs/audit_v4/RUN_LOG.md
**Краткое описание**: Журнал run’ов и путей к артефактам для воспроизводимости отчётов Audit v4.

**Полный документ**: [docs/audit_v4/RUN_LOG.md](audit_v4/RUN_LOG.md)

### Audit v4.2 — engineering log (после L2)

**Краткое описание**: Дополнительные документы, которые фиксируют изменения кода и телеметрию **после** эмпирических отчётов L2 (профилирование, оптимизации, env-флаги), со ссылками на исходные отчёты и артефакты статистики. Индекс папки: [docs/audit_v4/components/audit_4_2/README.md](audit_v4/components/audit_4_2/README.md). Пример для `asr_extractor`: [docs/audit_v4/components/audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md](audit_v4/components/audit_4_2/audio_processor/asr_extractor_engineering_log_v4_2.md).

### Компоненты, уже затронутые Audit v3 (на текущий момент)

- `core_clip`: `core_clip_npz_v2` (ModelManager-only specs, backend-friendly proxy outputs)
  - Док: `VisualProcessor/core/model_process/core_clip/README.md`
  - Schema: `VisualProcessor/schemas/core_clip_npz_v2.json` + `VisualProcessor/core/model_process/core_clip/SCHEMA.md`
- `core_depth_midas`: `core_depth_midas_npz_v3` (ModelManager spec identity in `models_used[]`, backend-friendly proxies, legacy keys removed)
  - Док: `VisualProcessor/core/model_process/core_depth_midas/README.md`
  - Schema: `VisualProcessor/schemas/core_depth_midas_npz_v3.json` + `VisualProcessor/core/model_process/core_depth_midas/SCHEMA.md`
- `core_object_detections`: `core_object_detections_npz_v2` (tracking removed, normalized geometry + frame-level aggregates, meta_json)
  - Док: `VisualProcessor/core/model_process/core_object_detections/README.md`
  - Schema: `VisualProcessor/schemas/core_object_detections_npz_v2.json` + `VisualProcessor/core/model_process/core_object_detections/SCHEMA.md`
  - Dev run (smoke): `docs/audit_v3/RUN_LOG.md` → `youtube/audit3_cod_smoke_2/audit3_cod_smoke_2`
- `ocr_extractor`: `ocr_extractor_npz_v2` (engine `ppocr_rec_onnx`/`tesseract`, privacy flag `retain_raw_ocr_text`, meta_json)
  - Док: `VisualProcessor/core/model_process/ocr_extractor/README.md`
  - Schema: `VisualProcessor/schemas/ocr_extractor_npz_v2.json` + `VisualProcessor/core/model_process/ocr_extractor/SCHEMA.md`
  - Dev run (smoke): `docs/audit_v3/RUN_LOG.md` → `youtube/audit3_cod_ocr_smoke_1/audit3_cod_ocr_smoke_1`
- `core_face_landmarks`: `core_face_landmarks_npz_v2` (person-mask gating from `core_object_detections`, raw+filtered landmarks, QA-friendly render assets)
  - Док: `VisualProcessor/core/model_process/core_face_landmarks/README.md`
  - Schema: `VisualProcessor/schemas/core_face_landmarks_npz_v2.json` + `VisualProcessor/core/model_process/core_face_landmarks/SCHEMA.md`
  - Dev run (smoke): `docs/audit_v3/RUN_LOG.md` → `youtube/audit3_core_face_landmarks_smoke_4/audit3_core_face_landmarks_smoke_4`
- `brand_semantics`: `brand_semantics_npz_v2` (semantic-head v1 contract, deterministic label-space + `db_digest`, offline render + assets)
  - Док: `VisualProcessor/core/model_process/core_identity/brand_semantics/README.md`
  - Schema: `VisualProcessor/schemas/brand_semantics_npz_v2.json` + `VisualProcessor/core/model_process/core_identity/brand_semantics/SCHEMA.md`
  - Dev run (contract smoke): `docs/audit_v3/RUN_LOG.md` → `youtube/audit3_brand_semantics_smoke_1/audit3_brand_semantics_smoke_1`
- `car_semantics`: `car_semantics_npz_v2` (semantic-head v1 contract, deterministic label-space + `db_digest`, offline render + assets, K=5, per-detection output)
  - Док: `VisualProcessor/core/model_process/core_identity/car_semantics/README.md`
  - Schema: `VisualProcessor/schemas/car_semantics_npz_v2.json` + `VisualProcessor/core/model_process/core_identity/car_semantics/SCHEMA.md`
- `content_domain`: `content_domain_npz_v2` (semantic-head v1 contract, offline domain DB + `db_digest`, K=5, `meta_json`, offline render)
  - Док: `VisualProcessor/core/model_process/core_identity/content_domain/README.md`
  - Schema: `VisualProcessor/schemas/content_domain_npz_v2.json` + `VisualProcessor/core/model_process/core_identity/content_domain/SCHEMA.md`
    - Примечание: run демонстрирует корректный **fail-fast** при пустой категории `brand` в Embedding Service; для OK run нужен Triton+seed базы.
- `behavioral`: `behavioral_npz_v1` (behavior analysis module; strict time-axis; offline render без CDN; debug fields сохранены как debug/analytics)
  - Док: `VisualProcessor/modules/behavioral/README.md`
  - Schema: `VisualProcessor/schemas/behavioral_npz_v1.json` + `VisualProcessor/modules/behavioral/SCHEMA.md`
- `cut_detection`: `cut_detection_npz_v1` + `cut_detection_model_facing_npz_v1` (baseline required module; strict time-axis; model-facing NPZ required; offline render без CDN; core_face_landmarks/core_object_detections soft quality deps for jump-cuts)
  - Док: `VisualProcessor/modules/cut_detection/README.md`
  - Schemas:
    - `VisualProcessor/schemas/cut_detection_npz_v1.json` + `VisualProcessor/modules/cut_detection/SCHEMA.md`
    - `VisualProcessor/schemas/cut_detection_model_facing_npz_v1.json` + `VisualProcessor/modules/cut_detection/SCHEMA_MODEL_FACING.md`
- `scene_classification`: `scene_classification_npz_v2` (Places365 scene segmentation/classification; Segmenter-owned axis; hard deps core_clip+cut_detection; CLIP semantics strictly from core_clip; offline render без CDN; UI payload + stage timings in meta)
  - Док: `VisualProcessor/modules/scene_classification/README.md`
  - Schema: `VisualProcessor/schemas/scene_classification_npz_v2.json` + `VisualProcessor/modules/scene_classification/SCHEMA.md`
- `emotion_face`: `emotion_face_npz_v3` (EmoNet facial emotion; axis aligned to Segmenter; compute gated by face_present + internal face-frame sampling; top-level model-facing time-series arrays; no-network ModelManager; offline render без CDN)
  - Док: `VisualProcessor/modules/emotion_face/README.md`
  - Schema: `VisualProcessor/schemas/emotion_face_npz_v3.json` + `VisualProcessor/modules/emotion_face/SCHEMA.md`
- `micro_emotion`: `micro_emotion_npz_v3` (OpenFace micro-expressions/AU; Segmenter-owned axis + face-gating via core_face_landmarks; model-facing per-frame vectors + tabular scalar aggregates `feature_names/feature_values`; events stream; offline render без CDN)
  - Док: `VisualProcessor/modules/micro_emotion/README.md`
  - Schema: `VisualProcessor/schemas/micro_emotion_npz_v3.json` + `VisualProcessor/modules/micro_emotion/SCHEMA.md`
- `detalize_face`: `detalize_face_npz_v3` (face feature proxies; hard dep core_face_landmarks; axis aligned to Segmenter; compute gated by face_present + optional internal face sampling; model-facing compact features + aggregated tabular stats; offline render без CDN; UI payload в `meta.ui_payload`)
  - Док: `VisualProcessor/modules/detalize_face/README.md`
  - Schema: `VisualProcessor/schemas/detalize_face_npz_v3.json` + `VisualProcessor/modules/detalize_face/SCHEMA.md`
- `color_light`: `color_light_npz_v2` (color & lighting analysis; Segmenter-owned axis; hard dep `scene_classification`; model-facing `frame_compact_features (M,16)` + `aggregated` + `video_features`; offline render без CDN; `store_debug_objects` для отключения тяжёлых `frames/scenes`)
  - Док: `VisualProcessor/modules/color_light/README.md`
  - Schema: `VisualProcessor/schemas/color_light_npz_v2.json` + `VisualProcessor/modules/color_light/SCHEMA.md`

---

## Модели и ML-система

### README.md
**Краткое описание**: Индекс документации по моделям, используемым в DataProcessor. Содержит структуру документации, ссылки на канонические документы в `Models/docs/`, планы и roadmap, контракты и спецификации, гайды по сборке, информацию о ModelManager, baseline и инвентаризацию моделей, resource costs. Указывает на устаревшие файлы-стабы и быструю навигацию по темам.

**Полный документ**: [docs/models_docs/README.md](models_docs/README.md)

### MODEL_INVENTORY.md
**Краткое описание**: Инвентаризация моделей в текущей кодовой базе — где модели загружаются и используются. Аудит репозитория для стандартизации через единый ModelManager. Описывает модели в VisualProcessor (core providers и modules), AudioProcessor (ASR, CLAP, diarization, source separation), TextProcessor (embeddings, NLP), shared/infra компоненты. Фиксирует network risk для каждой модели и требования к ModelManager.

**Полный документ**: [docs/models_docs/MODEL_INVENTORY.md](models_docs/MODEL_INVENTORY.md)

### MODEL_MANAGER_PLAN.md
**Краткое описание**: План реализации единого ModelManager для всего DataProcessor (Visual/Audio/Text). Определяет архитектуру (ModelSpec, ResolvedModel, ModelProvider, ModelManager), расположение в `dp_models/`, стратегию no-network enforcement, план миграции по фазам (TextProcessor → AudioProcessor → Visual modules → Visual core), правила добавления новых моделей, стандартные error codes, тесты и CI enforcement.

**Полный документ**: [docs/models_docs/MODEL_MANAGER_PLAN.md](models_docs/MODEL_MANAGER_PLAN.md)

### MODEL_MANAGER_Q.md
**Краткое описание**: Q&A по ModelManager (Round 0-12, все решения). Вопросы и ответы по границам ответственности, источнику правды (mapping, run manifests, профили), канонической схеме, no-network enforcement, weights_digest, model signature и models_used[], device policy, preprocessing, API ModelManager, ошибкам и taxonomy, физическому расположению артефактов, лицензиям, MVP scope.

**Полный документ**: [docs/models_docs/MODEL_MANAGER_Q.md](models_docs/MODEL_MANAGER_Q.md)

### BASELINE_MODELS.md
**Краткое описание**: Фиксирует baseline набор моделей и разделяет их на GPU (Triton) и CPU/in-process категории. Содержит полный список baseline компонентов DataProcessor (19 компонентов: 7 visual modules + 3 audio extractors + 9 core providers), scope и решения, описание GPU моделей (CLIP, MiDaS/RAFT branches, Places365), CPU/in-process моделей (Places365, YOLO, MediaPipe), pre-Triton workflow и benchmark entrypoint.

**Полный документ**: [docs/models_docs/BASELINE_MODELS.md](models_docs/BASELINE_MODELS.md)

### BASELINE_GPU_BRANCHES.md
**Краткое описание**: План baseline GPU веток (fixed-shape → dynamic batching) + Triton. Определяет базовый контракт (image input UINT8 NHWC, text input INT64, dynamic batching), два измерения ветвления моделей (input-size ветки и сложность архитектуры), правило выбора ветки (routing), список baseline GPU моделей и веток (MiDaS, RAFT, YOLO, CLIP), Triton план для каждой модели.

**Полный документ**: [docs/models_docs/BASELINE_GPU_BRANCHES.md](models_docs/BASELINE_GPU_BRANCHES.md)

### PRETRITON_BENCH_AND_EXPORT.md
**Краткое описание**: Гайд по pre-Triton бенчмаркам и ONNX экспорту для baseline GPU моделей. Определяет контракты (image input UINT8 NHWC, фиксированные размеры, offline policy), структуру DP_MODELS_ROOT, инструкции по запуску pre-Triton bench, примеры результатов (latency/memory), ONNX I/O спецификации для MiDaS/RAFT, план экспорта в Triton model repository.

**Полный документ**: [docs/models_docs/PRETRITON_BENCH_AND_EXPORT.md](models_docs/PRETRITON_BENCH_AND_EXPORT.md)

### GPU_VS_CPU_PERFORMANCE.md
**Краткое описание**: Анализ производительности GPU vs CPU для эмбеддингов. Объясняет проблему overhead переноса данных для коротких текстов, причины (overhead CPU↔GPU, размер батча, размер модели, первый запуск), рекомендации когда использовать CPU vs GPU, оптимизацию для title_embedder, технические детали overhead компонентов, измерения для multilingual-e5-large, рекомендации для production.

**Полный документ**: [docs/models_docs/GPU_VS_CPU_PERFORMANCE.md](models_docs/GPU_VS_CPU_PERFORMANCE.md)

### EMBEDDING_UNIFICATION_STRATEGY.md
**Краткое описание**: Стратегия унификации моделей эмбеддингов на multilingual-e5-large. Анализирует плюсы унификации (единая размерность 1024, лучшая поддержка русского, единая модель в памяти, упрощение конфигурации, высокое качество), минусы (производительность, избыточность для коротких текстов), сравнение по use cases, рекомендацию для production, план миграции, альтернативный подход (e5-base), ожидаемые результаты.

**Полный документ**: [docs/models_docs/EMBEDDING_UNIFICATION_STRATEGY.md](models_docs/EMBEDDING_UNIFICATION_STRATEGY.md)

### EMBEDDING_MODELS_COMPARISON.md
**Краткое описание**: Сравнение моделей эмбеддингов для TextProcessor. Описывает текущие модели (all-MiniLM-L6-v2, multilingual-e5-large), их характеристики (размерность, размер, скорость, качество, многоязычность), плюсы и минусы, рекомендации по выбору модели для production с русским текстом vs быстрых baseline тестов, альтернативные модели (e5-base, paraphrase-multilingual-mpnet, LaBSE), рекомендации для проекта.

**Полный документ**: [docs/models_docs/EMBEDDING_MODELS_COMPARISON.md](models_docs/EMBEDDING_MODELS_COMPARISON.md)

### MODEL_LICENSES.md
**Краткое описание**: Шаблон инвентаризации используемых моделей и их лицензий для коммерческого продукта. Содержит таблицу для фиксации model_name, model_version/revision, source URL, license, commercial use OK, notes/constraints. Правило: нельзя добавлять модель в прод без строки в таблице, для HuggingFace моделей фиксировать revision (commit/tag), не "latest".

**Полный документ**: [docs/models_docs/MODEL_LICENSES.md](models_docs/MODEL_LICENSES.md)

### SEMANTIC_HEADS_CONTRACTS_QA.md
**Краткое описание**: Q&A по контрактам semantic heads (решения Round 1-3). Единственная точка правды для функционала вокруг core_object_detections и semantic heads (brands/car make+segment/face identity/scene+place). Содержит канонические инварианты проекта, визию архитектуры (proposal vs semantics), контрактные артефакты, требования к качеству, Q&A по общим вопросам, специфике head'ов, базам данных, thresholds, reproducibility.

**Полный документ**: [docs/models_docs/SEMANTIC_HEADS_CONTRACTS_QA.md](models_docs/SEMANTIC_HEADS_CONTRACTS_QA.md)

### SCHEMA_SEMANTIC_HEADS_NPZ.md
**Краткое описание**: Единый контракт схемы NPZ для semantic heads (brands/cars/places/face identity) v1. Определяет общие правила (time-axis source-of-truth, no gating by thresholds, fail-fast, NaN-policy), общие ключи NPZ (time-axis, label space, track-level output, frame-level output, meta), специфику каждого head'а (core_brand_semantics, core_car_semantics, core_place_semantics, core_face_identity).

**Полный документ**: [docs/models_docs/SCHEMA_SEMANTIC_HEADS_NPZ.md](models_docs/SCHEMA_SEMANTIC_HEADS_NPZ.md)

### SEMANTIC_BASES_BUILD_GUIDE.md
**Краткое описание**: Гайд по сборке offline баз для semantic heads (no-network). Описывает общие правила (no-network в runtime, reproducibility, stable IDs, English canon), формат пакетов, инструкции по сборке баз для brands (v1=500), cars (make/model/segment/body_type/price buckets), celebs (v1=500), places/landmarks (gallery retrieval), preflight проверки перед запуском пайплайна.

**Полный документ**: [docs/models_docs/SEMANTIC_BASES_BUILD_GUIDE.md](models_docs/SEMANTIC_BASES_BUILD_GUIDE.md)

### SEMANTIC_HEADS_IMPLEMENTATION_PLAN.md
**Краткое описание**: План реализации semantic heads (v1). Опирается на зафиксированные решения в SEMANTIC_HEADS_CONTRACTS_QA.md, DynamicBatching, ENCODER_CONTRACT, MODEL_SYSTEM_RULES. Определяет глобальные инварианты, этапы реализации: контракты и схемы (без моделей), offline базы + digest/versioning, core_brand_semantics MVP→v1, core_car_semantics, core_face_identity, core_place_semantics, валидация качества, тесты.

**Полный документ**: [docs/models_docs/SEMANTIC_HEADS_IMPLEMENTATION_PLAN.md](models_docs/SEMANTIC_HEADS_IMPLEMENTATION_PLAN.md)

### OBJECT_DETECTIONS_AND_SEMANTICS_ROADMAP.md
**Краткое описание**: Полный roadmap работ по object detections и semantic heads до production-готовности. Фиксирует контекст и цель (богатая семантика для popularity-модели), инварианты/ограничения (no-network, Triton batching, shared primary sampling group; baseline Audit v3: tracking removed), артефакты и контракт, план работ по этапам: истина таксономии, доведение core_object_detections до production-контракта, семантические head'ы, оптимизации и масштабирование.

**Полный документ**: [docs/models_docs/OBJECT_DETECTIONS_AND_SEMANTICS_ROADMAP.md](models_docs/OBJECT_DETECTIONS_AND_SEMANTICS_ROADMAP.md)

### YOLO_FINETUNE_PLAN_V1.md
**Краткое описание**: План дообучения YOLO детектора (41 класс, кадры из 100k видео, GPU Tesla T4 16GB). Фиксирует финальный набор классов v1.0, ключевую идею архитектуры (YOLO как proposal layer, семантика в отдельных head'ах), выбор bbox vs segmentation, таксономию v1 для YOLO (40 классов), принципы (объектный уровень, region-proposal классы, контекст/наличие), финальный список классов с описанием.

**Полный документ**: [docs/models_docs/YOLO_FINETUNE_PLAN_V1.md](models_docs/YOLO_FINETUNE_PLAN_V1.md)

### CONTENT_DOMAIN_AND_FRANCHISE_QA.md
**Краткое описание**: Q&A по реализации компонентов content_domain и franchise_recognition (working doc). Цель: определить домен контента (мультик/аниме/игра/скрин-рекординг/реал) и конкретную франшизу/тайтл. Содержит контекст (связанные документы, реализованные компоненты), предлагаемые 2 компонента, contract proposal в стиле semantic head, вопросы по product scope, baseline/deployment, алгоритмам, базам данных, контрактам.

**Полный документ**: [docs/models_docs/CONTENT_DOMAIN_AND_FRANCHISE_QA.md](models_docs/CONTENT_DOMAIN_AND_FRANCHISE_QA.md)

### dp_models/MAIN_INDEX.md
**Краткое описание**: Главный индекс документации модуля `dp_models` — единой системы управления моделями для DataProcessor. Содержит описание основных компонентов (ModelManager, ModelCatalog, ModelSpec, провайдеры моделей, фабрики), каталога спецификаций моделей (spec_catalog/audio, spec_catalog/text, spec_catalog/vision), бандла моделей (bundled_models с артефактами для offline режима), использования компонентами DataProcessor (AudioProcessor, TextProcessor, VisualProcessor extractors), API и паттернов использования (get_global_model_manager(), ModelManager.get(), ResolvedModel), связанной документации. Обеспечивает единый интерфейс для загрузки моделей (in-process и Triton), валидацию локальных артефактов, управление провайдерами, offline режим (no-network enforcement), device policy, thread-safe кэширование.

**Полный документ**: [dp_models/MAIN_INDEX.md](../dp_models/MAIN_INDEX.md)

---

## Справочная документация

### GLOBAL.md
**Краткое описание**: Глобальные Q&A по проекту TrendFlow (Round 1). Содержит вопросы и ответы по продукту, пользователям, UX (MVP пользовательский сценарий, поддерживаемые платформы, вход/выход продукта, LLM-рендер), репозиториям и границам систем (где живёт сайт/бекенд, целевая архитектура MVP, генерация run_id и config_hash, политика повторных запросов), контрактам данных и строгим правилам (no-fallback scope, NPZ vs JSON, структура storage, run identity, aliases, valid empty outputs), Segmenter/sampling/frames_dir (analysis timeline, retention), Audio/Text процессорам (fallbacks, тяжёлые зависимости).

**Полный документ**: [docs/reference/GLOBAL.md](reference/GLOBAL.md)

### project_questions.md
**Краткое описание**: Вопросы по проекту TrendFlow/DataProcessor для фиксации контрактов. Содержит вопросы по целям продукта и таргетам (целевая платформа, целевые таргеты, мульти-таргет, нормализация таргетов, горизонт прогнозирования, пользовательский кейс, метрики качества модели), ответы пользователя, рекомендации ChatGPT, фиксацию решений (абсолютный прогноз, multi-target views+likes, multi-horizon с дельтами), вопросы по данным и их структуре, рекомендации по архитектуре и реализации.

**Полный документ**: [docs/reference/project_questions.md](reference/project_questions.md)

### stage_map.yaml
**Краткое описание**: Машиночитаемый source-of-truth файл, описывающий стадии выполнения системы (Global Orchestrator → DynamicBatch → DataProcessor → Processors). Определяет какие стадии существуют и в каком порядке выполняются, для каждой стадии описывает входы/выходы и какие knobs контролируются scheduler'ом. Содержит уровни: global_orchestrator (ingest_request, schedule_and_execute), dynamicbatch (load_inputs, schedule, execute_runs), dataprocessor (segment, audio, text, visual), processors (component-level stages). Важно: это не runtime state, а декларативное описание структуры.

**Полный документ**: [docs/reference/stage_map.yaml](reference/stage_map.yaml)

### knobs_contract.yaml
**Краткое описание**: Контракт параметров, контролируемых scheduler'ом (source-of-truth). Определяет интерфейс между scheduler, orchestrators и компонентами. Описывает откуда берётся каждый knob (scheduler/backend/profile), как он распространяется (CLI flags/config fields), где пишется "actual report" для сравнения plan vs fact. Содержит разделы: reports (scheduler_runtime_report_v1), knobs по уровням (L1: Inter-video DynamicBatch, L3: Visual component-level, L4: Audio/Text component-level), propagation paths, reported_in поля. Цель: сделать интерфейс явным и машиночитаемым.

**Полный документ**: [docs/reference/knobs_contract.yaml](reference/knobs_contract.yaml)

### component_graph.yaml
**Краткое описание**: DAG зависимостей между компонентами/модулями на уровне DataProcessor (source-of-truth, MVP). Декларативное описание зависимостей для построения "priority" как dependency-ordering и определения параллелизма. Содержит stages (baseline), nodes с component_name, owner_processor, depends_on_components, soft_dependencies, wait_on_checkpoints. Описывает зависимости для Segmenter, Visual core providers (core_clip, content_domain, franchise_recognition, ocr_extractor, core_object_detections, core_depth_midas, core_optical_flow, core_face_landmarks), Visual modules, Audio extractors, Text extractors. Важно: это не runtime state, а декларативное описание "что зависит от чего".

**Полный документ**: [docs/reference/component_graph.yaml](reference/component_graph.yaml)

---

## Инфраструктура и утилиты

### main.py
**Краткое описание**: Главная точка входа (entry point) DataProcessor для обработки видео. CLI интерфейс для запуска всего пайплайна обработки: Segmenter → AudioProcessor → TextProcessor → VisualProcessor. Парсит аргументы командной строки (video_path, output, global_config, visual_cfg_path, profile_path, dag_path, platform_id, video_id, run_id, sampling_policy_version, analysis_fps/resolution, chunk_size, run_audio/text флаги), координирует выполнение компонентов через orchestrator, управляет временными файлами и cleanup. Используется backend'ом через Celery задачи (`dp_queue/tasks.py`) для асинхронной обработки видео, поддерживает single-file и batch режимы. Интегрирован с глобальной конфигурацией (`configs/global_config.yaml`), профилями анализа (`profiles/`), DAG зависимостей (`dag/component_graph.py`), системой состояния (`state/`), хранилищем результатов (`storage/`).

**Расположение**: `DataProcessor/main.py`

### common/meta_builder.py
**Краткое описание**: Общие утилиты для работы с метаданными моделей, используемые во всех процессорах (AudioProcessor, TextProcessor, VisualProcessor). Содержит функции для канонизации списка используемых моделей (`model_used()`), вычисления детерминированной подписи моделей (`compute_model_signature()`), применения метаданных моделей к мета-словарю (`apply_models_meta()`). Обеспечивает стабильную сортировку и детерминированное хеширование для reproducibility. Используется для фиксации информации о моделях в артефактах компонентов (models_used[], model_signature).

**Расположение**: `DataProcessor/common/meta_builder.py`

### configs/README.md
**Краткое описание**: Документация по системе глобальной конфигурации DataProcessor. Описывает единый YAML конфиг для всех процессоров (AudioProcessor, TextProcessor, VisualProcessor) и их компонентов. Содержит структуру конфига (глобальные настройки, настройки процессоров, extractors/modules), приоритет настроек (глобальный конфиг → profile → CLI), валидацию, примеры конфигурации для всех extractors и модулей, batch processing настройки, dependency resolution для AudioProcessor, обратную совместимость.

**Полный документ**: [configs/README.md](../configs/README.md)

### configs/config_parser.py
**Краткое описание**: Парсер глобального конфига для DataProcessor. Класс `GlobalConfigParser` читает единый YAML конфиг и генерирует CLI аргументы для всех процессоров. Методы: `get_processor_config()` для получения конфига процессора, `is_processor_enabled()`/`is_processor_required()` для проверки статуса, `get_audio_cli_args()` для генерации CLI аргументов AudioProcessor (extractors, parallelism, batch processing, feature flags), `get_text_cli_args()` для TextProcessor (extractors, devices_config, extractor_params, batch processing), `get_visual_inline_config()` для VisualProcessor, `validate()` для валидации конфига. Поддерживает индивидуальные настройки параллелизма для каждого extractor'а, batch processing конфигурацию, feature flags для всех компонентов.

**Расположение**: `DataProcessor/configs/config_parser.py`

### configs/global_config.yaml
**Краткое описание**: Единый глобальный конфиг для всех процессоров DataProcessor (source-of-truth для конфигурации). Содержит секции: `global` (platform_id, sampling_policy_version, scheduler knobs), `processors.audio` (extractors с параметрами, parallelism, batch_processing, feature_flags), `processors.text` (extractors с параметрами, devices_config, batch_processing, feature_flags), `processors.visual` (inline_config с core_providers и modules). Поддерживает индивидуальные настройки параллелизма для каждого extractor'а, batch processing оптимизации, feature flags для включения дополнительных фичей, render настройки (enable_render, enable_html_render).

**Расположение**: `DataProcessor/configs/global_config.yaml`

### dag/component_graph.py
**Краткое описание**: Реализация графа зависимостей компонентов DataProcessor (DAG). Класс `ComponentGraph` загружает декларативное описание зависимостей из YAML (`component_graph.yaml`), валидирует граф (уникальность имен, существование зависимостей, отсутствие циклов), вычисляет топологический порядок выполнения компонентов (`topo_order()`). Класс `GraphNode` описывает узел графа (component_name, owner_processor, depends_on_components, soft_dependencies, wait_on_checkpoints). Используется orchestrator'ом для определения порядка выполнения компонентов и параллелизма. Поддерживает subset execution (выполнение подмножества компонентов с автоматическим добавлением транзитивных зависимостей).

**Расположение**: `DataProcessor/dag/component_graph.py`

### docker/worker/Dockerfile
**Краткое описание**: Dockerfile для worker контейнера DataProcessor (минимальный образ для PR-0 bootstrap checks). Базовый образ: `python:3.11-slim`. Устанавливает минимальные зависимости для bootstrap (boto3, celery, fastapi, redis, requests, uvicorn) без полных ML pipeline зависимостей. Запускает `bootstrap.py` для проверки подключения к Redis и MinIO bucket перед запуском основного worker процесса. Используется для health checks и валидации инфраструктуры перед запуском тяжелых ML компонентов.

**Расположение**: `DataProcessor/docker/worker/Dockerfile`

### docker/worker/bootstrap.py
**Краткое описание**: Bootstrap скрипт для worker контейнера. Выполняет pre-flight проверки инфраструктуры перед запуском основного worker процесса. Проверяет подключение к Redis (через `CELERY_BROKER_URL`), доступность MinIO bucket (через `S3_ENDPOINT` и `S3_BUCKET`). Использует boto3 для проверки S3 bucket, redis для проверки broker. Выводит статус проверок и время выполнения. Используется как CMD в Dockerfile для health checks и валидации окружения.

**Расположение**: `DataProcessor/docker/worker/bootstrap.py`

### dp_queue/celery_app.py
**Краткое описание**: Конфигурация Celery приложения для DataProcessor. Создает Celery app с именем "dataprocessor", настраивает broker и backend через переменные окружения (`CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, по умолчанию Redis). Конфигурирует сериализацию (JSON), timezone (UTC), отслеживание начала задач (`task_track_started=True`), автодискавери задач из модуля `dp_queue`. Используется backend'ом для отправки задач на обработку видео в очередь, worker'ами для получения и выполнения задач.

**Расположение**: `DataProcessor/dp_queue/celery_app.py`

### dp_queue/tasks.py
**Краткое описание**: Celery задачи для обработки видео. Содержит задачу `process_video_job` (имя "dataprocessor.process_video_job") с автоматическими retry (max_retries=3, retry_backoff, retry_jitter) для transient errors (RuntimeError). Задача принимает payload (ProcessVideoPayload), запускает DataProcessor root `main.py` в subprocess с CLI аргументами из payload, логирует команду и результат, возвращает статус выполнения. MVP реализация через subprocess (в будущем планируется замена на native Python runner). Используется backend'ом для асинхронной обработки видео через очередь.

**Расположение**: `DataProcessor/dp_queue/tasks.py`

### dp_queue/payloads.py
**Краткое описание**: Структуры данных для payload Celery задач. Содержит `ProcessVideoPayload` (frozen dataclass) с обязательными полями (video_path, platform_id, video_id, run_id) и опциональными параметрами (rs_base, output, sampling_policy_version, dataprocessor_version, analysis_fps/resolution, chunk_size, visual_cfg_path, profile_path, dag_path, dag_stage, run_audio/text флаги и параметры). Методы: `from_dict()` для десериализации из JSON, `to_cli_args()` для генерации CLI аргументов для `main.py`. Интегрирован с root `main.py` CLI интерфейсом, поддерживает все параметры конфигурации DataProcessor.

**Расположение**: `DataProcessor/dp_queue/payloads.py`

### dp_queue/__init__.py
**Краткое описание**: Публичный API модуля `dp_queue`. Экспортирует `celery_app` для использования backend'ом и worker'ами. Используется для импорта Celery приложения в других модулях системы.

**Расположение**: `DataProcessor/dp_queue/__init__.py`

### dp_triton/http_client.py
**Краткое описание**: Минимальный HTTP клиент для Triton Inference Server (без внешних зависимостей, использует только стандартную библиотеку Python). Класс `TritonHttpClient` реализует Triton HTTP v2 API для выполнения inference запросов. Методы: `ready()` для проверки готовности сервера (health check), `infer()` для inference с одним входным тензором и одним выходом, `infer_multi()` для inference с несколькими выходными тензорами, `infer_two_inputs()` для моделей с двумя входными тензорами (например, optical flow: prev_frame + cur_frame). Поддерживает различные типы данных (FP32, FP16, INT8-INT64, UINT8-UINT64), автоматическую конвертацию numpy массивов в JSON payload, обработку ошибок через исключение `TritonError` с кодами ошибок (triton_error, triton_http_error, triton_unavailable, triton_bad_request, triton_bad_response). Используется компонентами DataProcessor (VisualProcessor, AudioProcessor, embedding_service) для выполнения inference через Triton сервер. Результат inference возвращается как `TritonInferResult` (dataclass с output_name, output numpy array, datatype).

**Расположение**: `DataProcessor/dp_triton/http_client.py`

### dp_triton/__init__.py
**Краткое описание**: Публичный API модуля `dp_triton`. Экспортирует `TritonHttpClient` и `TritonError` для использования в компонентах DataProcessor. Используется для импорта Triton клиента в других модулях системы (VisualProcessor, AudioProcessor, embedding_service).

**Расположение**: `DataProcessor/dp_triton/__init__.py`

### triton/
**Краткое описание**: Директория с Triton Inference Server model repositories для GPU inference. Содержит модели в формате Triton (ONNX и native), организованные по репозиториям для различных конфигураций и размеров входов. Основные модели: CLIP (image/text, размеры 224/336/448), MiDaS depth estimation (256/384/512), RAFT optical flow (256/384/512), Places365 scene classification (224/336/448). Каждая модель включает preprocessing pipelines (preprocess_*), конфигурационные файлы `config.pbtxt`, версионированные артефакты в поддиректориях `1/`. Репозитории организованы по назначению: `models/` (полный набор), `models_clip_image_text_*/` (CLIP только), `models_core_low/` (минимальный набор для baseline), `models_midas_*/`, `models_raft_*/`, `models_places365/`, `models_t_1/` (тестовый набор). Используется компонентами VisualProcessor через `dp_triton/http_client.py` для выполнения inference на GPU через Triton сервер. Запускается через Docker контейнер `nvcr.io/nvidia/tritonserver` с монтированием выбранного model repository.

**Расположение**: `DataProcessor/triton/`

**Структура**: `triton/{model_repo}/{model_name}/{version}/`, `triton/{model_repo}/{model_name}/config.pbtxt`

### triton/README.md
**Краткое описание**: Документация по запуску Triton Inference Server для DataProcessor. Содержит команду Docker для запуска Triton сервера с GPU поддержкой, монтированием model repository, настройкой портов (8000 HTTP, 8001 gRPC, 8002 metrics). Используется для развертывания GPU моделей (CLIP, MiDaS, RAFT, YOLO, Places365) в production окружении. Описывает базовую конфигурацию для запуска Triton с выбранным model repository из директории `triton/`.

**Полный документ**: [triton/README.md](../triton/README.md)

### dp_results/
**Краткое описание**: Локальная директория для хранения результатов обработки (result_store) в режиме разработки и тестирования. Структура соответствует контракту result_store: `dp_results/<platform_id>/<video_id>/<run_id>/`. Внутри каждого run'а: `manifest.json` (манифест run'а с метаданными), директории компонентов (asr_extractor, clap_extractor, core_clip, etc.) с NPZ артефактами, `_render/` (HTML/JSON рендеры для визуализации), `_logs/` (логи процессоров), `_reports/` (scheduler_runtime_report.json), `_tmp_audio/` (временные аудио файлы), `state/` (state events в формате JSONL). В production используется MinIO или другое объектное хранилище вместо локальной директории. Соответствует контракту ARTIFACTS_AND_SCHEMAS.md (NPZ как source-of-truth, per-run storage, manifest.json).

**Расположение**: `DataProcessor/dp_results/`

**Структура**: `dp_results/<platform_id>/<video_id>/<run_id>/<component_name>/*.npz`

### faiss_indices/
**Краткое описание**: Директория для хранения FAISS индексов для быстрого векторного поиска. Содержит файлы `.faiss` (FAISS индексы) и `*_ids.npy` (маппинг индексов к ID объектов). Используется embedding_service для быстрого поиска похожих эмбеддингов по категориям (face, brand, car, place и др.), TextProcessor extractors для поиска похожих заголовков и текстов, VisualProcessor для face identity поиска. Индексы создаются автоматически при добавлении объектов и сохраняются на диск для персистентности. Поддерживает различные типы индексов (IndexFlatIP для cosine similarity, IndexHNSWFlat для больших корпусов). Используется для оптимизации производительности векторного поиска вместо полного перебора.

**Расположение**: `DataProcessor/faiss_indices/`

**Структура**: `faiss_indices/<model_name>.faiss`, `faiss_indices/<model_name>_ids.npy`

### profiles/
**Краткое описание**: Директория с YAML конфигурациями профилей анализа. Профили определяют конфигурацию DataProcessor для различных сценариев использования: какие процессоры включены/отключены (audio, text, visual), ссылки на конфигурации визуальных компонентов (visual.cfg_path), параметры обработки. Backend использует эти профили для настройки DataProcessor при обработке видео, автоматически создает публичные профили из YAML файлов при старте. Профили могут быть публичными (is_public=true) или пользовательскими, хранятся в БД backend'а с вычислением config_hash для детерминированного сравнения. Используется для гибкой настройки пайплайна обработки под разные задачи.

**Расположение**: `DataProcessor/profiles/`

**Структура**: `profiles/*.yaml` (например, `config.yaml`, `demo.yaml`)

### state/
**Краткое описание**: Модуль управления состоянием выполнения обработки видео. Содержит систему многоуровневого управления состоянием: `RunStateManager` (Level-2) для агрегированного состояния всего run'а (`run_state.json`), `ProcessorStateManager` (Level-3) для состояния каждого процессора (`state_<processor>.json` для audio, text, visual, segmenter), `_JournalWriter` для append-only журнала событий (`state_events.jsonl`). Определяет статусы выполнения через enum `Status` (waiting, running, success, empty, error, skipped). Отслеживает прогресс обработки, тайминги (started_at, finished_at, duration_ms), ошибки (error, error_code), состояние компонентов (Level-4), метаданные run'а. Состояние хранится в storage (FS или S3) по пути `state/<platform_id>/<video_id>/<run_id>/`. Используется orchestrator'ом для мониторинга прогресса, обработки ошибок, восстановления после сбоев, предоставления наблюдаемости для UI/backend.

**Расположение**: `DataProcessor/state/`

**Основные компоненты**: `managers.py` (RunStateManager, ProcessorStateManager), `enums.py` (Status), `__init__.py` (публичный API)

**Структура хранения**: `state/<platform_id>/<video_id>/<run_id>/run_state.json`, `state_<processor>.json`, `state_events.jsonl`

### storage/MAIN_INDEX.md
**Краткое описание**: Главный индекс модуля Storage — абстракции для работы с хранилищем данных (файловая система или S3/MinIO). Содержит описание базовых интерфейсов и типов (Storage Protocol, StorageError, NotFoundError, ObjectInfo), реализаций хранилища (FileSystemStorage для локальной файловой системы, S3Storage для S3/MinIO), конфигурации и настроек (StorageSettings, load_storage_settings из переменных окружения, KeyLayout для канонической структуры ключей), публичного API модуля. Модуль обеспечивает единый интерфейс для работы с хранилищем независимо от выбранного бэкенда, используется для хранения result_store, state, frames_dir в production (S3/MinIO) и development (локальная файловая система).

**Полный документ**: [storage/MAIN_INDEX.md](../storage/MAIN_INDEX.md)

---

## Скрипты и утилиты

### scripts/MAIN_INDEX.md
**Краткое описание**: Главный индекс всех скриптов и утилит DataProcessor. Содержит описание утилит и тестов (dp_models_selftest, venv_doctor, storage_smoke_test), скриптов загрузки и сохранения моделей (Whisper, emotion diarization, source separation, sentence transformers, pyannote), baseline демо-скриптов для проверки качества extractors (CLAP, tempo, loudness, cut detection, scene classification, shot quality, similarity metrics, uniqueness, video pacing), скриптов оптимизации моделей и ONNX экспорта (CLIP, MiDaS, RAFT, YOLO, Places365, MediaPipe, квантование, оптимизация), preflight проверок (check_semantic_bases), скриптов построения семантических кластеров и таксономии тем. Используется для навигации по всем вспомогательным скриптам проекта.

**Полный документ**: [scripts/MAIN_INDEX.md](../scripts/MAIN_INDEX.md)

---

## Сервисы

### embedding_service/MAIN_INDEX.md
**Краткое описание**: Главный индекс документации Embedding Service — единого сервиса для управления эмбеддингами разных категорий. Содержит описание документации (README.md, QUICK_START.md, SETUP.md), основных модулей (FastAPI API, конфигурация, embedding_manager, обработка ошибок), базы данных и индексов (PostgreSQL с pgvector, FAISS индексы), извлечения эмбеддингов (базовый класс, ArcFace для лиц, CLIP для изображений, фабрика extractors), менеджеров категорий (базовый менеджер, менеджеры для face/brand/car/place, фабрика менеджеров), утилит и скриптов (run_server.py, check_setup.py), конфигурации и зависимостей (requirements.txt, requirements-dev.txt), структуры данных (faiss_indices), инициализации модулей. Embedding Service предоставляет REST API для добавления объектов, поиска похожих, извлечения embedding, управления категориями (face, brand, car, place, franchise и др.), использует PostgreSQL с pgvector для хранения и FAISS для быстрого векторного поиска.

**Полный документ**: [embedding_service/MAIN_INDEX.md](../embedding_service/MAIN_INDEX.md)

---

## Процессоры

### Segmenter/README.md
**Краткое описание**: Документация Segmenter — компонента подготовки фреймов, аудио и метаданных для всех экстракторов. Segmenter является единственным источником frame_indices для всех компонентов системы. Описывает контракт входа (видео файл), контракт выхода (frames_dir с union-sampled кадрами в RGB, audio/audio.wav + audio/segments.json, metadata.json с frame_indices per-component), time-axis как source-of-truth для мультимодальной синхронизации (union_timestamps_sec), структуру frames_dir (только union sampled кадры, не все кадры видео), параметры analysis_fps/resolution, цветовое пространство RGB, извлечение аудио через ffmpeg. Определяет роль Segmenter в sampling (автоматическая генерация per-component budgets на основе VisualProcessor/config.yaml), двухпроходность, контракт audio/segments.json с универсальной кривой sampling. Используется как первый этап обработки перед запуском AudioProcessor, TextProcessor и VisualProcessor.

**Полный документ**: [Segmenter/README.md](../Segmenter/README.md)

### Segmenter/segmenter.py
**Краткое описание**: Реализация Segmenter — компонента для подготовки фреймов, аудио и метаданных. Класс Segmenter выполняет: `process_video()` — сохраняет фреймы батчами (batch_{id}.npy) и возвращает metadata.json, `extract_audio()` — извлекает аудио через ffmpeg в WAV формат, собирает метаданные (duration, sample_rate, total_samples), `create_extractor_metadata()` — формирует per-extractor метаданные (frame_indices для video, сегменты в ms и сэмплах для audio). Поддерживает union-sampled frames_dir (только кадры, требуемые компонентами), legacy режим полного извлечения кадров, обработку ошибок через SegmenterSkip. Требует opencv (cv2), numpy, ffmpeg/ffprobe CLI. Используется как entry point для подготовки данных перед обработкой экстракторами.

**Расположение**: `DataProcessor/Segmenter/segmenter.py`

### AudioProcessor/README.md
**Краткое описание**: Основная документация AudioProcessor — процессора аудио модальности. Описывает контракт входа (audio/audio.wav + audio/segments.json от Segmenter), контракт выхода (NPZ артефакты в per-run result_store), запуск через CLI (single-file и batch mode), конфигурацию через global_config.yaml, используемые модели (CLAP, Whisper, Speaker Diarization, Emotion Diarization, Source Separation), sampling requirements, параллелизм (внутренний и batch processing), архитектуру проекта (модульная структура, основные компоненты), orchestrator (error handling, progress reporting, stage timings), виртуальные extractors, batch processing (Stage 5).

**Полный документ**: [AudioProcessor/README.md](../AudioProcessor/README.md)

### AudioProcessor/docs/MAIN_INDEX.md
**Краткое описание**: Индекс документации всех extractors AudioProcessor. Содержит краткие описания и ссылки на README для всех 21 extractor'а: Tier-0 baseline (clap, tempo, loudness, asr) и Tier-1 optional extractors (speaker_diarization, emotion_diarization, source_separation, speech_analysis, spectral, quality, mfcc, mel, onset, chroma, rhythmic, key, band_energy, spectral_entropy, hpss, voice_quality, pitch). Каждое описание включает назначение, версию, требования к GPU/CPU, категорию.

**Полный документ**: [AudioProcessor/docs/MAIN_INDEX.md](../AudioProcessor/docs/MAIN_INDEX.md)

### AudioProcessor/docs/BATCH_PROCESSING_PLAN.md
**Краткое описание**: План реализации batch processing для AudioProcessor (Stage 5). Описывает двухуровневую параллельность (видео + сегменты), GPU batching для ML-моделей, CPU parallelism для signal processing extractors, изоляцию данных, валидацию, этапы реализации (Stage 0-5), примеры использования, производительность и оптимизации.

**Полный документ**: [AudioProcessor/docs/BATCH_PROCESSING_PLAN.md](../AudioProcessor/docs/BATCH_PROCESSING_PLAN.md)

### TextProcessor/docs/MAIN_INDEX.md
**Краткое описание**: Главный индекс документации TextProcessor. Содержит краткие описания и ссылки на README для всех 22 extractor'ов, организованных по уровням зависимостей (Tier-0: baseline независимые, Tier-1: embedding extractors, Tier-2: aggregation extractors, Tier-3: advanced metrics extractors). Описывает документацию (BATCH_PROCESSING_PLAN.md, LAST_FULL_RUN_LOG.md, TOOLS.md), архитектуру и core модули (MainProcessor, BaseExtractor, model_registry, metrics, utilities), схемы данных, конфигурацию, скрипты, CLI entry points, структуру проекта и интеграцию с DataProcessor. Включает статистику: 22 extractors (7 GPU, 15 CPU-only).

**Полный документ**: [TextProcessor/docs/MAIN_INDEX.md](../TextProcessor/docs/MAIN_INDEX.md)

### TextProcessor/docs/BATCH_PROCESSING_PLAN.md
**Краткое описание**: План адаптации TextProcessor для батчевой обработки. Описывает двухуровневую параллельность (видео + сегменты), GPU batching для ML-моделей (embedding extractors), CPU parallelism для независимых extractors, изоляцию данных, валидацию, этапы реализации (Stage 0-5), примеры использования, производительность и оптимизации. Статус: все стадии завершены (Stage 0-5), batch processing полностью интегрирован в CLI и готов к production использованию.

**Полный документ**: [TextProcessor/docs/BATCH_PROCESSING_PLAN.md](../TextProcessor/docs/BATCH_PROCESSING_PLAN.md)

### VisualProcessor/README.md
**Краткое описание**: Основная документация VisualProcessor — процессора визуальной модальности. Описывает контракт входа (frames_dir с кадрами видео + metadata.json от Segmenter), контракт выхода (NPZ артефакты в per-run result_store), запуск через CLI (single-file и batch mode), конфигурацию через global_config.yaml, архитектуру компонентов (core components и modules), используемые модели (CLIP, YOLO, RAFT, MiDaS, MediaPipe, Tesseract), sampling requirements, параллелизм (внутренний и batch processing), render system для визуализации результатов, профилирование и мониторинг. VisualProcessor содержит 12 core components (6 Tier-0 независимых + 6 Tier-1 semantic heads) и 17 modules для анализа визуальных признаков.

**Полный документ**: [VisualProcessor/README.md](../VisualProcessor/README.md)

### VisualProcessor/docs/MAIN_INDEX.md
**Краткое описание**: Главный индекс документации VisualProcessor. Содержит краткие описания и ссылки на README для всех 12 core components (Tier-0: core_clip, core_object_detections, core_optical_flow, core_depth_midas, core_face_landmarks, ocr_extractor; Tier-1: brand_semantics, car_semantics, face_identity, place_semantics, content_domain, franchise_recognition) и 17 modules (Tier-0: cut_detection, shot_quality, video_pacing, scene_classification; Tier-1: story_structure, emotion_face, detalize_face, behavioral, action_recognition, color_light, frames_composition, similarity_metrics, uniqueness, text_scoring, high_level_semantic, micro_emotion, optical_flow). Описывает документацию (BATCH_PROCESSING_PLAN.md, LAST_FULL_RUN_LOG.md), архитектуру и core модули (main.py, base_module.py, frame_manager, video_context, utils), batch processing utilities, дополнительную документацию (README_RUN_ALL_CORE.md, REQUIREMENTS.md, BRAND_DATABASE_GUIDE.md, SCHEMA_MODEL_FACING.md), FEATURES_DESCRIPTION.md файлы, CLI entry points, структуру проекта и интеграцию с DataProcessor. Включает статистику: 12 core components (6 Tier-0, 6 Tier-1), 17 modules (4 Tier-0, 13 Tier-1), 12 GPU components, 17 CPU-only components.

**Полный документ**: [VisualProcessor/docs/MAIN_INDEX.md](../VisualProcessor/docs/MAIN_INDEX.md)

### VisualProcessor/docs/BATCH_PROCESSING_PLAN.md
**Краткое описание**: План адаптации VisualProcessor для батчевой обработки. Описывает двухуровневую параллельность (видео + кадры), GPU batching для ML-моделей (CLIP, object detection, optical flow, depth, face landmarks, identity), CPU parallelism для независимых компонентов, изоляцию данных, валидацию NPZ файлов, этапы реализации (Stage 0-5), примеры использования, производительность и оптимизации. Статус: Stage 0-2 завершены (базовый каркас, изоляция артефактов, GPU batching для core_clip), Stage 3-5 в разработке.

**Полный документ**: [VisualProcessor/docs/BATCH_PROCESSING_PLAN.md](../VisualProcessor/docs/BATCH_PROCESSING_PLAN.md)

---

## Описания компонентов

### COMPONENTS_DESC.md
**Краткое описание**: Подробное описание всех компонентов DataProcessor (5261 строка). Содержит детальные описания AudioProcessor (структура модулей, API, Core, Schemas, Segment Policy, взаимосвязи), TextProcessor, VisualProcessor и всех их extractors/modules (60 компонентов всего: 21 AudioProcessor extractor, 22 TextProcessor extractor, 12 VisualProcessor core modules, 18 VisualProcessor modules). Для каждого компонента описывает: краткое описание, версию, категорию, извлекаемые фичи (основные, агрегаты, статистики), зависимости между фичами, upstream/downstream зависимости, взаимосвязи с модулями системы, алгоритмы, параметры конфигурации, форматы артефактов.

**Полный документ**: [docs/COMPONENTS_DESC.md](COMPONENTS_DESC.md)

### COMPONENTS_DESC_INDEX.md
**Краткое описание**: Индекс для быстрого поиска компонентов в `COMPONENTS_DESC.md`. Содержит номера строк для всех компонентов: основные процессоры (AudioProcessor, TextProcessor, VisualProcessor), AudioProcessor Extractors (21 компонент), TextProcessor Extractors (22 компонента), VisualProcessor Core Modules (12 компонентов), VisualProcessor Modules (18 компонентов). Включает статистику: всего 60 компонентов, общее количество строк в документе 5261.

**Полный документ**: [docs/COMPONENTS_DESC_INDEX.md](COMPONENTS_DESC_INDEX.md)

---