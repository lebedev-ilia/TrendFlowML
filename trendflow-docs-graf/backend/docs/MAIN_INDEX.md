# Главный индекс документации Backend

Этот документ служит единой точкой входа для навигации по всей документации backend-сервиса. Каждый раздел содержит краткое описание документов и ссылки на полные версии.

**Быстрый старт (Docker, порты, тесты):** [README.md](../README.md) в каталоге `backend/`.

**Демонстрация и портфолио (чеклист показа, карта документов, Q&A):** [DEMO_AND_PORTFOLIO.md](DEMO_AND_PORTFOLIO.md).

---

## Обзор и архитектура

### OVERVIEW.md
**Краткое описание**: Описывает границы backend-сервиса, ключевые решения и потоки данных. Определяет ответственность backend (REST API, оркестрация DataProcessor через Celery, индекс в PostgreSQL `core.*`, WebSocket-стрим событий, хранение файлов), потоки данных (Workspace → Video → **AnalysisJob**, IngestionRun → Fetcher → DataProcessor, Progress → UI), source-of-truth (manifest.json и NPZ артефакты), расположение кода в репозитории; уточняет наличие **legacy**-слоя БД в `public` параллельно с `core.*`.

**Полный документ**: [docs/OVERVIEW.md](OVERVIEW.md)

---

## Конфигурация и окружение

### CONFIGURATION.md
**Краткое описание**: Описывает переменные окружения и разрешение путей. Определяет обязательные переменные (TF_BACKEND_DB_DSN, TF_BACKEND_REDIS_URL, TF_BACKEND_JWT_SECRET, TF_BACKEND_ADMIN_EMAILS), **TF_BACKEND_CORS_ORIGINS**, **TF_BACKEND_DB_AUTO_CREATE**, пути хранилища, интеграцию с DataProcessor (TF_BACKEND_DATAPROC_ROOT, TF_BACKEND_VISUAL_CFG_DEFAULT) и **Fetcher** (TF_BACKEND_FETCHER_*, TF_BACKEND_RUN_TRIGGER_API_KEY, ingestion sync), правила разрешения путей относительно корня репозитория.

**Полный документ**: [docs/CONFIGURATION.md](CONFIGURATION.md)

---

## Хранилище и файловая система

### STORAGE_LAYOUT.md
**Краткое описание**: Описывает структуру хранилища на локальном filesystem. Определяет директории по умолчанию (storage/raw, storage/frames_dir, storage/result_store, example/example_videos, storage/profiles_cache, storage/state), структуру raw uploads (storage/raw/<video_id>/video.<ext>), result store как source-of-truth (manifest.json и NPZ артефакты), frames_dir для quality-скриптов, state_events.jsonl для прогресса.

**Полный документ**: [docs/STORAGE_LAYOUT.md](STORAGE_LAYOUT.md)

---

## База данных

### DATABASE.md
**Краткое описание**: Описывает доменную схему **`core.*`** по `app/dbv2/models.py` (users, oauth, security, workspaces, members, subscriptions, channels, videos, analysis_jobs, **ingestion_runs**, predictions и др.). Уточняет параллельный **legacy**-слой в `public`, который по-прежнему может сопровождаться Alembic; ENUM, индексы, миграции; пробелы относительно каноничных контрактов.

**Полный документ**: [docs/DATABASE.md](DATABASE.md)

### DATABASE_ARCH.md
**Краткое описание**: Полная спецификация базы данных для Core SaaS (PostgreSQL, Multi-Tenant, Enterprise-Ready). Содержит детальное описание всех таблиц, включая дополнительные таблицы (subscription_usage, billing_transactions, video_snapshots, video_comments, processing_configs, explainability_summary, recommendations, model_serving_log, api_keys, api_usage_logs, audit_logs, channel_owners), архитектурную иерархию и multi-tenant правила.

**Полный документ**: [docs/DATABASE_ARCH.md](DATABASE_ARCH.md)

---

## API и интерфейсы

### API.md
**Краткое описание**: Описывает REST и WebSocket endpoints. Определяет auth, workspaces, channels, videos, **analysis jobs** (в т.ч. создание с **HTTP 201**), **ingestion runs** (`POST/GET /api/runs`, trigger-processing, `WS .../events`), predictions, subscriptions, webhooks; правила авторизации (JWT для большей части REST; исключения и риски WS — в SECURITY/GAPS).

**Полный документ**: [docs/API.md](API.md)

---

## Загрузка и обработка видео

### UPLOADS_AND_VIDEOS.md
**Краткое описание**: Описывает upload flow для пользовательских видео и хранение метаданных. Определяет upload flow (init → upload → complete), дедуп через video_files.sha256_hex, метаданные через ffprobe (duration_sec, width, height), генерацию video_id для upload-видео (UUID backend'ом, пользователь не задаёт).

**Полный документ**: [docs/UPLOADS_AND_VIDEOS.md](UPLOADS_AND_VIDEOS.md)

---

## Профили анализа

### PROFILES.md
**Краткое описание**: Описывает профили анализа как JSON-конфигурации DataProcessor. Определяет нормализацию профиля (подстановка дефолтного visual.cfg_path, добавление processors если отсутствуют), вычисление config_hash (sha256 от JSON с сортировкой ключей), публичные профили (seed из DataProcessor/profiles/*.yaml), API endpoints для работы с профилями.

**Полный документ**: [docs/PROFILES.md](PROFILES.md)

---

## Runs и воркеры

### RUNS_AND_WORKERS.md
**Краткое описание**: Celery-задачи **`process_analysis_job`**, **`process_ingestion_run`**, **`sync_ingestion_run_status`** (beat); связь с **AnalysisJob** и **IngestionRun**; Fetcher и DataProcessor; прогресс (`state_events.jsonl`), качество отчётов; команды `celery -A app.worker:celery_app worker|beat` и заметка про отсутствие beat в `docker-compose` по умолчанию.

**Полный документ**: [docs/RUNS_AND_WORKERS.md](RUNS_AND_WORKERS.md)

---

## События и логирование

### EVENTS_AND_LOGGING.md
**Краткое описание**: Описывает WebSocket события и систему логирования. Определяет WebSocket endpoint (GET /api/runs/{run_id}/events, live-only поток), формат payload событий (run.status_changed, run.stage_changed, component.started, component.finished, log.line), Redis pubsub механику (publish_run_event, subscribe_run_events), логи DataProcessor (stdout/stderr → run_logs, live через WS, история через REST).

**Полный документ**: [docs/EVENTS_AND_LOGGING.md](EVENTS_AND_LOGGING.md)

---

## Безопасность

### SECURITY.md
**Краткое описание**: Auth через JWT (`deps.get_current_user`), admin (роль или `TF_BACKEND_ADMIN_EMAILS`), WebSocket ingestion run через **`?token=`** (см. `SECURITY.md` §3), артефакты по query `token`, **CORS через `TF_BACKEND_CORS_ORIGINS`**, **JWT secret в prod** (`TF_BACKEND_DEPLOYMENT_ENV`, `SECURITY.md` §6), **сервисные ключи** (`TF_BACKEND_RUN_TRIGGER_API_KEY`, webhooks DataProcessor).

**Полный документ**: [docs/SECURITY.md](SECURITY.md)

---

## Операции и деплой

### OPERATIONS.md
**Краткое описание**: Зависимости; **Docker Compose** (`backend/docker-compose.yml` — Postgres, Redis, api с `alembic upgrade head`, worker; узкий build context); локальный запуск API (`uvicorn`), **startup** (ensure_dirs, условный `create_all` для v2, seed профилей); Celery `worker` и **beat** (`app.worker:celery_app`); **health**; `.env` / **CONFIGURATION.md**.

**Полный документ**: [docs/OPERATIONS.md](OPERATIONS.md)

---

## Расхождения с каноничными контрактами

### GAPS_AND_ALIGNMENT.md
**Краткое описание**: Фиксирует расхождения между каноничными контрактами и реализацией backend. После Phases 0–5: YouTube-поток, idempotency `POST /api/runs`, отмена analysis (`POST /api/analysis/.../cancel` + DP cancel), health, JWT на WS. Идempotency для **`upload/complete`** — контракт в `UPLOADS_AND_VIDEOS.md` (HTTP upload в приложении может отсутствовать). Остаются billing, delete/retention, validation и др.

**Полный документ**: [docs/GAPS_AND_ALIGNMENT.md](GAPS_AND_ALIGNMENT.md)

---

## Интеграция Backend ↔ Fetcher (YouTube ingestion)

### FETCHER_INTEGRATION.md
**Краткое описание**: Настройка и использование интеграции с Fetcher (Phases 0–5). Конфигурация (TF_BACKEND_FETCHER_*), HTTP-клиент (create_run, get_run, manifest, artifacts), API runs (POST/GET /api/runs, trigger-processing, WebSocket events), синхронизация статуса (sync_ingestion_run_status, Celery beat), пути к артефактам и video_url (Phase 3).

**Полный документ**: [docs/FETCHER_INTEGRATION.md](FETCHER_INTEGRATION.md)

**E2E чеклист** (в корне репозитория): [docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md](../../docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md) — ручная проверка цикла YouTube URL → результат. Анализ и план фаз: [docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md](../../docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md).

**E2E без TextProcessor** (Backend → Fetcher → Segmenter + Audio + Visual, указатель скриптов): [E2E_PIPELINE_NO_TEXT.md](E2E_PIPELINE_NO_TEXT.md).

---

## Тестирование

### TESTING.md
**Краткое описание**: Описывает текущее состояние тестов backend. Определяет реализованные тесты (интеграция Backend ↔ DataProcessor: клиент API, адаптер v2→legacy, контракт payload; webhook в т.ч. **cancelled**; **идемпотентность POST /api/runs**), фикстуры и конфигурацию pytest, **Ruff** (`pyproject.toml`), команды запуска совпадающие с CI (**`ruff check app`**, pytest + **coverage.xml**), симметричные контрактные тесты в DataProcessor.

**Полный документ**: [docs/TESTING.md](TESTING.md)

### TESTING_PLAN.md
**Краткое описание**: План и чеклист тестирования backend. Определяет уровни тестов (unit, integration, contract, E2E), чеклисты по областям (интеграция с DataProcessor, REST API, БД, события/WebSocket, конфигурация, безопасность, профили, хранилище, CI), приоритеты реализации и команды запуска.

**Полный документ**: [docs/TESTING_PLAN.md](TESTING_PLAN.md)

---

## Справочная документация

### backend_qna_contracts.md
**Краткое описание**: Интерактивное интервью для фиксации backend-архитектуры TrendFlow. Содержит историю решений через раунды вопросов-ответов (Round 1-6), зафиксированные решения по deployment (Kubernetes), auth (email+password + OAuth), billing (единицы, hold, charge, refund), профилям анализа, запуску анализа (Runs), очередям и воркерам (Celery + Redis), прогрессу и логам, результатам, кэшированию, privacy/удалению, DB schema (DDL-черновик), REST API contract, WebSocket event spec, Fetcher/DataProcessor orchestration. Служит источником правды для всех архитектурных решений backend.

**Полный документ**: [docs/reference/backend_qna_contracts.md](reference/backend_qna_contracts.md)

### DATAPROCESSOR_CONTRACT.md
**Краткое описание**: Описывает контракт между Backend API и DataProcessor. Legacy Run/`process_run` vs **V2 AnalysisJob** / ingestion (`process_analysis_job`, `process_ingestion_run`), адаптер v2 → legacy payload, обработка `manifest.json` → AnalysisJob/Prediction, примеры вызовов.

**Полный документ**: [docs/reference/DATAPROCESSOR_CONTRACT.md](reference/DATAPROCESSOR_CONTRACT.md)

---

## Миграция и развитие

### MIGRATION_PLAN.md
**Краткое описание**: Описывает завершённую миграцию существующих `/api/*` endpoints на новую доменную модель (`core.*` schema). Определяет результат миграции (единственный API на core.*), выполненные этапы миграции, маппинг сущностей Legacy → V2, контракты с DataProcessor, архитектуру после миграции и следующие шаги (опционально).

**Полный документ**: [docs/MIGRATION_PLAN.md](MIGRATION_PLAN.md)

### Architecture Decision Records (ADR)

**Краткое описание**: Зафиксированные архитектурные решения (контекст, выбор, последствия). ADR 0001: **Celery** на **Redis** как брокер/backend, **Redis pub/sub** канал `run:{run_id}` для событий и WebSocket, **`manifest.json` и артефакты на диске** как source of truth результатов; БД — индекс.

**Индекс**: [docs/adr/README.md](adr/README.md) · **ADR 0001**: [docs/adr/0001-celery-redis-pubsub-manifest-source-of-truth.md](adr/0001-celery-redis-pubsub-manifest-source-of-truth.md)

---

## Портфолио и готовность к выносу в отдельный репозиторий

### DEMO_AND_PORTFOLIO.md
**Краткое описание**: **Итоговый гайд для демонстрации работодателю:** питч за 60 секунд, пошаговый чеклист демо (Docker, URL, что не входит в compose), таблица «вопрос → документ», честный блок ограничений, шпаргалка Q&A для собеседования.

**Полный документ**: [docs/DEMO_AND_PORTFOLIO.md](DEMO_AND_PORTFOLIO.md)

### PORTFOLIO_READINESS_CHECKLIST.md
**Краткое описание**: Подробный чеклист (репозиторий, безопасность, операционка, код, тесты, CI, документация, подготовка к собеседованиям) с отметками статуса. Точка входа для доведения backend до «нормального» портфолио-/prod-вида; **после закрытия блока документации по демо** сверяться с `DEMO_AND_PORTFOLIO.md`.

**Полный документ**: [docs/PORTFOLIO_READINESS_CHECKLIST.md](PORTFOLIO_READINESS_CHECKLIST.md)

### STANDALONE_REPOSITORY.md
**Краткое описание**: Вынос каталога `backend/` в отдельный git-репозиторий: список артефактов, логика `dataproc_root`, `docker-compose.standalone.yml`, профили в `profiles/`, перенос CI.

**Полный документ**: [docs/STANDALONE_REPOSITORY.md](STANDALONE_REPOSITORY.md)

---

## Каноничные источники

Для получения "полуфинальных" контрактов (архитектура/DB/API/WS/billing/retention) смотри:

- `DataProcessor/docs/contracts/*`
- `DataProcessor/docs/architecture/PRODUCTION_ARCHITECTURE.md`

---
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Vault](../../docs/MAIN_INDEX.md)
