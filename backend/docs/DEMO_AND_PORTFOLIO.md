# Демонстрация backend и портфолио

Документ **фиксирует итог** по каталогу `backend/`: как показать сервис работодателю, где что описано, что входит в стандартный Docker-стек и чего **намеренно нет** без дополнительных сервисов.

**Связанные материалы:** [README.md](../README.md) · [MAIN_INDEX.md](MAIN_INDEX.md) · [PORTFOLIO_READINESS_CHECKLIST.md](PORTFOLIO_READINESS_CHECKLIST.md) · [STANDALONE_REPOSITORY.md](STANDALONE_REPOSITORY.md) · [GAPS_AND_ALIGNMENT.md](GAPS_AND_ALIGNMENT.md)

---

## 1. Питч за 60 секунд

Backend — это **оркестратор платформы анализа видео**: REST API (FastAPI), **мультитенантность** (workspaces → channels → videos), фоновые задачи (**Celery + Redis**), **PostgreSQL** (`core.*`) как индекс, файлы и **manifest / NPZ** как source of truth по результатам. Два потока: **AnalysisJob** (видео в продукте) и **IngestionRun** (URL → **Fetcher** → DataProcessor). Есть **WebSocket** по событиям ingestion, **health/readiness**, Docker Compose для быстрого подъёма, **pytest + CI**.

---

## 2. Чеклист перед показом (рекомендуемый порядок)

1. **Из корня репозитория:** `docker compose -f backend/docker-compose.yml up --build`  
   Убедитесь, что рядом есть `DataProcessor/profiles/` (контекст сборки `profiles`).
2. Открыть **http://localhost:8080/health/ready** — ожидается **200** (`database` + `redis` = ok).
3. Открыть **http://localhost:8080/docs** — Swagger; health-маршруты **без** префикса `/api`.
4. (Опционально) Показать **GitHub Actions** (workflow **Backend CI**): шаги **Ruff** (`ruff check app`), **pytest** с **coverage** и артефакт **`coverage-xml`** (`backend/coverage.xml`). Локально из `backend/`: `.venv/bin/ruff check app` и полный прогон — см. [TESTING.md](TESTING.md) §3; быстрый срез: `pytest tests/ -m "unit or contract" -q`.
5. Явно проговорить **ограничения** (см. раздел 4 ниже и [GAPS_AND_ALIGNMENT.md](GAPS_AND_ALIGNMENT.md)) — так выглядит зрелость, а не слабость.

**Отдельный git-репозиторий (только backend):** шаги и список артефактов — в [STANDALONE_REPOSITORY.md](STANDALONE_REPOSITORY.md); локальный подъём: `docker compose -f docker-compose.standalone.yml up --build` из корня того репо (предварительно заполнить `profiles/`).

**Порты по умолчанию (compose):** API **8080**, PostgreSQL **5432**, Redis **6379**. В compose **нет** Fetcher, DataProcessor и **Celery beat**; полный E2E с ними — по [E2E_PIPELINE_NO_TEXT.md](E2E_PIPELINE_NO_TEXT.md) и смежным документам в корне репо.

---

## 3. Карта документации (что открыть по вопросу)

| Вопрос работодателя | Документ |
|--------------------|----------|
| Что за сервис и потоки данных? | [OVERVIEW.md](OVERVIEW.md) |
| Как поднять локально / Docker / env? | [README.md](../README.md), [OPERATIONS.md](OPERATIONS.md), [.env.example](../.env.example), [CONFIGURATION.md](CONFIGURATION.md) |
| Схема БД, ingestion_runs, legacy | [DATABASE.md](DATABASE.md) |
| REST и WS, коды ответов | [API.md](API.md) |
| Celery, analysis vs ingestion | [RUNS_AND_WORKERS.md](RUNS_AND_WORKERS.md) |
| Fetcher, beat, trigger | [FETCHER_INTEGRATION.md](FETCHER_INTEGRATION.md) |
| Контракт с DataProcessor (история vs актуальный код) | [reference/DATAPROCESSOR_CONTRACT.md](reference/DATAPROCESSOR_CONTRACT.md) |
| Безопасность, CORS, риски WS | [SECURITY.md](SECURITY.md) |
| Что ещё не сделано «как в каноне» | [GAPS_AND_ALIGNMENT.md](GAPS_AND_ALIGNMENT.md) |
| План дальнейшего доведения | [PORTFOLIO_READINESS_CHECKLIST.md](PORTFOLIO_READINESS_CHECKLIST.md) |
| Почему Celery + Redis + manifest (архитектурное решение) | [adr/0001-celery-redis-pubsub-manifest-source-of-truth.md](adr/0001-celery-redis-pubsub-manifest-source-of-truth.md) |
| Тесты | [TESTING.md](TESTING.md), [tests/README.md](../tests/README.md) |

Полное оглавление: [MAIN_INDEX.md](MAIN_INDEX.md).

---

## 4. Что сказать про ограничения (честно и коротко)

- **WebSocket** `/api/runs/{run_id}/events` требует **`?token=<JWT>`** и проверку владельца run ([SECURITY](SECURITY.md) §3; обновлено в `GAPS_AND_ALIGNMENT.md` §4).
- **Billing / ledger** не реализованы.
- **Cancel** analysis не гарантирует остановку subprocess DataProcessor — зафиксировано в GAPS.
- В **docker-compose** нет **beat** → периодический `sync_ingestion_run_status` для Fetcher нужен отдельным процессом при демо полного YouTube-потока.
- Вынос в **отдельный git-репозиторий** не сделан — в чеклисте п. 1.4; текущая инструкция рассчитана на **монорепозиторий** TrendFlowML.

---

## 5. Шпаргалка Q&A (собеседование)

- **Почему Celery?** Отделить тяжёлую обработку от HTTP, масштабировать workers, повторные попытки. Подробнее: [ADR 0001](adr/0001-celery-redis-pubsub-manifest-source-of-truth.md).
- **Зачем Redis?** Брокер/бэкенд Celery и **pub/sub** (`run:{run_id}`) для стрима событий в WS — см. тот же ADR.
- **Где истина по результатам анализа?** Файловое хранилище: **manifest.json** и артефакты; БД — для быстрых запросов и статусов (SoT в ADR 0001).
- **Как идемпотентность для runs?** Заголовок `Idempotency-Key` на `POST /api/runs` — см. API и GAPS.
- **Legacy vs core.*?** Новый API на **`core.*`**; **public** и старые модели — совместимость, см. `DATABASE.md`, `alembic/env.py`.

---

## 6. История

| Дата | Примечание |
|------|------------|
| 2026-04-02 | Финальный сводный документ для демо/портфолио; согласован с README, MAIN_INDEX, чеклистом. |
| 2026-04-02 | Исправлена ссылка в чеклисте (ограничения — §4); добавлены перекрёстные ссылки в GAPS, SECURITY, DATAPROCESSOR_CONTRACT. |
| 2026-04-02 | Чеклист демо §2: уточнены CI (Ruff, pytest, артефакт `coverage-xml`); Q&A: идемпотентность `POST /api/runs` со ссылкой на тест. См. [TESTING.md](TESTING.md), [PORTFOLIO_READINESS_CHECKLIST.md](PORTFOLIO_READINESS_CHECKLIST.md) п. 5.2–5.4. |
| 2026-04-02 | Карта документов §3: строка про ADR 0001 (Celery, Redis pub/sub, manifest SoT); п. 6.3 чеклиста портфолио. |
