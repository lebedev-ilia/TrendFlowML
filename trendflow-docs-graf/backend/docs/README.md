# TrendFlow Backend — документация

Этот раздел описывает **фактическую реализацию** backend‑сервиса в `backend/` и
фиксирует точки соответствия с каноничными контрактами из `DataProcessor/docs`.

**Запуск в Docker, порты, демо для ревьюера:** см. **[README.md](../README.md)** в каталоге `backend/` (не дублируется здесь полностью).

## Главный индекс

Для быстрой навигации по всей документации используйте **[MAIN_INDEX.md](MAIN_INDEX.md)** — единый документ-индекс с кратким описанием всех документов и ссылками на полные версии.

**Демонстрация работодателю / портфолио (чеклист, карта документов, Q&A):** [DEMO_AND_PORTFOLIO.md](DEMO_AND_PORTFOLIO.md).

## Индекс (быстрый список)

- `backend/docs/OVERVIEW.md` — границы backend, ключевые решения, поток данных.
- `backend/docs/CONFIGURATION.md` — переменные окружения и разрешение путей.
- `backend/docs/STORAGE_LAYOUT.md` — структура storage/result_store/frames_dir.
- `backend/docs/DATABASE.md` — схема **`core.*`** (`app/dbv2/models.py`), legacy в `public` через Alembic.
- `backend/docs/API.md` — REST/WS endpoints и контракт ответов.
- `backend/docs/UPLOADS_AND_VIDEOS.md` — upload flow, dedup, ffprobe.
- `backend/docs/PROFILES.md` — профили анализа, нормализация, config_hash, seed.
- `backend/docs/RUNS_AND_WORKERS.md` — Celery, DataProcessor, run lifecycle.
- `backend/docs/EVENTS_AND_LOGGING.md` — WS events, Redis pubsub, run_logs.
- `backend/docs/SECURITY.md` — auth/JWT, admin‑доступ, CORS, artifact token.
- `backend/docs/OPERATIONS.md` — запуск, зависимости, health checks, seed.
- `backend/docs/GAPS_AND_ALIGNMENT.md` — расхождения с канонами и TODO.
- `backend/docs/FETCHER_INTEGRATION.md` — Backend ↔ Fetcher (ingestion по URL).
- `backend/docs/PORTFOLIO_READINESS_CHECKLIST.md` — чеклист портфолио / prod-готовности.
- `backend/docs/DEMO_AND_PORTFOLIO.md` — **итоговый гайд по демо** (шаги, ссылки, ограничения, собеседование).
- `backend/docs/adr/` — архитектурные решения (ADR): Celery, Redis pub/sub, manifest as SoT — см. [adr/README.md](adr/README.md).
- `backend/docs/TESTING.md` — тестирование: что реализовано, как запускать тесты.
- `backend/docs/TESTING_PLAN.md` — план и чеклист реализации всех тестов backend.
- `backend/docs/reference/backend_qna_contracts.md` — история решений и Q&A по backend.

## Тесты

- `backend/tests/` — каталог тестов (pytest). Описание и команды: [backend/tests/README.md](../tests/README.md). План: [TESTING_PLAN.md](TESTING_PLAN.md), текущее состояние: [TESTING.md](TESTING.md). **CI:** Ruff + pytest + **`coverage.xml`** как артефакт — см. [.github/workflows/backend-ci.yml](../../.github/workflows/backend-ci.yml); конфиг линтера: `backend/pyproject.toml`.

## Каноничные источники

Если нужен “полуфинальный” контракт (архитектура/DB/API/WS/billing/retention),
смотри:

- `DataProcessor/docs/contracts/*`
- `DataProcessor/docs/architecture/PRODUCTION_ARCHITECTURE.md`
---

## Навигация

[Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
