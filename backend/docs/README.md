# TrendFlow Backend — документация

Этот раздел описывает **фактическую реализацию** backend‑сервиса в `backend/` и
фиксирует точки соответствия с каноничными контрактами из `DataProcessor/docs`.

## Индекс

- `backend/docs/OVERVIEW.md` — границы backend, ключевые решения, поток данных.
- `backend/docs/CONFIGURATION.md` — переменные окружения и разрешение путей.
- `backend/docs/STORAGE_LAYOUT.md` — структура storage/result_store/frames_dir.
- `backend/docs/DATABASE.md` — схема БД как реализована в `app/models.py`.
- `backend/docs/API.md` — REST/WS endpoints и контракт ответов.
- `backend/docs/UPLOADS_AND_VIDEOS.md` — upload flow, dedup, ffprobe.
- `backend/docs/PROFILES.md` — профили анализа, нормализация, config_hash, seed.
- `backend/docs/RUNS_AND_WORKERS.md` — Celery, DataProcessor, run lifecycle.
- `backend/docs/EVENTS_AND_LOGGING.md` — WS events, Redis pubsub, run_logs.
- `backend/docs/SECURITY.md` — auth/JWT, admin‑доступ, CORS, artifact token.
- `backend/docs/OPERATIONS.md` — запуск, зависимости, health checks, seed.
- `backend/docs/GAPS_AND_ALIGNMENT.md` — расхождения с канонами и TODO.
- `backend/docs/reference/backend_qna_contracts.md` — история решений и Q&A по backend.

## Каноничные источники

Если нужен “полуфинальный” контракт (архитектура/DB/API/WS/billing/retention),
смотри:

- `DataProcessor/docs/contracts/*`
- `DataProcessor/docs/architecture/PRODUCTION_ARCHITECTURE.md`


