# Gaps and alignment with canonical docs

Этот документ фиксирует расхождения между **каноничными контрактами**
(`DataProcessor/docs/backend/*`, `DataProcessor/docs/contracts/*`) и
фактической реализацией backend в `backend/`.

## 1) Billing

Контракт требует:

- `billing_ledger`
- hold/charge/release
- блокировку run при недостатке средств

В коде backend:

- billing таблиц нет
- нет расчёта cost_units и hold
- `estimated_cost_units` и `actual_cost_units` не используются

## 2) YouTube Fetcher

Контракт требует Fetcher worker (`fetch_video(run_id)`), который качает видео, собирает метаданные/комменты, кладёт в object storage, запускает обработку.

В коде backend (Phases 0–5 интеграции, см. `FETCHER_INTEGRATION.md`, `docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md`):

- **Реализовано:** создание run по YouTube URL (`POST /api/runs`), вызов Fetcher API, после finalize Fetcher вызывает Backend `trigger-processing`, Backend запускает DataProcessor по `video_url` (артефакты Fetcher). Синхронизация статуса из Fetcher (polling), WebSocket событий, обработка ошибок (502 при недоступности Fetcher, RUN_NOT_FOUND при 404 в sync).
- **Upload путь** по-прежнему поддерживается отдельно (Workspace → Channel → Video → Analysis).
- Связывание результата ingestion run с сущностями Backend (Video/AnalysisJob) для отображения артефактов в UI может быть добавлено отдельно.

## 3) Idempotency

Контракт требует `Idempotency-Key` для:

- `POST /api/runs`
- `POST /api/videos/upload/complete`

В коде backend: **идемпотентность для `POST /api/runs` реализована** (заголовок `Idempotency-Key`, при совпадении ключа возвращается существующий run). Регрессия: **`tests/api/test_runs.py`** (`TestRunsCreateIdempotency` — без повторного вызова Fetcher).

HTTP-роутеры для **`POST /api/videos/upload/*`** в текущем приложении **не подключены** (описание потока — `UPLOADS_AND_VIDEOS.md`). **Контракт идемпотентности** для `upload/complete` зафиксирован там же (заголовок `Idempotency-Key`, ключ в разрезе пользователя); при появлении реализации — повтор с тем же ключом возвращает тот же исход без дублирования побочных эффектов.

## 4) WebSocket auth

Контракт: аутентификация клиента (JWT или короткоживущий ticket).  
**Реализовано:** `WS /api/runs/{run_id}/events` принимает **`?token=<JWT>`** (тот же secret/алгоритм, что и для REST). Проверяется владелец run (`IngestionRun.user_id`). Без токена, с невалидным JWT или чужим run соединение закрывается с кодом **1008** до `accept()`. Подробности: `SECURITY.md`, `FETCHER_INTEGRATION.md` §7.3.

## 5) Cancel semantics

Контракт: «cancel доступен; дальнейшие компоненты не должны стартовать или pipeline должен завершиться отменой».

**Реализовано (AnalysisJob):**

- **`POST /api/analysis/{analysis_job_id}/cancel`** (см. `API.md`, `OPERATIONS.md`).
- **`queued`:** в БД сразу **`canceled`**; задача Celery при старте выходит, DataProcessor не вызывается.
- **`processing`:** backend вызывает **DataProcessor** `POST /api/v1/runs/{run_id}/cancel` с `run_id = analysis_job.id` (флаг отмены в Redis на стороне DP; worker DP останавливает обработку кооперативно). Финальный статус **`canceled`** в core — через poll/SSE задачи или webhook (как для других финальных статусов DP).
- **Завершённые** статусы: ответ **noop** (идемпотентность).
- **Backend не делает** жёсткий SIGKILL локального subprocess DataProcessor: в актуальном режиме интеграции используется **HTTP API** DataProcessor, не CLI `subprocess` из backend.

## 6) Delete endpoints и retention

Контракт: `DELETE /api/videos/{video_id}` удаляет артефакты/кэш/индексы.  
В коде endpoint отсутствует.  
Также отсутствует политика retention сырого видео.

## 7) Validation and limits

Контракты фиксируют:

- min/max длительность видео
- max resolution
- корректная обработка битых файлов

В коде:

- есть только `ffprobe`, без полноценной валидации

## 8) Admin/support roles

Контракт: роли `user/admin/support`.  
В коде: `role` есть, но используется только `admin`.

## 9) Health checks

Контракт: `/health` и `/health/live`.  

В коде: **реализовано** — `GET /health`, `GET /health/live` (liveness); `GET /health/ready` проверяет PostgreSQL и Redis (503 при сбое). См. `PORTFOLIO_READINESS_CHECKLIST.md`, `OPERATIONS.md`.

---

Эти пункты лучше переносить в roadmap/backlog по мере стабилизации backend.

---

## Как презентовать текущее состояние

Краткий гайд для демо и портфолио (чеклист, карта документов, Q&A): [DEMO_AND_PORTFOLIO.md](DEMO_AND_PORTFOLIO.md).

