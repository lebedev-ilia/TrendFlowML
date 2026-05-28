# Backend overview

Этот документ описывает **как устроен текущий backend** в репозитории и
какие основные потоки данных он реализует.

## 1) Границы сервиса

Backend в `backend/` отвечает за:

- REST API (FastAPI) для auth, workspaces, channels, videos, analysis jobs.
- Оркестрацию запуска DataProcessor через Celery task `process_analysis_job`.
- Индекс артефактов и логов в PostgreSQL (schema `core.*`).
- WebSocket‑стрим событий прогресса (Redis pubsub).
- Хранение файлов на локальном filesystem (storage/result_store/frames_dir).

**Поток YouTube (ingestion)** реализован (Phases 0–5): Backend создаёт run по URL (`POST /api/runs`), передаёт в Fetcher; Fetcher после finalize вызывает Backend `trigger-processing`; Backend запускает DataProcessor по артефактам Fetcher (`video_url`); статус синхронизируется из Fetcher (polling), события — через WebSocket. См. `backend/docs/FETCHER_INTEGRATION.md`, `docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md`, `docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md`. Billing‑ledger не реализован.

## 2) Потоки данных (фактическая реализация)

### 2.1 Workspace → Channel → Video

1. Пользователь создаёт workspace через `POST /api/workspaces`.
2. В workspace создаётся channel через `POST /api/workspaces/{workspace_id}/channels`.
3. В channel добавляется video через `POST /api/channels/{channel_id}/videos`.

### 2.2 Analysis Job → DataProcessor

1. `POST /api/workspaces/{workspace_id}/videos/{video_id}/analysis` создаёт `AnalysisJob`.
2. Celery task `process_analysis_job(analysis_job_id)` использует адаптер для преобразования
   в формат DataProcessor и запускает `DataProcessor/main.py` (subprocess).
3. После завершения:
   - читает `manifest.json`
   - создаёт `Prediction` записи из результатов
   - регистрирует артефакты в таблице `artifacts` (legacy, для совместимости)
   - запускает demo quality scripts

### 2.3 Runs (ingestion по YouTube URL)

1. `POST /api/runs` с `source_url` (YouTube и др.) создаёт запись в `ingestion_runs` и передаёт задачу в Fetcher (`POST /api/v1/runs`).
2. Celery beat запускает `sync_ingestion_run_status`: опрос Fetcher `GET /api/v1/runs/{run_id}`, обновление `ingestion_status`, `fetcher_stage`, ошибок; публикация событий в Redis.
3. После finalize Fetcher вызывает `POST /api/runs/{run_id}/trigger-processing`; Backend ставит задачу `process_ingestion_run(run_id)`.
4. Задача получает manifest/artifacts из Fetcher, передаёт в DataProcessor `video_url`; по завершении обновляет `ingestion_status` (completed/failed).
5. Клиент может смотреть статус через `GET /api/runs/{run_id}` и поток событий `WS /api/runs/{run_id}/events`.

### 2.4 Progress → UI

- DataProcessor пишет `state_events.jsonl`, backend tail‑ит этот файл
  и превращает строки в события `run.stage_changed` и `component.*`.
- Параллельно stdout/stderr DataProcessor пишется в `run_logs`
  и стримится в WebSocket.
- Для ingestion run события также публикуются при синхронизации статуса из Fetcher (Phase 4).

## 3) Source‑of‑truth

Как и в каноничных документах, **source‑of‑truth для результатов — это
`manifest.json` и NPZ артефакты**, а PostgreSQL — индекс и ускоритель.

## 4) Где смотреть код

- FastAPI приложение: `backend/app/main.py`
- Celery worker: `backend/app/worker.py`, `backend/app/tasks/`
- Маршруты API: `backend/app/routers/*` (auth, workspaces, channels, videos, analysis, runs)
- Модели БД: `backend/app/dbv2/models.py` (schema `core.*`)
- Адаптер DataProcessor: `backend/app/services/dataprocessor_adapter.py`
- Клиент Fetcher API: `backend/app/services/fetcher_client.py`
- Хранилище и ffprobe: `backend/app/services/storage.py`
- Redis pubsub: `backend/app/services/events.py`

## 5) Database (core.*)

Доменная модель API v2 находится в PostgreSQL schema **`core.*`**. В миграциях Alembic дополнительно поддерживается **legacy**-слой в `public` (совместимость); детали в `DATABASE.md` и `alembic/env.py`.

Таблицы **`core.*`**:
- Модели: `backend/app/dbv2/models.py`
- Миграции: `backend/alembic/versions/*`
- Подробности: `backend/docs/DATABASE.md`

## 6) API Endpoints

Основные endpoints:

- Auth:
  - `POST /api/auth/register`
  - `POST /api/auth/login`
  - `GET /api/auth/me`
- Workspaces:
  - `POST /api/workspaces`
  - `GET /api/workspaces`
  - `GET /api/workspaces/{workspace_id}`
  - `POST /api/workspaces/{workspace_id}/members`
- Channels:
  - `POST /api/workspaces/{workspace_id}/channels`
  - `GET /api/workspaces/{workspace_id}/channels`
- Videos:
  - `POST /api/channels/{channel_id}/videos`
  - `GET /api/channels/{channel_id}/videos`
- Analysis:
  - `POST /api/workspaces/{workspace_id}/videos/{video_id}/analysis`
  - `GET /api/workspaces/{workspace_id}/analysis`
  - `GET /api/analysis/{analysis_job_id}/predictions`
- Runs (ingestion по URL):
  - `POST /api/runs` (body: `source_url`, header: `Idempotency-Key?`)
  - `GET /api/runs`, `GET /api/runs/{run_id}`
  - `WS /api/runs/{run_id}/events`
  - `POST /api/runs/{run_id}/trigger-processing` (вызов от Fetcher)

Полный список: `backend/docs/API.md`

