# Backend overview

Этот документ описывает **как устроен текущий backend** в репозитории и
какие основные потоки данных он реализует.

## 1) Границы сервиса

Backend в `backend/` отвечает за:

- REST API (FastAPI) для auth, видео, профилей, runs и админки.
- Оркестрацию запуска DataProcessor через Celery task `process_run`.
- Индекс артефактов и логов в PostgreSQL.
- WebSocket‑стрим событий прогресса (Redis pubsub).
- Хранение файлов на локальном filesystem (storage/result_store/frames_dir).

Важно: сервис Fetcher (YouTube download) и billing‑ledger пока **не реализованы** —
это фиксируется в `backend/docs/GAPS_AND_ALIGNMENT.md`.

## 2) Потоки данных (фактическая реализация)

### 2.1 Upload → Video

1. `POST /api/videos/upload/init` создаёт `Video` (`platform_id="upload"`)
   и `Upload`.
2. `PUT /api/videos/upload/{upload_id}` сохраняет файл во временную папку.
3. `POST /api/videos/upload/complete`:
   - переносит файл в `storage/raw/<video_id>/video.<ext>`
   - вычисляет `sha256`, создаёт `VideoFile` и `VideoSource`
   - делает `UserVideoLink` (access control)

### 2.2 Run → DataProcessor

1. `POST /api/runs` создаёт `Run` и минимальные `RunComponent`
   (сейчас только `segmenter` и `visual` для UI‑прогресса).
2. Celery task `process_run(run_id)` запускает `DataProcessor/main.py`
   (subprocess), пишет логи и события.
3. После завершения:
   - читает `manifest.json`
   - регистрирует артефакты в таблице `artifacts`
   - запускает demo quality scripts

### 2.3 Progress → UI

- DataProcessor пишет `state_events.jsonl`, backend tail‑ит этот файл
  и превращает строки в события `run.stage_changed` и `component.*`.
- Параллельно stdout/stderr DataProcessor пишется в `run_logs`
  и стримится в WebSocket.

## 3) Source‑of‑truth

Как и в каноничных документах, **source‑of‑truth для результатов — это
`manifest.json` и NPZ артефакты**, а PostgreSQL — индекс и ускоритель.

## 4) Где смотреть код

- FastAPI приложение: `backend/app/main.py`
- Celery worker: `backend/app/worker.py`, `backend/app/tasks.py`
- Маршруты API: `backend/app/routers/*`
- Хранилище и ffprobe: `backend/app/services/storage.py`
- Redis pubsub: `backend/app/services/events.py`

