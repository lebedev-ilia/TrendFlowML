# Runs, analysis jobs и Celery workers

Документ описывает **текущую** фактическую реализацию фоновой обработки в пакете `backend/app/tasks/` (`analysis.py`, `ingestion.py`, `events.py`, `manifest.py`) и связанные HTTP-маршруты. Legacy-модель отдельного `process_run` для старых `runs` в этом тексте не используется — актуальные задачи: **`process_analysis_job`**, **`process_ingestion_run`**, **`sync_ingestion_run_status`**.

См. также: [OVERVIEW.md](OVERVIEW.md), [FETCHER_INTEGRATION.md](FETCHER_INTEGRATION.md), [reference/DATAPROCESSOR_CONTRACT.md](reference/DATAPROCESSOR_CONTRACT.md). Архитектурное обоснование очереди, Redis pub/sub и **manifest как source of truth:** [adr/0001-celery-redis-pubsub-manifest-source-of-truth.md](adr/0001-celery-redis-pubsub-manifest-source-of-truth.md).

---

## 1) Analysis Job (путь: workspace → video → analysis)

**HTTP:** `POST /api/workspaces/{workspace_id}/videos/{video_id}/analysis` — создаёт запись **AnalysisJob** (`core.analysis_jobs`), ставит в очередь Celery.

**Задача:** `process_analysis_job(analysis_job_id)` (`app/tasks/analysis.py`)

Назначение: преобразовать контекст v2 в payload для DataProcessor (адаптер `dataprocessor_adapter`), запустить обработку (HTTP API DataProcessor и/или subprocess — см. код и `dataprocessor.py`), стримить логи и события, по завершении разобрать `manifest.json`, обновить job, при необходимости создать **Prediction** и зарегистрировать артефакты.

Отмена: **`POST /api/analysis/{analysis_job_id}/cancel`** — для `queued` сразу `canceled` в БД; для `processing` — вызов **DataProcessor** `POST /api/v1/runs/{run_id}/cancel` (`run_id` = id job); терминальные статусы — **noop**. Детали: [OPERATIONS.md](OPERATIONS.md) §6, [GAPS_AND_ALIGNMENT.md](GAPS_AND_ALIGNMENT.md) §5.

---

## 2) Ingestion run (путь: URL → Fetcher → DataProcessor)

**HTTP:**

- `POST /api/runs` — создаёт **IngestionRun**, вызывает Fetcher (`fetcher_client`), при необходимости **Idempotency-Key**.
- `GET /api/runs`, `GET /api/runs/{run_id}` — статус и детали.
- `POST /api/runs/{run_id}/trigger-processing` — вызов **от Fetcher** после finalize; ставит `process_ingestion_run` (опционально защита `X-API-Key`, см. `TF_BACKEND_RUN_TRIGGER_API_KEY`).

**Задача:** `process_ingestion_run(run_id)` — получает manifest/артефакты из Fetcher, собирает payload с `video_url`, запускает DataProcessor.

**Периодическая синхронизация:** `sync_ingestion_run_status` — опрос Fetcher, обновление полей run в БД, события. Расписание в `app/worker.py` (`beat_schedule`); требуется процесс **`celery -A app.worker:celery_app beat`**. В `backend/docker-compose.yml` **beat по умолчанию не запущен** — только `worker` и `api`.

---

## 3) DataProcessor: прогресс и логи

При работе воркера:

- DataProcessor может писать **`state_events.jsonl`** — backend tail’ит и публикует события (`run.stage_changed`, `component.*`) в Redis.
- stdout/stderr могут попадать в **run_logs** (legacy) и транслироваться в WebSocket.

Детали payload командной строки и путей: см. код `app/tasks/`, [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md), [DATAPROCESSOR_CONTRACT.md](reference/DATAPROCESSOR_CONTRACT.md).

---

## 4) Quality reports

После успешного прогона analysis job backend может искать скрипты вида `**/quality_report/demo_*_quality.py` и запускать их (см. `app/services/quality.py` и `app/tasks/analysis.py`).

---

## 5) Запуск воркера и beat (локально)

Из каталога **`backend/`**:

```bash
celery -A app.worker:celery_app worker --loglevel=INFO
celery -A app.worker:celery_app beat --loglevel=INFO
```

Docker: см. [README.md](../README.md) и [OPERATIONS.md](OPERATIONS.md).
