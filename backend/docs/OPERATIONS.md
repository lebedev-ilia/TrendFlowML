# Operations (dev)

## Docker (демо для ревьюера)

Из **корня репозитория** TrendFlowML:

```bash
docker compose -f backend/docker-compose.yml up --build
```

Поднимаются PostgreSQL, Redis, API (порт **8080**, после `alembic upgrade head`) и Celery worker. Контекст сборки — каталог `backend/` и build-context `DataProcessor/profiles`, не весь репозиторий (иначе при передаче контекста в демон возможны десятки ГБ и строки вида `transferring context` на очень долго). Подробности: `backend/README.md`, `backend/docker-compose.yml`.

## 1) Зависимости

- PostgreSQL
- Redis
- ffprobe (из ffmpeg)
- Python packages из `backend/requirements.txt`

## 2) Запуск API

Пример:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

На старте выполняется (`app.main:on_startup`):

- создание директорий storage (`ensure_dirs`);
- если `TF_BACKEND_DB_AUTO_CREATE=true` — создание схемы `core` и `create_all` только для моделей **v2**; в production обычно `false` и схема применяется через Alembic;
- попытка seed публичных профилей из `<dataproc_root>/profiles/*.yaml`; при недоступной БД — предупреждение в лог, процесс не завершается.

В Docker (`backend/docker-compose.yml`) для API вызывается `alembic upgrade head` перед `uvicorn`.

## 3) Запуск Celery worker

```
celery -A app.worker:celery_app worker --loglevel=INFO
```

Worker нужен для задач анализа и ingestion (`process_analysis_job`, `process_ingestion_run` и др.). Периодический опрос Fetcher для ingestion — задача `sync_ingestion_run_status`; для неё отдельно запускают **beat**:

```
celery -A app.worker:celery_app beat --loglevel=INFO
```

В `docker-compose` по умолчанию поднимаются только **api** и **worker**; **beat** в файл не включён.

## 4) Что хранить в .env

Минимум:

- `TF_BACKEND_DB_DSN`
- `TF_BACKEND_REDIS_URL`
- `TF_BACKEND_JWT_SECRET`
- `TF_BACKEND_ADMIN_EMAILS` (опционально)
- `TF_BACKEND_CORS_ORIGINS` — см. `CONFIGURATION.md`

Полный перечень: `docs/CONFIGURATION.md`, пример: `backend/.env.example`.

## 5) Health checks

Health checks:

- `GET /health` и `GET /health/live` — liveness (процесс отвечает).
- `GET /health/ready` — readiness: проверка PostgreSQL (`SELECT 1`) и Redis (`PING`); при недоступности зависимости — HTTP 503 и поле `detail.checks_failed`.

См. также `docs/PORTFOLIO_READINESS_CHECKLIST.md`, тесты `tests/api/test_health.py`.

## 6) Отмена analysis job (`POST /api/analysis/{id}/cancel`)

Семантика согласована с возможностями DataProcessor (отмена через **HTTP**, не SIGKILL из backend):

| Статус job в БД | Поведение |
|-----------------|-----------|
| `queued` | Сразу **`canceled`** в `core.analysis_jobs`; Celery-задача при старте завершается без вызова DataProcessor. |
| `processing` | Вызов **DataProcessor** `POST /api/v1/runs/{run_id}/cancel`, где `run_id` = UUID analysis job (тот же id, что при старте обработки). Worker DataProcessor периодически проверяет флаг в Redis и завершает пайплайн; финальный `canceled` в core приходит через существующий poll/webhook. |
| `completed` / `failed` / `canceled` | Идемпотентный ответ **`noop`**. |

Подробности: `GAPS_AND_ALIGNMENT.md` §5, `app/routers/analysis.py`, `app/services/dataprocessor.py` (`request_dataprocessor_cancel`).

