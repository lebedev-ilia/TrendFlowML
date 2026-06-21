# Configuration and env vars

Настройки определены в `backend/app/config.py` и читаются из окружения
с префиксом `TF_BACKEND_`.

## 1) Переменные окружения

Обязательные/часто используемые:

- `TF_BACKEND_DB_DSN`  
  DSN для PostgreSQL.  
  По умолчанию: `postgresql+psycopg://trendflow:trendflow@localhost:5432/trendflow`  
  **Если используете тот же Postgres, что и Fetcher (порт 5433):** создайте пользователя и БД для backend (см. раздел «Один Postgres для Backend и Fetcher» ниже) и задайте, например:  
  `TF_BACKEND_DB_DSN=postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow`  
  **Важно:** переменная должна быть задана во всех процессах Backend: и для **API (uvicorn)**, и для **Celery worker**, и для **Celery beat** — иначе API будет подключаться к порту по умолчанию (5432) и вернёт 500 при обращении к БД.
- `TF_BACKEND_REDIS_URL`  
  Redis для Celery и WS pubsub.  
  По умолчанию: `redis://localhost:6379/0`
- `TF_BACKEND_DEPLOYMENT_ENV`  
  `development` (по умолчанию), `production` или `staging`. В **production** и **staging** при **слабом или пустом** `TF_BACKEND_JWT_SECRET` (см. список в `app.config.is_weak_jwt_secret` / `SECURITY.md` §6) приложение **не стартует**. В development слабый secret допускается с **предупреждением** в лог.
- `TF_BACKEND_JWT_SECRET`  
  Секрет для JWT. По умолчанию `"change-me"`.
- `TF_BACKEND_JWT_ALGORITHM`  
  По умолчанию `HS256`.
- `TF_BACKEND_JWT_EXP_MINUTES`  
  Время жизни токена, по умолчанию 7 дней.
- `TF_BACKEND_ADMIN_EMAILS`  
  Список email через запятую для admin‑доступа.
- `TF_BACKEND_CORS_ORIGINS`  
  CORS: `"*"` (по умолчанию, dev) или явные origin через запятую, например `http://localhost:3000,https://app.example.com`. См. `SECURITY.md`.

Пути хранилища (могут быть переопределены):

- `TF_BACKEND_STORAGE_ROOT`
- `TF_BACKEND_RESULT_STORE_BASE`
- `TF_BACKEND_FRAMES_DIR_BASE`
- `TF_BACKEND_RAW_UPLOADS_DIR`
- `TF_BACKEND_EXAMPLE_VIDEOS_DIR`

Интеграция с DataProcessor:

- `TF_BACKEND_DATAPROC_ROOT`
- `TF_BACKEND_VISUAL_CFG_DEFAULT`

**DataProcessor API настройки** (для Этапа 6):

- `TF_BACKEND_DATAPROCESSOR_API_URL`  
  URL DataProcessor API для HTTP запросов.  
  По умолчанию: `http://localhost:8001`  
  В production: `http://dataprocessor:8000` (если в docker-compose)

- `TF_BACKEND_DATAPROCESSOR_API_KEY`  
  API Key для аутентификации при запросах к DataProcessor API.  
  Опционально, но рекомендуется для production.  
  По умолчанию: `None` (development mode разрешает доступ без ключа)

- `TF_BACKEND_DATAPROCESSOR_POLL_INTERVAL`  
  Интервал между запросами статуса при polling (секунды).  
  По умолчанию: `5` секунд

- `TF_BACKEND_DATAPROCESSOR_TIMEOUT_SECONDS`  
  Максимальное время ожидания завершения обработки (секунды).  
  По умолчанию: `3600` секунд (1 час)

## 2) Как разрешаются пути

Если env не задан, пути рассчитываются относительно корня репозитория:

- `storage_root` → `<repo>/storage`
- `result_store_base` → `<storage_root>/result_store`
- `frames_dir_base` → `<storage_root>/frames_dir`
- `raw_uploads_dir` → `<storage_root>/raw`
- `example_videos_dir` → `<repo>/example/example_videos`
- `dataproc_root` → если существует каталог `<repo>/DataProcessor`, используется он (монорепозиторий); иначе `<repo>` (автономный backend-репозиторий; профили в `<repo>/profiles/`). Подробнее: [STANDALONE_REPOSITORY.md](STANDALONE_REPOSITORY.md).
- `visual_cfg_default` → `<dataproc_root>/configs/visual_triton_baseline_gpu_local.yaml`

## 3) Где используется

- Создание папок: `backend/app/services/storage.py::ensure_dirs`
- Запуск DataProcessor: `backend/app/tasks/analysis.py`, `ingestion.py`
- Seed профилей: `backend/app/routers/profiles.py`

## 4) Один Postgres для Backend и Fetcher (порт 5433)

Если Fetcher уже использует Postgres на порту 5433 (логин `fetcher` / БД `fetcher_db`), на том же инстансе нужно завести отдельные пользователя и БД для Backend.

**Создание пользователя и БД (один раз):**

Через контейнер Fetcher (если Postgres запущен через `docker compose` в каталоге Fetcher; контейнер по умолчанию — `fetcher-postgres`). Выполнять **по одной команде**: `CREATE DATABASE` в PostgreSQL нельзя выполнять внутри транзакции вместе с другими DDL.

```bash
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "CREATE USER trendflow WITH PASSWORD 'trendflow';"
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "CREATE DATABASE trendflow OWNER trendflow;"
```

Или напрямую с хоста (если есть `psql`):

```bash
PGPASSWORD=fetcher_password psql -h localhost -p 5433 -U fetcher -d fetcher_db -c "CREATE USER trendflow WITH PASSWORD 'trendflow';"
PGPASSWORD=fetcher_password psql -h localhost -p 5433 -U fetcher -d fetcher_db -c "CREATE DATABASE trendflow OWNER trendflow;"
```

Если роль `trendflow` уже есть, достаточно создать БД: `CREATE DATABASE trendflow OWNER trendflow;`

**Переменная окружения для Backend и Celery:**

```bash
export TF_BACKEND_DB_DSN="postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow"
```

После этого выполните миграции Alembic из каталога `backend` и перезапустите Celery worker.

**Проверка E2E (run до completed):** скрипт `backend/scripts/e2e_run_to_complete.py` создаёт пользователя (или логинится), создаёт run по YouTube URL и опрашивает статус до `completed` или `failed`. Запуск из каталога `backend` с активированным venv и заданным `TF_BACKEND_DB_DSN`; должны быть запущены Backend API, Fetcher API, Fetcher worker, Backend Celery worker и beat. **Полный runbook** (все команды, переменные, типичные проблемы): [E2E_RUNBOOK.md](E2E_RUNBOOK.md).

**Если миграция упала с ошибкой `DuplicateObject` (тип или объект уже существует):** БД в частично применённом состоянии. Проще всего пересоздать БД и применить миграции заново:

```bash
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'trendflow';"
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "DROP DATABASE trendflow;"
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "CREATE DATABASE trendflow OWNER trendflow;"
cd /path/to/backend && source .venv/bin/activate && export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow' && alembic upgrade head
```

## 5) Fetcher: один Redis для API и worker

При E2E (run по URL) Backend вызывает Fetcher API (POST `/api/v1/runs`). Fetcher API создаёт run в своей БД и ставит задачу `fetch_video` в очередь Celery (Redis). Задачу забирает **Fetcher Celery worker**.

**Если run так и висит на pending (0/7), воркер задач не получает.** Частая причина — **разный Redis** у Fetcher API и у Fetcher worker:

- API в Docker использует, например, `redis://redis:6379/0` (сервис в docker-compose).
- Worker на хосте (`./scripts/run_worker_on_host.sh`) по умолчанию использует `redis://localhost:6379/0`.

В итоге задачи пишутся в один Redis, а воркер слушает другой.

**Что сделать:** чтобы API и worker использовали один и тот же брокер:

- Либо запускайте **и Fetcher API, и Fetcher worker на хосте** (одинаковый `FETCHER_REDIS_URL` / `CELERY_BROKER_URL`, например `redis://localhost:6379/0`).
- Либо оба в Docker с одним и тем же `CELERY_BROKER_URL`.
- Если API в Docker, а worker на хосте — в контейнере API задайте Redis на хосте: `FETCHER_REDIS_URL=redis://host.docker.internal:6379/0` (или IP хоста вместо `host.docker.internal`).
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
