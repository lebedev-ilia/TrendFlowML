# E2E Run: полный запуск и что мы делали

Документ описывает запуск цепочки **Backend → Fetcher → DataProcessor** до успешного завершения ingestion run (статус `completed`), все нужные переменные окружения, команды и исправления, которые были внесены.

Актуальный worklog с последними правками и валидированными run-id см. в `backend/docs/E2E_WORKLOG_2026-03-13.md`.

**Цепочка без TextProcessor (Segmenter + Audio + Visual):** указатель команд и скриптов — [`E2E_PIPELINE_NO_TEXT.md`](E2E_PIPELINE_NO_TEXT.md).

**Фиксы E2E / DataProcessor / cold cache / Audio NPZ (2026-04):** [`E2E_DP_FIXES_2026-04.md`](E2E_DP_FIXES_2026-04.md).

**Визуальные семантические головы, Embedding Service seed, тайминги сьюта (2026-04):** [`E2E_WORKLOG_VISUAL_SEMANTICS_2026-04.md`](E2E_WORKLOG_VISUAL_SEMANTICS_2026-04.md).

**Полный max-E2E** (Segmenter + все процессоры из `global_config.yaml`, Triton, артефакты в `storage/e2e_full_max/`): см. **раздел 9** ниже.

**Два режима E2E:**
- **Только Fetcher**: скрипт без `--with-dataprocessor` ждёт `ingestion_status=completed`, который выставляется синком из Fetcher (Fetcher COMPLETED → completed). DataProcessor не требуется.
- **Полный E2E (Fetcher + DataProcessor)**: скрипт с `--with-dataprocessor` ждёт перехода в `processing` (задача `process_ingestion_run` запустилась), затем `completed` после завершения DataProcessor. Нужны Backend, Fetcher и **DataProcessor API + worker**.

---

## 1. Что должно быть запущено

### 1.1 Минимальный E2E (Backend + Fetcher)

Для прохождения E2E до первого `ingestion_status=completed` (по синку из Fetcher) нужны:

| Сервис | Назначение |
|--------|------------|
| **PostgreSQL** | Одна инстанция на порту 5433 (общая для Fetcher и Backend при настройке «один Postgres»). |
| **Redis** | Один инстанс (порт 6379). **Один и тот же** Redis должен использоваться и Fetcher API, и Fetcher worker. |
| **MinIO/S3** | Хранилище для Fetcher (артефакты, manifest). Порт 9000 при запуске на хосте. |
| **Fetcher API** | HTTP API (uvicorn), порт 8000. Создаёт run и ставит задачу `fetch_video` в Celery. |
| **Fetcher Celery worker** | Обрабатывает очереди: `fetcher.high`, `fetcher.normal`, `fetcher.low`, `fetch.metadata`, `fetch.video`, `fetch.comments`, `fetch.finalize`, `fetch.maintenance`. |
| **Backend API** | FastAPI (uvicorn), порт 8001. Создаёт ingestion run и вызывает Fetcher. |
| **Backend Celery worker** | Выполняет `sync_ingestion_run_status` и задачу `process_ingestion_run` (после trigger-processing). |
| **Backend Celery beat** | По расписанию запускает `sync_ingestion_run_status` (каждые 20 с по умолчанию). |

### 1.2 Полный E2E (с DataProcessor)

Дополнительно к списку выше:

| Сервис | Назначение |
|--------|------------|
| **DataProcessor API** | HTTP API (uvicorn), порт 8002 (на хосте — иной порт, т.к. Fetcher уже 8000). Принимает POST /api/v1/process от Backend, отдаёт 202. |
| **DataProcessor worker** | Обрабатывает очередь запусков (Redis Streams / воркер из `DataProcessor/api`). Скачивает видео по video_url, пишет manifest и артефакты. |

---

## 2. Переменные окружения

### 2.1 Backend (все процессы: API, Celery worker, Celery beat)

```bash
# Обязательно при использовании Postgres Fetcher (порт 5433)
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'

# Опционально (если не по умолчанию)
export TF_BACKEND_REDIS_URL='redis://localhost:6379/0'
export TF_BACKEND_JWT_SECRET='change-me'
# URL Fetcher API (Backend дергает его при создании run)
export TF_BACKEND_FETCHER_API_URL='http://localhost:8000'
export TF_BACKEND_FETCHER_TIMEOUT_SECONDS=30
# Интервал синка статуса из Fetcher (секунды)
export TF_BACKEND_INGESTION_SYNC_INTERVAL_SECONDS=20

# HTTP таймаут на POST /api/v1/process (сек): DP может долго отвечать, пока качает video_url в кеш
export TF_BACKEND_DATAPROCESSOR_ENQUEUE_TIMEOUT_SECONDS=600

# Для полного E2E (Backend → DataProcessor): URL DataProcessor API (обязательно для process_ingestion_run)
export TF_BACKEND_DATAPROCESSOR_API_URL='http://localhost:8002'
# Опционально: API Key, если DataProcessor требует X-API-Key
# export TF_BACKEND_DATAPROCESSOR_API_KEY='your-key'
```

### 2.2 Fetcher API (uvicorn)

```bash
# Тот же Redis, что и у Fetcher worker (иначе задачи не подхватит воркер)
export FETCHER_REDIS_URL='redis://localhost:6379/0'
export FETCHER_POSTGRES_DSN='postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db'
export FETCHER_S3_ENDPOINT_URL='http://localhost:9000'
export FETCHER_S3_ACCESS_KEY='minioadmin'
export FETCHER_S3_SECRET_KEY='minioadmin123'
export FETCHER_BUCKET_RAW='video-analytics-raw'
# Без сети к YouTube — нормализация по URL без yt-dlp
export FETCHER_YOUTUBE_USE_YT_DLP=false
# Для полного E2E (Fetcher → Backend trigger-processing → DataProcessor): URL Backend API
export FETCHER_BACKEND_BASE_URL='http://localhost:8001'
# Опционально: если Backend требует X-API-Key для trigger-processing
# export FETCHER_BACKEND_TRIGGER_API_KEY='your-key'
```

### 2.3 Fetcher Celery worker (тот же Redis, что и у Fetcher API)

Скрипт `Fetcher/scripts/run_worker_on_host.sh` уже выставляет:

- `FETCHER_REDIS_URL` / `CELERY_BROKER_URL` = `redis://localhost:6379/0`
- `FETCHER_POSTGRES_DSN`, `FETCHER_S3_*`, `FETCHER_BUCKET_RAW`, `FETCHER_YOUTUBE_USE_YT_DLP`

Если Fetcher API запущен в другом окружении (например, в Docker), в нём нужно задать **тот же** Redis, что и у воркера (например, `redis://host.docker.internal:6379/0` для доступа к Redis на хосте).

---

## 3. Подготовка БД (один раз)

### 3.1 Postgres: пользователь и БД для Backend (порт 5433)

Если Backend использует тот же Postgres, что и Fetcher:

```bash
# Через docker (контейнер fetcher-postgres)
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "CREATE USER trendflow WITH PASSWORD 'trendflow';"
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "CREATE DATABASE trendflow OWNER trendflow;"

# Или с хоста
PGPASSWORD=fetcher_password psql -h localhost -p 5433 -U fetcher -d fetcher_db -c "CREATE USER trendflow WITH PASSWORD 'trendflow';"
PGPASSWORD=fetcher_password psql -h localhost -p 5433 -U fetcher -d fetcher_db -c "CREATE DATABASE trendflow OWNER trendflow;"
```

### 3.2 Миграции Backend

```bash
cd /path/to/TrendFlowML/backend
source .venv/bin/activate
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'
alembic upgrade head
```

При ошибке `DuplicateObject` или частично применённых миграциях можно пересоздать БД:

```bash
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'trendflow';"
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "DROP DATABASE trendflow;"
docker exec -it fetcher-postgres psql -U fetcher -d fetcher_db -c "CREATE DATABASE trendflow OWNER trendflow;"
cd /path/to/TrendFlowML/backend && source .venv/bin/activate && export TF_BACKEND_DB_DSN='...' && alembic upgrade head
```

### 3.3 MinIO (бакеты для Fetcher)

В каталоге Fetcher (один раз):

```bash
cd /path/to/TrendFlowML/Fetcher
PYTHONPATH="$PWD" python scripts/init_minio_buckets.py
```

(При необходимости задать `FETCHER_S3_*` и endpoint.)

---

## 4. Команды запуска

Все команды ниже предполагают, что Postgres, Redis и MinIO уже подняты (например, через `docker compose up -d postgres redis minio` в каталоге Fetcher).

### 4.1 Backend API

```bash
cd /path/to/TrendFlowML/backend
source .venv/bin/activate
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 4.2 Backend Celery worker

```bash
cd /path/to/TrendFlowML/backend
source .venv/bin/activate
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'
export TF_BACKEND_REDIS_URL='redis://localhost:6379/0'
celery -A app.worker:celery_app worker --loglevel=info
```

### 4.3 Backend Celery beat

```bash
cd /path/to/TrendFlowML/backend
source .venv/bin/activate
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'
export TF_BACKEND_REDIS_URL='redis://localhost:6379/0'
celery -A app.worker:celery_app beat --loglevel=info
```

### 4.4 Fetcher API

```bash
cd /path/to/TrendFlowML/Fetcher
source .venv/bin/activate   # или .fetcher_venv
export FETCHER_REDIS_URL='redis://localhost:6379/0'
export FETCHER_POSTGRES_DSN='postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db'
export FETCHER_S3_ENDPOINT_URL='http://localhost:9000'
export FETCHER_YOUTUBE_USE_YT_DLP=false
uvicorn fetcher.api:app --host 0.0.0.0 --port 8000
```

### 4.5 Fetcher Celery worker (обязательно тот же Redis, что и у Fetcher API)

```bash
cd /path/to/TrendFlowML/Fetcher
source .venv/bin/activate   # или .fetcher_venv
./scripts/run_worker_on_host.sh
```

Скрипт сам выставляет `FETCHER_REDIS_URL`, `CELERY_BROKER_URL`, Postgres, S3, очереди. Для работы без сети к YouTube перед запуском можно задать:

```bash
export FETCHER_YOUTUBE_USE_YT_DLP=false
./scripts/run_worker_on_host.sh
```

### 4.6 DataProcessor API (для полного E2E)

На одном хосте с Fetcher (8000) и Backend (8001) запускайте DataProcessor API на порту **8002**:

```bash
cd /path/to/TrendFlowML/DataProcessor/api
source .venv/bin/activate   # или создайте виртуальное окружение
# Redis для очереди (можно тот же, что у Fetcher/Backend, или отдельный)
export REDIS_URL='redis://localhost:6379/1'
# Порт, доступный для Backend
uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
```

### 4.7 DataProcessor worker (для полного E2E)

Воркер, обрабатывающий задачи из очереди (см. `DataProcessor/api/README.md`, скрипты запуска воркера):

```bash
cd /path/to/TrendFlowML/DataProcessor/api
source .venv/bin/activate
export REDIS_URL='redis://localhost:6379/1'
# Запуск воркера — см. документацию DataProcessor (python -m api.worker или скрипт из репо)
python -m api.worker
```

Убедитесь, что Backend при вызове `process_ingestion_run` достучится до DataProcessor API по `TF_BACKEND_DATAPROCESSOR_API_URL` (например `http://localhost:8002`).

---

## 5. Запуск E2E-скрипта

После запуска всех сервисов из п. 4:

```bash
cd /path/to/TrendFlowML/backend
source .venv/bin/activate
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'

# Минимум (E2E до завершения Fetcher: ingestion_status=completed по синку)
python scripts/e2e_run_to_complete.py --source-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# С прогрессом по этапам Fetcher и подробным ответом Backend
python scripts/e2e_run_to_complete.py --source-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --fetcher-url http://localhost:8000 --verbose

# Полный E2E (Fetcher + DataProcessor): ждём processing → completed
export TF_BACKEND_DATAPROCESSOR_API_URL='http://localhost:8002'
python scripts/e2e_run_to_complete.py --source-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --with-dataprocessor --fetcher-url http://localhost:8000 --verbose
```

Скрипт регистрирует/логинит пользователя (по умолчанию `e2e@example.com` / `e2etest123`), создаёт run, опрашивает Backend и при `--fetcher-url` — Fetcher.

- **Без `--with-dataprocessor`**: выход по первому `ingestion_status=completed` (синк из Fetcher). DataProcessor может не успеть запуститься.
- **С `--with-dataprocessor`**: скрипт сначала ждёт перехода в `processing` (задача `process_ingestion_run` стартовала), затем ждёт `completed` или `failed` после DataProcessor. Таймаут на этап «ожидание processing» и «ожидание completed после processing» задаётся общим `--timeout`.

### 5.1 Расширенный smoke для DataProcessor (segmenter + audio)

Этот запуск нужен, чтобы подтвердить, что `DataProcessor` способен поднимать **не только `segmenter`**, но и **`audio`** (по умолчанию: `clap`, `tempo`, `loudness`) в текущем окружении.

Требования:

- `Fetcher` run уже завершён (артефакт `video_file` в состоянии READY).
- Запущены `DataProcessor API` и `DataProcessor worker`.

Запуск (segmenter + audio, **без** Text и **без** Visual):

```bash
cd /path/to/TrendFlowML/backend
source scripts/e2e_env.sh
.venv/bin/python -u scripts/e2e_dataprocessor_audio_smoke.py \
  --fetcher-run-id 3bee7a40-e835-49bc-a135-e3a3092d8953
```

Тот же сценарий **с VisualProcessor** (по-прежнему `run_text=false`, `text` в профиле выключен):

```bash
.venv/bin/python -u scripts/e2e_dataprocessor_audio_smoke.py \
  --fetcher-run-id <FETCHER_RUN_UUID> \
  --with-visual
```

Опционально: **`--visual-cfg-path /abs/path/...yaml`** (по умолчанию — `DataProcessor/configs/audit_v3/visual/visual_core_5_only.yaml`; для части узлов нужен Triton/GPU — см. YAML).

Скрипт:

- возьмёт signed `download_url` для `video_file` из `Fetcher /api/v1/runs/<run_id>/artifacts`;
- вызовет `DataProcessor /api/v1/process` с `run_audio=true`, `run_text=false`, и `profile_config.processors.audio.enabled=true`; при **`--with-visual`** добавляет `processors.visual.enabled=true`;
- будет печатать progress до `status=success` или `status=error` (в логе есть `segmenter` / `audio` / `visual`);
- покажет путь в `storage/result_store/.../<run_id>/`, где должны лежать audio `*_extractor/_render/render.html` и артефакты visual по включённым модулям.

Примечание про перегрузку (backpressure):

- Если `DataProcessor` перегружен, `POST /api/v1/process` вернёт `503 Service Unavailable` и заголовок `Retry-After`.
- В этом случае просто повторите запуск после указанного времени; `DataProcessor` теперь отклоняет перегрузку **до** скачивания `video_url` в кэш.

---

## 6. Что мы делали (фиксы и доработки)

Краткий список изменений, чтобы run доходил до `completed` и не зависал.

### Backend

- **БД**: использование одного Postgres с Fetcher (порт 5433), пользователь `trendflow`, БД `trendflow`; `TF_BACKEND_DB_DSN` задаётся для API, Celery worker и beat.
- **Миграции**: в первой миграции enum-типы создаются с `checkfirst=True`, в таблицах используются `ENUM(..., create_type=False)`, чтобы не было «type already exists».
- **Синк статуса**: в `sync_ingestion_run_status` добавлена пауза 0.15 с между GET-запросами к Fetcher и повтор при 429 (одна попытка через 5 с).

### Fetcher

- **Orchestrator**: после `normalize_source()` обновление `VideoSource.normalized_video_id` и `Run.status` выполняется в **той же** сессии с повторным запросом сущностей из БД (чтобы объекты были attached и коммит сохранял изменения).
- **Cache miss**: после успешного завершения каждой из задач `fetch_metadata_task`, `download_video_task`, `fetch_comments_task` вызывается `_maybe_enqueue_finalize_after_cache_miss(run_id)` — при готовности всех артефактов run переводится в FINALIZING и ставится `finalize_task`.
- **«Multiple rows»**: в `utils.py` и `workers/artifacts.py` запросы к `Video` и `Artifact`, где ожидалась одна запись, заменены на `.order_by(...).first()` вместо `.one()`/`.one_or_none()`.
- **Manifest**: сериализация в JSON через `manifest.dict()` (Pydantic 1.x) и `json.dumps(..., default=str)` для полей UUID/datetime.
- **Finalize при ошибке**: переход run в FAILED выполняется только когда **не** будет retry (`if not will_retry`); при retry run остаётся в FINALIZING, чтобы повторная попытка могла перевести его в COMPLETED.
- **Восстановление**: в finalize при установке COMPLETED разрешён переход из FAILED в COMPLETED (recovery), поле `run.error` очищается.
- **Периодика**: задача `requeue_stuck_finalize` (каждые 2 мин) перепоставляет finalize для run в статусе FINALIZING без `finished_at`, созданных более 2 минут назад.
- **Идемпотентность finalize**: в начале finalize при статусе run уже COMPLETED задача сразу завершается.

### E2E и документация

- **E2E-скрипт**: при зависании на pending (≥60 с, 0/7) выводится подсказка про общий Redis для Fetcher API и worker.
- **CONFIGURATION.md**: раздел про один Redis для Fetcher API и worker; раздел про один Postgres для Backend и Fetcher.

---

## 7. Типичные проблемы

| Симптом | Причина | Что сделать |
|--------|--------|-------------|
| Run висит на pending (0/7), воркер не пишет «fetch_video received» | Fetcher API и Fetcher worker используют разные Redis | Задать один и тот же `FETCHER_REDIS_URL` / `CELERY_BROKER_URL` для API и worker. Если API в Docker — указать Redis на хосте (например `host.docker.internal:6379`). |
| Backend 500 при работе с БД | Нет/неверный `TF_BACKEND_DB_DSN` или он не задан для части процессов | Задать `TF_BACKEND_DB_DSN` для uvicorn, Celery worker и beat. |
| Fetcher: «Multiple rows were found when one or none was required» | Уже исправлено через `.first()` | Перезапустить Fetcher worker, чтобы подхватить актуальный код. |
| Fetcher: «Object of type UUID is not JSON serializable» | Уже исправлено через `default=str` в `json.dumps` | Перезапустить Fetcher worker. |
| Finalize падает с «Invalid status transition: FAILED → COMPLETED» | Run уже перевели в FAILED при первой ошибке; retry не мог обновить статус | Исправлено: при retry run не переводится в FAILED; добавлен recovery FAILED→COMPLETED. Перезапустить воркер, для старых run — создать новый. |
| 429 от Fetcher при синке статуса | Слишком частые GET от Backend | Уже добавлены пауза между запросами и retry при 429. |
| Run застрял в FINALIZING | Задача finalize не была поставлена или потеряна | Периодическая задача `requeue_stuck_finalize` перепоставит finalize. Убедиться, что воркер слушает очередь `fetch.maintenance`. |
| Полный E2E: «DataProcessor did not start» / таймаут на processing | Fetcher не вызывает trigger-processing, или Backend worker не обрабатывает задачу, или DataProcessor API недоступен | Задать **Fetcher**: `FETCHER_BACKEND_BASE_URL='http://localhost:8001'` (или URL Backend), чтобы после finalize вызывался `POST .../trigger-processing`. Проверить: Backend Celery worker запущен; `TF_BACKEND_DATAPROCESSOR_API_URL` задан; DataProcessor API + worker запущены. |
| Visual: `cut_detection` / `content_domain` падают на Triton, в логах `localhost:8000` | Воркер без `TRITON_HTTP_URL`; из YAML подставлялась заглушка порта Fetcher | Задать `TRITON_HTTP_URL=http://127.0.0.1:8010` для процессов DataProcessor (`e2e_env.sh`); убедиться, что Triton слушает этот порт. В коде: подстановка из `inline_config.global` вместо заглушки (`VisualProcessor/main.py`). |
| После ошибки `cut_detection` сыпятся `shot_quality`, `high_level_semantic`, … | Нет артефакта `cut_detection/*.npz` | Сначала починить Triton/сэмплинг для `cut_detection`; остальное — каскад. |

---

## 8. Сводка переменных (копипаста)

**Backend (все процессы); для полного E2E добавить DATAPROCESSOR_API_URL:**

```bash
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'
export TF_BACKEND_REDIS_URL='redis://localhost:6379/0'
export TF_BACKEND_FETCHER_API_URL='http://localhost:8000'
# Полный E2E (Backend → DataProcessor):
export TF_BACKEND_DATAPROCESSOR_API_URL='http://localhost:8002'
```

**Fetcher (API и worker — одинаково для Redis); для полного E2E добавить BACKEND_BASE_URL:**

```bash
export FETCHER_REDIS_URL='redis://localhost:6379/0'
export CELERY_BROKER_URL="$FETCHER_REDIS_URL"
export CELERY_RESULT_BACKEND="$FETCHER_REDIS_URL"
export FETCHER_POSTGRES_DSN='postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db'
export FETCHER_S3_ENDPOINT_URL='http://localhost:9000'
export FETCHER_S3_ACCESS_KEY='minioadmin'
export FETCHER_S3_SECRET_KEY='minioadmin123'
export FETCHER_BUCKET_RAW='video-analytics-raw'
export FETCHER_YOUTUBE_USE_YT_DLP=false
# Полный E2E: Fetcher вызывает Backend trigger-processing после finalize
export FETCHER_BACKEND_BASE_URL='http://localhost:8001'
```

Подробнее: [CONFIGURATION.md](CONFIGURATION.md).

---

## 9. Полный max-E2E (`e2e_full_max_run.py`)

Сценарий прогоняет цепочку **Backend → Fetcher → DataProcessor** с **максимальным** профилем из [`DataProcessor/configs/global_config.yaml`](../../DataProcessor/configs/global_config.yaml): Segmenter, AudioProcessor (все включённые экстракторы), TextProcessor, VisualProcessor (включая Triton-ядра CLIP / MiDaS / RAFT и полный набор модулей из шаблона).

### 9.1 Подготовка

1. Инфраструктура и app-стек (Postgres, Redis, MinIO, все сервисы из чеклиста) — рекомендуется `./backend/scripts/start_e2e_stack.sh --with-infra`; детали в [E2E_FULL_CHECKLIST.md](E2E_FULL_CHECKLIST.md).
2. Один shell с `source backend/scripts/e2e_env.sh` **до** запуска процессов DataProcessor, чтобы в окружении был **`TRITON_HTTP_URL=http://127.0.0.1:8010`** (или ваш порт).  
   - Порты **8000 / 8001 / 8002** на хосте — это Fetcher / Backend / DataProcessor API; **не** подменяйте ими URL Triton.  
   - В `global_config.yaml` исторически встречается заглушка `http://localhost:8000` для полей `triton_http_url` у отдельных компонентов; код VisualProcessor при отсутствии `TRITON_HTTP_URL` в окружении подставляет URL из `inline_config.global` вместо этой заглушки (см. `DataProcessor/VisualProcessor/main.py`: `_triton_http_url_for_subprocess_env`).
3. **Triton:** либо `python scripts/e2e_full_max_run.py --with-triton-docker` (поднимает контейнер на `TRITON_E2E_HTTP_PORT`, по умолчанию **8010**), либо заранее запущенный сервер и `export TRITON_HTTP_URL=...`.

### 9.2 Команда

Из каталога `backend` (venv активирован):

```bash
source scripts/e2e_env.sh
source .venv/bin/activate
python -u scripts/e2e_full_max_run.py --with-triton-docker --offline-example
```

Лог в файл:

```bash
python -u scripts/e2e_full_max_run.py --with-triton-docker --offline-example >> run_e2e.txt 2>&1
```

**Подмножество процессоров** (Segmenter всегда выполняется): `--processors text`, `--processors visual`, `--processors audio,text`, … (по умолчанию `audio,text,visual`). Если `visual` не входит в список, Triton для скрипта не обязателен, `--with-triton-docker` для visual не стартует.

### 9.3 Артефакты прогона

После запуска создаётся каталог:

`<repo>/storage/e2e_full_max/<run_tag>/`

В нём:

| Файл / каталог | Назначение |
|----------------|------------|
| `global_config_e2e.yaml` | Патченная копия global config для этого прогона |
| `summary.json` | Метаданные: `processors`, `triton_http_url`, exit code, пути |
| `text_input_video_document.json` | Копия VideoDocument при оффлайн-сценарии (если включён text) |
| `e2e_stack_logs/` | Снимок `backend/.e2e/logs/latest` |

Пока прогон идёт, маркер **`storage/e2e_full_max/active_global_config`** указывает путь к YAML для `process_ingestion_run`; после успеха маркер удаляется (если не передан `--keep-active-global-config`).

### 9.4 Ожидаемый результат (валидированный прогон 2026-04-05)

При успешной конфигурации:

- **`ingestion_status`**: `completed`.
- **`dataproc` в прогресс-таблице**: успех, строка вида `seg+ aud+ vis+ tex+`, **overall 100%** (или эквивалент для выбранного `--processors`).
- **Аудио**: все включённые в конфиге экстракторы в статусе `success` (типовое время порядка **8–9 минут** суммарно по сабкомпонентам при полном наборе).
- **Визуал**: большинство сабкомпонентов `success`; типовое время визуального процессора порядка **10–12 минут** на коротком оффлайн-ролике.

**Необязательный внешний сервис — Embedding Service** (`embedding_service_url` в конфиге, по умолчанию **`http://localhost:8005`**). Если сервис не поднят, следующие корни VisualProcessor завершаются с **ошибкой**, при этом общий ingestion может остаться **`completed`** (политика пайплайна — не валить весь run):

| Компонент | Симптом в логах |
|-----------|-----------------|
| `franchise_recognition` | `[Embedding Service] localhost:8005 — not running` |
| `brand_semantics` | то же |
| `car_semantics` | то же |
| `face_identity` / `core_face_identity` | то же |
| `place_semantics` | то же |

Чтобы закрыть и их зелёным статусом, поднимите сервис эмбеддингов на **8005** или отключите соответствующие `core_providers` в профиле visual.

**Опционально:** предупреждение `brand_semantics | tracks not found in detections.npz` — следствие отсутствия трекинга в YOLO-выходе; пайплайн может продолжить с деградацией качества.

### 9.5 Связанные файлы

| Файл | Роль |
|------|------|
| [`backend/scripts/e2e_full_max_run.py`](../scripts/e2e_full_max_run.py) | Патч YAML, Triton docker, вызов `e2e_run_to_complete.py` |
| [`backend/scripts/e2e_env.sh`](../scripts/e2e_env.sh) | `STORAGE_ROOT`, `TRITON_HTTP_URL`, таймауты DP |
| [`backend/scripts/e2e_triton_docker.sh`](../scripts/e2e_triton_docker.sh) | Старт/ожидание Triton в Docker |
| [`DataProcessor/scripts/run_visual_full_test.py`](../../DataProcessor/scripts/run_visual_full_test.py) | Локальный только-visual прогон без Backend (см. docstring) |
