# Полный E2E Backend → Fetcher → DataProcessor: чеклист

Пошаговый список того, что нужно сделать для прогона полной цепочки до `ingestion_status=completed` (с DataProcessor).

Подробности: [E2E_RUNBOOK.md](E2E_RUNBOOK.md).

---

## 1. Инфраструктура (один раз)

**Вариант «всё одной командой»:** из корня репозитория:

```bash
./backend/scripts/setup_e2e_infra.sh
```

Скрипт поднимает Postgres/Redis/MinIO (docker compose), создаёт БД `trendflow`, запускает миграции Backend и создаёт бакеты MinIO. Требования: `docker`, `psql` (если Postgres доступен только с хоста), `backend/.venv` с установленным alembic.

**Вручную:**

- [ ] **PostgreSQL** — поднять (например `Fetcher/docker-compose.yml`: `docker compose up -d postgres`). Порт **5433** на хосте.
- [ ] **Redis** — поднять (`docker compose up -d redis`). Порт **6379**.
- [ ] **MinIO** — поднять (`docker compose up -d minio`). Порт **9000**.
- [ ] **БД Backend**: в том же Postgres создать пользователя и БД `trendflow` (см. E2E_RUNBOOK, п. 3.1).
- [ ] **Миграции Backend**: `cd backend && export TF_BACKEND_DB_DSN='...' && alembic upgrade head`.
- [ ] **MinIO бакеты**: `cd Fetcher && PYTHONPATH="$PWD" python scripts/init_minio_buckets.py`.

---

## 2. Переменные окружения

**Вариант «одним скриптом»:** подставить переменные в текущий shell:

```bash
source backend/scripts/e2e_env.sh    # из корня репо
# или из backend:
source scripts/e2e_env.sh
```

Ниже — тот же набор вручную.

### Backend (все процессы: API, worker, beat)

- [ ] `TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'`
- [ ] `TF_BACKEND_REDIS_URL='redis://localhost:6379/0'`
- [ ] `TF_BACKEND_FETCHER_API_URL='http://localhost:8000'`
- [ ] **Для полного E2E:** `TF_BACKEND_DATAPROCESSOR_API_URL='http://localhost:8002'`
- [ ] **Для полного E2E (авторизация DataProcessor API):** `TF_BACKEND_DATAPROCESSOR_API_KEY='dev-key'`

### Fetcher (API и worker — один и тот же Redis)

- [ ] `FETCHER_REDIS_URL='redis://localhost:6379/0'` (и для worker: `CELERY_BROKER_URL` тот же)
- [ ] `FETCHER_POSTGRES_DSN='postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db'`
- [ ] `FETCHER_S3_*`, `FETCHER_BUCKET_RAW`, при необходимости `FETCHER_YOUTUBE_USE_YT_DLP=false`
- [ ] Для режима **YouTube Data API v3 + мок‑видео (рекомендуется для E2E без реального трафика к YouTube)**:
  - `FETCHER_YOUTUBE_DATA_ENABLED=true`
  - `FETCHER_YOUTUBE_DATA_API_KEY='<dev-key>'` (ключ для YouTube Data API)
  - `FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD=true`
  - `FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_DIR='/path/to/sample_videos'` (директория с `sample_0.mp4`, `sample_1.mp4`, …)
  - `FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_COUNT=8` (или другое количество sample‑файлов)
- [ ] **Для полного E2E:** `FETCHER_BACKEND_BASE_URL='http://localhost:8001'` (чтобы Fetcher вызывал trigger-processing после finalize)
- [ ] *(опционально, только для локального E2E без жёсткой зависимости от comments_file)* `FETCHER_ALLOW_FINALIZE_WITHOUT_COMMENTS=true`

### DataProcessor

- [ ] `REDIS_URL='redis://localhost:6379/1'` (можно отдельная DB или тот же Redis).
- [ ] `STORAGE_TYPE='fs'`
- [ ] `STORAGE_ROOT='<repo>/storage'` (единый storage для `state/`, `result_store/`, кэша `video_url`).
- [ ] `ALLOWED_VIDEO_PATHS='<repo>/storage/videos,<repo>/storage/uploads'`
- [ ] *(опционально, для dev)* `DATAPROCESSOR_API_KEY='dev-key'`
- [ ] *(для dev, чтобы Backend проходил аутентификацию в DataProcessor API)* `API_KEY='dev-key'`
- [ ] **Для полного Visual + Triton:** `TRITON_HTTP_URL='http://127.0.0.1:8010'` (или порт из `TRITON_E2E_HTTP_PORT`). Без этого подпроцессы VisualProcessor могут ходить на неверный HTTP-порт.

`backend/scripts/e2e_env.sh` в локальном E2E выставляет `STORAGE_ROOT=<repo>/storage` и по умолчанию `TRITON_HTTP_URL` на `127.0.0.1:${TRITON_E2E_HTTP_PORT}`, чтобы `DataProcessor API`/worker и VisualProcessor согласовали Triton с `DataProcessor/main.py`.

Создание директорий для `STORAGE_ROOT`:

```bash
mkdir -p ./storage/videos ./storage/uploads ./storage/videos/_url_cache
```

---

## 3. Запуск сервисов (порядок не критичен, но все должны быть запущены)

Команды Backend нужно выполнять **из каталога backend** (иначе `ModuleNotFoundError: No module named 'app'`).

**Вариант "одной командой" для всего app-стека:**

```bash
./backend/scripts/start_e2e_stack.sh
```

Скрипт делает следующее:

- подхватывает переменные из `backend/scripts/e2e_env.sh`;
- создаёт каталоги `STORAGE_ROOT/videos` и `STORAGE_ROOT/uploads`;
- поднимает `Backend API`, `Backend worker`, `Backend beat`, `Fetcher API`, `Fetcher worker`, `DataProcessor API`, `DataProcessor worker`;
- пишет `pid`-файлы в `backend/.e2e/pids/`;
- создаёт отдельную директорию логов для каждого процесса в `backend/.e2e/logs/<run-id>/...`.

Инфраструктура (`postgres/redis/minio`) поднимается отдельно, один раз:

```bash
./backend/scripts/setup_e2e_infra.sh
```

Если нужно совместить это с запуском app-стека, используйте явный флаг:

```bash
./backend/scripts/start_e2e_stack.sh --with-infra
```

Остановить app-стек:

```bash
./backend/scripts/stop_e2e_stack.sh
```

Если нужно также остановить `postgres/redis/minio` из `Fetcher/docker-compose.yml`:

```bash
./backend/scripts/stop_e2e_stack.sh --with-infra
```

**Backend (в venv `.venv`):**

- [ ] API:

```bash
cd backend
source .venv/bin/activate
source scripts/e2e_env.sh
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

- [ ] Celery worker:

```bash
cd backend
source .venv/bin/activate
source scripts/e2e_env.sh
celery -A app.worker:celery_app worker --loglevel=info
```

- [ ] Celery beat:

```bash
cd backend
source .venv/bin/activate
source scripts/e2e_env.sh
celery -A app.worker:celery_app beat --loglevel=info
```

**Fetcher (в venv `.fetcher_venv`):**

- [ ] API:

```bash
cd Fetcher
source .fetcher_venv/bin/activate
uvicorn fetcher.api:app --host 0.0.0.0 --port 8000
```

- [ ] Celery worker (host):

```bash
cd Fetcher
source .fetcher_venv/bin/activate
./scripts/run_worker_on_host.sh
```

**DataProcessor (в venv `.data_venv`):**  
Перед запуском в этом терминале подставить переменные (`REDIS_URL`, `STORAGE_ROOT`, `ALLOWED_VIDEO_PATHS`), иначе API/worker не стартуют. Из корня репо: `source backend/scripts/e2e_env.sh`. Каталоги для storage создать один раз: `mkdir -p ./storage/videos ./storage/uploads ./storage/videos/_url_cache`.

- [ ] API:

```bash
cd DataProcessor
source .data_venv/bin/activate
source ../backend/scripts/e2e_env.sh   # из корня репо: source backend/scripts/e2e_env.sh
.data_venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8002
```

- [ ] Worker:

```bash
cd DataProcessor
source .data_venv/bin/activate
source ../backend/scripts/e2e_env.sh
.data_venv/bin/python -m api.worker
```

Порты: Backend **8001**, Fetcher **8000**, DataProcessor **8002**.

---

## 4. Запуск E2E-скрипта

```bash
cd backend
source .venv/bin/activate
source scripts/e2e_env.sh

python scripts/e2e_run_to_complete.py \
  --source-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --with-dataprocessor \
  --fetcher-url http://localhost:8000 \
  --verbose
```

- [ ] Скрипт создаёт run, опрашивает Backend и Fetcher.
- [ ] Ожидание: сначала `ingestion_status=completed` (синк из Fetcher), затем переход в `processing` (Backend вызвал DataProcessor), затем снова `completed` после DataProcessor.

### 4.1 Полный max-E2E с `global_config.yaml` (`e2e_full_max_run.py`)

Оркестрация «всё из шаблона» (Segmenter + Audio + Text + Visual + Triton), патч активного `global_config`, артефакты в `storage/e2e_full_max/<run_tag>/`.

Требования:

- [ ] Запущены те же сервисы, что для п. 4 (Backend, Fetcher, DataProcessor API + worker).
- [ ] Поднят **Triton** на **`127.0.0.1:${TRITON_E2E_HTTP_PORT}`** (по умолчанию **8010**), либо флаг `--with-triton-docker` в команде ниже.
- [ ] В окружении процессов DataProcessor задан **`TRITON_HTTP_URL`** (в `e2e_env.sh` это делается по умолчанию; при отсутствии Triton временно снимите экспорт или не запускайте visual).

Команда (оффлайн-пример без сети к YouTube, см. mock в `e2e_env.sh`):

```bash
cd backend
source .venv/bin/activate
source scripts/e2e_env.sh
python -u scripts/e2e_full_max_run.py --with-triton-docker --offline-example
```

Подмножество процессоров: `--processors text`, `--processors visual`, `audio,text`, и т.д.

Ожидаемый итог: `ingestion_status=completed`, в таблице прогресса успех по segmenter / audio / text / visual (для выбранных процессоров). Отдельные visual-компоненты с **`embedding_service_url` → localhost:8005** могут быть в **error**, если сервис эмбеддингов не запущен — см. [E2E_RUNBOOK.md § 9](E2E_RUNBOOK.md#9-полный-max-e2e-e2e_full_max_runpy).

### 4.2 Расширенный smoke для DataProcessor (segmenter + audio [+ visual])

Нужен, чтобы подтвердить, что `DataProcessor` в текущем окружении может поднять не только `segmenter`, но и `audio` (extractors: `clap`, `tempo`, `loudness`). Опционально **`--with-visual`**: тот же прогон плюс **VisualProcessor**, **без Text** (`run_text=false`).

Полный указатель без Text: [`E2E_PIPELINE_NO_TEXT.md`](E2E_PIPELINE_NO_TEXT.md). Полный max-conфиг через Backend: [§ 4.1](#41-полный-max-e2e-с-global_configyaml-e2e_full_max_runpy).

Требования:

- [ ] Есть завершённый `Fetcher run_id` (артефакт `video_file` READY).
- [ ] Запущены `DataProcessor API` и `DataProcessor worker`.

Запуск (только audio):

```bash
cd backend
source .venv/bin/activate
source scripts/e2e_env.sh
.venv/bin/python -u scripts/e2e_dataprocessor_audio_smoke.py \
  --fetcher-run-id 3bee7a40-e835-49bc-a135-e3a3092d8953
```

С Visual (см. `--visual-cfg-path` при необходимости):

```bash
.venv/bin/python -u scripts/e2e_dataprocessor_audio_smoke.py \
  --fetcher-run-id <FETCHER_RUN_UUID> \
  --with-visual
```

---

## 5. Если что-то не работает

- Run висит на **pending** (0/7) → Fetcher API и Fetcher worker должны использовать **один и тот же** Redis.
- **DataProcessor не стартует** (таймаут на processing) → задать у Fetcher `FETCHER_BACKEND_BASE_URL='http://localhost:8001'`; проверить, что Backend worker и DataProcessor API + worker запущены и доступны.

### 5.1. Известные особенности и фиксы (март 2026)

- **Зависание на `ingestion=running, stage=download_video  (3/7)`**  
  - Причина: в Fetcher функция `all_artifacts_ready` требовала наличия трёх артефактов (`metadata_file`, `video_file`, `comments_file`) со статусом `COMPLETED`, поэтому `finalize` никогда не ставился в очередь, если не было `comments_file`.  
  - Для локального E2E добавлен флаг `FETCHER_ALLOW_FINALIZE_WITHOUT_COMMENTS=true` и настройка `allow_finalize_without_comments` в `FetcherSettings`: при его включении достаточно `metadata_file` и `video_file`, чтобы запустить `finalize`. **В проде оставлять `False`.**

- **`GET /api/v1/runs/{run_id}/manifest` возвращает 503 при статусе Fetcher `COMPLETED`**  
  - Причина: в Fetcher API проверялось `run.status not in ("completed", "finalizing")` (нижний регистр), а state‑machine использует верхний (`COMPLETED`, `FINALIZING`).  
  - Исправлено: `get_run_manifest` теперь сравнивает `status_upper` с `RUN_STATUS_COMPLETED` / `RUN_STATUS_FINALIZING`.

- **401 Unauthorized при вызове DataProcessor API (`POST /api/v1/process`) из Backend**  
  - DataProcessor проверяет заголовок `X-API-Key` через `api.security.verify_api_key`, беря валидный ключ из `config.api_key` (env `API_KEY`).  
  - Backend отправляет этот заголовок только если задан `TF_BACKEND_DATAPROCESSOR_API_KEY`.  
  - Для dev‑E2E используем согласованный ключ:  
    - на стороне Backend: `TF_BACKEND_DATAPROCESSOR_API_KEY='dev-key'`,  
    - на стороне DataProcessor: `API_KEY='dev-key'` (и при желании `DATAPROCESSOR_API_KEY='dev-key'`).

- **500 Internal Server Error в DataProcessor (`parameter request must be an instance of starlette.requests.Request`)**  
  - Это ошибка интеграции с `slowapi` (rate limiting) для эндпоинта `/api/v1/process`.  
  - Для локального E2E rate limiting отключён в `api/endpoints/process.py` через `rate_limit_decorator`, который теперь возвращает исходную функцию без обёртки `limiter.limit(...)`. В проде можно вернуть лимитер, настроив его корректно.

- **KeyError `"Attempt to overwrite 'message' in LogRecord"` в Backend при логировании ответа DataProcessor**  
  - Причина: в `backend/app/services/dataprocessor.py` в `logger.info(..., extra={...})` использовался ключ `message`, который конфликтует с полем `LogRecord`.  
  - Исправлено: поля переименованы в `dataprocessor_status` и `dataprocessor_message`.

- **DataProcessor worker пишет `Run not found in TaskManager, skipping`, но E2E всё равно проходит**  
  - Сейчас `TaskManager` в DataProcessor реализован как in‑memory реестр на процесс, поэтому API и worker, запущенные в разных процессах, не делят одно и то же состояние. Для локального E2E этого достаточно, так как Backend считает ingestion завершённым после успешного приёма запроса `/api/v1/process`.  
  - Для полноценной прод‑обработки требуется доработка: либо перенести `TaskManager` в Redis (используя `redis_schema`), либо в worker опираться только на метаданные и состояние в Redis, а не на in‑memory `TaskManager`.

### 5.1. Полная очистка и переинициализация БД Fetcher (dev)

Если Fetcher начал отвечать ошибками вида `Multiple rows were found when one or none was required` или `relation "runs" does not exist`, можно в dev полностью пересоздать его БД:

1. Убедиться, что сервис `postgres` из `Fetcher/docker-compose.yml` запущен (порт 5433 на хосте).
2. Дропнуть/создать БД `fetcher_db` (через psql или docker exec) — см. подробности в [Fetcher/docs/DATABASE.md](../../Fetcher/docs/DATABASE.md).
3. Применить миграции Fetcher:

```bash
cd Fetcher
source .fetcher_venv/bin/activate
export FETCHER_POSTGRES_DSN="postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db"
python -m alembic upgrade head
```

После этого перезапустить Fetcher API и Fetcher worker и повторно прогнать E2E.

Полный список типичных проблем: [E2E_RUNBOOK.md](E2E_RUNBOOK.md) § 7.
