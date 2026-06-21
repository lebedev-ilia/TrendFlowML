# E2E: ручная настройка и исправления (Backend → Fetcher → Celery)

Документ фиксирует все изменения и шаги, выполненные для запуска ручного E2E-сценария: **Backend** создаёт run по YouTube URL, вызывает **Fetcher API**, Fetcher ставит задачу в **Celery**, **Fetcher worker** обрабатывает пайплайн (нормализация URL → metadata → video → comments → finalize).

Связанные чеклисты: [E2E_YOUTUBE_INGESTION_CHECKLIST.md](E2E_YOUTUBE_INGESTION_CHECKLIST.md), [Fetcher/docs/E2E_FETCHER_DATAPROCESSOR.md](../Fetcher/docs/E2E_FETCHER_DATAPROCESSOR.md).

---

## 1. Архитектура и порядок запуска

- **Backend** (локально: uvicorn на 8001, Celery worker + beat для синхронизации статуса с Fetcher).
- **Fetcher** (Docker Compose): API (8000), Postgres, Redis, MinIO, Celery worker (и при необходимости beat).
- **DataProcessor** (опционально для полного E2E): для проверки цепочки до Fetcher достаточно Backend + Fetcher.

Порядок запуска:

1. Поднять Fetcher: `cd Fetcher && docker compose up -d postgres redis minio fetcher-api fetcher-worker`
2. Запустить Backend: БД (Postgres на 5434), Redis для Celery, затем `uvicorn app.main:app --host 0.0.0.0 --port 8001`
3. Запустить Backend Celery worker и beat (отдельные терминалы)
4. При необходимости — DataProcessor

---

## 2. Backend: исправления и настройка

Эти правки были внесены ранее в рамках настройки E2E (здесь — краткая сводка).

| Проблема | Решение |
|----------|--------|
| `AmbiguousForeignKeysError` для `User.memberships` | В `backend/app/dbv2/models.py` в relationship `User.memberships` добавлен явный `foreign_keys="WorkspaceMember.user_id"`. |
| Ошибка при регистрации (bcrypt / password length) | В `backend/app/auth.py` схема хеширования сменена на `pbkdf2_sha256` в `CryptContext(schemes=["pbkdf2_sha256"])`. |
| `ModuleNotFoundError: email-validator` | Установка: `pip install "pydantic[email]"` в venv Backend. |
| Backend с SQLite и `CREATE SCHEMA` | Backend ожидает PostgreSQL; для локального запуска используется отдельный Postgres (например, порт 5434), переменная `TF_BACKEND_DB_DSN`. |

---

## 3. Fetcher (Docker): переменные окружения

`FetcherSettings` использует префикс **`FETCHER_`**. В docker-compose для сервисов Fetcher должны быть заданы именно эти переменные, иначе внутри контейнера приложение подключается к `localhost` (Postgres/Redis) и падает.

### 3.1. Обязательные переменные для `fetcher-api` и `fetcher-worker`

В `Fetcher/docker-compose.yml` для `fetcher-api` и `fetcher-worker` заданы:

- **БД**
  - `FETCHER_POSTGRES_DSN=postgresql+psycopg2://fetcher:fetcher_password@postgres:5432/fetcher_db`
- **Redis**
  - `FETCHER_REDIS_URL=redis://redis:6379/0`
- **S3/MinIO**
  - `FETCHER_S3_ENDPOINT_URL=http://minio:9000`
  - `FETCHER_S3_ACCESS_KEY=minioadmin`
  - `FETCHER_S3_SECRET_KEY=minioadmin123`
  - `FETCHER_BUCKET_RAW=video-analytics-raw`
  - `FETCHER_S3_USE_SSL=false`, `FETCHER_S3_VERIFY_SSL=false`
- **Нормализация YouTube без сети (локальная разработка)**
  - `FETCHER_YOUTUBE_USE_YT_DLP=false` — при создании run не выполняется сетевой запрос к YouTube для извлечения video_id; ID берётся из URL (см. п. 4).

Старые переменные без префикса (`POSTGRES_HOST`, `REDIS_URL`, `S3_*`) оставлены для совместимости, но ключевыми являются `FETCHER_*`.

### 3.2. Очереди Celery и команда воркера

API при создании run отправляет задачу `fetch_metadata_task` в очередь **`fetcher.normal`** (или `fetcher.high` / `fetcher.low` в зависимости от priority). Остальные этапы пайплайна попадают в очереди `fetch.video`, `fetch.comments`, `fetch.finalize`, `fetch.maintenance`.

Если воркер запускать без `-Q`, он слушает только дефолтную очередь и **не обрабатывает** задачи из `fetcher.normal`, поэтому run остаётся в статусе pending.

В `docker-compose.yml` для `fetcher-worker` команда задана явно:

```yaml
command: celery -A fetcher.celery_app worker --loglevel=info --concurrency=4 -Q fetcher.high,fetcher.normal,fetcher.low,fetch.metadata,fetch.video,fetch.comments,fetch.finalize,fetch.maintenance
```

---

## 4. Fetcher: нормализация YouTube URL без сети

Для локального E2E без доступа контейнера к интернету добавлена возможность не вызывать yt-dlp при нормализации URL.

- **Конфиг** (`fetcher/config.py`): параметр `youtube_use_yt_dlp: bool = True`. При `False` нормализация не делает HTTP-запросов к YouTube.
- **Переменная окружения**: `FETCHER_YOUTUBE_USE_YT_DLP=false` (в docker-compose для api/worker).
- **Логика** (`fetcher/orchestrator.py`, `normalize_source`):
  - при `youtube_use_yt_dlp=True` — поведение как раньше (yt-dlp, сеть);
  - при `False` — извлечение video_id только из URL: для `youtu.be/<id>` — из path, для `youtube.com/watch?v=<id>` — из query-параметра `v`. При невозможности извлечь ID выбрасывается `ValueError`.

Это устраняет таймаут при `POST /api/v1/runs` из контейнера без доступа к YouTube. Этапы **metadata / video / comments** по-прежнему обращаются к YouTube через yt-dlp; при отсутствии сети они будут падать по таймауту или connection error.

---

## 5. Fetcher: исправления импортов в задачах Celery

В `fetcher/tasks.py` при выполнении `fetch_metadata_task` возникали `NameError` из-за отсутствующих импортов. Добавлены:

- **Модель**: `from .models import Run, VideoSource` (используется в начале задачи для чтения `platform` / `platform_video_id` из `VideoSource`).
- **События**: `from .events import publish_job_failed, publish_job_finished, publish_job_started` (используются при старте/успехе/падении job в задачах metadata и других).

Файлы: `fetcher/tasks.py` (импорты в начале файла).

---

## 6. Ручная проверка E2E (curl)

Предполагается: Backend на 8001, Fetcher API на 8000, получен JWT в `TOKEN`.

```bash
# 1. Регистрация/логин Backend (если ещё не сделано)
# POST /api/auth/register, затем POST /api/auth/login, сохранить token в TOKEN

# 2. Создать run (Backend проксирует в Fetcher)
curl -s -X POST "http://localhost:8001/api/runs" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"run_id": "a1b2c3d4-1111-2222-3333-444455556666", "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}' | jq .

# Ожидание: 201, в теле run_id, ingestion_status: "pending", message: "Run created and ingestion queued"

# 3. Статус run в Backend
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8001/api/runs/<RUN_ID>" | jq .

# 4. Логи Fetcher (задача и воркер)
docker logs fetcher-api --tail=50
docker logs fetcher-worker --tail=60
```

В логах воркера при успешной постановке задачи должны появиться строки вида: `Task fetcher.fetch_metadata[...] received`, `Starting metadata task for run_id=...`, `Starting metadata worker for run_id=...`. Дальше либо успешное выполнение, либо ошибка (например, таймаут к YouTube при `FETCHER_YOUTUBE_USE_YT_DLP=true` или при запросах metadata/video/comments).

---

## 7. Worker на хосте (если контейнер не имеет доступа к YouTube)

Если контейнер `fetcher-worker` не может достучаться до YouTube (SSL handshake timeout, firewall), а на **хосте** интернет есть, воркер можно запускать **на хосте**. Он подключается к тем же Postgres, Redis и MinIO по портам, проброшенным из Docker.

**Шаги:**

1. Оставить в Docker только сервисы без выхода в интернет: API, Postgres, Redis, MinIO. Воркер в Docker не запускать (или остановить):
   ```bash
   cd Fetcher
   docker compose up -d postgres redis minio fetcher-api
   # fetcher-worker не поднимать или: docker compose stop fetcher-worker
   ```

2. Создать бакеты в MinIO (один раз, те же переменные что у воркера — localhost:9000):
   ```bash
   cd Fetcher
   source .venv/bin/activate
   export FETCHER_S3_ENDPOINT_URL=http://localhost:9000
   export FETCHER_S3_ACCESS_KEY=minioadmin
   export FETCHER_S3_SECRET_KEY=minioadmin123
   PYTHONPATH=. python scripts/init_minio_buckets.py
   ```

3. На хосте запустить воркер:
   ```bash
   ./scripts/run_worker_on_host.sh
   ```

Скрипт `scripts/run_worker_on_host.sh` выставляет переменные для подключения к localhost:5433 (Postgres), localhost:6379 (Redis), localhost:9000 (MinIO) и запускает Celery с тем же списком очередей. По умолчанию на хосте включён реальный доступ к YouTube (`FETCHER_YOUTUBE_USE_YT_DLP=true`). Если MinIO ещё не имел бакетов, один раз выполните `scripts/init_minio_buckets.py` (см. шаг 2 выше).

**Проверка:** создать run через Backend, в логах воркера на хосте должны появиться `Task fetcher.fetch_metadata received`, затем успешная загрузка метаданных с YouTube и запись в S3 (или следующая ошибка по шагам пайплайна).

---

## 8. Прокси для доступа к YouTube (ограниченный доступ)

Если YouTube недоступен с хоста (например, при ограничениях по региону), запросы к YouTube можно пускать через HTTP/SOCKS5-прокси. Fetcher передаёт прокси в yt-dlp для metadata, download и comments.

**Переменные окружения:**

- `FETCHER_ENABLE_PROXIES=true` — включить использование прокси.
- `FETCHER_PROXIES` — один URL или несколько через запятую. Примеры:
  - `http://127.0.0.1:1080`
  - `socks5://user:password@proxy.example.com:1080`
  - `http://host1:8080,http://host2:8080` (round-robin).

**Пример запуска воркера на хосте с прокси:**

```bash
cd Fetcher
export FETCHER_ENABLE_PROXIES=true
export FETCHER_PROXIES="http://127.0.0.1:1080"   # подставьте свой прокси
./scripts/run_worker_on_host.sh
```

Прокси должен быть запущен локально или на доступном хосте (VPN, корпоративный прокси, резидентный прокси и т.п.). В Docker то же самое: задать `FETCHER_ENABLE_PROXIES=true` и `FETCHER_PROXIES=...` в `environment` сервиса `fetcher-worker` в docker-compose.

---

## 9. Известные ограничения

- **Синхронизация статуса Backend ↔ Fetcher**: Backend обновляет `ingestion_status`, `fetcher_stage`, `fetcher_error_*` через периодическую задачу Celery (`sync_ingestion_run_status`). До первого успешного sync ответ `GET /api/runs/{run_id}` может содержать `fetcher_stage: null` и `ingestion_status: "pending"`.
- **Доступ к YouTube из контейнера**: при отсутствии исходящего доступа к youtube.com этапы metadata/video/comments будут падать (SSL handshake timeout или connection refused). **Обход:** запускать Fetcher Celery worker на хосте (см. раздел 7); для проверки только цепочки Backend → Fetcher API → Celery → воркер достаточно и в контейнере.
- **Часовой пояс / drift**: в логах Celery может появляться предупреждение о большом drift между хостом и контейнером (например, 36000 s); на работу очередей это не влияет.

---

## 10. Краткий чеклист изменений в репозитории

| Компонент | Изменение |
|-----------|-----------|
| **Backend** | `dbv2/models.py`: `foreign_keys="WorkspaceMember.user_id"` в `User.memberships`. `auth.py`: `CryptContext(schemes=["pbkdf2_sha256"])`. |
| **Fetcher docker-compose** | Для api/worker/beat: `FETCHER_POSTGRES_DSN`, `FETCHER_REDIS_URL`, `FETCHER_S3_*`, `FETCHER_BUCKET_RAW`, `FETCHER_YOUTUBE_USE_YT_DLP=false`. Команда воркера: `-Q fetcher.high,fetcher.normal,fetcher.low,fetch.metadata,fetch.video,fetch.comments,fetch.finalize,fetch.maintenance`. |
| **Fetcher config** | `youtube_use_yt_dlp: bool = True`, описание и переменная `FETCHER_YOUTUBE_USE_YT_DLP`. |
| **Fetcher orchestrator** | В `normalize_source` для YouTube: ветка по `settings.youtube_use_yt_dlp`; при False — парсинг URL без yt-dlp. |
| **Fetcher tasks** | Импорты: `VideoSource` из `.models`; `publish_job_started`, `publish_job_finished`, `publish_job_failed` из `.events`. |

---

## 11. Связанные документы

- [E2E_YOUTUBE_INGESTION_CHECKLIST.md](E2E_YOUTUBE_INGESTION_CHECKLIST.md) — общий чеклист E2E по шагам (создание run, синхронизация, trigger, DataProcessor).
- [Fetcher/docs/E2E_FETCHER_DATAPROCESSOR.md](../Fetcher/docs/E2E_FETCHER_DATAPROCESSOR.md) — E2E Fetcher → DataProcessor, примеры curl к Fetcher напрямую.
- [Fetcher/docs/TESTING_PLAN.md](../Fetcher/docs/TESTING_PLAN.md) — стратегия тестирования Fetcher, unit/integration/chaos/e2e, журнал прогресса.
---

## Навигация

[Vault](MAIN_INDEX.md)
