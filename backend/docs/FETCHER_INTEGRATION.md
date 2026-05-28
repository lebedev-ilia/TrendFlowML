# Интеграция Backend ↔ Fetcher

Документ описывает настройку и использование интеграции Backend с сервисом **Fetcher** (ingestion видео с YouTube и других платформ). Общий анализ и план фаз — в [docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md](../../docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md).

---

## 1. Назначение

- **Fetcher** отвечает за сбор данных о видео: метаданные, комментарии, скачивание, сохранение в object storage и формирование `manifest.json` для DataProcessor.
- **Backend** по контракту создаёт run (UUID), передаёт его в Fetcher и в дальнейшем может запрашивать статус, manifest и артефакты для запуска DataProcessor.

Текущая реализация в Backend:

- **Phase 0**: конфигурация URL и API key Fetcher; HTTP-клиент (`fetcher_client`: create_run, get_run, manifest, artifacts).
- **Phase 1**: канонический сценарий «run по YouTube URL»:
  - `POST /api/runs` — создание run в БД (таблица `core.ingestion_runs`), вызов Fetcher `POST /api/v1/runs` с тем же run_id; опционально заголовок `Idempotency-Key`.
  - `GET /api/runs` — список run'ов пользователя (опционально фильтр по workspace_id).
  - `GET /api/runs/{run_id}` — детали run'а (доступ только к своим).
- **Phase 2**: после finalize Fetcher вызывает Backend, Backend запускает DataProcessor:
  - `POST /api/runs/{run_id}/trigger-processing` — вызывается **Fetcher'ом** после перевода run в COMPLETED; ставит задачу `process_ingestion_run(run_id)` в очередь. Задача забирает manifest и артефакты из Fetcher, скачивает видео и отправляет в DataProcessor.
  - Опциональная аутентификация: заголовок `X-API-Key` (значение `TF_BACKEND_RUN_TRIGGER_API_KEY`), если ключ задан в Backend.
  - В Fetcher задаются `FETCHER_BACKEND_BASE_URL` и при необходимости `FETCHER_BACKEND_TRIGGER_API_KEY`.

---

## 2. Конфигурация (Backend)

Переменные окружения (префикс `TF_BACKEND_`):

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `TF_BACKEND_FETCHER_API_URL` | Базовый URL Fetcher API (без завершающего `/`) | `http://localhost:8000` |
| `TF_BACKEND_FETCHER_API_KEY` | API key для заголовка `X-API-Key` (если Fetcher требует аутентификацию) | — (не задан) |
| `TF_BACKEND_FETCHER_TIMEOUT_SECONDS` | Таймаут HTTP-запросов к Fetcher (секунды) | `30.0` |
| `TF_BACKEND_RUN_TRIGGER_API_KEY` | (Phase 2) Если задан, то `POST .../trigger-processing` принимается только с заголовком `X-API-Key` с этим значением (вызов от Fetcher) | — (не задан) |
| `TF_BACKEND_INGESTION_SYNC_INTERVAL_SECONDS` | (Phase 4) Интервал вызова задачи `sync_ingestion_run_status` при использовании Celery beat (секунды) | `20` |

Файл: `backend/app/config.py` (класс `Settings`).

Пример для локальной разработки (Fetcher без auth):

```bash
export TF_BACKEND_FETCHER_API_URL=http://localhost:8000
# TF_BACKEND_FETCHER_API_KEY не задаём, если Fetcher не требует
```

Пример с аутентификацией:

```bash
export TF_BACKEND_FETCHER_API_URL=http://fetcher:8000
export TF_BACKEND_FETCHER_API_KEY=your-fetcher-api-key
export TF_BACKEND_FETCHER_TIMEOUT_SECONDS=60
```

---

## 3. HTTP-клиент (модуль fetcher_client)

Модуль: **`backend/app/services/fetcher_client.py`**.

### 3.1 Методы

- **`create_run(run_id, source_url, ...)`** / **`create_run_async(...)`**  
  `POST /api/v1/runs` — создание run в Fetcher. Обязательные аргументы: `run_id` (UUID), `source_url` (URL видео). Опционально: `platform`, `priority`, `webhook_url`, `idempotency_key`. Возвращает ответ Fetcher (run_id, status, message и т.д.).

- **`get_run(run_id)`** / **`get_run_async(run_id)`**  
  `GET /api/v1/runs/{run_id}` — получение статуса run (status, stage, timestamps, error_code и т.д.).

- **`get_run_manifest(run_id)`** / **`get_run_manifest_async(run_id)`**  
  `GET /api/v1/runs/{run_id}/manifest` — получение manifest (контракт Fetcher ↔ DataProcessor: пути к video_file, meta_file, comments_file в storage). Доступен после успешного завершения ingestion (status=COMPLETED).

- **`get_run_artifacts(run_id)`** / **`get_run_artifacts_async(run_id)`**  
  `GET /api/v1/runs/{run_id}/artifacts` — получение артефактов с signed URLs для скачивания.

Во всех функциях опциональный аргумент **`settings`**: если не передан, используется `Settings()` (текущие переменные окружения).

### 3.2 Ошибки

- **`httpx.HTTPStatusError`** — при 4xx/5xx от Fetcher (в т.ч. 404, 409).
- **`httpx.RequestError`** — при сетевой ошибке или таймауте.

Обработку ошибок и повторные попытки вызывающий код должен реализовывать сам.

### 3.3 Пример использования (синхронный)

```python
from uuid import uuid4
from app.services.fetcher_client import create_run, get_run, get_run_manifest

run_id = uuid4()
# Создать run в Fetcher
resp = create_run(
    run_id,
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    platform="youtube",
    webhook_url="https://backend.example.com/webhooks/fetcher",
)
assert resp["status"] in ("PENDING", "COMPLETED")  # или уже completed из кеша

# Позже — получить статус
status = get_run(run_id)
# Позже — получить manifest для передачи в DataProcessor
if status.get("status") == "COMPLETED":
    manifest = get_run_manifest(run_id)
```

### 3.4 Контракт с Fetcher

- Запросы/ответы соответствуют **Fetcher API**: `Fetcher/fetcher/schemas/api.py`, `Fetcher/docs/BACKEND_CONTRACTS.md`.
- Аутентификация: заголовок **`X-API-Key`** (если в Backend задан `fetcher_api_key`).
- Идемпотентность создания: заголовок **`Idempotency-Key`** (опционально).

---

## 4. API Runs (Phase 1)

Endpoint'ы Backend для создания и просмотра run'ов ингестиции:

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/runs` | Создать run по `source_url` (YouTube и др.). Body: `{ "source_url": "https://...", "workspace_id": "uuid?" }`. Заголовок `Idempotency-Key` (опционально) — при повторном запросе возвращается существующий run. Требует `Authorization: Bearer <token>`. |
| `GET` | `/api/runs` | Список run'ов текущего пользователя. Query: `workspace_id?`, `limit?` (по умолчанию 50). |
| `GET` | `/api/runs/{run_id}` | Детали run'а (только свой). Ответ включает Phase 4: `fetcher_stage`, `fetcher_error_code`, `fetcher_error_message`. |
| `POST` | `/api/runs/{run_id}/trigger-processing` | (Phase 2) Вызов от Fetcher после finalize. При заданном `TF_BACKEND_RUN_TRIGGER_API_KEY` требуется `X-API-Key`. Ответ: 202 Accepted. (Phase 5.4: идемпотентность — при уже запущенной обработке, `ingestion_status=processing`, повторный вызов возвращает 202 без постановки задачи.) |
| `WS` | `/api/runs/{run_id}/events` | (Phase 4) WebSocket поток событий: `run.status_changed`, `run.stage_changed` и др. Обязателен query-параметр `token` (JWT); доступ только к своим run'ам (по user_id). |

Модель БД: `core.ingestion_runs` (run_id PK, user_id, source_url, workspace_id, ingestion_status, idempotency_key, fetcher_stage, fetcher_error_code, fetcher_error_message, created_at, updated_at). Миграции: `0003_add_ingestion_runs.py`, `0004_ingestion_runs_fetcher_fields.py`. См. `backend/app/dbv2/models.py` (IngestionRun).

### Задача process_ingestion_run (Phase 2, Phase 3)

Celery-задача `process_ingestion_run(run_id)` выполняется после вызова trigger-processing. **Phase 3:** использует `build_ingestion_payload_from_fetcher(run_id)` для получения manifest и артефактов из Fetcher; передаёт в DataProcessor **video_url** (signed URL) — DataProcessor сам скачивает видео в свой кэш. Backend больше не скачивает видео во временный файл. Профиль — ingestion-default. См. `backend/app/tasks/ingestion.py`, `backend/app/services/dataprocessor_adapter.py`, `docs/PHASE3_ARTIFACTS_CONTRACT.md`.

---

## 6. Phase 3: Пути к артефактам и video_url

### 6.1 Единая точка формирования payload (ingestion)

Модуль **`backend/app/services/dataprocessor_adapter.py`**:

- **`build_ingestion_payload_from_fetcher(run_id, settings=None)`** — запрашивает у Fetcher `get_run_manifest` и `get_run_artifacts`, извлекает signed URL для `video_file`, возвращает **`IngestionPayloadFromFetcher`** (run_id, platform_id, video_id, profile_config, video_url). Не скачивает видео.
- **`IngestionPayloadFromFetcher`** — dataclass с полями `video_url` (обязательно для варианта B) и опционально `video_path` (если Backend сам скачал — fallback вариант A).

Контракт вариантов A/B и кэш DataProcessor: **`docs/PHASE3_ARTIFACTS_CONTRACT.md`**.

### 6.2 Вызов DataProcessor с video_path или video_url

**`backend/app/services/dataprocessor.py`**:

- **`run_dataprocessor_async(..., video_path=None, video_url=None, ...)`** — принимает либо локальный путь к видео (`video_path`), либо URL для скачивания (`video_url`). В JSON-запросе к DataProcessor API передаётся соответствующее поле; при наличии `video_url` DataProcessor скачивает файл в свой кэш и обрабатывает его.

### 6.3 DataProcessor: приём video_url и кэш

- **DataProcessor API** (`DataProcessor/api/schemas/requests.py`): в **ProcessRequest** добавлено опциональное поле **`video_url`**. Допускается ровно один из `video_path` или `video_url`.
- В endpoint'е `/api/v1/process` при наличии `video_url` запрос скачивается в кэш (`api/utils/video_url_cache.py`), путь подставляется в `video_path`, далее пайплайн работает с локальным файлом.
- Конфиг DataProcessor: **`video_url_cache_dir`** (переменная окружения или по умолчанию `{первая из allowed_video_paths}/_url_cache`). См. `DataProcessor/api/config.py`.

Конфигурация Fetcher для вызова Backend (Phase 2): в Fetcher задаются `FETCHER_BACKEND_BASE_URL` и при необходимости `FETCHER_BACKEND_TRIGGER_API_KEY`. После COMPLETED в finalize вызывается `POST {base_url}/api/runs/{run_id}/trigger-processing`. См. `Fetcher/fetcher/config.py`, `Fetcher/fetcher/tasks.py` (finalize_task).

---

## 7. Phase 4: События и статусы ingestion

**Цель:** Backend знает статус и стадию Fetcher и отдаёт их в UI (REST и WebSocket).

### 7.1 Транспорт: polling

События Fetcher → Backend передаются через **polling**: периодическая задача опрашивает Fetcher `GET /api/v1/runs/{run_id}` для run'ов в статусе `pending` или `running`, обновляет БД и публикует события в Redis. Альтернативы (Redis pubsub с Fetcher, Kafka) не требуются на текущем этапе.

### 7.2 Задача sync_ingestion_run_status

- **Модуль:** `backend/app/tasks/ingestion.py` (Celery task `sync_ingestion_run_status`).
- **Поведение:** выбирает из БД run'ы с `ingestion_status in ('pending', 'running')` (лимит `batch_size`, по умолчанию 50), для каждого вызывает `fetcher_client.get_run(run_id)`, маппит статус Fetcher (PENDING/RUNNING/COMPLETED/FAILED) в `ingestion_status`, записывает `fetcher_stage`, `fetcher_error_code`, `fetcher_error_message`, коммитит, публикует `run.status_changed` и при смене стадии `run.stage_changed` через `publish_run_event(run_id, payload)`.
- **Расписание:** запуск по Celery beat каждые `TF_BACKEND_INGESTION_SYNC_INTERVAL_SECONDS` секунд (по умолчанию 20). Запуск beat: `celery -A app.worker:celery_app beat -l info`; worker: `celery -A app.worker:celery_app worker -l info`. Конфиг beat: `backend/app/worker.py` (`beat_schedule`). В `backend/docker-compose.yml` сервис **beat не поднимается** — только при отдельном запуске.

### 7.3 WebSocket GET /api/runs/{run_id}/events

- **Роутер:** `backend/app/routers/runs.py` (endpoint `ws_run_events`).
- **Авторизация:** обязателен query-параметр `token` (JWT). Проверяется, что run принадлежит текущему пользователю (user_id); при отсутствии или невалидном token или при доступе к чужому run соединение закрывается с кодом 1008 (policy violation).
- **Поведение:** клиент подключается по WebSocket к `/api/runs/{run_id}/events?token=<JWT>`; сервер подписывается на Redis-канал `run:{run_id}` (`subscribe_run_events`) и пересылает каждое событие клиенту в виде JSON. События приходят при обновлении статуса (задача `sync_ingestion_run_status`) и при прогрессе DataProcessor (задача `process_analysis_job` и др.).
- **Формат событий:** `{ "type": "run.status_changed" | "run.stage_changed" | ..., "run_id": "...", "ts": "...", "payload": { ... } }`. См. `backend/docs/EVENTS_AND_LOGGING.md`.

### 7.4 Поля ответа GET /api/runs/{run_id}

В ответ добавлены опциональные поля (Phase 4): `fetcher_stage`, `fetcher_error_code`, `fetcher_error_message` (синхронизируются из Fetcher задачей `sync_ingestion_run_status`).

---

## 5. Тесты

Unit-тесты клиента: **`backend/tests/test_fetcher_client.py`**.

Проверяется:

- Отправка POST с `run_id`, `source_url`, заголовками `X-API-Key` и при необходимости `Idempotency-Key`.
- Отправка GET на `/api/v1/runs/{run_id}`, `/manifest`, `/artifacts` и разбор ответа.
- Проброс `httpx.HTTPStatusError` при 4xx/5xx.
- Асинхронные варианты (`create_run_async`, `get_run_async`, `get_run_manifest_async`, `get_run_artifacts_async`).

Запуск (из каталога `backend`):

```bash
pytest tests/test_fetcher_client.py -v
```

Тесты используют моки `httpx.Client` / `httpx.AsyncClient`, без реального Fetcher.

---

## 8. Связанные документы

- [docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md](../../docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md) — анализ интеграции и план фаз (0–5).
- [docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md](../../docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md) — чеклист E2E: YouTube URL → результат (Phase 5).
- [docs/PHASE3_ARTIFACTS_CONTRACT.md](../../docs/PHASE3_ARTIFACTS_CONTRACT.md) — контракт доступа к артефактам Fetcher (video_path / video_url, кэш DataProcessor).
- [GAPS_AND_ALIGNMENT.md](./GAPS_AND_ALIGNMENT.md) — разрывы между контрактом и реализацией (в т.ч. «fetcher отсутствует»; Phase 0 закрывает клиент и конфиг).
- [reference/backend_qna_contracts.md](./reference/backend_qna_contracts.md) — контракт Fetcher orchestration (run_id, fetch_video, process_run).
- Fetcher: `Fetcher/docs/BACKEND_CONTRACTS.md`, `Fetcher/fetcher/api.py` (эндпоинты).
