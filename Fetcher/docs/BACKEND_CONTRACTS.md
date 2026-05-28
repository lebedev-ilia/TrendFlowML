## Backend контракты Fetcher

Этот документ описывает **контракты между Backend, Fetcher и DataProcessor** для ingestion‑pipeline’а.

Основан на:

- архитектуре Fetcher (`plan.md`);
- чеклисте Phase 0 (`checklist.md`);
- стиле документации `DataProcessor/api/docs`.

Документ фокусируется на:

- жизненном цикле `run_id`;
- схеме событий pipeline;
- контракте `manifest.json` (версионирование схемы);
- интерфейсе платформенных адаптеров (`PlatformAdapter`);
- привязке контрактов к pydantic‑схемам в `Fetcher/schemas/*`.

---

## 1. Жизненный цикл run_id

### 1.1. Общий поток

Логический поток между Backend, Fetcher и DataProcessor:

```text
User / Backend
   │  POST /runs (YouTube URL)
   ▼
Backend API
   │  создаёт run_id (UUID), запись в backend DB
   │  записывает video_source (url, platform='youtube')
   │  enqueue fetch_video(run_id) → очередь Fetcher
   ▼
Fetcher Orchestrator
   │  state machine Fetcher (PENDING → ... → FINALIZING)
   │  пишет артефакты и manifest.json в storage
   │  публикует события в Redis / backend events
   ▼
DataProcessor
   │  запускается по manifest.json (через backend / DP API)
   ▼
Backend
   │  агрегирует статус (ingestion + processing)
```

### 1.2. Статусы run_id на уровне Fetcher

Fetcher хранит собственный state machine (см. `plan.md`, раздел 4.2). Для контрактов с Backend важно:

- **Список статусов Fetcher**:
  - `PENDING`
  - `NORMALIZING_SOURCE`
  - `CHECKING_CACHE`
  - `FETCHING_METADATA`
  - `FETCHING_CHANNEL`
  - `FETCHING_COMMENTS`
  - `DOWNLOADING_VIDEO`
  - `UPLOADING_ARTIFACTS`
  - `FINALIZING`
  - `COMPLETED`
  - `FAILED`
- **Инварианты**:
  - `run_id` создаётся строго на стороне Backend (source of truth — backend DB).
  - В Fetcher `run_id` используется как внешний ключ (таблица `runs` в Fetcher БД).
  - Переходы между состояниями Fetcher **идемпотентны** (повторный запуск шага не ломает артефакты).

### 1.3. Отображение статусов в Backend

Backend хранит свой статус run’а (например, `ingestion_status`, `processing_status`, `overall_status`). Для интеграции с Fetcher:

- **Минимальный контракт маппинга**:
  - `FETCHING_METADATA` / `FETCHING_CHANNEL` / `FETCHING_COMMENTS` / `DOWNLOADING_VIDEO` / `UPLOADING_ARTIFACTS` → `ingestion_status = "running"`.
  - `COMPLETED` → `ingestion_status = "success"`.
  - `FAILED` → `ingestion_status = "error"`, с возможностью сохранить код причины.
- **Ошибки Fetcher**:
  - Fetcher обязан возвращать машинно‑читаемый `error_code` и `error_reason` (см. события ниже).
  - Backend не интерпретирует детали, но может маппить в человекочитаемое описание для UI.

### 1.4. Идентификаторы и инварианты

- `run_id`: UUID, генерируется в Backend, неизменяем, используется во всех системах.
- `platform`: строка (`"youtube"`, `"tiktok"`, ...), задаётся Backend или вычисляется Fetcher при нормализации URL.
- `platform_video_id`: строка, нормализованный идентификатор видео на платформе (например, YouTube video id).
- **Инварианты**:
  - для каждого `(platform, platform_video_id)` в Fetcher существует не более одной записи `videos` (глобальный кеш);
  - все артефакты, связанные с run’ом, ссылаются на один `video_id` в Fetcher БД;
  - Backend никогда не записывает напрямую в Fetcher БД, только через очередь / API.

---

## 2. Схема событий pipeline

### 2.1. Цели и источники событий

События Fetcher используются для:

- live‑обновлений в UI (через Backend WebSocket: `run.status_changed`, `log.line`, и т.п.);
- отладки и аудита;
- мониторинга (метрики/логи).

Fetcher публикует события:

- в Redis pubsub / Streams (`fetcher:run:{run_id}` или аналогичный канал);
- опционально — в backend сервис событий, который уже ретранслирует их в WebSocket (`backend/docs/EVENTS_AND_LOGGING.md`).

### 2.2. Базовый формат события

**Envelope события** (логический контракт, не навязывающий конкретную реализацию транспорта):

```json
{
  "event_version": "1.0",
  "source": "fetcher",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "platform": "youtube",
  "platform_video_id": "dQw4w9WgXcQ",
  "type": "run.status_changed",
  "stage": "DOWNLOADING_VIDEO",
  "status": "running",
  "payload": {},
  "ts": "2026-03-05T12:00:00.000Z"
}
```

**Обязательные поля**:

- `event_version` — строка, версионирование схемы (начинаем с `"1.0"`).
- `source` — строка, `"fetcher"`.
- `run_id` — UUID в строковом виде.
- `type` — строка, тип события (см. ниже).
- `ts` — ISO 8601 timestamp в UTC.

**Рекомендуемые поля**:

- `platform`, `platform_video_id` — помогают backend и UI не ходить в БД за базовой идентификацией.
- `stage` — текущий шаг state machine Fetcher.
- `status` — агрегированный статус шага/потока.
- `payload` — структура, зависящая от `type` (см. подтипы).

### 2.3. Типы событий

Fetcher должен поддерживать как минимум следующие типы (согласованные с backend’ом):

- `run.status_changed`
  - **Когда**: при изменении статуса Fetcher run’а (см. 1.2).
  - **payload**:
    - `old_status`: предыдущий статус Fetcher.
    - `new_status`: новый статус Fetcher.
    - `reason`: опциональная строка (для `FAILED`).
    - `error_code`: опциональный машинно‑читаемый код (`YOUTUBE_429`, `VIDEO_NOT_FOUND`, `DOWNLOAD_TIMEOUT`, ...).
- `run.stage_changed`
  - **Когда**: при смене `stage` ingestion pipeline (например, `FETCHING_METADATA` → `DOWNLOADING_VIDEO`).
  - **payload**:
    - `old_stage`
    - `new_stage`
- `job.started`
  - **Когда**: старт конкретного job’а (`fetch_metadata`, `download_video`, `fetch_comments`, `finalize`).
  - **payload**:
    - `job_type`: строка.
    - `job_id`: UUID (идентификатор записи в `fetch_jobs`).
- `job.finished`
  - **Когда**: успешное завершение job’а.
  - **payload**:
    - `job_type`
    - `job_id`
    - `duration_ms`
- `job.failed`
  - **Когда**: job завершился с ошибкой.
  - **payload**:
    - `job_type`
    - `job_id`
    - `error_code`
    - `error_message`
- `log.line`
  - **Когда**: важная строка лога ingestion (агрегируется/фильтруется по уровню).
  - **payload**:
    - `level`: `"info" | "warning" | "error"`.
    - `message`: текст.

### 2.4. Совместимость с backend WebSocket

Backend уже использует следующие типы событий (`backend/docs/EVENTS_AND_LOGGING.md`):

- `run.status_changed`
- `run.stage_changed`
- `component.started`
- `component.finished`
- `log.line`

Fetcher **не обязан** публиковать `component.*` (они относятся к DataProcessor), но:

- типы `run.status_changed` / `run.stage_changed` / `log.line` должны быть согласованы по полям;
- Backend может использовать события Fetcher для того же WebSocket‑канала, что и события DataProcessor (единый поток событий run’а).

---

## 3. Контракт manifest.json и версионирование

### 3.1. Роль manifest.json

`manifest.json` — основной контракт между Fetcher и DataProcessor:

- Fetcher **гарантирует структуру и доступность** артефактов (`video.mp4`, `meta.json`, `comments.json`, ...).
- DataProcessor читает **только manifest**, не зная внутренней структуры бакетов Fetcher.

### 3.2. Базовая схема manifest.json

Минимальная схема (MVP, совместима с `plan.md`, раздел 6.3).  
Фактическая Python‑схема описана в `Fetcher/schemas/manifest.py` (`FetcherManifest`, `ArtifactInfo`):

```json
{
  "manifest_version": "1.0",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "abc123",
  "platform": "youtube",
  "duration_seconds": 540,
  "storage_layout_version": "1.0",
  "artifacts": {
    "video_file": {
      "path": "raw/youtube/2026/03/05/abc123/video.mp4",
      "checksum": "sha256:...",
      "size_bytes": 123456789
    },
    "meta_file": {
      "path": "raw/youtube/2026/03/05/abc123/meta.json",
      "checksum": "sha256:...",
      "size_bytes": 12345
    },
    "comments_file": {
      "path": "raw/youtube/2026/03/05/abc123/comments.json",
      "checksum": "sha256:...",
      "size_bytes": 67890,
      "comment_count": 100
    }
  }
}
```

**Обязательные поля**:

- `manifest_version` — строка, версионирование **контракта** между Fetcher и DataProcessor.
- `run_id` — UUID, тот же, что в Backend/Fetcher.
- `video_id` — нормализованный идентификатор видео (совпадает с `platform_video_id` или маппится однозначно).
- `platform` — `"youtube"` и т.п.
- `duration_seconds` — целое/float, длительность видео.
- `artifacts.video_file.path`, `.meta_file.path`, `.comments_file.path` — относительные пути внутри `video-analytics-raw` бакета.

**Рекомендуемые поля**:

- `storage_layout_version` — версионирование схемы путей в object storage (на случай будущих изменений layout’а).
- `checksum` для критичных артефактов (`video_file`, `meta_file`, `comments_file`).
- `size_bytes` — размер файлов.

### 3.3. Версионирование манифеста

Контракт версионирования:

- `manifest_version`:
  - **`1.x`** — текущее поколение контракта Fetcher → DataProcessor;
  - минорные изменения (добавление необязательных полей) не ломают потребителей;
  - мажорное изменение (изменение/удаление обязательных полей) требует поддержки нескольких версий на стороне DataProcessor.
- `storage_layout_version`:
  - позволяет менять layout бакетов (директории, схемы ключей) без изменения логического контракта;
  - при изменении layout Fetcher гарантирует, что DataProcessor с нужной версией понимает новый layout.

### 3.4. Инварианты manifest.json

- manifest записывается **только после** успешного завершения всех обязательных ingestion‑шагов (metadata/video/comments).
- manifest должен быть **атомарен** с точки зрения читателя:
  - либо отсутствует (run ещё не готов / провалился),
  - либо содержит консистентные ссылки на существующие и проверенные артефакты.
- checksum и size в manifest соответствуют фактическим значениям в storage (проверяются Fetcher при upload’е).

---

## 4. Интерфейс PlatformAdapter

### 4.1. Цели

Platform adapters инкапсулируют особенности конкретных платформ (YouTube, TikTok, Instagram, …) и предоставляют **единый интерфейс** Fetcher Orchestrator’у.

Backend и DataProcessor **не должны знать** деталей адаптеров, только:

- `platform` (строка);
- допустимые значения `platform_video_id`;
- инварианты по артефактам и manifest.

### 4.2. Базовый интерфейс

Логический Python‑интерфейс (см. `plan.md`, раздел 2.3):

```python
class PlatformAdapter:
    def fetch_metadata(self, source: str, *, run_id: str) -> None: ...
    def download_video(self, source: str, *, run_id: str) -> None: ...
    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None: ...
```

**Контракт уровня платформы**:

- `source` — исходный URL или `platform_video_id` (зависит от этапа; Orchestrator отвечает за нормализацию).
- `run_id` — обязателен, используется для логирования/событий/связи с артефактами.
- `limit` — верхняя граница числа комментариев (по умолчанию соответствует чеклисту: ≤100).

### 4.3. Общие инварианты адаптеров

Все реализации `PlatformAdapter` (например, `YouTubeAdapter`) обязаны:

- **Идемпотентность**:
  - повторный вызов метода для того же `(platform, platform_video_id)` не создаёт дубликатов артефактов;
  - перед записью проверяется наличие arifact’а / записи в БД.
- **Ошибки**:
  - различать **retryable** (`RateLimit`, сетевые, временные `5xx`) и **non‑retryable** (`VIDEO_NOT_FOUND`, `PRIVATE_VIDEO`, `AGE_RESTRICTED` без cookies);
  - маппить ошибки в нормализованные `error_code`, используемые в событиях и логах.
- **Лимиты**:
  - уважать конфигурируемые лимиты времени (`timeout_seconds`) и попыток (`max_retries`);
  - использовать Redis‑rate‑лимитер и proxy‑pool (см. чеклист Phase 0/1).

### 4.4. Контракт включения/выключения платформ

Backend и DevOps должны иметь возможность включать/выключать платформы через конфигурацию:

- конфиг‑ключи уровня Fetcher (пример):  
  - `FETCHER_ENABLED_PLATFORMS = ["youtube"]`
  - `FETCHER_YOUTUBE_ENABLED = true`
- Orchestrator:
  - при попытке обработки `platform`, не входящей в `ENABLED_PLATFORMS`, возвращает контролируемую ошибку (`PLATFORM_DISABLED`) и не стартует ingestion.

---

## 5. Релиз и эволюция контрактов

### 5.1. Правила изменения контрактов

- Любое изменение:
  - `run_id` lifecycle (новые статусы),  
  - схемы событий,  
  - структуры `manifest.json`,  
  - интерфейса `PlatformAdapter`  
  
должно быть отражено:

- в этом документе;
- в связанной документации Backend / DataProcessor (если затрагиваются их интерфейсы);
- в соответствующих чеклистах (обновление фаз/задач).

### 5.2. Совместимость

- Добавление **новых необязательных полей** (с разумными значениями по умолчанию) считается backwards‑compatible.
- Изменение/удаление **обязательных полей** требует:
  - увеличения `manifest_version` (мажорная версия);
  - поддержки нескольких версий на стороне DataProcessor на переходный период;
  - согласованного релизного плана Backend/Fetcher/DataProcessor.

### 5.3. Связанные документы

- `Fetcher/docs/plan.md` — архитектура Fetcher и pipeline.
- `Fetcher/docs/checklist.md` — Phase 0…7, включая backend‑контракты.
- `DataProcessor/api/docs/DATAPROCESSOR_API_ARCHITECTURE.md` — архитектура DataProcessor API.
- `backend/docs/EVENTS_AND_LOGGING.md` — события и WebSocket‑протокол Backend.
 - `Fetcher/schemas/manifest.py` — pydantic‑схемы `FetcherManifest` и артефактов.
 - `Fetcher/schemas/events.py` — pydantic‑схемы событий Fetcher.


