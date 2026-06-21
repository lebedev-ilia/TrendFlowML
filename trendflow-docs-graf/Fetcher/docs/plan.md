## Fetcher: Data Ingestion Platform для YouTube и других платформ

Этот документ описывает **целевую архитектуру сервиса Fetcher** как production‑grade ingestion‑платформу, а не просто скрипт скачивания YouTube‑видео.  
Fetcher отвечает за надёжный и масштабируемый сбор **видео, метаданных, временных снэпшотов и комментариев** с YouTube (и других платформ), сохранение их в object storage/БД и запуск обработки в `DataProcessor`.

---

## 1. Роль Fetcher в общей архитектуре

**Общий поток:**

```text
User / Backend
   │  POST /runs (YouTube URL)
   ▼
Backend API
   │  создаёт run_id, пишет video_source
   │  enqueue fetch_video(run_id)
   ▼
Fetcher Orchestrator
   │  state machine, cache, планирование
   ▼
Fetcher Workers (metadata / video / comments)
   │  собирают артефакты, пишут в S3/MinIO + DB
   ▼
Artifact Builder
   │  собирает manifest.json для DataProcessor
   │  enqueue process_run(run_id)
   ▼
DataProcessor
   │  Segmenter / Audio / Text / Visual
   ▼
Result Store (NPZ + manifest.json)
```

**Основные задачи Fetcher:**

- **Download & normalize**:
  - скачивание видео (yt‑dlp),
  - нормализация контейнера/кодеков/качества через ffmpeg (например, до 720p).
- **Metadata ingestion**:
  - базовая meta: `title`, `description`, `tags`, `language`, `duration_seconds`, `published_at`,
  - данные о канале: `channel_id`, `channel_title`, `subscriber_count`, `video_count`, `view_count_channel`,
  - временные снэпшоты: `view_count`, `like_count`, `comment_count`, `subscriber_count` по времени.
- **Comments ingestion**:
  - сбор top‑N (обычно 100) релевантных комментариев с лайками, числом ответов, автором, датой.
- **Persistence**:
  - object storage (S3/MinIO) — `video.mp4`, `meta.json`, `comments.json`, `manifest.json`,
  - PostgreSQL — ссылки (`storage_key`), кеш по `platform_video_id`, состояние pipeline, логи.
- **Orchestration & reliability**:
  - state machine по шагам ingestion,
  - очередь задач (Celery/Redis на MVP, Kafka как опция для продакшена),
  - idempotency и cache‑reuse (не скачивать одно и то же видео по 100 раз),
  - устойчивость к ошибкам YouTube и сетевым сбоям.

---

## 2. Высокоуровневая архитектура Fetcher

### 2.1. Production‑архитектура (Celery/Redis)

```text
                Backend API
                     │
                     │ create run (YouTube URL)
                     ▼
              Message Queue
               (Redis/Celery)
                     │
                     ▼
           ┌─────────────────────┐
           │ Fetcher Orchestrator│
           └──────────┬──────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     Metadata      Video       Comments
      Worker       Worker        Worker
          │           │           │
          └──────┬────┴────┬──────┘
                 ▼         ▼
            Artifact Builder
                 │
                 ▼
            Object Storage
             (MinIO / S3)
                 │
                 ▼
        enqueue process_run(run_id)
                 │
                 ▼
            DataProcessor
```

### 2.2. Ключевые компоненты

- **Fetcher Orchestrator**
  - единственная публичная операция: `fetch_video(run_id)`;
  - читает источник (`video_sources`), нормализует URL → `platform_video_id` (например, YouTube `video_id`);
  - проверяет глобальный кеш:
    - если видео уже скачано → переиспользует артефакты и сразу enqueue `process_run(run_id)`;
    - если нет — ставит задачи `fetch_metadata`, `download_video`, `fetch_comments`.

- **Metadata Worker**
  - получает YouTube URL / `platform_video_id`;
  - использует `yt-dlp --dump-json` и/или YouTube Data API;
  - собирает meta:
    - `title`, `description`, `tags`, `language`, `duration_seconds`, `published_at`,
    - сведения о канале: `channel_id`, `channel_title`, `subscriber_count`, `video_count`, `view_count_channel`,
    - `thumbnails`, `formats`, `captions`, `chapters`, `age_limit`, `country` и т.п.;
  - сохраняет `meta.json` и `channel.json` в object storage + запись в таблицы `video_metadata`/`channel_metadata`.

- **Video Download Worker**
  - скачивает и нормализует видео:
    - `yt-dlp` для загрузки лучшего видео+аудио трека (с ограничением высоты, например `height<=720`),
    - `ffmpeg` для merge/normalize контейнера/кодеков/fps;
  - кладёт `video.mp4` в bucket `video-analytics-raw` (S3/MinIO);
  - удаляет временный файл из `/tmp`;
  - создаёт запись в таблице `artifacts` (`artifact_type = 'video_file'`).

- **Comments Worker**
  - собирает top‑N (обычно 100) релевантных комментариев:
    - текст, лайки, число ответов, автор, дата, `raw_json`;
  - может работать постранично (pagination/streaming) для уменьшения нагрузки и обхода лимитов;
  - складывает комментарии:
    - либо в таблицу `comments` (PostgreSQL) с колонкой `raw_json`,
    - либо в `comments.json` в S3 и ссылку в `artifacts`.

- **Artifact Builder**
  - ждёт завершения всех обязательных задач (metadata/video/comments);
  - собирает `manifest.json` для DataProcessor:
    - ссылки на `video.mp4`, `meta.json`, `comments.json`,
    - `video_id`, `platform`, `duration_seconds`, и др.;
  - публикует задачу `process_run(run_id)` в очередь DataProcessor.

### 2.3. Platform adapters (YouTube, TikTok, Instagram, …)

Fetcher изначально проектируется как платформенный:

- базовый интерфейс адаптера:

```python
class PlatformAdapter:
    def fetch_metadata(self, source: str) -> None: ...
    def download_video(self, source: str) -> None: ...
    def fetch_comments(self, source: str) -> None: ...
```

- реализация для YouTube:

```python
class YouTubeAdapter(PlatformAdapter):
    ...
```

Структура:

```text
platforms/
  base.py          # PlatformAdapter, общие модели
  youtube/
    metadata.py
    downloader.py
    comments.py
  tiktok/
    ...
  instagram/
    ...
```

Выбор адаптера и включение платформы может контролироваться feature‑флагами и конфигурацией.

---

## 3. Data model (PostgreSQL)

Fetcher имеет **свою БД**, отделённую от backend и DataProcessor.  
В ней хранятся:

- лёгкие метаданные и статусы,
- кеш и идентификаторы видео,
- пути к артефактам,
- логи и события pipeline.

Тяжёлые данные (видео, крупные JSON) лежат в object storage.

### 3.1. Основные сущности

- `runs` — связь с backend‑run (по `run_id`).
- `video_sources` — исходный URL и нормализованный `platform_video_id`.
- `videos` — глобальный кеш по `(platform, platform_video_id)`.
- `video_metadata` — сырая meta YouTube (title/description/... + `raw_json`).
- `channel_metadata` — метаданные канала (id/title, статистика, `raw_json`).
- `video_snapshots` — временные метрики (views/likes/comments/subs по снэпшотам).
- `comments` — комментарии (по одному ряду на комментарий, с `raw_json`).
- `artifacts` — артефакты в S3/MinIO (video/meta/comments/manifest/thumbnail/...).
- `fetch_jobs` — отдельные шаги pipeline (`fetch_metadata`, `download_video`, ...).
- `fetch_logs` — event‑лог pipeline по `run_id`/stage/level/message.

### 3.2. Примеры таблиц (упрощённо)

```sql
CREATE TABLE runs (
    id UUID PRIMARY KEY,                 -- run_id из backend
    source_type VARCHAR(20) NOT NULL,    -- youtube / tiktok / ...
    source_url  TEXT        NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    error       TEXT
);

CREATE TABLE video_sources (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(id),
    platform VARCHAR(20) NOT NULL,          -- 'youtube'
    url TEXT         NOT NULL,
    normalized_video_id VARCHAR(50),        -- 'abc123'
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE videos (
    id UUID PRIMARY KEY,
    platform VARCHAR(20) NOT NULL,
    platform_video_id VARCHAR(100) NOT NULL,
    channel_id VARCHAR(100),
    duration_seconds INT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(platform, platform_video_id)
);
```

```sql
CREATE TABLE video_metadata (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    title TEXT,
    description TEXT,
    language VARCHAR(10),
    duration_seconds INT,
    published_at TIMESTAMP,
    raw_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE channel_metadata (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    channel_id VARCHAR(100),
    channel_title TEXT,
    subscriber_count BIGINT,
    video_count INT,
    view_count BIGINT,
    raw_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

```sql
CREATE TABLE video_snapshots (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    snapshot_index INT,          -- 0,1,2,3 ...
    view_count    BIGINT,
    like_count    BIGINT,
    comment_count BIGINT,
    subscriber_count BIGINT,
    collected_at TIMESTAMP
);
```

```sql
CREATE TABLE comments (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    author TEXT,
    text   TEXT,
    like_count  INT,
    reply_count INT,
    published_at TIMESTAMP,
    raw_json JSONB
);
```

```sql
CREATE TABLE artifacts (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    artifact_type VARCHAR(30),   -- video_file / metadata_file / comments_file / manifest / thumbnail
    storage_path TEXT,           -- S3/MinIO key
    size_bytes BIGINT,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING / UPLOADING / COMPLETED / FAILED
    checksum TEXT,                                  -- SHA256 для валидации целостности
    created_at TIMESTAMP DEFAULT NOW()
);
```

```sql
CREATE TABLE fetch_jobs (
    id UUID PRIMARY KEY,
    run_id UUID REFERENCES runs(id),
    job_type VARCHAR(30),        -- fetch_metadata / download_video / fetch_comments / finalize
    status  VARCHAR(20),         -- pending / running / completed / failed
    retries INT DEFAULT 0,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    error TEXT
);
```

```sql
CREATE TABLE fetch_logs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID,
    stage  VARCHAR(50),
    level  VARCHAR(10),
    message   TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 4. Pipeline и state machine

### 4.1. Основные этапы pipeline

Логический pipeline для одного `run_id`:

```text
INIT
  ↓
NORMALIZE_SOURCE      (нормализация URL → platform_video_id)
  ↓
CHECK_CACHE           (глобальный кеш по videos/artifacts)
  ├── CACHE_HIT  → FINALIZE → TRIGGER_DATAPROCESSOR
  └── CACHE_MISS → FETCH_METADATA → FETCH_CHANNEL → FETCH_COMMENTS → DOWNLOAD_VIDEO
                                               ↓
                                         UPLOAD_ARTIFACTS → FINALIZE
                                                                ↓
                                                     TRIGGER_DATAPROCESSOR
```

### 4.2. Состояния run

Пример state machine:

```text
PENDING
NORMALIZING_SOURCE
CHECKING_CACHE
FETCHING_METADATA
FETCHING_CHANNEL
FETCHING_COMMENTS
DOWNLOADING_VIDEO
UPLOADING_ARTIFACTS
FINALIZING
COMPLETED
FAILED
```

**Ключевой инвариант:** каждый шаг **идемпотентен**.  
Если task выполнится дважды, артефакты не ломаются:

- перед каждым шагом проверяем, существует ли уже нужный артефакт/запись;
- если есть — пропускаем шаг и обновляем state.

Дополнительно для скачивания видео используется **distributed lock**, чтобы избежать дублирующихся скачиваний для одного и того же `platform_video_id`:

- при входе в `DOWNLOAD_VIDEO`:
  - пытаемся установить Redis‑ключ `lock:video:{platform}:{platform_video_id}` (например, через `SETNX`),
  - если lock установлен — скачиваем и загружаем видео, по завершении — снимаем lock,
  - если lock уже существует — либо ждём завершения (poll статуса артефакта), либо переиспользуем уже готовый артефакт.

### 4.3. Orchestrator (идея интерфейса)

```python
class FetchOrchestrator:
    def run(self, run_id: str) -> None:
        state = self.repo.get_state(run_id)

        if state == "PENDING":
            self.normalize_source(run_id)
        elif state == "NORMALIZING_SOURCE":
            self.check_cache(run_id)
        elif state == "CHECKING_CACHE":
            self.fetch_metadata(run_id)
        elif state == "FETCHING_METADATA":
            self.fetch_channel(run_id)
        elif state == "FETCHING_CHANNEL":
            self.fetch_comments(run_id)
        elif state == "FETCHING_COMMENTS":
            self.download_video(run_id)
        elif state == "DOWNLOADING_VIDEO":
            self.upload_artifacts(run_id)
        elif state == "UPLOADING_ARTIFACTS":
            self.finalize(run_id)
```

Каждый переход может быть реализован как отдельная Celery‑задача, которая:

- читает текущий state из БД,
- выполняет свой шаг,
- обновляет state и, при необходимости, ставит следующую задачу.

---

## 5. Интеграция с YouTube (yt‑dlp, лимиты, прокси)

### 5.1. Ограничения YouTube

- **Rate limits** (практические оценки):
  - ~200–500 metadata‑запросов / IP / час,
  - ~50–100 скачиваний видео / IP / час,
  - при превышении — HTTP 429 / 403.
- **Bot detection**:
  - анализ IP reputation, частоты запросов, cookies, user‑agent, TLS‑fingerprint.
- **Geo‑restrictions и age‑restriction**:
  - часть видео доступна только из определённых стран или только при наличии cookies.

### 5.2. yt‑dlp: базовые команды

**Получение метаданных:**

```bash
yt-dlp \
  --dump-json \
  --no-playlist \
  --skip-download \
  URL
```

**Скачивание видео (максимум 720p):**

```bash
yt-dlp \
  -f "bestvideo[height<=720]+bestaudio/best" \
  -o "/tmp/%(id)s.%(ext)s" \
  --no-playlist \
  --geo-bypass \
  --retries 5 \
  --fragment-retries 5 \
  --socket-timeout 30 \
  URL
```

**Python‑обёртка:**

```python
import yt_dlp

def fetch_metadata(url: str) -> dict:
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info
```

### 5.3. Прокси и anti‑rate‑limit

- Типы прокси:
  - datacenter — дешёвые и быстрые, но чаще банятся,
  - residential/mobile — дороже, но стабильнее.
- Таблица `proxies` (или внешний конфиг):
  - `proxy_url`, `country`, `health_score`, `success_rate`, `last_used`.
- Rotation:
  - при 429/403/timeout — выбор другого прокси,
  - Redis‑rate‑лимитер по IP/прокси и типу операции (metadata vs download).
- Поддержка cookies (`--cookies cookies.txt`) для age‑restricted видео.

### 5.4. Rate limiting layer и circuit breaker

Поверх rotation/proxy‑pool нужен **централизованный rate‑limiter**, чтобы не “выстрелить себе в ногу” 1000 запросами в минуту с одного IP.

Рекомендуемый подход — **Redis‑based token bucket / leaky bucket**, с отдельными лимитами для:

- metadata‑запросов (`rate:youtube:metadata:{ip}`),
- download‑запросов (`rate:youtube:download:{ip}`).

Упрощённый пример:

```python
def acquire_token(key: str, limit: int, window_sec: int) -> bool:
    count = redis.incr(key)
    if count == 1:
        redis.expire(key, window_sec)
    return count <= limit
```

Дополнительно вводится **circuit breaker**:

- считаем долю ошибок 429/403 за окно (например, последние 1–5 минут);
- если `429_rate` превышает порог:
  - временно блокируем download/metadata (например, на 5 минут),
  - продолжаем только легковесные операции (логирование, проверка кеша).

---

## 6. Storage архитектура (S3 / MinIO / layout)

### 6.1. Общая идея

- Fetcher **не хранит** видео долговременно на локальном диске:
  - `download → tmp (/tmp) → upload → rm tmp`;
- все тяжёлые артефакты живут в object storage;
- БД хранит только `storage_key` и лёгкие метаданные.

### 6.2. Buckets и layout

Рекомендуемая структура:

```text
video-analytics-raw/         # raw видео + сырые JSON
  raw/youtube/YYYY/MM/DD/VIDEO_ID/video.mp4
  raw/youtube/YYYY/MM/DD/VIDEO_ID/meta.json
  raw/youtube/YYYY/MM/DD/VIDEO_ID/comments.json

video-analytics-processed/   # артефакты DataProcessor
  processed/youtube/VIDEO_ID/...
```

Плюсы:

- равномерное распределение файлов по директориям,
- лёгкая чистка по дате,
- человекочитаемые пути (удобно дебажить).

### 6.3. Manifest для DataProcessor

Fetcher строит `manifest.json` (для конкретного видео/платформы), который является контрактом с DataProcessor:

```json
{
  "manifest_version": "1.0",
  "video_id": "abc123",
  "platform": "youtube",
  "duration_seconds": 540,
  "video_file": "raw/youtube/2026/03/05/abc123/video.mp4",
  "meta_file": "raw/youtube/2026/03/05/abc123/meta.json",
  "comments_file": "raw/youtube/2026/03/05/abc123/comments.json"
}
```

DataProcessor читает **только manifest**, не зная деталей внутреннего layout Fetcher.

### 6.4. Python‑пример upload

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
)

def upload_video(local_path: str, storage_key: str) -> None:
    s3.upload_file(
        local_path,
        "video-analytics-raw",
        storage_key,
    )
```

После upload Fetcher:

- считает SHA256‑checksum локального файла,
- записывает `checksum` и `status='COMPLETED'` в таблицу `artifacts`,
- удаляет временный файл из `/tmp`.

Если upload оборвался или checksum не совпала — выставляется `status='FAILED'`, и pipeline либо ретраит шаг, либо помечает run как `FAILED`.

### 6.5. Lifecycle‑политики хранения

При реальной нагрузке (десятки тысяч видео в день) raw‑storage быстро растёт до терабайт.  
Рекомендуется на уровне S3/MinIO включить **lifecycle‑политику**:

- для `video-analytics-raw` (raw‑видео):
  - удалить или перевести в “cold storage” через N дней (например, удалить через 30 дней),
- для `video-analytics-processed` (фичи/результаты):
  - хранить сильно дольше (либо “навсегда”, либо с отдельной политикой),
- временные buckets (`temp`) чистить агрессивно (TTL ~7 дней).

При этом ML‑pipeline и backend должны опираться на **derived features/NPZ**, а не на гарантированную доступность исходных `video.mp4`.

---

## 7. Queue и orchestration (Celery / Kafka)

### 7.1. MVP: Celery + Redis

Для MVP достаточно Celery + Redis:

- отдельные очереди:
  - `fetch.metadata`,
  - `fetch.video`,
  - `fetch.comments`,
  - `fetch.finalize`;
- простой роутинг задач по worker’ам (metadata/download/comments),
- приоритеты:
  - **high**: metadata + finalize (быстрое принятие решения и завершение run),
  - **medium**: comments,
  - **low**: download (самая тяжёлая операция, не должна “задавить” остальные).

В Kubernetes/Helm манифестах ресурсы воркеров делятся на “лёгкие” и “тяжёлые”:

- `fetcher-metadata` / `fetcher-comments`:
  - CPU: ~0.2–0.5,
  - RAM: ~256–512 MB;
- `fetcher-download`:
  - CPU: 1–2,
  - RAM: 1–2 GB,
  - ограниченное количество pod’ов для контроля нагрузки на сеть/диск.

Пример Celery‑задачи:

```python
@celery.task(bind=True, max_retries=5)
def fetch_metadata_task(self, run_id: str) -> None:
    try:
        service.fetch_metadata(run_id)
    except RateLimitError as exc:
        raise self.retry(exc=exc, countdown=60)
```

### 7.2. Production: Kafka (event‑driven, опционально)

Для масштабов 100k+ видео/день можно перейти на Kafka:

- топики:
  - `video.downloaded`,
  - `video.features`,
  - `video.failed` и т.п.;
- consumer‑группы обрабатывают разные стадии (frame/audio/comment processing и т.д.);
- Kafka даёт высокую пропускную способность и реплей событий.

---

## 8. Наблюдаемость и устойчивость

### 8.1. Метрики (Prometheus)

Минимальный набор:

- `fetcher_videos_downloaded_total`,
- `fetcher_videos_failed_total`,
- `fetcher_download_time_seconds` (histogram),
- `fetcher_metadata_latency_seconds`,
- `fetcher_comments_latency_seconds`,
- `fetcher_cache_hits_total`, `fetcher_cache_miss_total` (для вычисления `cache_hit_ratio`),
- `fetcher_youtube_429_total`, `fetcher_youtube_403_total`,
- `fetcher_video_size_bytes` (histogram).

### 8.2. Логи

- таблица `fetch_logs` + текстовые логи (stdout/stderr);
- каждая запись:
  - `run_id`, `stage`, `level`, `message`, `created_at`;
- ключевые стадии: `download_video`, `fetch_metadata`, `fetch_comments`, `upload_artifacts`, `finalize`.

### 8.3. Fault‑tolerance

- **Retry‑политика**:
  - ретраим rate‑limit/сетевые/прокси‑ошибки;
  - не ретраим `video removed/private/age_restricted` (non‑retryable).
- **Resume**:
  - pipeline продолжает выполнение после рестартов воркеров (state machine в БД).
- **Dead‑letter queue**:
  - все окончательно упавшие run’ы сводим в отдельный поток / таблицу для ручного анализа.

---

## 9. Tech stack и деплой

- **Язык**: Python.
- **Очередь**:
  - MVP: Celery + Redis,
  - Production (опция): Kafka.
- **Download/Meta**: `yt-dlp`, `ffmpeg`.
- **Storage**: MinIO (self‑hosted) / S3 (cloud).
- **БД**: PostgreSQL (для крупных таблиц `comments`/`fetch_logs` можно добавить partitioning).
- **Мониторинг**: Prometheus + Grafana.
- **Деплой** (Kubernetes):
  - `fetcher-orchestrator`,
  - `fetcher-metadata`,
  - `fetcher-download`,
  - `fetcher-comments` — отдельные deployments, масштабируемые независимо.

---

## 10. Нефункциональные требования и инварианты

- **Idempotency**:
  - каждый шаг pipeline можно выполнять повторно без побочных эффектов;
  - прежде чем писать артефакт, проверяем наличие по ключу.
- **Cache‑aware**:
  - глобальный кеш по `(platform, platform_video_id)` в таблицах `videos`/`artifacts`;
  - повторные run’ы для одного URL переиспользуют уже скачанные артефакты.
- **Scalability**:
  - горизонтальное масштабирование воркеров по типам задач (metadata / download / comments);
  - отдельные очереди для тяжёлых задач (download).
- **Security & privacy**:
  - PII‑scanner для комментариев (e‑mail/телефоны/ссылки),
  - retention‑политика для raw комментариев и, при необходимости, для raw видео,
  - (опционально) OAuth‑верификация владельца канала для долгого хранения raw‑данных.

---

## 11. Roadmap (MVP → Production)

**MVP (целевой первый этап):**

- Поддержка YouTube как единственной платформы (`platform = 'youtube'`).
- Очередь Celery + Redis.
- Компоненты:
  - Orchestrator с простой state machine,
  - Metadata Worker (yt‑dlp `--dump-json`),
  - Video Download Worker (720p, базовые флаги),
  - Comments Worker (top‑N комментариев),
  - Artifact Builder + `manifest.json` для DataProcessor.
- Простая БД‑схема (`runs`, `video_sources`, `videos`, `video_metadata`, `artifacts`, `fetch_jobs`, `fetch_logs`).
- Базовые метрики + логи.

**Production v1:**

- Расширенный state machine (подробные состояния и причины ошибок).
- Продуманный cache reuse (videos/artifacts).
- Proxy‑pool, rate‑limiter и продвинутый retry.
- Снэпшоты метрик (`video_snapshots`) для Temporal‑фичей.
- Устойчивость к 10k видео/день (горизонтальное масштабирование).

**Production v2:**

- Поддержка нескольких платформ (YouTube/TikTok/Instagram/… через модуль `platforms/*`).
- Event‑driven архитектура на Kafka (при необходимости).
- Partitioning крупных таблиц (`comments`, `fetch_logs`).
- Расширенные политики безопасности (PII, OAuth‑верификация владельца канала).

В такой конфигурации Fetcher становится **полноценной ingestion‑платформой**, надёжно снабжающей DataProcessor и ML‑систему всеми необходимыми данными о видео, канале и динамике метрик.

---

## 12. Тестирование и качество

Для обеспечения надёжности и предсказуемости работы Fetcher используется многоуровневая стратегия тестирования:

- unit‑тесты на ключевые модули (`api`, `orchestrator`, `workers`, `platforms`, `idempotency`, `rate_limiter`, `backpressure` и т.д.);
- интеграционные тесты (API + БД + Redis + S3 + Celery + Kafka stub);
- e2e/сценарные тесты для критичных пользовательских флоу;
- chaos‑тесты и базовые нагрузочные тесты.

Детальный план и чеклист по покрытию тестами описаны в документе `TESTING_PLAN.md`. Он должен использоваться как рабочий список задач при развитии тестов и как справочник по критериям качества.
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
