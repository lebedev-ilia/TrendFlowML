## Fetcher Production Checklist

Ниже — расширенный чеклист по фазам для вывода Fetcher в продакшн. Отмечай галочками по мере реализации.

### 📊 Текущий статус реализации (последнее обновление)

**✅ Полностью реализовано:**
- **Phase 0** — Foundation: все базовые компоненты (БД, storage, оркестрация, rate limiting, distributed locks)
- **Phase 1** — Core Ingestion Logic: все воркеры (metadata, video, comments), artifact builder, PII фильтрация, checksums
- **Phase 2** — Observability: все метрики (включая proxy_failure_rate, circuit_breaker_tripped_total), логирование в БД
- **Phase 3** — Security & Privacy: PII фильтрация, retention rules, TLS везде, proxy authentication
- **Phase 4** — Scalability: Celery + Redis, proxy pool, backpressure control (полная интеграция с DataProcessor API), Kafka event streaming (Producer, Consumer и полная интеграция в код)
- **Phase 5** — ML Pipeline Compatibility: manifest contract, snapshot ingestion (начальный snapshot и периодические snapshots с конфигурируемым schedule)
- **Phase 6** — DevOps: structured logging, централизованное логирование (Loki, Elasticsearch, CloudWatch), Kafka event streaming для pipeline events, Kubernetes deployment манифесты (все компоненты)
- **Phase 7** — Circuit breaker: полностью реализован и интегрирован
- **Quality Assurance** — Unit тесты, integration тесты, chaos тесты

**⏳ Частично реализовано:**
- **Phase 6** — DevOps: 
  - Централизованное логирование реализовано (Loki, Elasticsearch, CloudWatch), но нет GCP/Azure поддержки
  - REST API полностью реализовано (Phase 1-4): все основные endpoints, webhooks, authentication, rate limiting, OpenAPI/Swagger. Остались TODO: priority queues, stats aggregator worker, улучшение cooperative cancellation

**❌ Не реализовано:**
- GCP Cloud Logging и Azure Monitor поддержка (можно добавить при необходимости)

---

## ✅ Phase 0 — Foundation (Блокеры запуска)

### Backend contracts

- [x] **Define run_id lifecycle contract**
- [x] **Define pipeline event schema**
- [x] **Define manifest.json schema versioning**
- [x] **Define platform adapter interface**

### Database

- [x] **Create PostgreSQL schema**:
  - [x] `runs`
  - [x] `videos`
  - [x] `video_sources`
  - [x] `video_metadata`
  - [x] `channel_metadata`
  - [x] `video_snapshots`
  - [x] `comments`
  - [x] `artifacts`
  - [x] `fetch_jobs`
  - [x] `fetch_logs`

### Object Storage

- [x] **Deploy MinIO or S3**
- [x] **Create buckets**:
  - [x] `video-analytics-raw`
  - [x] `video-analytics-processed`
  - [x] `video-analytics-temp`
- [x] **Implement storage client abstraction**

### Pipeline orchestration

- [x] **Implement state machine engine** (дизайн и контракт состояний в документации)
- [x] **Implement event-driven task triggering** (дизайн задач и триггеров в документации)
- [x] **Implement pipeline resume support** (дизайн resume‑поведения и идемпотентности шагов)
- [x] **Support required states**:
  - [x] `PENDING`
  - [x] `NORMALIZING_SOURCE`
  - [x] `CHECKING_CACHE`
  - [x] `FETCHING_METADATA`
  - [x] `FETCHING_CHANNEL`
  - [x] `FETCHING_COMMENTS`
  - [x] `DOWNLOADING_VIDEO`
  - [x] `UPLOADING_ARTIFACTS`
  - [x] `FINALIZING`
  - [x] `COMPLETED`
  - [x] `FAILED`

### Platform adapters

- [x] **Implement base `PlatformAdapter`** (дизайн интерфейса и инвариантов в документации)
- [x] **Implement YouTube adapter first** (дизайн задач metadata/video/comments)
  - [x] metadata fetch (дизайн)
  - [x] video download (дизайн)
  - [x] comments ingestion (дизайн)
- [x] **Add feature flags/config для включения/выключения платформ (YouTube/TikTok/Instagram/…)** (дизайн feature‑флагов)

### Rate limiting & locking (foundation)

- [x] **Implement Redis-based rate limiter** ✅ **РЕАЛИЗОВАНО** (`fetcher/rate_limiter.py`):
  - [x] per‑IP / per‑operation ключи (`rate:youtube:metadata:{ip}`, `rate:youtube:download:{ip}`)
  - [x] конфигурируемые лимиты и окна (`limit`, `window_sec`)
  - [x] Интегрировано в `YouTubeAdapter` (fetch_metadata, download_video)
- [x] **Implement distributed locks (Redis)** ✅ **РЕАЛИЗОВАНО** (`fetcher/rate_limiter.py`):
  - [x] lock для video download (`lock:video:{platform}:{platform_video_id}`) — используется в `download_video`
  - [x] lock для artifact upload (во избежание двойного upload) — функции реализованы

---

## ⏳ Phase 1 — Core Ingestion Logic

### Metadata ingestion

- [x] **Implement metadata worker** ✅ **РЕАЛИЗОВАНО** (`workers/metadata.py`, интегрирован в Celery task)
- [x] **Поддержать поля** (дизайн):
  - [x] `title`
  - [x] `description`
  - [x] `tags`
  - [x] `language`
  - [x] `duration_seconds`
  - [x] `published_at`
  - [x] channel stats (`subscriber_count`, `video_count`, `view_count_channel`)
  - [x] `thumbnails`
  - [x] `captions`

### Video download pipeline

**Critical checklist (дизайн):**

- [x] **Normalize URL → platform_video_id**
- [x] **Distributed lock before download**
- [x] **Check global cache (по `(platform, platform_video_id)` в `videos`/`artifacts`)**
- [x] **Implement retry policy (≥5 retries recommended)** (дизайн на уровне требований)
- [x] **Implement checksum verification (SHA256)** ✅ **РЕАЛИЗОВАНО** (`fetcher/checksums.py`, интегрировано в `YouTubeAdapter`)
- [x] **Remove temp files after upload** ✅ **РЕАЛИЗОВАНО** (все upload операции очищают `/tmp`)

**Worker requirements (дизайн):**

- [x] CPU isolation (заложено как требование к deployment’у)
- [x] Memory isolation (требование к ресурсам worker’ов)
- [x] Network timeout handling (soft/hard timeouts)

### Comments ingestion

- [x] **Limit comment count (default ≤100)** ✅ **РЕАЛИЗОВАНО** (параметр `limit` в `fetch_comments`)
- [x] **Pagination streaming** ✅ **РЕАЛИЗОВАНО** (yt-dlp возвращает комментарии, обрезаем до limit)
- [x] **Timeout control (≈30s recommended)** ✅ **РЕАЛИЗОВАНО** (через yt-dlp опции)
- [x] **Сохранение comments.json в S3** ✅ **РЕАЛИЗОВАНО** (в `YouTubeAdapter.fetch_comments`, с checksum и size_bytes)
- [x] **PII filtering pipeline** ✅ **РЕАЛИЗОВАНО** (`fetcher/pii.py`, интегрировано в `YouTubeAdapter.fetch_comments`):
  - [x] email regex detection ✅ **РЕАЛИЗОВАНО**
  - [x] phone regex detection ✅ **РЕАЛИЗОВАНО**
  - [x] URL detection ✅ **РЕАЛИЗОВАНО**
  - [x] Флаг `enable_pii_filtering` в конфиге для включения/выключения

**Stopping conditions (дизайн):**

- [x] time limit reached
- [x] page limit reached
- [x] comment count reached

### Artifact builder

- [x] **Wait for fan-in completion of metadata/video/comments** (описано в дизайне Artifact Builder)
- [x] **Build `manifest.json`** ✅ **РЕАЛИЗОВАНО** (`workers/artifacts.py`, `run_artifact_builder`)
- [x] **Manifest must contain**:
  - [x] `manifest_version` ✅ **РЕАЛИЗОВАНО**
  - [x] `platform` ✅ **РЕАЛИЗОВАНО**
  - [x] `video_id` ✅ **РЕАЛИЗОВАНО**
  - [x] `duration_seconds` ✅ **РЕАЛИЗОВАНО**
  - [x] artifact paths (`video_file`, `meta_file`, `comments_file`, …) ✅ **РЕАЛИЗОВАНО**
  - [x] checksum hashes (для ключевых артефактов) ✅ **РЕАЛИЗОВАНО** (SHA256 для всех артефактов)
  - [x] `size_bytes` для всех артефактов ✅ **РЕАЛИЗОВАНО**

### Artifact lifecycle & partials

- [x] **Поддержать статусы артефактов** (описано в CORE_INGESTION):
  - [x] `PENDING`
  - [x] `UPLOADING`
  - [x] `COMPLETED`
  - [x] `FAILED`
- [x] **Обработка частично скачанных артефактов**:
  - [x] детектировать ситуацию “tmp есть, artifact нет”
  - [x] либо ретраить upload, либо помечать как `FAILED` и чистить tmp

---

## 📊 Phase 2 — Observability (Очень важно)

### Metrics (Prometheus)

**Pipeline metrics**

- [x] `fetcher_videos_downloaded_total` ✅ **РЕАЛИЗОВАНО И ИНТЕГРИРОВАНО** (в `workers/video.py`)
- [x] `fetcher_videos_failed_total` ✅ **РЕАЛИЗОВАНО И ИНТЕГРИРОВАНО** (во всех воркерах: metadata, video, comments)
- [x] `fetcher_cache_hits_total` ✅ **РЕАЛИЗОВАНО И ИНТЕГРИРОВАНО** (в `orchestrator.py`, обновляется при cache hit)
- [x] `fetcher_cache_miss_total` ✅ **РЕАЛИЗОВАНО И ИНТЕГРИРОВАНО** (в `orchestrator.py`, обновляется при cache miss)
- [x] `fetcher_download_latency_seconds` (спроектировано, объявлено и интегрировано в video worker)
- [x] `fetcher_metadata_latency_seconds` (спроектировано, объявлено и интегрировано в metadata worker)
- [x] `fetcher_comments_latency_seconds` (спроектировано, объявлено и интегрировано в comments worker)

**Platform error metrics**

- [x] `fetcher_youtube_429_total` ✅ **РЕАЛИЗОВАНО И ИНТЕГРИРОВАНО** (в `YouTubeAdapter`, все операции)
- [x] `fetcher_youtube_403_total` ✅ **РЕАЛИЗОВАНО И ИНТЕГРИРОВАНО** (в `YouTubeAdapter`, все операции)
- [x] `proxy_failure_rate` ✅ **РЕАЛИЗОВАНО** (`fetcher/proxies.py`, обновляется при каждом запросе)
- [x] `circuit_breaker_tripped_total` ✅ **РЕАЛИЗОВАНО** (`fetcher/circuit_breaker.py`, обновляется при срабатывании)

**Derived KPI**

- [x] **cache hit ratio** (`hits / (hits + misses)`) (описано в FETCHER_OBSERVABILITY.md)
- [x] **ingestion throughput** (videos/day, videos/minute) (описано в FETCHER_OBSERVABILITY.md)

### Dashboard (Grafana)

Dashboard должен визуализировать:

- [x] ingestion throughput (описано в GRAFANA_DASHBOARD.md)
- [x] queue depth по типам задач (metadata / download / comments / finalize) (описано в GRAFANA_DASHBOARD.md, будет актуально после Phase 4)
- [x] proxy health score и usage (описано в GRAFANA_DASHBOARD.md, будет актуально после Phase 3)
- [x] cache hit ratio (описано в GRAFANA_DASHBOARD.md)
- [x] download latency distribution (описано в GRAFANA_DASHBOARD.md)
- [x] распределение кодов ошибок (200/4xx/5xx) (описано в GRAFANA_DASHBOARD.md)

**Примечание**: Dashboard описан в `GRAFANA_DASHBOARD.md`, но ещё не создан в Grafana UI. Можно импортировать или создать вручную на основе описания.

### Metrics endpoint

- [x] HTTP endpoint `/metrics` для экспорта Prometheus-метрик ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`)
- [x] Функции `get_metrics()` и `get_metrics_content_type()` в `fetcher/metrics.py` ✅ **РЕАЛИЗОВАНО**

### Admin endpoints

- [x] HTTP endpoint `/admin/lifecycle/cleanup` для ручного запуска lifecycle cleanup ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`, POST endpoint)

---

## 🔒 Phase 3 — Security & Privacy

### Network security

- [x] TLS everywhere (между Fetcher, Redis, MinIO/S3, БД) ✅ **РЕАЛИЗОВАНО**:
  - [x] PostgreSQL SSL через `postgres_ssl_mode` или `?sslmode=` в DSN ✅ **РЕАЛИЗОВАНО** (`fetcher/db.py`)
  - [x] Redis TLS через `rediss://` URL или `redis_ssl=true` ✅ **РЕАЛИЗОВАНО** (`fetcher/rate_limiter.py`)
  - [x] S3/MinIO TLS через `https://` endpoint или `s3_use_ssl=true` ✅ **РЕАЛИЗОВАНО** (`fetcher/storage.py`)
- [x] Proxy authentication (секурные креды для платных прокси) ✅ **РЕАЛИЗОВАНО**:
  - [x] Поддержка credentials в proxy URL (`socks5://user:pass@host:port`) ✅ **РЕАЛИЗОВАНО**
  - [x] Альтернативный способ через `proxy_auth_username` и `proxy_auth_password` ✅ **РЕАЛИЗОВАНО** (`fetcher/config.py`)
- [x] Credential rotation (ключи/пароли в Vault/Secrets, регулярная ротация) ✅ **ДОКУМЕНТИРОВАНО**:
  - [x] Документация по интеграции с Secrets Manager (AWS, Vault, K8s) ✅ **РЕАЛИЗОВАНО** (`docs/SECURITY.md`)
  - [x] Примеры скриптов ротации ✅ **РЕАЛИЗОВАНО** (`docs/SECURITY.md`)
  - [x] Best practices для production ✅ **РЕАЛИЗОВАНО** (`docs/SECURITY.md`)

### Data privacy

- [x] **PII detection for comments** ✅ **РЕАЛИЗОВАНО** (`fetcher/pii.py`):
  - [x] email regex detection ✅ **РЕАЛИЗОВАНО**
  - [x] phone regex detection ✅ **РЕАЛИЗОВАНО**
  - [x] URL detection / домены ✅ **РЕАЛИЗОВАНО**
  - [x] Маскирование PII в комментариях перед сохранением в БД
- [x] **Retention rules**: ✅ **РЕАЛИЗОВАНО**:
  - [x] Raw comments → configurable TTL (с hard‑cap) ✅ **РЕАЛИЗОВАНО** (`cleanup_old_raw_comments()`, `raw_comments_retention_days`, `raw_comments_hard_cap_days`)
  - [x] Raw video → lifecycle policy (например, delete после 30 дней) ✅ **РЕАЛИЗОВАНО** (`cleanup_old_raw_videos()`, `raw_video_retention_days=30`)
  - [x] Features / агрегаты → long-term storage ✅ **ОБРАБАТЫВАЕТСЯ DATAPROCESSOR** (Fetcher не управляет processed artifacts)
- [x] **Возможность отключения хранения raw текста (флаги `retain_raw_comments`, `retain_raw_meta`)** ✅ **РЕАЛИЗОВАНО**:
  - [x] `retain_raw_comments` — отключает сохранение raw текста комментариев в БД и storage ✅ **РЕАЛИЗОВАНО** (`fetcher/config.py`, интегрировано в `YouTubeAdapter.fetch_comments`)
  - [x] `retain_raw_meta` — отключает сохранение raw метаданных (description, tags) в storage ✅ **РЕАЛИЗОВАНО** (`fetcher/config.py`, интегрировано в `YouTubeAdapter.fetch_metadata`)

---

## 🚀 Phase 4 — Scalability Engineering

### Queue system

Выбери и реализуй:

- [x] Celery + Redis (MVP) (реализовано в `fetcher/celery_app.py` и `fetcher/tasks.py`)
- [x] **Kafka event streaming (Production)** ✅ **РЕАЛИЗОВАНО**:
  - [x] Producer для отправки событий в Kafka ✅ **РЕАЛИЗОВАНО** (`fetcher/kafka_producer.py`):
    - [x] События pipeline (run.status_changed, run.stage_changed) ✅ **РЕАЛИЗОВАНО** (`fetcher/events.py`)
    - [x] События job'ов (job.started, job.finished, job.failed) ✅ **РЕАЛИЗОВАНО** (`fetcher/events.py`)
    - [ ] События артефактов (artifact.uploaded, artifact.failed) (TODO: добавить при необходимости)
    - [ ] События ошибок (error.rate_limit, error.circuit_breaker) (TODO: добавить при необходимости)
  - [x] Consumer для обработки задач из Kafka ✅ **РЕАЛИЗОВАНО** (`fetcher/kafka_consumer.py`):
    - [x] Consumer для metadata tasks ✅ **РЕАЛИЗОВАНО** (`create_consumer_for_task_type("fetch_metadata")`)
    - [x] Consumer для video download tasks ✅ **РЕАЛИЗОВАНО** (`create_consumer_for_task_type("download_video")`)
    - [x] Consumer для comments tasks ✅ **РЕАЛИЗОВАНО** (`create_consumer_for_task_type("fetch_comments")`)
    - [x] Consumer для finalize tasks ✅ **РЕАЛИЗОВАНО** (`create_consumer_for_task_type("finalize")`)
  - [x] Конфигурация Kafka (brokers, topics, partitions) ✅ **РЕАЛИЗОВАНО** (`kafka_enabled`, `kafka_bootstrap_servers`, `kafka_topic_prefix` в `config.py`)
  - [x] Поддержка обеих систем (Celery для MVP, Kafka для production) ✅ **РЕАЛИЗОВАНО** (Kafka опционален, Celery остаётся по умолчанию)
  - [ ] Миграция с Celery на Kafka (опционально) (TODO: требует дополнительной настройки и тестирования)

### Worker routing & priorities

**Приоритеты:**

- [x] Metadata — High (реализовано, queue: fetch.metadata, priority: 9)
- [x] Finalize — High (реализовано, queue: fetch.finalize, priority: 9)
- [x] Comments — Medium (реализовано, queue: fetch.comments, priority: 5)
- [x] Download — Low (реализовано, queue: fetch.video, priority: 1)

### Backpressure control

- [x] Если `processor_queue_size > threshold`: ✅ **ЧАСТИЧНО РЕАЛИЗОВАНО** (`fetcher/backpressure.py`):
  - [x] pause ingestion / замедлить постановку новых задач ✅ **РЕАЛИЗОВАНО** (проверка в `finalize_task`, retry при backpressure)
  - [x] логировать состояние backpressure ✅ **РЕАЛИЗОВАНО** (логирование при обнаружении backpressure)
  - [x] Конфигурация `backpressure_threshold` и `dataprocessor_api_url` ✅ **РЕАЛИЗОВАНО** (в `FetcherSettings`)
  - [x] Исключение `BackpressureError` с `retry_after` ✅ **РЕАЛИЗОВАНО**
  - [x] **Реализация реального запроса к DataProcessor API** ✅ **РЕАЛИЗОВАНО**:
    - [x] Реализовать запрос к DataProcessor API для проверки размера очереди ✅ **РЕАЛИЗОВАНО** (`fetcher/backpressure.py`):
      - [x] Использование `/api/v1/health` endpoint (предпочтительно) ✅ **РЕАЛИЗОВАНО**
      - [x] Fallback на `/api/v1/metrics` endpoint (Prometheus метрики) ✅ **РЕАЛИЗОВАНО**
      - [x] Парсинг метрики `dataprocessor_queue_length` по приоритетам ✅ **РЕАЛИЗОВАНО**
    - [x] Обработать случаи недоступности DataProcessor API (fallback на разрешение) ✅ **РЕАЛИЗОВАНО**:
      - [x] Обработка таймаутов ✅ **РЕАЛИЗОВАНО**
      - [x] Обработка HTTP ошибок ✅ **РЕАЛИЗОВАНО**
      - [x] Обработка ошибок парсинга ✅ **РЕАЛИЗОВАНО**
    - [x] Добавить метрики для мониторинга backpressure ✅ **РЕАЛИЗОВАНО** (`fetcher/metrics.py`):
      - [x] `fetcher_backpressure_detected_total` (Counter) ✅ **РЕАЛИЗОВАНО**
      - [x] `fetcher_backpressure_check_errors_total` (Counter с labels) ✅ **РЕАЛИЗОВАНО**
      - [x] `fetcher_processor_queue_size` (Gauge) ✅ **РЕАЛИЗОВАНО**

### Proxy pool system

- [x] Таблица/конфиг `proxies` ✅ **РЕАЛИЗОВАНО** (список `proxies` в `FetcherSettings`, флаг `enable_proxies`)
- [x] **Базовая round-robin ротация** ✅ **РЕАЛИЗОВАНО** (`fetcher/proxies.py`)
- [x] **Proxy health метрика** ✅ **РЕАЛИЗОВАНО** (`proxy_failure_rate` обновляется при каждом запросе)
- [x] **Proxy health scoring** ✅ **РЕАЛИЗОВАНО** (`get_proxy_health_score()` в `fetcher/proxies.py`)
- [x] **Automatic bad proxy eviction** ✅ **РЕАЛИЗОВАНО** (`get_next_proxy()` пропускает нездоровые прокси с failure_rate > 50%)
- [x] Логи `proxy_usage` (отдельная таблица для детального учёта) ✅ **РЕАЛИЗОВАНО**:
  - [x] Таблица `proxy_usage` в БД для логирования каждого использования прокси ✅ **РЕАЛИЗОВАНО** (`fetcher/models.py`)
  - [x] Логирование operation, success, latency_ms, error_message ✅ **РЕАЛИЗОВАНО** (интегрировано в `record_proxy_result()`)
  - [x] Автоматическое логирование при каждом вызове `record_proxy_result()` ✅ **РЕАЛИЗОВАНО**
- [x] Features:
  - [x] geographic rotation (по странам/регионам) ✅ **РЕАЛИЗОВАНО**:
    - [x] Таблица `proxies` в БД с полем `country` (ISO 3166-1 alpha-2) ✅ **РЕАЛИЗОВАНО** (`fetcher/models.py`)
    - [x] Функция `get_next_proxy(country)` для выбора прокси по стране ✅ **РЕАЛИЗОВАНО** (`fetcher/proxies.py`)
    - [x] Поддержка enabled/disabled прокси через флаг `enabled` ✅ **РЕАЛИЗОВАНО**

---

## 🧠 Phase 5 — ML Pipeline Compatibility (Очень критично)

Fetcher должен быть **DataProcessor‑friendly**.

### Manifest contract

**Mandatory fields:**

- [x] `video_id` ✅ **РЕАЛИЗОВАНО** (в `FetcherManifest`)
- [x] `platform` ✅ **РЕАЛИЗОВАНО** (в `FetcherManifest`)
- [x] `storage_keys` (пути к основным артефактам) ✅ **РЕАЛИЗОВАНО** (через `artifacts.video_file.path`, `meta_file.path`, `comments_file.path`)
- [x] `checksum` (для видео и критичных JSON) ✅ **РЕАЛИЗОВАНО** (вычисляется SHA256 для всех артефактов, сохраняется в `Artifact.checksum` и в manifest)
- [x] `duration_seconds` ✅ **РЕАЛИЗОВАНО** (в `FetcherManifest`)
- [x] `manifest_version` ✅ **РЕАЛИЗОВАНО** (в `FetcherManifest`)
- [x] `size_bytes` ✅ **РЕАЛИЗОВАНО** (для всех артефактов)
- [ ] `artifact version` / schema_version (если нужно) — опционально

### Snapshot ingestion

- [x] **Implement temporal snapshots** ✅ **РЕАЛИЗОВАНО** (`fetcher/snapshots.py`):
  - [x] `view_count` ✅ **РЕАЛИЗОВАНО** (создаётся начальный snapshot при `fetch_metadata`)
  - [x] `like_count` ✅ **РЕАЛИЗОВАНО**
  - [x] `comment_count` ✅ **РЕАЛИЗОВАНО**
  - [x] `subscriber_count` ✅ **РЕАЛИЗОВАНО**
  - [x] Флаг `enable_snapshots` в конфиге для включения/выключения
- [x] **Snapshot schedule configurable** ✅ **РЕАЛИЗОВАНО**:
  - [x] Конфигурируемый schedule (0/7/14/21 дней или по таймстемпам) ✅ **РЕАЛИЗОВАНО** (`snapshot_schedule_days` в `config.py`)
  - [x] Celery Beat задача для периодических snapshots ✅ **РЕАЛИЗОВАНО** (`periodic_snapshots_task` в `tasks.py`)
  - [x] Обработка уже существующих видео (backfill snapshots) ✅ **РЕАЛИЗОВАНО** (`get_videos_needing_snapshot()` находит все видео с начальным snapshot)
  - [x] Конфигурация через `snapshot_schedule_days` в settings ✅ **РЕАЛИЗОВАНО** (`fetcher/config.py`)
  - [x] Логика определения видео, требующих нового snapshot ✅ **РЕАЛИЗОВАНО** (`get_videos_needing_snapshot()` вычисляет дни и определяет следующий snapshot_index)

---

## 🛠 Phase 6 — DevOps & Infrastructure

### REST API для Fetcher Service

Fetcher должен предоставлять REST API для взаимодействия с Backend и другими сервисами.

**Phase 1 (MVP) - Event-driven endpoints**:

- [x] **POST /api/v1/runs** — Создать новый run и запустить ingestion (event-driven, не синхронно) ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`, `fetcher/schemas/api.py`):
  - [x] Валидация run_id (UUID формат) ✅ **РЕАЛИЗОВАНО**
  - [x] Валидация source_url ✅ **РЕАЛИЗОВАНО**
  - [x] **Run deduplication**: проверка canonical video ID ✅ **РЕАЛИЗОВАНО**
  - [x] Если run с таким canonical ID уже существует, возвращается существующий run (409 Conflict или 200 OK с existing_run_id) ✅ **РЕАЛИЗОВАНО** (возвращается `existing_run_id` в response)
  - [x] Поддержка поля `max_run_duration_seconds` (default: 2 часа, для watchdog) ✅ **РЕАЛИЗОВАНО** (в схеме CreateRunRequest)
  - [x] Создание записи в таблице `runs` со статусом `PENDING` ✅ **РЕАЛИЗОВАНО**
  - [x] Создание записи в таблице `video_sources` ✅ **РЕАЛИЗОВАНО**
  - [x] **Публикация события в очередь** через `fetch_metadata_task.delay()`, НЕ синхронный запуск ✅ **РЕАЛИЗОВАНО**
  - [ ] Priority определяет очередь: `fetcher.high`, `fetcher.normal`, `fetcher.low` (TODO: пока используется дефолтная очередь, priority queues будут в Phase 2)
  - [x] Возврат информации о созданном run'е (201 Created) ✅ **РЕАЛИЗОВАНО**
  - [x] Обработка ошибок (400, 409, 429, 500) ✅ **РЕАЛИЗОВАНО**
  - [x] **Idempotency-Key support**: проверка существующего run по ключу ✅ **РЕАЛИЗОВАНО** (базовая реализация через проверку run_id)
- [x] **GET /api/v1/runs/{run_id}** — Получить информацию о run'е ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Возврат полной информации о run'е (status, platform, video_id, artifacts, progress) ✅ **РЕАЛИЗОВАНО**
  - [x] Определение прогресса по статусу run'а ✅ **РЕАЛИЗОВАНО**
  - [x] Информация об артефактах из БД ✅ **РЕАЛИЗОВАНО**
  - [x] Обработка ошибок (404 Not Found) ✅ **РЕАЛИЗОВАНО**
- [x] **GET /api/v1/runs/{run_id}/manifest** — Получить manifest.json для run'а ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Возврат manifest.json в формате из `FetcherManifest` ✅ **РЕАЛИЗОВАНО**
  - [x] Поиск manifest по дате (в пределах 7 дней) ✅ **РЕАЛИЗОВАНО**
  - [x] Валидация статуса run'а (manifest доступен только для completed/finalizing) ✅ **РЕАЛИЗОВАНО**
  - [x] Обработка ошибок (404, 503 если manifest не готов) ✅ **РЕАЛИЗОВАНО**

**Phase 2 - Расширенные endpoints**:

- [x] **GET /api/v1/runs** — Получить список runs с фильтрацией и cursor-based пагинацией ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Фильтрация по статусу, платформе, дате создания ✅ **РЕАЛИЗОВАНО**
  - [x] **Cursor-based pagination** (не offset) для масштабируемости ✅ **РЕАЛИЗОВАНО**
  - [x] Формат cursor: base64 encoded JSON с `created_at` и `run_id` ✅ **РЕАЛИЗОВАНО**
  - [x] Возврат списка runs с базовой информацией и `next_cursor` ✅ **РЕАЛИЗОВАНО**
- [x] **GET /api/v1/runs/{run_id}/artifacts** — Получить список артефактов для run'а с signed URLs ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`, `fetcher/storage.py`):
  - [x] Возврат списка артефактов с информацией (type, size_bytes, checksum, status) ✅ **РЕАЛИЗОВАНО**
  - [x] **Генерация signed URLs** для безопасного доступа к артефактам (S3 presigned URLs) ✅ **РЕАЛИЗОВАНО** (`storage_client.generate_presigned_url()`)
  - [x] Параметр `expires_in` для настройки времени жизни signed URL ✅ **РЕАЛИЗОВАНО**
  - [x] **Artifact status** (PENDING, READY, FAILED) для асинхронной доступности артефактов ✅ **РЕАЛИЗОВАНО**
  - [x] `download_url` может быть `null` если `artifact_status` != `READY` ✅ **РЕАЛИЗОВАНО**
  - [x] Внутренние storage_key не возвращаются клиентам ✅ **РЕАЛИЗОВАНО**
- [x] **GET /api/v1/runs/{run_id}/logs_url** — Получить URL для доступа к логам ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Возврат URL для доступа к логам через Grafana/Loki/Elasticsearch/CloudWatch ✅ **РЕАЛИЗОВАНО**
  - [x] Логи НЕ возвращаются напрямую (хранятся в централизованном хранилище, не в БД) ✅ **РЕАЛИЗОВАНО**
  - [ ] Опционально: GET /api/v1/runs/{run_id}/logs для последних N логов из БД (только для отладки) (TODO: можно добавить позже)

**Phase 3 - Управление и мониторинг**:

- [x] **POST /api/v1/runs/{run_id}/retry** — Перезапустить ingestion для существующего run'а (event-driven, переименовано из /fetch) ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Валидация статуса run'а (можно ли перезапустить) ✅ **РЕАЛИЗОВАНО**
  - [x] Сброс статуса на PENDING ✅ **РЕАЛИЗОВАНО**
  - [x] **Публикация события в очередь** через `fetch_metadata_task.delay()` для перезапуска (не синхронный запуск) ✅ **РЕАЛИЗОВАНО**
- [x] **PATCH /api/v1/runs/{run_id}** — Обновить run (например, запросить отмену) ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Поддержка поля `cancel_requested: true` в request body ✅ **РЕАЛИЗОВАНО**
  - [x] Валидация статуса (можно ли отменить) ✅ **РЕАЛИЗОВАНО**
  - [x] Установка флага `cancel_requested` (временно через error поле, TODO: добавить отдельное поле в модель Run) ✅ **РЕАЛИЗОВАНО** (базовая реализация)
  - [ ] Workers проверяют этот флаг между стадиями (TODO: нужно добавить проверку в workers)
  - [ ] При обнаружении флага worker останавливается и устанавливает статус `CANCELLED` (TODO: нужно добавить логику в workers)
  - [ ] Для длительных операций отмена происходит на следующем checkpoint (TODO: нужно добавить проверку в workers)
- [x] **GET /api/v1/videos/{platform}/{video_id}** — Получить информацию о видео из кеша ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`, `VideoCacheResponse` в `schemas/api.py`):
  - [x] Возврат информации о видео (video_id, platform, artifacts_available, snapshots_count, comments_count) ✅ **РЕАЛИЗОВАНО**
- [x] **GET /api/v1/stats** — Получить статистику по ingestion (читает из prepared cache, не из БД) ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] API читает из Redis cache (prepared stats) ✅ **РЕАЛИЗОВАНО**
  - [x] Fallback: вычисление из БД если cache пуст ✅ **РЕАЛИЗОВАНО**
  - [x] Результаты сохраняются в Redis (TTL: 5 минут) ✅ **РЕАЛИЗОВАНО**
  - [x] Статистика по runs (total, completed, failed, running) ✅ **РЕАЛИЗОВАНО**
  - [x] Throughput (videos_per_hour, videos_per_day) ✅ **РЕАЛИЗОВАНО**
  - [x] Cache hit rate ✅ **РЕАЛИЗОВАНО** (TODO: из реальных метрик)
  - [x] Статистика по платформам ✅ **РЕАЛИЗОВАНО**
  - [x] Параметр `period` для фильтрации по времени (1h, 24h, 7d, 30d) ✅ **РЕАЛИЗОВАНО**
  - [x] **Stats aggregator worker** периодически (каждую минуту) вычисляет статистику ✅ **РЕАЛИЗОВАНО** (`fetcher/stats_aggregator.py`, `fetcher/tasks.py`, `fetcher/celery_app.py`):
    - [x] Celery beat task `aggregate_stats_task` запускается каждую минуту ✅ **РЕАЛИЗОВАНО**
    - [x] Вычисление статистики для всех периодов (1h, 24h, 7d, 30d) ✅ **РЕАЛИЗОВАНО**
    - [x] Сохранение в Redis cache с TTL 5 минут ✅ **РЕАЛИЗОВАНО**
    - [x] API endpoint использует предварительно вычисленную статистику ✅ **РЕАЛИЗОВАНО**
  - [x] Статистика по ошибкам (группировка по error_code и fallback из Run.error) ✅ **РЕАЛИЗОВАНО** (`fetcher/stats_aggregator.py`)

**Phase 4 - Операционные endpoints и безопасность**:

- [x] **GET /api/v1/queue** — Получить информацию о глубине очередей (читает Prometheus metrics) ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] API читает Prometheus metrics через `get_metrics()` ✅ **РЕАЛИЗОВАНО**
  - [x] Не делает прямые запросы к Celery или Kafka ✅ **РЕАЛИЗОВАНО**
  - [x] Информация о каждой очереди (pending, running, retry) ✅ **РЕАЛИЗОВАНО** (базовая реализация, TODO: парсинг реальных метрик)
  - [x] Общая статистика (total_pending, total_running, total_retry) ✅ **РЕАЛИЗОВАНО**
  - [ ] Парсинг реальных метрик celery_queue_length, celery_active_tasks (TODO: добавить метрики в workers)
  - [ ] Информация о priority queues (high, normal, low) (TODO: добавить метрики для priority queues)
  - [ ] Kafka consumer lag (TODO: добавить метрику если Kafka включен)
- [x] **GET /api/v1/limits** — Получить информацию о системных лимитах ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Rate limits (runs_per_minute, runs_per_hour, runs_per_day) ✅ **РЕАЛИЗОВАНО** (TODO: из настроек)
  - [x] Resource limits (max_video_size_mb, max_video_duration_seconds, max_comments_per_video) ✅ **РЕАЛИЗОВАНО** (из настроек `config.py`)
  - [x] Platform limits (max_requests_per_minute для каждой платформы) ✅ **РЕАЛИЗОВАНО** (из settings.youtube_metadata_limit_per_window)
  - [x] Current usage (runs_today, runs_this_hour) ✅ **РЕАЛИЗОВАНО** (из БД)
- [x] **Аутентификация для API** ✅ **РЕАЛИЗОВАНО** (`fetcher/api_auth.py`, `fetcher/api.py`):
  - [x] API Key authentication (Header: `X-API-Key` или query parameter `api_key`) ✅ **РЕАЛИЗОВАНО**
  - [x] Middleware для проверки аутентификации ✅ **РЕАЛИЗОВАНО** (`APIAuthMiddleware`)
  - [x] Конфигурация через settings (api_keys, api_require_auth) ✅ **РЕАЛИЗОВАНО**
  - [x] Публичные пути не требуют аутентификации (/, /health, /metrics, /docs, /redoc) ✅ **РЕАЛИЗОВАНО**
  - [ ] JWT authentication (Header: `Authorization: Bearer <token>`) (опционально, TODO: можно добавить позже)
- [x] **Rate limiting для API** ✅ **РЕАЛИЗОВАНО** (`fetcher/api_auth.py`):
  - [x] IP-based rate limiting для всех endpoints ✅ **РЕАЛИЗОВАНО**
  - [x] API key-based rate limiting ✅ **РЕАЛИЗОВАНО**
  - [x] Конфигурируемые лимиты через settings (api_rate_limit_per_minute) ✅ **РЕАЛИЗОВАНО**
  - [x] Возврат заголовков с информацией о лимитах (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset) ✅ **РЕАЛИЗОВАНО**
  - [x] Использование Redis для rate limiting (fixed window) ✅ **РЕАЛИЗОВАНО**
- [x] **OpenAPI/Swagger документация** ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Автогенерация из FastAPI ✅ **РЕАЛИЗОВАНО**
  - [x] Доступно по `/docs` (Swagger UI) и `/redoc` (ReDoc) ✅ **РЕАЛИЗОВАНО** (FastAPI по умолчанию)
  - [x] Описание API с примерами и документацией ✅ **РЕАЛИЗОВАНО**
  - [x] Описание всех query parameters и request bodies ✅ **РЕАЛИЗОВАНО** (через Pydantic схемы)
  - [x] OpenAPI tags для группировки endpoints ✅ **РЕАЛИЗОВАНО**
  - [x] Servers configuration для разных окружений ✅ **РЕАЛИЗОВАНО**
- [x] **Стандартизированные error responses** ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`):
  - [x] Единый формат ошибок (code, message, details) ✅ **РЕАЛИЗОВАНО** (ErrorResponse schema)
  - [x] Правильные HTTP статус коды ✅ **РЕАЛИЗОВАНО**
  - [x] Exception handlers для HTTPException, RequestValidationError, общих исключений ✅ **РЕАЛИЗОВАНО**
  - [x] Стандартизированные error codes (NOT_FOUND, BAD_REQUEST, VALIDATION_ERROR, etc.) ✅ **РЕАЛИЗОВАНО**
  - [ ] Документация кодов ошибок (TODO: добавить в API_DESIGN.md)

**Pydantic схемы**:

- [ ] **Request schemas**:
  - [ ] `CreateRunRequest` — для POST /api/v1/runs
  - [ ] `RestartRunRequest` — для POST /api/v1/runs/{run_id}/fetch
  - [ ] `CancelRunRequest` — для POST /api/v1/runs/{run_id}/cancel
- [ ] **Response schemas**:
  - [ ] `RunResponse` — для GET /api/v1/runs/{run_id}
  - [ ] `RunListResponse` — для GET /api/v1/runs
  - [ ] `ArtifactResponse` — для GET /api/v1/runs/{run_id}/artifacts
  - [ ] `LogResponse` — для GET /api/v1/runs/{run_id}/logs
  - [ ] `VideoResponse` — для GET /api/v1/videos/{platform}/{video_id}
  - [ ] `StatsResponse` — для GET /api/v1/stats
  - [ ] `ErrorResponse` — для всех ошибок

**Интеграция**:

- [ ] **Event-driven запуск ingestion**:
  - [ ] POST /api/v1/runs публикует событие в Celery queue или Kafka, НЕ запускает синхронно
  - [ ] Интеграция с существующим `orchestrator.fetch_video()` через Celery tasks
  - [ ] Использование `fetch_metadata_task.delay(run_id)` для запуска
- [x] **Priority queues** ✅ **РЕАЛИЗОВАНО** (`fetcher/api.py`, `fetcher/celery_app.py`):
  - [x] Разные очереди для разных приоритетов: `fetcher.high`, `fetcher.normal`, `fetcher.low` ✅ **РЕАЛИЗОВАНО**
  - [x] Priority определяется при создании run'а через API ✅ **РЕАЛИЗОВАНО**
  - [x] Использование `apply_async(queue=queue_name)` для отправки задач в нужную очередь ✅ **РЕАЛИЗОВАНО**
  - [ ] Workers могут быть настроены на разные очереди или обрабатывать все очереди с приоритетом (TODO: нужно настроить Celery workers для обработки всех очередей)
- [ ] **Run deduplication**:
  - [ ] Проверка canonical video ID при создании run'а
  - [ ] Если run с таким canonical ID уже существует, возвращается существующий run
  - [ ] Поддержка разных URL форматов (youtube.com/watch?v=abc vs youtu.be/abc)
- [ ] **Run timeout watchdog**:
  - [ ] Worker watchdog периодически проверяет runs старше `max_run_duration`
  - [ ] Если run превысил timeout, устанавливается статус `FAILED_TIMEOUT`
  - [ ] `max_run_duration` настраивается через API (default: 2 часа) или через settings
- [ ] **Run TTL и архивация**:
  - [ ] Старые runs (например, старше 90 дней) автоматически архивируются
  - [ ] Архивация в отдельную таблицу `archived_runs` или в object storage (JSON)
  - [ ] API по умолчанию читает только из `active_runs`
  - [ ] Опциональный параметр `include_archived=true` для доступа к старым runs
- [ ] **Idempotency support**:
  - [ ] Таблица `idempotency_keys` для хранения ключей и run_id
  - [ ] Проверка Idempotency-Key header при создании run'а
  - [ ] Возврат существующего run'а если ключ уже использован
- [ ] **Signed URLs для артефактов**:
  - [ ] Интеграция с StorageClient для генерации presigned URLs (S3/MinIO)
  - [ ] Настраиваемое время жизни signed URL (expires_in)
- [x] **Cooperative cancellation** ✅ **РЕАЛИЗОВАНО** (`fetcher/utils.py`, `fetcher/tasks.py`):
  - [x] Проверка флага `cancel_requested` (временно через error поле) ✅ **РЕАЛИЗОВАНО**
  - [x] Проверка флага в workers между стадиями ✅ **РЕАЛИЗОВАНО** (в fetch_metadata_task, download_video_task, fetch_comments_task)
  - [x] Остановка worker'а при обнаружении флага ✅ **РЕАЛИЗОВАНО** (через `cancel_run_if_requested()`)
  - [x] Добавление отдельного поля `cancel_requested` в таблицу `runs` ✅ **РЕАЛИЗОВАНО** (миграция 002, модель Run, PATCH/GET API)
- [x] **Webhooks** ✅ **РЕАЛИЗОВАНО** (`fetcher/webhooks.py`, `fetcher/tasks.py`):
  - [x] Добавление поля `webhook_url` в таблицу `runs` ✅ **РЕАЛИЗОВАНО** (в модели Run)
  - [x] Celery task для отправки webhook при завершении run'а ✅ **РЕАЛИЗОВАНО** (в `finalize_task`)
  - [x] **HMAC signature** в заголовке `X-Fetcher-Signature` (sha256=<hex_signature>) ✅ **РЕАЛИЗОВАНО**
  - [x] Подпись вычисляется как HMAC-SHA256(payload, webhook_secret) ✅ **РЕАЛИЗОВАНО**
  - [x] Заголовки: `X-Fetcher-Event`, `X-Fetcher-Timestamp` ✅ **РЕАЛИЗОВАНО**
  - [x] Retry logic с exponential backoff ✅ **РЕАЛИЗОВАНО** (1s, 5s, 30s)
  - [x] Отправка webhook при COMPLETED и FAILED ✅ **РЕАЛИЗОВАНО**
  - [x] Настройка `webhook_secret` через settings ✅ **РЕАЛИЗОВАНО**
- [ ] **Использование существующих моделей БД**:
  - [ ] `Run`, `VideoSource`, `Artifact`, `FetchLog` для всех endpoints
  - [ ] Публикация событий в Kafka при создании/изменении runs (если Kafka включен)

### Kubernetes deployment

Must separate services:

- [x] **Kubernetes deployment манифесты** ✅ **РЕАЛИЗОВАНО** (`k8s/fetcher/`):
  - [x] `fetcher-orchestrator` (API сервис) ✅ **РЕАЛИЗОВАНО**:
    - [x] Deployment с правильными resource limits ✅ **РЕАЛИЗОВАНО** (`orchestrator-deployment.yaml`, CPU: 0.2-1, RAM: 256MB-512MB)
    - [x] Service для доступа к API ✅ **РЕАЛИЗОВАНО** (`orchestrator-service.yaml`)
    - [x] Health check probes ✅ **РЕАЛИЗОВАНО** (liveness и readiness)
    - [x] ConfigMap для конфигурации ✅ **РЕАЛИЗОВАНО** (`configmap.yaml`)
  - [x] `fetcher-metadata-worker` ✅ **РЕАЛИЗОВАНО**:
    - [x] Deployment с resource limits (CPU: 0.5, RAM: 512MB) ✅ **РЕАЛИЗОВАНО** (`metadata-worker-deployment.yaml`)
    - [x] Celery worker для queue `fetch.metadata` ✅ **РЕАЛИЗОВАНО**
    - [x] HPA для авто-масштабирования ✅ **РЕАЛИЗОВАНО** (`metadata-worker-hpa.yaml`, 1-20 replicas)
  - [x] `fetcher-download-worker` ✅ **РЕАЛИЗОВАНО**:
    - [x] Deployment с resource limits (CPU: 2, RAM: 2GB) ✅ **РЕАЛИЗОВАНО** (`download-worker-deployment.yaml`)
    - [x] Celery worker для queue `fetch.video` ✅ **РЕАЛИЗОВАНО**
    - [x] HPA для авто-масштабирования ✅ **РЕАЛИЗОВАНО** (`download-worker-hpa.yaml`, 1-10 replicas)
  - [x] `fetcher-comments-worker` ✅ **РЕАЛИЗОВАНО**:
    - [x] Deployment с resource limits (CPU: 1, RAM: 1GB) ✅ **РЕАЛИЗОВАНО** (`comments-worker-deployment.yaml`)
    - [x] Celery worker для queue `fetch.comments` ✅ **РЕАЛИЗОВАНО**
    - [x] HPA для авто-масштабирования ✅ **РЕАЛИЗОВАНО** (`comments-worker-hpa.yaml`, 1-15 replicas)
  - [x] `fetcher-finalize-worker` ✅ **РЕАЛИЗОВАНО**:
    - [x] Deployment для artifact builder ✅ **РЕАЛИЗОВАНО** (`finalize-worker-deployment.yaml`)
    - [x] Celery worker для queue `fetch.finalize` ✅ **РЕАЛИЗОВАНО**
  - [x] `fetcher-beat` ✅ **РЕАЛИЗОВАНО**:
    - [x] Deployment для Celery Beat (периодические задачи) ✅ **РЕАЛИЗОВАНО** (`beat-deployment.yaml`)
    - [x] Один replica (singleton) ✅ **РЕАЛИЗОВАНО**
  - [x] **Service манифесты** ✅ **РЕАЛИЗОВАНО**:
    - [x] Service для orchestrator API ✅ **РЕАЛИЗОВАНО** (`orchestrator-service.yaml`)
  - [x] **ConfigMap и Secrets** ✅ **РЕАЛИЗОВАНО**:
    - [x] ConfigMap для конфигурации Fetcher ✅ **РЕАЛИЗОВАНО** (`configmap.yaml`)
    - [x] Secrets для credentials (БД, Redis, S3, прокси) ✅ **РЕАЛИЗОВАНО** (`secrets-example.yaml` с примерами)
  - [x] **HPA (Horizontal Pod Autoscaler)** ✅ **РЕАЛИЗОВАНО**:
    - [x] HPA для metadata worker (метрики: CPU, память) ✅ **РЕАЛИЗОВАНО** (`metadata-worker-hpa.yaml`)
    - [x] HPA для download worker ✅ **РЕАЛИЗОВАНО** (`download-worker-hpa.yaml`)
    - [x] HPA для comments worker ✅ **РЕАЛИЗОВАНО** (`comments-worker-hpa.yaml`)
  - [x] **Документация** ✅ **РЕАЛИЗОВАНО** (`k8s/fetcher/README.md` с инструкциями по развёртыванию)

**Resource limits:**

| Worker   | CPU | RAM   |
|----------|-----|-------|
| Metadata | 0.5 | 512MB |
| Comments | 1   | 1GB   |
| Download | 2   | 2GB   |

### Logging

- [x] Structured logs (JSON‑формат) ✅ **РЕАЛИЗОВАНО** (`fetcher/logging.py`)
- [x] **Запись логов в таблицу `fetch_logs`** ✅ **РЕАЛИЗОВАНО** (интегрировано в `log_with_context`)
- [x] **Central log storage** ✅ **РЕАЛИЗОВАНО**:
  - [x] Интеграция с Loki (Grafana Loki) ✅ **РЕАЛИЗОВАНО** (`fetcher/logging_handlers.py`, `LokiHandler`)
  - [x] Интеграция с Elasticsearch ✅ **РЕАЛИЗОВАНО** (`ElasticsearchHandler`)
  - [x] Интеграция с AWS CloudWatch ✅ **РЕАЛИЗОВАНО** (`CloudWatchHandler`)
  - [x] Настройка отправки structured logs в JSON формате ✅ **РЕАЛИЗОВАНО** (используется `StructuredFormatter`)
  - [x] Индексация и поиск по логам ✅ **РЕАЛИЗОВАНО** (через Loki/Elasticsearch/CloudWatch)
  - [ ] Интеграция с GCP Cloud Logging (TODO: добавить поддержку)
  - [ ] Интеграция с Azure Monitor (TODO: добавить поддержку)
- [x] **Pipeline event logs** ✅ **РЕАЛИЗОВАНО**:
  - [x] Отдельная система для событий pipeline (Kafka) ✅ **РЕАЛИЗОВАНО** (`fetcher/kafka_producer.py`, `fetcher/events.py`)
  - [x] Интеграция с event streaming (Kafka для production) ✅ **РЕАЛИЗОВАНО** (полная интеграция в `orchestrator.py` и `tasks.py`)
  - [x] Event schema для всех типов событий ✅ **РЕАЛИЗОВАНО** (`schemas/events.py` с полной схемой событий)

**Required fields:**

- [x] `run_id` (интегрировано в воркеры)
- [x] `stage` (интегрировано в воркеры)
- [x] `level` (интегрировано в воркеры)
- [x] `timestamp` (интегрировано в воркеры)
- [x] `platform` / `platform_video_id` (интегрировано в воркеры)

### Monitoring alerts

Alert if:

- [x] Failure rate > threshold (описано в MONITORING_ALERTS.md)
- [x] Queue depth exploding (описано в MONITORING_ALERTS.md, будет актуально после Phase 4)
- [x] 429 rate spike (описано в MONITORING_ALERTS.md)
- [x] Cache hit ratio drop (описано в MONITORING_ALERTS.md)
- [x] Proxy health ухудшается (большой процент неуспешных запросов) (описано в MONITORING_ALERTS.md, будет актуально после Phase 3)

**Примечание**: Все алерты описаны в `MONITORING_ALERTS.md` с PromQL запросами и примерами конфигурации Prometheus Alertmanager.

### Health check

- [x] HTTP endpoint `/health` для проверки здоровья сервиса (реализовано в `fetcher/api.py`)
- [x] Проверка PostgreSQL базы данных
- [x] Проверка Redis (опционально)
- [x] Проверка S3/MinIO storage
- [x] Отслеживание uptime сервиса
- [x] Возврат статусов: healthy, degraded, unhealthy (503 для unhealthy)

---

## 🔥 Phase 7 — Post-MVP Production Hardening (Очень важно)

### Circuit breaker

Trigger conditions:

- [x] Too many 429 responses ✅ **РЕАЛИЗОВАНО** (`fetcher/circuit_breaker.py`)
- [x] Proxy failure spike ✅ **РЕАЛИЗОВАНО** (через общий счётчик ошибок)
- [x] Network timeout surge ✅ **РЕАЛИЗОВАНО** (через общий счётчик ошибок)

Behavior:

- [x] Temporarily disable соответствующий ingestion stage (metadata/download/comments) ✅ **РЕАЛИЗОВАНО** (интегрировано в `YouTubeAdapter`, проверка `breaker.is_open()` перед каждой операцией)
- [x] Логировать срабатывания и причину ✅ **РЕАЛИЗОВАНО** (логирование при переходе состояний)
- [x] Метрика `circuit_breaker_tripped_total` ✅ **РЕАЛИЗОВАНО** (обновляется при каждом срабатывании)

### Distributed locks

Use Redis locks for:

- [x] video download ✅ **РЕАЛИЗОВАНО** (используется в `YouTubeAdapter.download_video`)
- [ ] pipeline execution (при необходимости) — опционально, пока не требуется
- [x] artifact upload ✅ **РЕАЛИЗОВАНО** (функции `acquire_artifact_lock`/`release_artifact_lock` в `rate_limiter.py`, можно использовать при необходимости)

### Lifecycle storage policies

Configure:

- [x] Raw video: ✅ **РЕАЛИЗОВАНО** (`fetcher/lifecycle.py`, `cleanup_old_raw_videos()`):
  - [x] delete after ~30 days ✅ **РЕАЛИЗОВАНО** (конфигурируется через `raw_video_retention_days`, по умолчанию 30)
  - [x] Метод `delete_object()` в StorageClient для удаления артефактов
- [ ] **Улучшение lifecycle cleanup**:
  - [x] Реализовать `list_objects()` в StorageClient для temp bucket ✅ **РЕАЛИЗОВАНО** (`fetcher/storage.py`)
  - [x] Реализовать реальную очистку temp bucket ✅ **РЕАЛИЗОВАНО** (`cleanup_old_temp_files()` в `lifecycle.py`)
  - [ ] Архивация/удаление processed artifacts (когда будет требование)
- [ ] Features:
  - [ ] long-term storage (либо отдельная, мягкая политика) — обрабатывается DataProcessor
- [x] Temp files / temp buckets: ✅ **РЕАЛИЗОВАНО** (`cleanup_old_temp_files()`):
  - [x] TTL ≤ 7 days ✅ **РЕАЛИЗОВАНО** (конфигурируется через `temp_files_retention_days`, по умолчанию 7)
- [x] Failed runs cleanup ✅ **РЕАЛИЗОВАНО** (`cleanup_old_failed_runs()`, retention по умолчанию 7 дней)
- [x] **Периодическая задача для автоматической очистки** ✅ **РЕАЛИЗОВАНО** (`lifecycle_cleanup_task` в Celery, запускается ежедневно через Beat)
- [x] **Конфигурация retention policies** ✅ **РЕАЛИЗОВАНО** (в `FetcherSettings`: `raw_video_retention_days`, `temp_files_retention_days`, `failed_runs_retention_days`)

---

## ⭐ Quality Assurance Checklist

Перед релизом убедиться:

- [x] ✅ Idempotency of pipeline stages ✅ **РЕАЛИЗОВАНО**:
  - [x] Модуль `idempotency.py` с функциями проверки идемпотентности ✅ **РЕАЛИЗОВАНО**
  - [x] Интеграция в workers (metadata, video, comments) для пропуска уже выполненных stages ✅ **РЕАЛИЗОВАНО**
  - [x] Проверка существования артефактов в БД и storage ✅ **РЕАЛИЗОВАНО**
  - [x] Опциональная проверка checksum для валидации целостности ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Resume after worker crash ✅ **РЕАЛИЗОВАНО**:
  - [x] Модуль `resume.py` с функциями для определения следующей stage ✅ **РЕАЛИЗОВАНО**
  - [x] Функция `get_incomplete_runs()` для поиска незавершённых run'ов ✅ **РЕАЛИЗОВАНО**
  - [x] Функция `determine_next_stage()` для определения следующей stage для resume ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Manifest validation (JSON‑schema / internal validator) ✅ **РЕАЛИЗОВАНО** (`fetcher/manifest_validator.py`, валидация перед сохранением)
- [x] ✅ Checksum validation (несовпадение приводит к fail/retry) ✅ **РЕАЛИЗОВАНО**:
  - [x] Функция `validate_artifact_checksum()` для проверки checksum артефактов ✅ **РЕАЛИЗОВАНО** (`fetcher/idempotency.py`)
  - [x] Опциональная проверка checksum в `is_stage_idempotent()` ✅ **РЕАЛИЗОВАНО**
  - [x] Checksum вычисляется и сохраняется при upload всех артефактов ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Cache reuse correctness (нет лишних скачиваний) ✅ **РЕАЛИЗОВАНО**:
  - [x] Проверка кеша в `orchestrator.check_cache()` перед постановкой задач ✅ **РЕАЛИЗОВАНО**
  - [x] Проверка идемпотентности в workers для пропуска уже выполненных stages ✅ **РЕАЛИЗОВАНО**
  - [x] Distributed lock для video download предотвращает дубликаты ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Retry safety (нет бесконечных циклов / дубликатов) ✅ **РЕАЛИЗОВАНО** (`fetcher/errors.py`, разделение retryable/non-retryable ошибок, проверка max_retries)
- [x] ✅ Event ordering correctness (state machine не "прыгает" назад) ✅ **РЕАЛИЗОВАНО**:
  - [x] Модуль `state_machine.py` с таблицей разрешенных переходов ✅ **РЕАЛИЗОВАНО**
  - [x] Функция `validate_transition()` для валидации переходов ✅ **РЕАЛИЗОВАНО**
  - [x] Интеграция валидации в `orchestrator.py` и `tasks.py` ✅ **РЕАЛИЗОВАНО**
  - [x] Предотвращение недопустимых переходов (например, COMPLETED → PENDING) ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Proxy rotation correctness ✅ **РЕАЛИЗОВАНО**:
  - [x] Модуль `validation.py` с функцией `validate_proxy_rotation()` ✅ **РЕАЛИЗОВАНО**
  - [x] Проверка равномерного распределения прокси (round-robin) ✅ **РЕАЛИЗОВАНО**
  - [x] Endpoint `/admin/validation` для проверки корректности ✅ **РЕАЛИЗОВАНО**
  - [x] Логирование отклонений от равномерного распределения ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Rate limiter enforcement (нет всплесков выше заданных лимитов) ✅ **РЕАЛИЗОВАНО**:
  - [x] Функция `validate_rate_limiter_enforcement()` для проверки лимитов ✅ **РЕАЛИЗОВАНО**
  - [x] Проверка, что rate limiter не позволяет превысить лимит в окне времени ✅ **РЕАЛИЗОВАНО**
  - [x] Endpoint `/admin/validation` для проверки корректности ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Circuit breaker работает и снимается после окна "cooldown" ✅ **РЕАЛИЗОВАНО**:
  - [x] Функция `validate_circuit_breaker_cooldown()` для проверки cooldown ✅ **РЕАЛИЗОВАНО**
  - [x] Проверка перехода OPEN → HALF_OPEN после истечения cooldown ✅ **РЕАЛИЗОВАНО** (`circuit_breaker.py`, метод `is_open()`)
  - [x] Endpoint `/admin/validation` для проверки корректности ✅ **РЕАЛИЗОВАНО**
- [x] ✅ Load‑тесты на целевую нагрузку (например, 10k видео/день) ✅ **РЕАЛИЗОВАНО**:
  - [x] Скрипт `scripts/load_test.py` для load-тестирования ✅ **РЕАЛИЗОВАНО**
  - [x] Поддержка целевой нагрузки (10k видео/день) ✅ **РЕАЛИЗОВАНО**
  - [x] Сбор метрик (throughput, latency, success/failure rate) ✅ **РЕАЛИЗОВАНО**
  - [x] Документация по load testing (`docs/LOAD_TESTING.md`) ✅ **РЕАЛИЗОВАНО**

---

## Дополнительные рекомендации (опциональные, но полезные)

- [x] **Schema migration strategy** (Alembic): чёткий процесс эволюции БД‑схемы без даунтайма ✅ **РЕАЛИЗОВАНО**:
  - [x] Настроен Alembic для управления миграциями ✅ **РЕАЛИЗОВАНО** (`alembic.ini`, `alembic/env.py`)
  - [x] Создана начальная миграция с полной схемой БД ✅ **РЕАЛИЗОВАНО** (`alembic/versions/001_initial_schema.py`)
  - [x] Документация по использованию Alembic ✅ **РЕАЛИЗОВАНО** (`docs/DEVELOPMENT.md`)
- [x] **Local dev окружение** (docker-compose для Fetcher + Redis + MinIO + Postgres) ✅ **РЕАЛИЗОВАНО**:
  - [x] Docker Compose файл с всеми сервисами ✅ **РЕАЛИЗОВАНО** (`docker-compose.yml`)
  - [x] Dockerfile для Fetcher ✅ **РЕАЛИЗОВАНО** (`Dockerfile`)
  - [x] Документация по local development ✅ **РЕАЛИЗОВАНО** (`docs/DEVELOPMENT.md`)
- [x] **Автотесты** ✅ **РЕАЛИЗОВАНО**:
  - [x] Структура тестов (unit, integration) ✅ **РЕАЛИЗОВАНО** (`tests/`)
  - [x] Конфигурация pytest ✅ **РЕАЛИЗОВАНО** (`pytest.ini`)
  - [x] Общие фикстуры ✅ **РЕАЛИЗОВАНО** (`tests/conftest.py`)
  - [x] Unit тесты для state machine ✅ **РЕАЛИЗОВАНО** (`tests/unit/test_state_machine.py`)
  - [x] Unit тесты для идемпотентности ✅ **РЕАЛИЗОВАНО** (`tests/unit/test_idempotency.py`)
  - [x] Unit тесты для resume ✅ **РЕАЛИЗОВАНО** (`tests/unit/test_resume.py`)
  - [x] Integration тесты для идемпотентности ✅ **РЕАЛИЗОВАНО** (`tests/integration/test_idempotency.py`)
  - [x] Integration тесты для resume ✅ **РЕАЛИЗОВАНО** (`tests/integration/test_resume.py`)
  - [x] Документация по тестированию ✅ **РЕАЛИЗОВАНО** (`tests/README.md`)
  - [x] Зависимости для тестирования ✅ **РЕАЛИЗОВАНО** (`requirements-test.txt`)
  - [x] **Unit тесты на адаптеры платформ** ✅ **ЧАСТИЧНО РЕАЛИЗОВАНО**:
    - [x] Тесты для `YouTubeAdapter.fetch_metadata()` с моками yt-dlp ✅ **РЕАЛИЗОВАНО** (`tests/unit/test_youtube_adapter.py`)
    - [x] Тесты для `YouTubeAdapter.download_video()` с моками yt-dlp ✅ **РЕАЛИЗОВАНО**
    - [x] Тесты для `YouTubeAdapter.fetch_comments()` с моками yt-dlp ✅ **РЕАЛИЗОВАНО**
    - [x] Тесты для обработки ошибок (429, 403, network errors) ✅ **РЕАЛИЗОВАНО**
    - [x] Тесты для PII фильтрации в комментариях ✅ **РЕАЛИЗОВАНО**
    - [x] Тесты для checksum вычисления ✅ **РЕАЛИЗОВАНО**
    - [x] Тесты для circuit breaker интеграции ✅ **РЕАЛИЗОВАНО**
    - [x] Тесты для snapshot creation ✅ **РЕАЛИЗОВАНО**
    - [x] Тесты для retain_raw_comments/retain_raw_meta флагов ✅ **РЕАЛИЗОВАНО**
    - [ ] Тесты для proxy rotation (интеграция с get_next_proxy)
    - [ ] Тесты для всех edge cases (пустые комментарии, отсутствие метаданных, etc.)
  - [x] **Интеграционные тесты на полный pipeline** ✅ **РЕАЛИЗОВАНО**:
    - [x] Тест полного pipeline с фейковым YouTube (моки yt-dlp) ✅ **РЕАЛИЗОВАНО** (`tests/integration/test_full_pipeline.py`)
    - [x] Тест идемпотентности (повторный запуск не создаёт дубликаты) ✅ **РЕАЛИЗОВАНО** (`tests/integration/test_idempotency.py`)
    - [x] Тест с моками БД и storage ✅ **РЕАЛИЗОВАНО**
    - [x] Тест cache hit scenario ✅ **РЕАЛИЗОВАНО**
    - [x] Тест обработки ошибок (429, таймауты) ✅ **РЕАЛИЗОВАНО**
    - [ ] Тест resume после сбоя worker'а (TODO: реализовать в test_resume.py)
- [x] **Chaos‑тесты**: искусственные падения воркеров/сетевые ошибки для проверки устойчивости ✅ **РЕАЛИЗОВАНО**:
  - [x] Структура для chaos-тестов ✅ **РЕАЛИЗОВАНО** (`tests/chaos/`)
  - [x] Тесты для падений worker'ов ✅ **РЕАЛИЗОВАНО** (`tests/chaos/test_worker_failures.py`)
  - [x] Тесты для сетевых ошибок ✅ **РЕАЛИЗОВАНО** (`tests/chaos/test_network_failures.py`)
  - [x] Маркер `chaos` для pytest ✅ **РЕАЛИЗОВАНО** (`pytest.ini`)
  - [x] **Реализация chaos тестов** ✅ **РЕАЛИЗОВАНО**:
    - [x] Тест восстановления после падения metadata worker'а ✅ **РЕАЛИЗОВАНО** (`tests/chaos/test_worker_failures.py`)
    - [x] Тест восстановления после падения video worker'а ✅ **РЕАЛИЗОВАНО**
    - [x] Тест восстановления после падения comments worker'а ✅ **РЕАЛИЗОВАНО**
    - [x] Тест устойчивости к потере подключения к Redis ✅ **РЕАЛИЗОВАНО** (`tests/chaos/test_network_failures.py`)
    - [x] Тест устойчивости к потере подключения к Storage ✅ **РЕАЛИЗОВАНО**
    - [x] Тест устойчивости к потере подключения к БД ✅ **РЕАЛИЗОВАНО**
    - [x] Тест устойчивости к таймаутам YouTube API ✅ **РЕАЛИЗОВАНО**
    - [x] Тест устойчивости к rate limit errors (429) ✅ **РЕАЛИЗОВАНО**
    - [ ] Тест восстановления после падения finalize worker'а (TODO: когда будет finalize worker)
- [x] **Runbooks & playbooks**: что делать при падении прокси, росте 429, экспоненциальном росте очереди ✅ **РЕАЛИЗОВАНО**:
  - [x] Документация по runbooks ✅ **РЕАЛИЗОВАНО** (`docs/RUNBOOKS.md`)
  - [x] Руководства по решению типичных проблем ✅ **РЕАЛИЗОВАНО**:
    - [x] Падение прокси ✅ **РЕАЛИЗОВАНО**
    - [x] Рост 429 ошибок ✅ **РЕАЛИЗОВАНО**
    - [x] Экспоненциальный рост очереди ✅ **РЕАЛИЗОВАНО**
    - [x] Проблемы с БД ✅ **РЕАЛИЗОВАНО**
    - [x] Проблемы с Storage ✅ **РЕАЛИЗОВАНО**
    - [x] Circuit Breaker открыт ✅ **РЕАЛИЗОВАНО**
    - [x] Backpressure от DataProcessor ✅ **РЕАЛИЗОВАНО**

---

## 🔧 Дополнительные задачи разработки

### Улучшение rate limiter

- [ ] **Логирование ошибок Redis**:
  - [ ] Добавить логирование и отдельные метрики ошибок Redis
  - [ ] Обработка различных типов ошибок (connection, timeout, etc.)
  - [ ] Метрика `fetcher_redis_errors_total` для мониторинга
- [ ] **Более гибкая схема rate limiting**:
  - [ ] Token bucket или leaky bucket вместо fixed-window
  - [ ] Поддержка разных стратегий через конфигурацию
  - [ ] Обратная совместимость с текущей реализацией

### Поддержка других платформ

- [ ] **TikTok adapter**:
  - [ ] Реализация `TikTokAdapter` на основе `PlatformAdapter`
  - [ ] Интеграция с TikTok API или scraping
  - [ ] Тесты для TikTok adapter
- [ ] **Instagram adapter**:
  - [ ] Реализация `InstagramAdapter` на основе `PlatformAdapter`
  - [ ] Интеграция с Instagram API
  - [ ] Тесты для Instagram adapter
- [ ] **Расширяемая архитектура**:
  - [ ] Документация по добавлению новых платформ
  - [ ] Примеры и шаблоны для новых адаптеров

### Улучшение observability

- [ ] **Дополнительные метрики**:
  - [ ] Метрики для каждого типа ошибки (по коду ошибки)
  - [ ] Метрики для latency каждого этапа (p50, p95, p99)
  - [ ] Метрики для использования ресурсов (CPU, память)
  - [ ] Метрики для размера очереди по типам задач
- [ ] **Grafana Dashboard**:
  - [ ] Создать реальный dashboard в Grafana (сейчас только описание в `GRAFANA_DASHBOARD.md`)
  - [ ] Импортировать готовый dashboard JSON
  - [ ] Настроить алерты в Grafana

### Улучшение безопасности

- [ ] **Аутентификация для admin endpoints**:
  - [ ] Добавить аутентификацию для `/admin/*` endpoints
  - [ ] Интеграция с OAuth/JWT
  - [ ] Middleware для проверки токенов
- [ ] **Rate limiting для API**:
  - [ ] Защита API endpoints от злоупотреблений
  - [ ] IP-based rate limiting для `/admin/*` endpoints
  - [ ] Конфигурируемые лимиты для разных endpoints

### Улучшение документации

- [ ] **API документация**:
  - [ ] OpenAPI/Swagger спецификация (автогенерация из FastAPI)
  - [ ] Примеры запросов и ответов для всех endpoints
  - [ ] Описание всех query parameters и request bodies
- [ ] **Deployment guide**:
  - [ ] Подробное руководство по развёртыванию в production
  - [ ] Troubleshooting guide с типичными проблемами
  - [ ] Performance tuning guide
- [ ] **Architecture diagrams**:
  - [ ] Визуальные диаграммы архитектуры (Mermaid или PlantUML)
  - [ ] Sequence diagrams для основных потоков
  - [ ] Component diagrams для всех сервисов

### Следующие шаги (roadmap)

- [x] **Приоритетные очереди и Kafka‑наблюдаемость** (частично):
  - [x] Метрики по приоритетам: `fetcher_celery_queue_pending{queue="..."}` (Gauge), обновляются при GET /api/v1/queue; реальная глубина из Redis LLEN ✅ **РЕАЛИЗОВАНО** (`fetcher/celery_queues.py`, `fetcher/metrics.py`, `fetcher/api.py`)
  - [x] GET /api/v1/queue возвращает реальные pending по всем очередям (fetcher.high/normal/low, fetch.video/comments/finalize/maintenance/metadata) ✅ **РЕАЛИЗОВАНО**
  - [x] В `celery_app.py` добавлены комментарии с рекомендуемой командой запуска воркеров по очередям ✅ **РЕАЛИЗОВАНО**
  - [ ] Настроить Celery‑воркеры в k8s/deploy под `fetcher.high`, `fetcher.normal`, `fetcher.low` (ручная настройка при деплое)
  - [ ] Добавить метрику Kafka consumer lag и вывести в наблюдаемость (если Kafka включена; Gauge объявлен в `metrics.py`)
- [x] **End‑to‑end интеграция с DataProcessor** (документация и чеклист):
  - [x] Чеклист E2E зафиксирован ✅ **РЕАЛИЗОВАНО** (`docs/E2E_FETCHER_DATAPROCESSOR.md`): сценарии POST/GET runs, manifest, artifacts, отмена, кеш; примеры curl и опция автоматизации
  - [ ] Прогнать e2e‑сценарии на реальном стенде и при необходимости дополнить чеклист
- [x] **Мониторинг и алерты в Grafana** (готово к импорту):
  - [x] Дашборд по описанию из `GRAFANA_DASHBOARD.md` ✅ **РЕАЛИЗОВАНО** (`docs/grafana/fetcher-dashboard.json`): Ingestion Overview, Queue depth (`fetcher_celery_queue_pending`), Cache, Download latency, Platform health, Errors, DataProcessor queue
  - [x] README по импорту ✅ **РЕАЛИЗОВАНО** (`docs/grafana/README.md`)
  - [x] Алерты по ключевым метрикам ✅ **РЕАЛИЗОВАНО** (`docs/grafana/fetcher_alerts.yaml`): High failure rate, Circuit breaker, Proxy failure, Low cache hit ratio, 429 rate, Queue depth high, High download latency
  - [ ] Импортировать дашборд и алерты в реальном Grafana/Prometheus и при необходимости скорректировать пороги
- [ ] **Усиление безопасности API**:
  - [ ] Добавить JWT‑аутентификацию при необходимости и разграничить доступ к admin‑эндпоинтам
  - [ ] Пересмотреть значения лимитов (`runs_per_minute/hour/day`, `max_video_size_mb`, `max_video_duration_seconds`) под реальные бизнес‑ограничения
- [x] **Платформы и эксплуатация** (частично):
  - [x] Спланировать добавление платформ ✅ **РЕАЛИЗОВАНО** (`docs/ADDING_PLATFORMS.md`): чеклист для TikTok/Instagram (интерфейс, нормализация URL, конфиг, оркестратор, тесты)
  - [ ] Реализовать `TikTokAdapter` / `InstagramAdapter` по чеклисту
  - [ ] Доработать REST API при необходимости (фильтры/поля)
  - [x] CI: миграции Alembic + тесты на каждый merge ✅ **РЕАЛИЗОВАНО** (`.github/workflows/fetcher-ci.yml`): Lint (ruff), Alembic check, unit-тесты, integration-тесты с PostgreSQL и Redis)
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
