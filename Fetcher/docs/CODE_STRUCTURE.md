## Структура кода Fetcher

Этот документ описывает текущую структуру Python‑кода Fetcher и соответствие модулей архитектурным документам.

---

## 1. Пакет `fetcher/`

Корневой пакет сервиса Fetcher:

- `fetcher/__init__.py`
  - корневой модуль пакета, содержит краткое описание назначения;
  - точка входа для импорта общих сущностей (`config`, `models`, и т.д. в будущем).

---

## 2. Конфигурация

- `fetcher/config.py`
  - класс `FetcherSettings` (на базе `pydantic.BaseSettings`);
  - читает переменные окружения с префиксом `FETCHER_` (`env_prefix="FETCHER_"`);
  - важные поля:
    - `postgres_dsn` — DSN для БД Fetcher;
    - `redis_url` — URL Redis для rate limiting/locks/очередей;
    - `s3_endpoint_url`, `s3_access_key`, `s3_secret_key`, `s3_region` — настройки S3/MinIO;
    - `bucket_raw`, `bucket_processed`, `bucket_temp` — имена бакетов (см. `STORAGE_LAYOUT.md`);
    - `enabled_platforms`, `youtube_enabled` — управление платформами (см. `PLATFORM_ADAPTERS.md`);
    - `youtube_metadata_limit_per_window`, `youtube_download_limit_per_window`, окна — базовые лимиты (см. `RATE_LIMITING_AND_LOCKS.md`);
    - `youtube_data_api_key`, `youtube_data_enabled`, `youtube_data_max_comments` — включение и настройка режима YouTube Data API v3;
    - `youtube_metadata_cache_ttl_seconds` — TTL in‑memory кэша метадаты YouTube Data API;
    - `youtube_daily_quota_limit`, `youtube_rate_limit_rps` — лимиты квоты и RPS для YouTube Data API;
    - `youtube_mock_video_download`, `youtube_mock_sample_video_dir`, `youtube_mock_sample_video_count` — параметры мокового скачивания YouTube‑видео для dev/e2e‑режима.
  - экспортирует singleton `settings = FetcherSettings()`.

---

## 3. База данных

- `fetcher/db.py`
  - создаёт `engine` SQLAlchemy (`create_engine(settings.postgres_dsn, ...)`);
  - `SessionLocal` — фабрика сессий;
  - `session_scope()` — контекстный менеджер для безопасной работы с транзакциями.
  - соответствует схеме из `DATABASE.md`.

- `fetcher/models.py`
  - содержит SQLAlchemy‑модели:
    - `Run`, `VideoSource`, `Video`, `VideoMetadata`, `ChannelMetadata`,
      `VideoSnapshot`, `Comment`, `Artifact`, `FetchJob`, `FetchLog`;
    - `Base`, `UUIDPrimaryKeyMixin`, `TimestampMixin`.
  - структуры выровнены с SQL‑контрактом из `Fetcher/docs/DATABASE.md`.

---

## 4. Object Storage

- `fetcher/storage.py`
  - протокол `StorageClient`:
    - `upload_file(local_path, bucket, key)`;
    - `download_file(bucket, key, local_path)`;
    - `object_exists(bucket, key)`.
  - реализация `S3StorageClient`:
    - обёртка над `boto3.client("s3", ...)`, использует настройки из `FetcherSettings`;
    - базовые методы upload/download/head_object;
  - singleton `storage_client` — экземпляр `S3StorageClient`.
  - реализует контракт layout’а из `STORAGE_LAYOUT.md`.

---

## 5. Platform adapters и сервисные клиенты

- `fetcher/platforms/base.py`
  - абстрактный класс `PlatformAdapter` с методами:
    - `fetch_metadata(source, *, run_id)`;
    - `download_video(source, *, run_id)`;
    - `fetch_comments(source, *, run_id, limit=100)`;
  - соответствует контракту из `PLATFORM_ADAPTERS.md`.

- `fetcher/platforms/youtube/__init__.py`
  - экспортирует `YouTubeAdapter`.

- `fetcher/platforms/provider_mode.py`, `fetcher/platforms/dual_provider.py`
  - `ProviderMode` (api_first, api_only, sdk_only, parallel) и `fetch_with_fallback()`.

- `fetcher/schemas/platform_video.py`
  - `PlatformVideoDto` — каноническая модель метаданных для всех платформ.

- `fetcher/platforms/adapter_utils.py`, `fetcher/platforms/download_utils.py`
  - общая персистенция metadata/comments и yt-dlp download.

- `fetcher/platforms/youtube/adapter.py`
  - класс `YouTubeAdapter(PlatformAdapter)`:
    - `fetch_metadata(source, run_id)`:
      - `youtube_provider_mode=api_first`: YouTube Data API с auto-fallback на yt-dlp при quota/429;
      - `sdk_only`: только yt-dlp;
    - `download_video(source, run_id)`:
      - при `settings.youtube_mock_video_download=True` использует мок‑режим: выбирает sample‑видео из директории по детерминированному индексу и загружает его как артефакт `video_file`;
      - в остальных случаях использует `yt-dlp` для скачивания видео ≤720p во временный файл, загружает файл в storage (`video-analytics-raw`) и регистрирует артефакт `video_file`.
    - `fetch_comments(source, run_id, limit=100)`:
      - при `settings.youtube_data_enabled=True` использует `YouTubeDataClient.iter_comments(...)` для стриминга комментариев, по мере прихода пишет их в таблицу `Comment` и формирует артефакт `comments_file` в формате JSON;
      - при `settings.youtube_data_enabled=False` использует `yt-dlp` для извлечения комментариев, ограничивает количество комментариев (top‑N), сохраняет комментарии в таблицу `Comment` и регистрирует артефакт `comments_file`.
  - вспомогательные методы `_get_or_create_video` и `_ensure_artifact` показывают интеграцию с БД.

- `fetcher/platforms/tiktok/`, `instagram/`, `rutube/`, `twitch/` — адаптеры с dual-mode.
- `fetcher/platforms/registry.py` — `get_adapter(platform)` для всех 5 платформ.
- `fetcher/platforms/platform_clients.py` — фабрики API/SDK клиентов из credentials.

- `fetcher/credentials/` — шаблоны и README для API keys/tokens (без правок кода).
- `fetcher/services/credentials.py` — `CredentialsStore`.

- `fetcher/services/__init__.py`, `fetcher/services/youtube_data_client.py`
  - `YouTubeDataClient` — YouTube Data API v3.
- `fetcher/services/tiktok_display_client.py`, `tiktok_sdk_client.py` — TikTok API + TikTokApi.
- `fetcher/services/instagram_graph_client.py`, `instagram_sdk_client.py` — Graph API + Instaloader.
- `fetcher/services/twitch_helix_client.py`, `twitch_sdk_client.py` — Helix + twitchAPI.
- `fetcher/services/rutube_ytdlp_client.py` — RuTube через yt-dlp.

---

## 6. Observability

- `fetcher/metrics.py`
  - Prometheus-метрики для Fetcher:
    - `fetcher_videos_downloaded_total` — счётчик успешных скачиваний;
    - `fetcher_videos_failed_total` — счётчик неудачных ингестий;
    - `fetcher_cache_hits_total`, `fetcher_cache_miss_total` — метрики кеша;
    - `fetcher_download_latency_seconds`, `fetcher_metadata_latency_seconds`, `fetcher_comments_latency_seconds` — гистограммы latency;
    - `fetcher_youtube_429_total`, `fetcher_youtube_403_total` — метрики ошибок YouTube API.
  - соответствует требованиям из `FETCHER_OBSERVABILITY.md`.

- `fetcher/logging.py`
  - structured logging для Fetcher:
    - `StructuredFormatter` — JSON formatter с полями `run_id`, `stage`, `level`, `timestamp`, `platform`;
    - `setup_logging()` — настройка логирования (JSON или text формат);
    - `log_with_context()` — helper для логирования с контекстом Fetcher.
  - используется всеми воркерами для логирования с контекстом.

- `fetcher/api.py`
  - FastAPI приложение для HTTP endpoints Fetcher (MVP):
    - `GET /metrics` — экспорт Prometheus-метрик в формате text/plain;
    - `GET /health` — health check endpoint с проверкой зависимостей (database, redis, storage);
    - `GET /` — root endpoint с информацией о сервисе;
    - обработка ошибок при генерации метрик и health check.
  - используется для экспорта метрик для Prometheus scraping и health checks.

- `fetcher/health.py`
  - функции проверки здоровья зависимостей Fetcher:
    - `check_database_health()` — проверка PostgreSQL;
    - `check_redis_health()` — проверка Redis (опционально);
    - `check_storage_health()` — проверка S3/MinIO;
    - `get_uptime_seconds()` — время работы сервиса;
    - `set_startup_time()` — установка времени запуска.
  - используется в health check endpoint.

- `fetcher/schemas.py`
  - Pydantic схемы для Fetcher API:
    - `HealthResponse` — response для health check endpoint.

---

## 7. Workers

- `fetcher/workers/__init__.py`
  - экспортирует функции воркеров: `run_metadata_worker`, `run_video_download_worker`, `run_comments_worker`, `run_artifact_builder`.

- `fetcher/workers/metadata.py`
  - `run_metadata_worker(run_id)`:
    - получает `VideoSource` для run'а;
    - вызывает `YouTubeAdapter.fetch_metadata()`;
    - интегрированы метрики (`fetcher_metadata_latency_seconds`) и structured logging.

- `fetcher/workers/video.py`
  - `run_video_download_worker(run_id)`:
    - получает `VideoSource` для run'а;
    - вызывает `YouTubeAdapter.download_video()`;
    - интегрированы метрики (`fetcher_download_latency_seconds`, `fetcher_videos_downloaded_total`) и structured logging.

- `fetcher/workers/comments.py`
  - `run_comments_worker(run_id, limit=100)`:
    - получает `VideoSource` для run'а;
    - вызывает `YouTubeAdapter.fetch_comments()` с указанным лимитом;
    - интегрированы метрики (`fetcher_comments_latency_seconds`) и structured logging.

- `fetcher/workers/artifacts.py`
  - `run_artifact_builder(run_id)`:
    - ожидает завершения других артефактов (video, metadata, comments);
    - строит `FetcherManifest` и загружает `manifest.json` в storage;
    - регистрирует артефакт `manifest_file`.

---

## 9. Queue & Orchestration

- `fetcher/celery_app.py`
  - Celery приложение для Fetcher:
    - broker и backend на Redis;
    - конфигурация (serialization, timezone, time limits);
    - роутинг задач по очередям с приоритетами.
  - соответствует требованиям из `QUEUE_ORCHESTRATION.md`.

- `fetcher/tasks.py`
  - Celery задачи для очередей:
    - `fetch_metadata_task` (queue: fetch.metadata, priority: 9, max_retries: 5);
    - `download_video_task` (queue: fetch.video, priority: 1, max_retries: 3);
    - `fetch_comments_task` (queue: fetch.comments, priority: 5, max_retries: 3);
    - `finalize_task` (queue: fetch.finalize, priority: 9, max_retries: 3).
  - каждая задача обёрнута вокруг существующего воркера;
  - retry с exponential backoff;
  - логирование с контекстом.

- `fetcher/orchestrator.py`
  - оркестратор для запуска ingestion pipeline:
    - `fetch_video(run_id)` — главная функция оркестратора;
    - `normalize_source(url)` — нормализация URL → platform_video_id;
    - `check_cache(platform, platform_video_id)` — проверка глобального кеша;
    - логика cache hit → сразу finalize, cache miss → постановка задач в очереди;
    - обновление статуса run в state machine.

- `fetcher/utils.py`
  - утилиты для Fetcher:
    - `all_artifacts_ready(run_id)` — проверка готовности всех обязательных артефактов.

---

## 8. Связь с архитектурой

- Архитектурные документы (`BACKEND_CONTRACTS.md`, `DATABASE.md`, `STORAGE_LAYOUT.md`, `PLATFORM_ADAPTERS.md`, `PIPELINE_ORCHESTRATION.md`, `CORE_INGESTION.md`, `RATE_LIMITING_AND_LOCKS.md`, `FETCHER_OBSERVABILITY.md`, `QUEUE_ORCHESTRATION.md`) описывают:
  - контракты между Fetcher, Backend и DataProcessor;
  - целевую схему БД и layout storage;
  - поведение adapter'ов и worker'ов;
  - метрики и observability требования.
- Код в пакете `fetcher` постепенно реализует эти контракты:
  - `config.py` и `storage.py` — фундамент для Object Storage;
  - `models.py` и `db.py` — слой доступа к данным;
  - `platforms/*` — адаптеры платформ, которые будут вызываться Orchestrator'ом и worker'ами;
  - `workers/*` — конкретные шаги ingestion‑pipeline'а (metadata, video download, comments, artifact builder);
  - `metrics.py` и `logging.py` — observability (метрики и structured logging);
  - `celery_app.py`, `tasks.py`, `orchestrator.py` — queue & orchestration (Celery + Redis).

По мере добавления новых модулей (orchestrator, rate limiter, locks) этот документ должен обновляться.


