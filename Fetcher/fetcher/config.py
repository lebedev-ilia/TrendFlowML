from __future__ import annotations

from typing import List, Union

from pydantic import AnyHttpUrl, Field, validator

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    from pydantic import BaseSettings


class FetcherSettings(BaseSettings):
    """Настройки Fetcher.

    Источник истинны для:
    - подключения к БД и Redis;
    - параметров S3/MinIO;
    - включённых платформ;
    - лимитов запросов к платформам.
    """

    # Database
    postgres_dsn: str = Field(
        "postgresql+psycopg2://fetcher:fetcher@localhost:5432/fetcher",
        description="DSN для PostgreSQL, используемой Fetcher. Для TLS используйте параметр ?sslmode=require.",
    )
    postgres_ssl_mode: str | None = Field(
        None,
        description="SSL режим для PostgreSQL (disable, allow, prefer, require, verify-ca, verify-full). Если None, берётся из postgres_dsn.",
    )

    # Redis
    redis_url: str = Field(
        "redis://localhost:6379/0",
        description="URL Redis для rate limiting, locks и очередей. Для TLS используйте rediss:// вместо redis://.",
    )
    redis_ssl: bool = Field(
        False,
        description="Включить SSL/TLS для Redis (автоматически определяется из redis_url, если начинается с rediss://).",
    )
    redis_ssl_cert_reqs: str = Field(
        "required",
        description="Требования к сертификату Redis SSL (none, optional, required).",
    )

    # Storage (S3/MinIO)
    s3_endpoint_url: AnyHttpUrl | None = Field(
        None,
        description="Endpoint S3/MinIO (например, https://minio:9000 для TLS). Если None — используется дефолт SDK.",
    )
    s3_access_key: str | None = Field(
        None,
        description="Access key для S3/MinIO.",
    )
    s3_secret_key: str | None = Field(
        None,
        description="Secret key для S3/MinIO.",
    )
    s3_region: str | None = Field(
        None,
        description="Регион S3 (опционально).",
    )
    s3_use_ssl: bool = Field(
        True,
        description="Использовать SSL/TLS для S3/MinIO (автоматически определяется из s3_endpoint_url, если начинается с https://).",
    )
    s3_verify_ssl: bool = Field(
        True,
        description="Проверять SSL сертификат для S3/MinIO.",
    )

    bucket_raw: str = Field(
        "video-analytics-raw",
        description="Bucket для raw‑видео и сырых JSON.",
    )
    bucket_processed: str = Field(
        "video-analytics-processed",
        description="Bucket для обработанных артефактов DataProcessor.",
    )
    bucket_temp: str = Field(
        "video-analytics-temp",
        description="Bucket для временных файлов.",
    )

    # Platforms
    enabled_platforms: List[str] = Field(
        default_factory=lambda: ["youtube"],
        description='Список включённых платформ (например, ["youtube"]).',
    )
    youtube_enabled: bool = Field(
        True,
        description="Флаг включения YouTube адаптера.",
    )
    tiktok_enabled: bool = Field(
        False,
        description="Флаг включения TikTok адаптера.",
    )
    instagram_enabled: bool = Field(
        False,
        description="Флаг включения Instagram адаптера.",
    )
    rutube_enabled: bool = Field(
        False,
        description="Флаг включения RuTube адаптера.",
    )
    twitch_enabled: bool = Field(
        False,
        description="Флаг включения Twitch адаптера.",
    )

    # Credentials directory (API keys / tokens without editing code)
    credentials_dir: str = Field(
        "fetcher/credentials",
        description="Каталог с JSON/txt файлами credentials для платформ.",
    )

    # Rate limiting (YouTube, базовые значения; точные подберём экспериментально)
    youtube_metadata_limit_per_window: int = Field(
        400,
        description="Максимум metadata‑запросов к YouTube на IP/прокси за окно.",
    )
    youtube_metadata_window_sec: int = Field(
        3600,
        description="Длительность окна для metadata‑лимита (в секундах).",
    )
    youtube_download_limit_per_window: int = Field(
        80,
        description="Максимум download‑запросов к YouTube на IP/прокси за окно.",
    )
    youtube_download_window_sec: int = Field(
        3600,
        description="Длительность окна для download‑лимита (в секундах).",
    )

    # Rate limiting (TikTok, базовые значения; точные подберём экспериментально)
    tiktok_metadata_limit_per_window: int = Field(
        400,
        description="Максимум metadata‑запросов к TikTok на IP/прокси за окно.",
    )
    tiktok_metadata_window_sec: int = Field(
        3600,
        description="Длительность окна для metadata‑лимита TikTok (в секундах).",
    )
    tiktok_download_limit_per_window: int = Field(
        80,
        description="Максимум download‑запросов к TikTok на IP/прокси за окно.",
    )
    tiktok_download_window_sec: int = Field(
        3600,
        description="Длительность окна для download‑лимита TikTok (в секундах).",
    )
    tiktok_comments_limit_per_window: int = Field(
        400,
        description="Максимум comments‑операций для TikTok на IP/прокси за окно.",
    )
    tiktok_comments_window_sec: int = Field(
        3600,
        description="Длительность окна для comments‑лимита TikTok (в секундах).",
    )

    # YouTube normalization strategy
    youtube_use_yt_dlp: bool = Field(
        True,
        description=(
            "Использовать ли yt-dlp для нормализации YouTube URL (network call к youtube.com). "
            "Для локальной разработки/CI без доступа в интернет можно выставить в False, тогда "
            "ID будет извлекаться простым парсером URL без HTTP‑запросов."
        ),
    )

    # TikTok normalization strategy
    tiktok_use_yt_dlp: bool = Field(
        False,
        description=(
            "Использовать ли yt-dlp для нормализации TikTok URL (может требовать сетевой запрос). "
            "По умолчанию False: парсим только URL формата /@user/video/<id> без сети."
        ),
    )

    # YouTube Data API v3 (метадата и комментарии)
    youtube_data_api_key: str | None = Field(
        None,
        description="API key для YouTube Data API v3.",
    )
    youtube_data_enabled: bool = Field(
        False,
        description="Включить использование YouTube Data API v3 для метадаты и комментариев.",
    )
    youtube_data_max_comments: int = Field(
        1000,
        description="Максимальное количество комментариев на видео, запрашиваемых через YouTube Data API.",
    )
    youtube_metadata_cache_ttl_seconds: int = Field(
        24 * 60 * 60,
        description="TTL (в секундах) для кэша метадаты YouTube Data API в памяти.",
    )
    youtube_daily_quota_limit: int = Field(
        10_000,
        description="Суточный лимит квоты YouTube Data API (units).",
    )
    youtube_rate_limit_rps: int = Field(
        5,
        description="Базовый лимит запросов в секунду к YouTube Data API.",
    )
    youtube_provider_mode: str = Field(
        "api_first",
        description="Режим провайдера YouTube: api_first, api_only, sdk_only, parallel.",
    )

    # Моковое скачивание видео для E2E/локальной разработки
    youtube_mock_video_download: bool = Field(
        False,
        description="Включить моковое скачивание YouTube‑видео (sample файлы вместо реального трафика).",
    )
    youtube_mock_sample_video_dir: str | None = Field(
        None,
        description="Директория с sample‑видео для мокового скачивания YouTube‑роликов.",
    )
    youtube_mock_sample_video_count: int = Field(
        8,
        description="Количество sample‑файлов в директории мок‑видео.",
    )

    # Моковое скачивание видео TikTok для E2E/локальной разработки
    tiktok_mock_video_download: bool = Field(
        False,
        description="Включить моковое скачивание TikTok‑видео (sample файлы вместо реального трафика).",
    )
    tiktok_mock_sample_video_dir: str | None = Field(
        None,
        description="Директория с sample‑видео для мокового скачивания TikTok‑роликов.",
    )
    tiktok_mock_sample_video_count: int = Field(
        8,
        description="Количество sample‑файлов в директории мок‑видео TikTok.",
    )

    # TikTok Display API + TikTokApi SDK
    tiktok_data_enabled: bool = Field(
        False,
        description="Включить TikTok Display API для метаданных.",
    )
    tiktok_provider_mode: str = Field(
        "api_first",
        description="Режим провайдера TikTok: api_first, api_only, sdk_only, parallel.",
    )
    tiktok_client_key: str | None = Field(None, description="TikTok OAuth Client Key.")
    tiktok_client_secret: str | None = Field(None, description="TikTok OAuth Client Secret.")
    tiktok_access_token: str | None = Field(None, description="TikTok User/Client Access Token.")
    tiktok_open_id: str | None = Field(None, description="TikTok user open_id из OAuth.")
    tiktok_ms_token: str | None = Field(None, description="TikTok msToken cookie для TikTokApi SDK.")

    # Instagram Graph API + Instaloader SDK
    instagram_data_enabled: bool = Field(
        False,
        description="Включить Instagram Graph API для метаданных.",
    )
    instagram_provider_mode: str = Field(
        "api_first",
        description="Режим провайдера Instagram: api_first, api_only, sdk_only, parallel.",
    )
    instagram_access_token: str | None = Field(None, description="Instagram Graph API long-lived token.")
    instagram_ig_user_id: str | None = Field(None, description="Instagram Business/Creator user id.")
    instagram_instaloader_session: str | None = Field(
        None,
        description="Путь к session-файлу Instaloader.",
    )

    # Twitch Helix API + twitchAPI SDK
    twitch_data_enabled: bool = Field(
        False,
        description="Включить Twitch Helix API.",
    )
    twitch_provider_mode: str = Field(
        "api_first",
        description="Режим провайдера Twitch: api_first, api_only, sdk_only, parallel.",
    )
    twitch_client_id: str | None = Field(None, description="Twitch application Client-ID.")
    twitch_client_secret: str | None = Field(None, description="Twitch application Client Secret.")
    twitch_access_token: str | None = Field(None, description="Twitch OAuth/App access token.")

    # RuTube (только yt-dlp SDK — официального API нет)
    rutube_provider_mode: str = Field(
        "sdk_only",
        description="Режим провайдера RuTube (только sdk_only поддерживается).",
    )

    # Rate limiting (Instagram)
    instagram_metadata_limit_per_window: int = Field(200, description="Instagram metadata requests per window.")
    instagram_metadata_window_sec: int = Field(3600, description="Instagram metadata window seconds.")

    # Rate limiting (Twitch)
    twitch_metadata_limit_per_window: int = Field(800, description="Twitch metadata requests per window.")
    twitch_metadata_window_sec: int = Field(60, description="Twitch metadata window seconds.")

    # Rate limiting (RuTube)
    rutube_metadata_limit_per_window: int = Field(400, description="RuTube metadata requests per window.")
    rutube_metadata_window_sec: int = Field(3600, description="RuTube metadata window seconds.")
    rutube_download_limit_per_window: int = Field(80, description="RuTube download requests per window.")
    rutube_download_window_sec: int = Field(3600, description="RuTube download window seconds.")

    # Локальный E2E / dev‑режим: позволять завершать пайплайн без comments_file
    allow_finalize_without_comments: bool = Field(
        False,
        description=(
            "Разрешить завершать finalize, когда metadata_file и video_file готовы, "
            "но comments_file отсутствует. Полезно для локального E2E и отладки."
        ),
    )

    # Proxy / network
    enable_proxies: bool = Field(
        False,
        description="Включить использование proxy-пула для запросов к платформам.",
    )
    # Union[str, List[str]] чтобы из env приходила строка без JSON-парсинга (http://... не валидный JSON)
    proxies: Union[str, List[str]] = Field(
        default_factory=list,
        description="Список proxy URL (например, socks5://user:pass@host:port или http://user:pass@host:port). В env — строка или через запятую.",
    )

    @validator("proxies", pre=True)
    def parse_proxies_from_env(cls, v: object) -> List[str]:
        """Парсит FETCHER_PROXIES: строка (один URL) или через запятую — в список."""
        if v is None:
            return []
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

    # Proxy authentication (альтернативный способ через отдельные переменные)
    proxy_auth_username: str | None = Field(
        None,
        description="Username для proxy authentication (если не указан в proxy URL).",
    )
    proxy_auth_password: str | None = Field(
        None,
        description="Password для proxy authentication (если не указан в proxy URL).",
    )
    cookie_files_dir: str | None = Field(
        None,
        description="Директория с Netscape cookie .txt файлами для yt-dlp; файлы ротируются между запросами.",
    )
    cookie_file_glob: str = Field(
        "*.txt",
        description="Glob-шаблон cookie-файлов внутри cookie_files_dir.",
    )

    # Privacy / PII & snapshots
    enable_pii_filtering: bool = Field(
        True,
        description="Включить базовую PII‑фильтрацию комментариев (email/phone/url).",
    )
    enable_snapshots: bool = Field(
        False,
        description="Включить запись initial video_snapshots (snapshot_index=0). По умолчанию отключено для упрощения локальной разработки и unit‑тестов.",
    )
    snapshot_schedule_days: List[int] = Field(
        default_factory=lambda: [0, 7, 14, 21],
        description="Расписание для периодических snapshots в днях от первого snapshot (например, [0, 7, 14, 21] для snapshots на 0, 7, 14, 21 день).",
    )
    # Raw text retention flags
    retain_raw_comments: bool = Field(
        False,
        description="Сохранять raw текст комментариев в БД и storage (по умолчанию False для privacy).",
    )
    retain_raw_meta: bool = Field(
        True,
        description="Сохранять raw метаданные (description, tags) в storage (по умолчанию True, можно отключить для privacy).",
    )

    # Lifecycle / Retention policies
    raw_video_retention_days: int = Field(
        30,
        description="Количество дней для хранения raw видео перед удалением (по умолчанию 30).",
    )
    raw_comments_retention_days: int = Field(
        30,
        description="Количество дней для хранения raw комментариев перед удалением (по умолчанию 30).",
    )
    raw_comments_hard_cap_days: int = Field(
        60,
        description="Hard cap для хранения raw комментариев (максимум, даже если пользователь выбрал больше, по умолчанию 60).",
    )
    temp_files_retention_days: int = Field(
        7,
        description="Количество дней для хранения temp файлов перед удалением (по умолчанию 7).",
    )
    failed_runs_retention_days: int = Field(
        7,
        description="Количество дней для хранения failed runs перед удалением/архивацией (по умолчанию 7).",
    )

    # Backpressure control
    dataprocessor_api_url: str | None = Field(
        None,
        description="URL DataProcessor API для проверки размера очереди (например, http://dataprocessor:8000).",
    )
    backpressure_threshold: int = Field(
        1000,
        description="Порог размера очереди DataProcessor для срабатывания backpressure (по умолчанию 1000).",
    )

    # Centralized logging
    logging_backend: str | None = Field(
        None,
        description="Backend для централизованного логирования (loki, elasticsearch, cloudwatch). Если None — логи только в stdout и БД.",
    )
    logging_loki_url: str | None = Field(
        None,
        description="URL Grafana Loki API (например, http://loki:3100). Используется если logging_backend=loki.",
    )
    logging_elasticsearch_url: str | None = Field(
        None,
        description="URL Elasticsearch API (например, http://elasticsearch:9200). Используется если logging_backend=elasticsearch.",
    )
    logging_elasticsearch_index: str = Field(
        "fetcher-logs",
        description="Имя индекса Elasticsearch для логов.",
    )
    logging_cloudwatch_log_group: str = Field(
        "/aws/fetcher",
        description="Имя log group в AWS CloudWatch. Используется если logging_backend=cloudwatch.",
    )
    logging_cloudwatch_region: str | None = Field(
        None,
        description="AWS регион для CloudWatch (если None, используется дефолтный).",
    )

    # Kafka event streaming (опционально для production)
    kafka_enabled: bool = Field(
        False,
        description="Включить Kafka event streaming (если False, используется только Celery + Redis).",
    )
    kafka_bootstrap_servers: str | list[str] | None = Field(
        None,
        description="Kafka brokers (например, 'localhost:9092' или ['kafka1:9092', 'kafka2:9092']).",
    )
    kafka_topic_prefix: str = Field(
        "fetcher",
        description="Префикс для Kafka topics (например, 'fetcher' → 'fetcher.events', 'fetcher.tasks.*').",
    )

    # Webhooks
    webhook_secret: str | None = Field(
        None,
        description="Секретный ключ для подписи webhook payload (HMAC-SHA256). Если None, webhooks отправляются без подписи.",
    )

    # API Authentication
    api_keys: str | List[str] | None = Field(
        None,
        description="Список валидных API keys (comma-separated строка или список). Если None, аутентификация отключена.",
    )
    api_require_auth: bool = Field(
        False,
        description="Требовать ли аутентификацию для всех API endpoints (кроме публичных). Если False и api_keys не настроены, аутентификация отключена.",
    )

    # API Rate Limiting
    api_rate_limit_per_minute: int = Field(
        60,
        description="Rate limit для API endpoints (запросов в минуту на IP или API key).",
    )

    # API run creation limits (для GET /api/v1/limits)
    runs_per_minute: int = Field(
        100,
        description="Максимум созданий run'ов в минуту (отображается в /api/v1/limits).",
    )
    runs_per_hour: int = Field(
        1000,
        description="Максимум созданий run'ов в час (отображается в /api/v1/limits).",
    )
    runs_per_day: int = Field(
        10000,
        description="Максимум созданий run'ов в день (отображается в /api/v1/limits).",
    )
    max_video_size_mb: int = Field(
        500,
        description="Максимальный размер видео в МБ (отображается в /api/v1/limits).",
    )
    max_video_duration_seconds: int = Field(
        3600,
        description="Максимальная длительность видео в секундах (отображается в /api/v1/limits).",
    )
    max_comments_per_video: int = Field(
        100000,
        description="Максимум комментариев на видео (отображается в /api/v1/limits).",
    )

    # Backend trigger (Phase 2): вызов Backend после finalize для запуска DataProcessor
    # См. docs/BACKEND_CONTRACTS.md, docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md (в корне репо)
    backend_base_url: str | None = Field(
        None,
        description="Базовый URL Backend API (например http://backend:8000). Если задан, после COMPLETED вызывается POST {url}/api/runs/{run_id}/trigger-processing.",
    )
    backend_trigger_api_key: str | None = Field(
        None,
        description="API key для заголовка X-API-Key при вызове Backend trigger-processing (должен совпадать с TF_BACKEND_RUN_TRIGGER_API_KEY).",
    )

    class Config:
        env_prefix = "FETCHER_"
        env_file = ".env"
        case_sensitive = False


settings = FetcherSettings()


__all__ = ["FetcherSettings", "settings"]


