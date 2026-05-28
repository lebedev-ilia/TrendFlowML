from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Значения, при которых в production/staging процесс не стартует (см. validate_security_at_startup).
_WEAK_JWT_SECRETS = frozenset(
    {
        "change-me",
        "demo-change-me-in-production",
    }
)


def is_weak_jwt_secret(secret: str) -> bool:
    """Пустая строка или известные демо/дефолтные секреты."""
    s = (secret or "").strip().lower()
    return not s or s in _WEAK_JWT_SECRETS


def validate_security_at_startup(settings: "Settings") -> None:
    """
    Production/staging: fail-fast при слабом JWT secret.
    Development: одно предупреждение в лог при слабом секрете.
    """
    if settings.is_production_like() and is_weak_jwt_secret(settings.jwt_secret):
        raise RuntimeError(
            "TF_BACKEND_JWT_SECRET is missing, empty, or a known demo/default value while "
            "TF_BACKEND_DEPLOYMENT_ENV is production or staging. Set a long random secret "
            "(e.g. openssl rand -hex 32). See SECURITY.md."
        )
    if is_weak_jwt_secret(settings.jwt_secret):
        logger.warning(
            "TF_BACKEND_JWT_SECRET uses a weak or default value; do not set "
            "TF_BACKEND_DEPLOYMENT_ENV=production until you change it. See SECURITY.md."
        )


class Settings(BaseSettings):
    app_name: str = "TrendFlow Backend"
    debug: bool = False
    # production | staging — жёсткая проверка JWT secret при старте; иначе development.
    deployment_env: str = "development"

    # CORS: "*" (dev) или список origin через запятую, напр. "http://localhost:3000,https://app.example.com"
    cors_origins: str = "*"

    db_dsn: str = "postgresql+psycopg://trendflow:trendflow@localhost:5432/trendflow"
    # If true, the app will call `metadata.create_all()` on startup for legacy and v2 schemas.
    # Prefer running Alembic migrations and keep this disabled in production.
    db_auto_create: bool = False
    redis_url: str = "redis://localhost:6379/0"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_exp_minutes: int = 60 * 24 * 7
    admin_emails: str = ""

    storage_root: Optional[str] = None
    result_store_base: Optional[str] = None
    frames_dir_base: Optional[str] = None
    raw_uploads_dir: Optional[str] = None
    example_videos_dir: Optional[str] = None

    dataproc_root: Optional[str] = None
    visual_cfg_default: Optional[str] = None

    # DataProcessor API настройки (для Этапа 6)
    dataprocessor_api_url: str = "http://localhost:8001"  # URL DataProcessor API
    dataprocessor_api_key: Optional[str] = None  # API Key для аутентификации
    dataprocessor_poll_interval: int = 5  # Интервал polling статуса (секунды)
    dataprocessor_timeout_seconds: int = 3600  # Timeout для обработки (секунды)
    # POST /api/v1/process может держать соединение открытым, пока DP качает video_url в кеш (cold cache).
    dataprocessor_enqueue_timeout_seconds: float = 600.0
    # Повтор POST /api/v1/process при 503 (backpressure); пауза из Retry-After, не выше cap.
    dataprocessor_enqueue_max_retries: int = 12
    dataprocessor_enqueue_retry_after_cap_seconds: int = 120
    # Unified DataProcessor global_config (--global-config). Также: storage/e2e_full_max/active_global_config
    dataprocessor_global_config_path: Optional[str] = None

    # Fetcher API настройки (Backend ↔ Fetcher, Phase 0 интеграции)
    # См. docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md и backend/docs/FETCHER_INTEGRATION.md
    fetcher_api_url: str = "http://localhost:8000"  # URL Fetcher API (POST/GET /api/v1/runs); Docker Fetcher — порт 8000
    fetcher_api_key: Optional[str] = None  # API Key для X-API-Key (если Fetcher требует аутентификацию)
    fetcher_timeout_seconds: float = 30.0  # Timeout HTTP-запросов к Fetcher (секунды)

    # Trigger processing (Phase 2): вызов от Fetcher после finalize
    # См. backend/docs/FETCHER_INTEGRATION.md, docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md
    run_trigger_api_key: Optional[str] = None  # Если задан, POST .../trigger-processing требует X-API-Key (для вызова из Fetcher)

    # Phase 4: синхронизация статуса ingestion из Fetcher (polling)
    # Интервал вызова задачи sync_ingestion_run_status при использовании Celery beat (секунды)
    ingestion_sync_interval_seconds: int = 20
    ingestion_sync_lookback_hours: int = 6

    class Config:
        env_prefix = "TF_BACKEND_"
        env_file = ".env"

    def resolve_paths(self) -> "ResolvedPaths":
        repo_root = Path(__file__).resolve().parents[2]
        storage_root = Path(self.storage_root) if self.storage_root else repo_root / "storage"
        result_store_base = (
            Path(self.result_store_base)
            if self.result_store_base
            else storage_root / "result_store"
        )
        frames_dir_base = (
            Path(self.frames_dir_base) if self.frames_dir_base else storage_root / "frames_dir"
        )
        raw_uploads_dir = (
            Path(self.raw_uploads_dir) if self.raw_uploads_dir else storage_root / "raw"
        )
        example_videos_dir = (
            Path(self.example_videos_dir)
            if self.example_videos_dir
            else repo_root / "example" / "example_videos"
        )
        if self.dataproc_root:
            dataproc_root = Path(self.dataproc_root)
        else:
            # Монорепозиторий: .../TrendFlowML/DataProcessor. Автономный репо: только .../profiles/ у корня.
            _monolith_dp = repo_root / "DataProcessor"
            dataproc_root = _monolith_dp if _monolith_dp.is_dir() else repo_root
        visual_cfg_default = (
            Path(self.visual_cfg_default)
            if self.visual_cfg_default
            else dataproc_root / "configs" / "audit_v3" / "visual" / "visual_core_5_only.yaml"
        )
        return ResolvedPaths(
            repo_root=repo_root,
            storage_root=storage_root,
            result_store_base=result_store_base,
            frames_dir_base=frames_dir_base,
            raw_uploads_dir=raw_uploads_dir,
            example_videos_dir=example_videos_dir,
            dataproc_root=dataproc_root,
            visual_cfg_default=visual_cfg_default,
        )

    def admin_email_set(self) -> set[str]:
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}

    def cors_allow_origins(self) -> list[str]:
        """Список заголовка Access-Control-Allow-Origin (один "*" или явные URL)."""
        raw = self.cors_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    def is_production_like(self) -> bool:
        return self.deployment_env.strip().lower() in ("production", "staging")


class ResolvedPaths(BaseModel):
    repo_root: Path
    storage_root: Path
    result_store_base: Path
    frames_dir_base: Path
    raw_uploads_dir: Path
    example_videos_dir: Path
    dataproc_root: Path
    visual_cfg_default: Path

