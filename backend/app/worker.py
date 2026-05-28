from __future__ import annotations

from celery import Celery

from .config import Settings

settings = Settings()

celery_app = Celery(
    "trendflow_backend",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Импорт модуля с задачами (sync_ingestion_run_status и др.), иначе worker не регистрирует их
    include=["app.tasks"],
)

# Phase 4: периодическая синхронизация статуса ingestion из Fetcher (polling)
# Worker: celery -A app.worker:celery_app worker -l info
# Beat (периодические задачи): celery -A app.worker:celery_app beat -l info
_interval = getattr(settings, "ingestion_sync_interval_seconds", 20)
celery_app.conf.beat_schedule = {
    "sync-ingestion-run-status": {
        "task": "sync_ingestion_run_status",
        "schedule": float(_interval),
        "options": {"queue": "celery"},
    },
}

