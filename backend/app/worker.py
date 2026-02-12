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
)

