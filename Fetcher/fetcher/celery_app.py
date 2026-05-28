"""Celery приложение для Fetcher.

Используется для управления очередями задач (metadata, video, comments, finalize).
"""

from __future__ import annotations

import os
from celery import Celery

from .config import settings


def _get_broker_url() -> str:
    """Получить URL брокера для Celery.

    Использует CELERY_BROKER_URL из env или redis_url из настроек.
    """
    return os.getenv("CELERY_BROKER_URL", settings.redis_url)


def _get_result_backend() -> str:
    """Получить URL result backend для Celery.

    Использует CELERY_RESULT_BACKEND из env или redis_url из настроек.
    """
    return os.getenv("CELERY_RESULT_BACKEND", settings.redis_url)


# Создание Celery приложения
celery_app = Celery(
    "fetcher",
    broker=_get_broker_url(),
    backend=_get_result_backend(),
)

# Конфигурация Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 час для video download
    task_soft_time_limit=3300,  # 55 минут soft limit
    worker_prefetch_multiplier=1,  # Не prefetch много задач
    worker_max_tasks_per_child=50,  # Перезапуск worker после N задач
    # Роутинг задач по очередям
    # Примечание: fetch_metadata_task может быть отправлен в разные очереди на основе priority
    # (fetcher.high, fetcher.normal, fetcher.low), поэтому роутинг определяется динамически
    task_routes={
        "fetcher.tasks.fetch_metadata_task": {"queue": "fetcher.normal"},  # Default queue, может быть переопределён при вызове
        "fetcher.tasks.download_video_task": {"queue": "fetch.video", "priority": 1},
        "fetcher.tasks.fetch_comments_task": {"queue": "fetch.comments", "priority": 5},
        "fetcher.tasks.finalize_task": {"queue": "fetch.finalize", "priority": 9},
        "fetcher.tasks.lifecycle_cleanup_task": {"queue": "fetch.maintenance", "priority": 1},
        "fetcher.tasks.periodic_snapshots_task": {"queue": "fetch.maintenance", "priority": 1},
        "fetcher.requeue_stuck_finalize": {"queue": "fetch.maintenance", "priority": 2},
        "fetcher.tasks.aggregate_stats_task": {"queue": "fetch.maintenance", "priority": 1},
    },
    # Приоритеты очередей
    task_default_queue="fetch.metadata",
    task_default_priority=5,
    task_default_exchange="fetcher",
    task_default_exchange_type="direct",
    task_default_routing_key="fetcher",
    # Периодические задачи (beat schedule)
    beat_schedule={
        "lifecycle-cleanup-daily": {
            "task": "fetcher.tasks.lifecycle_cleanup_task",
            "schedule": 86400.0,  # Каждые 24 часа
        },
        "periodic-snapshots-daily": {
            "task": "fetcher.tasks.periodic_snapshots_task",
            "schedule": 86400.0,  # Каждые 24 часа
        },
        "aggregate-stats-minute": {
            "task": "fetcher.tasks.aggregate_stats_task",
            "schedule": 60.0,  # Каждую минуту
        },
        "requeue-stuck-finalize": {
            "task": "fetcher.requeue_stuck_finalize",
            "schedule": 120.0,  # Каждые 2 минуты — перепоставить finalize для застрявших run'ов
        },
    },
)

# Автообнаружение задач
celery_app.autodiscover_tasks(["fetcher"])

# Рекомендуемые очереди для воркеров (приоритетные + воркерские):
# - Metadata: celery -A fetcher.celery_app worker -Q fetcher.high,fetcher.normal,fetcher.low,fetch.metadata -n metadata@%%h
# - Video:    celery -A fetcher.celery_app worker -Q fetch.video -n video@%%h
# - Comments: celery -A fetcher.celery_app worker -Q fetch.comments -n comments@%%h
# - Finalize: celery -A fetcher.celery_app worker -Q fetch.finalize -n finalize@%%h
# - Maintenance: celery -A fetcher.celery_app worker -Q fetch.maintenance -n maintenance@%%h

__all__ = ["celery_app"]

