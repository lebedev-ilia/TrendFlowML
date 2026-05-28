"""Утилиты для наблюдения за очередями Celery (глубина очередей в Redis).

Используется в GET /api/v1/queue и для обновления метрик fetcher_celery_queue_pending.
"""

from __future__ import annotations

from typing import Dict

from .rate_limiter import get_redis_client
from .metrics import fetcher_celery_queue_pending

# Все очереди Fetcher: приоритетные для metadata + воркерские
CELERY_QUEUE_NAMES = [
    "fetcher.high",
    "fetcher.normal",
    "fetcher.low",
    "fetch.metadata",
    "fetch.video",
    "fetch.comments",
    "fetch.finalize",
    "fetch.maintenance",
]


def get_celery_queue_lengths() -> Dict[str, int]:
    """Получить текущую глубину очередей Celery из Redis (LLEN по каждому имени очереди).

    Returns:
        Словарь {queue_name: pending_count}. Отсутствующие ключи в Redis дают 0.
    """
    result: Dict[str, int] = {}
    try:
        redis_client = get_redis_client()
        for queue_name in CELERY_QUEUE_NAMES:
            try:
                count = redis_client.llen(queue_name)
                result[queue_name] = int(count) if count is not None else 0
            except Exception:
                result[queue_name] = 0
        return result
    except Exception:
        return {q: 0 for q in CELERY_QUEUE_NAMES}


def update_celery_queue_metrics() -> None:
    """Обновить Prometheus Gauges fetcher_celery_queue_pending по текущим глубинам из Redis."""
    lengths = get_celery_queue_lengths()
    for queue_name, pending in lengths.items():
        fetcher_celery_queue_pending.labels(queue=queue_name).set(pending)


__all__ = ["get_celery_queue_lengths", "update_celery_queue_metrics", "CELERY_QUEUE_NAMES"]
