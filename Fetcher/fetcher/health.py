"""Функции проверки здоровья зависимостей Fetcher."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from sqlalchemy import text

from .config import settings
from .db import engine
from .storage import storage_client

logger = logging.getLogger(__name__)


def check_database_health() -> Dict[str, Any]:
    """Проверить доступность PostgreSQL.

    Returns:
        Dict с ключами:
        - status: "healthy" | "unhealthy"
        - error: сообщение об ошибке (если есть)
    """
    try:
        with engine.connect() as conn:
            # Простой запрос для проверки соединения
            conn.execute(text("SELECT 1"))
        return {"status": "healthy"}
    except Exception as e:
        logger.exception("Database health check failed")
        return {"status": "unhealthy", "error": str(e)}


def check_redis_health() -> Dict[str, Any]:
    """Проверить доступность Redis.

    Returns:
        Dict с ключами:
        - status: "healthy" | "unhealthy" | "not_configured"
        - error: сообщение об ошибке (если есть)
    """
    try:
        import redis

        client = redis.from_url(settings.redis_url, socket_connect_timeout=2)
        # Простой ping для проверки соединения
        client.ping()
        return {"status": "healthy"}
    except ImportError:
        return {"status": "not_configured", "error": "redis package not installed"}
    except Exception as e:
        logger.exception("Redis health check failed")
        return {"status": "unhealthy", "error": str(e)}


def check_storage_health() -> Dict[str, Any]:
    """Проверить доступность S3/MinIO storage.

    Выполняет тестовую операцию (head_object на несуществующий объект)
    для проверки доступности storage.

    Returns:
        Dict с ключами:
        - status: "healthy" | "unhealthy"
        - error: сообщение об ошибке (если есть)
        - type: "s3" | "minio"
    """
    try:
        # Пытаемся проверить существование несуществующего объекта
        # Это проверяет доступность storage без создания реальных объектов
        test_bucket = settings.bucket_raw
        test_key = "__health_check__"

        # Если bucket не существует, это тоже ошибка
        # Для MVP просто проверяем, что можем выполнить операцию
        exists = storage_client.object_exists(test_bucket, test_key)
        # Результат не важен, важно что операция выполнилась без ошибок

        # Определяем тип storage по endpoint
        storage_type = "minio" if settings.s3_endpoint_url else "s3"

        return {"status": "healthy", "type": storage_type}
    except Exception as e:
        logger.exception("Storage health check failed")
        return {"status": "unhealthy", "error": str(e)}


# Глобальная переменная для отслеживания времени запуска
_startup_time: float | None = None


def set_startup_time() -> None:
    """Установить время запуска сервиса."""
    global _startup_time
    _startup_time = time.time()


def get_uptime_seconds() -> float:
    """Получить время работы сервиса в секундах.

    Returns:
        Время работы в секундах (0 если сервис ещё не запущен).
    """
    if _startup_time is None:
        return 0.0
    return time.time() - _startup_time


__all__ = [
    "check_database_health",
    "check_redis_health",
    "check_storage_health",
    "set_startup_time",
    "get_uptime_seconds",
]

