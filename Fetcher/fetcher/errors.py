"""Классификация ошибок для Fetcher.

Разделяет ошибки на retryable (можно повторить) и non-retryable (fail-fast).
Соответствует Quality Assurance Checklist (Retry safety).
"""

from __future__ import annotations

import logging
from typing import Type

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Базовый класс для ошибок, которые можно повторить (retry)."""

    pass


class NonRetryableError(Exception):
    """Базовый класс для ошибок, которые нельзя повторить (fail-fast)."""

    pass


# Retryable ошибки
class NetworkError(RetryableError):
    """Сетевые ошибки (connection timeout, DNS failure, etc.)."""

    pass


class TimeoutError(RetryableError):
    """Timeout ошибки (component превысил timeout)."""

    pass


class RateLimitError(RetryableError):
    """Rate limit ошибки (429 Too Many Requests)."""

    pass


class TransientStorageError(RetryableError):
    """Временные ошибки storage (503 Service Unavailable, connection errors)."""

    pass


class CircuitBreakerOpenError(RetryableError):
    """Circuit breaker открыт (временная блокировка операций)."""

    pass


# Non-retryable ошибки
class VideoNotFoundError(NonRetryableError):
    """Видео не найдено (удалено, не существует)."""

    pass


class PrivateVideoError(NonRetryableError):
    """Видео приватное (нет доступа)."""

    pass


class AgeRestrictedError(NonRetryableError):
    """Видео с возрастными ограничениями (требуется авторизация)."""

    pass


class InvalidInputError(NonRetryableError):
    """Невалидный вход (повреждённое видео, неподдерживаемый формат)."""

    pass


class LogicError(NonRetryableError):
    """Логическая ошибка (алгоритм вернул ошибку валидации)."""

    pass


class AuthenticationError(NonRetryableError):
    """Ошибка аутентификации (неверный API key)."""

    pass


class MissingDependencyError(NonRetryableError):
    """Отсутствует зависимость (например, frame_indices отсутствуют)."""

    pass


def is_retryable_error(error: Exception) -> bool:
    """Проверить, является ли ошибка retryable.

    Args:
        error: Исключение для проверки

    Returns:
        True если ошибка retryable, False если non-retryable
    """
    # Проверяем по типу исключения
    if isinstance(error, RetryableError):
        return True
    if isinstance(error, NonRetryableError):
        return False

    # Проверяем по строковому представлению (для ошибок от yt-dlp и других библиотек)
    error_str = str(error).lower()

    # Retryable паттерны
    retryable_patterns = [
        "timeout",
        "connection",
        "network",
        "429",  # Too Many Requests
        "503",  # Service Unavailable
        "502",  # Bad Gateway
        "504",  # Gateway Timeout
        "temporary",
        "transient",
        "rate limit",
        "too many requests",
        "circuit breaker",
    ]

    # Non-retryable паттерны
    non_retryable_patterns = [
        "not found",
        "404",  # Not Found
        "private",
        "age restricted",
        "invalid",
        "authentication",
        "401",  # Unauthorized
        "403",  # Forbidden (может быть и retryable, но обычно нет)
        "missing",
        "removed",
        "deleted",
        "unavailable",  # Может быть и retryable, но для видео обычно нет
    ]

    # Сначала проверяем non-retryable (более специфичные)
    for pattern in non_retryable_patterns:
        if pattern in error_str:
            return False

    # Затем проверяем retryable
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True

    # По умолчанию считаем retryable (консервативный подход)
    # Лучше повторить попытку, чем пропустить из-за временной ошибки
    logger.warning(
        f"Unknown error type, treating as retryable: {type(error).__name__}: {error_str}"
    )
    return True


def get_error_category(error: Exception) -> str:
    """Получить категорию ошибки для логирования и метрик.

    Args:
        error: Исключение

    Returns:
        Категория ошибки: "retryable" или "non_retryable"
    """
    if isinstance(error, RetryableError):
        return "retryable"
    if isinstance(error, NonRetryableError):
        return "non_retryable"

    return "retryable" if is_retryable_error(error) else "non_retryable"


__all__ = [
    "RetryableError",
    "NonRetryableError",
    "NetworkError",
    "TimeoutError",
    "RateLimitError",
    "TransientStorageError",
    "CircuitBreakerOpenError",
    "VideoNotFoundError",
    "PrivateVideoError",
    "AgeRestrictedError",
    "InvalidInputError",
    "LogicError",
    "AuthenticationError",
    "MissingDependencyError",
    "is_retryable_error",
    "get_error_category",
]

