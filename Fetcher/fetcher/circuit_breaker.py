"""Circuit breaker для Fetcher.

Реализует паттерн circuit breaker для временной блокировки операций при всплесках ошибок.
Соответствует Phase 7 чеклиста (Post-MVP Production Hardening).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .metrics import circuit_breaker_tripped_total

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Состояния circuit breaker."""

    CLOSED = "closed"  # Нормальная работа
    OPEN = "open"  # Блокировка операций
    HALF_OPEN = "half_open"  # Тестовый режим после cooldown


@dataclass
class CircuitBreakerConfig:
    """Конфигурация circuit breaker."""

    failure_threshold: int = 10  # Количество ошибок для открытия
    success_threshold: int = 3  # Количество успешных запросов для закрытия (half-open -> closed)
    window_seconds: int = 60  # Окно времени для подсчёта ошибок
    cooldown_seconds: int = 300  # Время блокировки (5 минут)


class CircuitBreaker:
    """Circuit breaker для операций Fetcher.

    Отслеживает частоту ошибок за окно времени и временно блокирует операции
    при превышении порога.
    """

    def __init__(
        self,
        operation: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Инициализация circuit breaker.

        Args:
            operation: Тип операции (metadata, download, comments)
            config: Конфигурация (если None, используется дефолтная)
        """
        self.operation = operation
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_timestamps: deque[float] = deque()
        self.success_count = 0
        self.opened_at: Optional[float] = None

    def record_success(self) -> None:
        """Записать успешный запрос."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                logger.info(
                    f"Circuit breaker {self.operation}: HALF_OPEN -> CLOSED "
                    f"(success_count={self.success_count})"
                )
                self.state = CircuitState.CLOSED
                self.success_count = 0
        elif self.state == CircuitState.OPEN:
            # Проверяем, не истёк ли cooldown
            if self.opened_at and (time.time() - self.opened_at) >= self.config.cooldown_seconds:
                logger.info(
                    f"Circuit breaker {self.operation}: OPEN -> HALF_OPEN (cooldown expired)"
                )
                self.state = CircuitState.HALF_OPEN
                self.success_count = 1
                self.opened_at = None

    def record_failure(self, reason: str = "unknown") -> None:
        """Записать неуспешный запрос.

        Args:
            reason: Причина ошибки (429, 403, timeout, etc.)
        """
        now = time.time()

        # Очищаем старые ошибки (старше window_seconds)
        while (
            self.failure_timestamps
            and (now - self.failure_timestamps[0]) > self.config.window_seconds
        ):
            self.failure_timestamps.popleft()

        self.failure_timestamps.append(now)

        if self.state == CircuitState.HALF_OPEN:
            # В half-open любая ошибка снова открывает circuit
            logger.warning(
                f"Circuit breaker {self.operation}: HALF_OPEN -> OPEN (failure in test mode)"
            )
            self.state = CircuitState.OPEN
            self.opened_at = now
            self.success_count = 0
            circuit_breaker_tripped_total.labels(operation=self.operation, reason=reason).inc()
        elif self.state == CircuitState.CLOSED:
            # Проверяем, не превышен ли порог
            if len(self.failure_timestamps) >= self.config.failure_threshold:
                logger.error(
                    f"Circuit breaker {self.operation}: CLOSED -> OPEN "
                    f"(failures={len(self.failure_timestamps)} >= threshold={self.config.failure_threshold}, reason={reason})"
                )
                self.state = CircuitState.OPEN
                self.opened_at = now
                circuit_breaker_tripped_total.labels(operation=self.operation, reason=reason).inc()

    def is_open(self) -> bool:
        """Проверить, открыт ли circuit breaker (блокирует ли операции).

        Returns:
            True если операции заблокированы, False иначе
        """
        if self.state == CircuitState.OPEN:
            # Проверяем cooldown
            if self.opened_at and (time.time() - self.opened_at) >= self.config.cooldown_seconds:
                # Переходим в half-open
                logger.info(
                    f"Circuit breaker {self.operation}: OPEN -> HALF_OPEN (cooldown expired)"
                )
                self.state = CircuitState.HALF_OPEN
                self.opened_at = None
                return False  # Разрешаем тестовые запросы
            return True
        return False

    def get_state(self) -> CircuitState:
        """Получить текущее состояние circuit breaker."""
        # Обновляем состояние если нужно
        self.is_open()
        return self.state


# Глобальные circuit breakers для каждой операции
_metadata_breaker = CircuitBreaker("metadata")
_download_breaker = CircuitBreaker("download")
_comments_breaker = CircuitBreaker("comments")


def get_circuit_breaker(operation: str) -> CircuitBreaker:
    """Получить circuit breaker для операции.

    Args:
        operation: Тип операции (metadata, download, comments)

    Returns:
        CircuitBreaker для указанной операции
    """
    if operation == "metadata":
        return _metadata_breaker
    elif operation == "download":
        return _download_breaker
    elif operation == "comments":
        return _comments_breaker
    else:
        # Создаём новый для неизвестной операции
        return CircuitBreaker(operation)


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "get_circuit_breaker",
]

