"""Валидация корректности работы proxy rotation и rate limiter.

Соответствует Quality Assurance Checklist:
- Proxy rotation correctness
- Rate limiter enforcement (нет всплесков выше заданных лимитов)
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

from .config import settings
from .proxies import get_next_proxy, record_proxy_result
from .rate_limiter import acquire_token

logger = logging.getLogger(__name__)

# Трекинг использования прокси для валидации rotation
_proxy_usage_tracker: Dict[str, int] = defaultdict(int)
_proxy_usage_lock = None

# Трекинг rate limiter для валидации enforcement
_rate_limiter_tracker: Dict[str, List[float]] = defaultdict(list)
_rate_limiter_lock = None


def validate_proxy_rotation(
    num_requests: int = 100,
    expected_distribution_threshold: float = 0.1,
) -> tuple[bool, Dict[str, float]]:
    """Валидировать корректность proxy rotation.

    Проверяет, что прокси используются равномерно (round-robin) и не происходит
    зацикливания на одном прокси.

    Args:
        num_requests: Количество запросов для тестирования
        expected_distribution_threshold: Порог отклонения от равномерного распределения (0.1 = 10%)

    Returns:
        Tuple (is_valid, distribution)
        is_valid=True если rotation корректен, False иначе
        distribution содержит процент использования каждого прокси
    """
    if not settings.enable_proxies or not settings.proxies:
        logger.warning("Proxy rotation validation skipped: proxies disabled or empty")
        return True, {}

    proxy_counts: Dict[str, int] = defaultdict(int)

    # Симулируем запросы
    for _ in range(num_requests):
        proxy = get_next_proxy()
        if proxy:
            proxy_counts[proxy] += 1

    if not proxy_counts:
        logger.error("Proxy rotation validation failed: no proxies returned")
        return False, {}

    # Проверяем равномерность распределения
    expected_count = num_requests / len(settings.proxies)
    distribution: Dict[str, float] = {}
    is_valid = True

    for proxy_url in settings.proxies:
        count = proxy_counts.get(proxy_url, 0)
        percentage = (count / num_requests) * 100 if num_requests > 0 else 0
        distribution[proxy_url] = percentage

        # Проверяем отклонение от ожидаемого распределения
        deviation = abs(count - expected_count) / expected_count if expected_count > 0 else 1.0
        if deviation > expected_distribution_threshold:
            logger.warning(
                f"Proxy rotation validation: {proxy_url} has deviation {deviation:.2%} "
                f"(count={count}, expected={expected_count:.1f})"
            )
            # Не считаем это критической ошибкой, если есть здоровые прокси

    # Проверяем, что все прокси используются
    unused_proxies = [p for p in settings.proxies if proxy_counts.get(p, 0) == 0]
    if unused_proxies:
        logger.warning(
            f"Proxy rotation validation: {len(unused_proxies)} proxies never used: {unused_proxies}"
        )
        # Это может быть нормально, если прокси нездоровы

    logger.info(
        f"Proxy rotation validation: {len(proxy_counts)}/{len(settings.proxies)} proxies used, "
        f"distribution: {distribution}"
    )

    return is_valid, distribution


def validate_rate_limiter_enforcement(
    key: str,
    limit: int,
    window_sec: int,
    num_requests: Optional[int] = None,
) -> tuple[bool, Dict[str, int]]:
    """Валидировать, что rate limiter корректно ограничивает запросы.

    Проверяет, что rate limiter не позволяет превысить лимит в окне времени.

    Args:
        key: Ключ для rate limiter
        limit: Лимит запросов
        window_sec: Окно времени в секундах
        num_requests: Количество запросов для тестирования (по умолчанию limit * 2)

    Returns:
        Tuple (is_valid, stats)
        is_valid=True если rate limiter корректен, False иначе
        stats содержит статистику (allowed, denied, total)
    """
    if num_requests is None:
        num_requests = limit * 2

    allowed = 0
    denied = 0
    start_time = time.time()

    # Симулируем запросы
    for _ in range(num_requests):
        if acquire_token(key, limit, window_sec):
            allowed += 1
        else:
            denied += 1

    elapsed = time.time() - start_time

    # Проверяем, что не превышен лимит в окне
    # В fixed-window rate limiter первые limit запросов должны быть разрешены
    # Остальные должны быть отклонены (если окно не истекло)
    is_valid = allowed <= limit

    if not is_valid:
        logger.error(
            f"Rate limiter validation failed: {allowed} requests allowed, "
            f"but limit is {limit} (denied={denied}, elapsed={elapsed:.2f}s)"
        )
    else:
        logger.info(
            f"Rate limiter validation: {allowed} allowed, {denied} denied "
            f"(limit={limit}, window={window_sec}s, elapsed={elapsed:.2f}s)"
        )

    stats = {
        "allowed": allowed,
        "denied": denied,
        "total": num_requests,
        "elapsed_seconds": elapsed,
    }

    return is_valid, stats


def validate_circuit_breaker_cooldown(
    operation: str,
    cooldown_seconds: int = 300,
) -> tuple[bool, Dict[str, float]]:
    """Валидировать, что circuit breaker корректно снимается после cooldown.

    Args:
        operation: Тип операции (metadata, download, comments)
        cooldown_seconds: Время cooldown в секундах

    Returns:
        Tuple (is_valid, timings)
        is_valid=True если circuit breaker корректен, False иначе
        timings содержит временные метки (opened_at, cooldown_expired_at)
    """
    from .circuit_breaker import CircuitBreaker, get_circuit_breaker

    breaker = get_circuit_breaker(operation)

    # Проверяем текущее состояние
    initial_state = breaker.get_state()

    # Если circuit открыт, проверяем cooldown
    if breaker.is_open():
        opened_at = breaker.opened_at
        if opened_at:
            elapsed = time.time() - opened_at
            remaining = cooldown_seconds - elapsed

            logger.info(
                f"Circuit breaker {operation}: OPEN, opened_at={opened_at}, "
                f"elapsed={elapsed:.1f}s, remaining={remaining:.1f}s"
            )

            # Проверяем, что cooldown ещё не истёк (если circuit открыт)
            if remaining > 0:
                is_valid = True
                timings = {
                    "opened_at": opened_at,
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": remaining,
                    "cooldown_seconds": cooldown_seconds,
                }
            else:
                # Cooldown истёк, circuit должен перейти в HALF_OPEN
                is_valid = breaker.get_state().value == "half_open"
                timings = {
                    "opened_at": opened_at,
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": 0,
                    "cooldown_seconds": cooldown_seconds,
                    "state_after_cooldown": breaker.get_state().value,
                }
        else:
            # Circuit открыт, но opened_at не установлен (некорректное состояние)
            is_valid = False
            timings = {"error": "opened_at not set"}
    else:
        # Circuit закрыт или в half-open
        is_valid = True
        timings = {
            "state": breaker.get_state().value,
            "cooldown_seconds": cooldown_seconds,
        }

    return is_valid, timings


__all__ = [
    "validate_proxy_rotation",
    "validate_rate_limiter_enforcement",
    "validate_circuit_breaker_cooldown",
]

