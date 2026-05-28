"""Backpressure control для Fetcher.

Реализует проверку размера очереди DataProcessor перед finalize,
чтобы не перегружать DataProcessor при высокой нагрузке.
Соответствует Phase 4 чеклиста (Backpressure control).
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from .config import settings
from .metrics import (
    fetcher_backpressure_detected_total,
    fetcher_backpressure_check_errors_total,
    fetcher_processor_queue_size,
)

logger = logging.getLogger(__name__)


class BackpressureError(Exception):
    """Исключение, выбрасываемое при обнаружении backpressure."""

    def __init__(self, message: str, retry_after: int = 300):
        super().__init__(message)
        self.retry_after = retry_after


def get_processor_queue_size(processor_api_url: Optional[str] = None) -> int:
    """Получить размер очереди DataProcessor через HTTP API.

    Использует два варианта:
    1. `/api/v1/health` - возвращает JSON с `metrics.queue_length` (предпочтительно)
    2. `/api/v1/metrics` - Prometheus метрики, парсит `dataprocessor_queue_length` (fallback)

    Args:
        processor_api_url: URL DataProcessor API (по умолчанию из настроек)

    Returns:
        Размер очереди (сумма всех приоритетов) или 0 при ошибке

    Raises:
        BackpressureError: При недоступности DataProcessor API (опционально)
    """
    if processor_api_url is None:
        processor_api_url = getattr(settings, "dataprocessor_api_url", None)

    if not processor_api_url:
        # Если URL не настроен, считаем что backpressure проверка отключена
        logger.debug("DataProcessor API URL not configured, skipping backpressure check")
        return 0

    try:
        with httpx.Client(timeout=5.0) as client:
            # Вариант 1: Через health endpoint (предпочтительно)
            # DataProcessor возвращает queue_length в metrics.queue_length
            try:
                health_url = f"{processor_api_url.rstrip('/')}/api/v1/health"
                response = client.get(health_url)
                response.raise_for_status()
                health_data = response.json()
                
                # Извлекаем queue_length из metrics
                if "metrics" in health_data and "queue_length" in health_data["metrics"]:
                    queue_length = health_data["metrics"]["queue_length"]
                    logger.debug(f"DataProcessor queue size from health endpoint: {queue_length}")
                    return int(queue_length)
                else:
                    logger.warning("Health endpoint does not contain queue_length, trying metrics endpoint")
            except (httpx.HTTPStatusError, KeyError, ValueError) as e:
                logger.debug(f"Failed to get queue size from health endpoint: {e}, trying metrics endpoint")

            # Вариант 2: Через Prometheus metrics endpoint (fallback)
            try:
                metrics_url = f"{processor_api_url.rstrip('/')}/api/v1/metrics"
                response = client.get(metrics_url)
                response.raise_for_status()
                metrics_text = response.text
                
                # Парсим метрику dataprocessor_queue_length из Prometheus формата
                # Формат: dataprocessor_queue_length{priority="high"} 5.0
                total_queue_length = 0
                for line in metrics_text.split("\n"):
                    if line.startswith("dataprocessor_queue_length"):
                        # Извлекаем значение из строки вида: dataprocessor_queue_length{priority="high"} 5.0
                        try:
                            # Разделяем по пробелу и берем последнее значение
                            parts = line.split()
                            if len(parts) >= 2:
                                value = float(parts[-1])
                                total_queue_length += int(value)
                        except (ValueError, IndexError):
                            continue
                
                if total_queue_length > 0:
                    logger.debug(f"DataProcessor queue size from metrics endpoint: {total_queue_length}")
                    # Обновляем метрику
                    fetcher_processor_queue_size.set(total_queue_length)
                    return total_queue_length
                else:
                    logger.warning("Could not parse queue length from metrics endpoint")
                    fetcher_backpressure_check_errors_total.labels(error_type="parse_error").inc()
                    return 0
            except (httpx.HTTPStatusError, ValueError) as e:
                logger.warning(f"Failed to get queue size from metrics endpoint: {e}")
                fetcher_backpressure_check_errors_total.labels(error_type="metrics_endpoint_error").inc()
                return 0

    except httpx.TimeoutException:
        logger.warning("DataProcessor API timeout, assuming queue is not overloaded")
        fetcher_backpressure_check_errors_total.labels(error_type="timeout").inc()
        return 0
    except httpx.RequestError as e:
        logger.warning(f"DataProcessor API request failed: {e}, assuming queue is not overloaded")
        fetcher_backpressure_check_errors_total.labels(error_type="request_error").inc()
        return 0
    except Exception as e:
        logger.error(f"Unexpected error checking processor queue size: {e}")
        fetcher_backpressure_check_errors_total.labels(error_type="unexpected_error").inc()
        return 0


def check_backpressure(
    threshold: Optional[int] = None,
    processor_api_url: Optional[str] = None,
) -> bool:
    """Проверить, не переполнена ли очередь DataProcessor.

    Args:
        threshold: Порог размера очереди (по умолчанию из настроек)
        processor_api_url: URL DataProcessor API (по умолчанию из настроек)

    Returns:
        True если очередь переполнена (backpressure), False иначе
    """
    if threshold is None:
        threshold = getattr(settings, "backpressure_threshold", 1000)

    queue_size = get_processor_queue_size(processor_api_url)

    is_backpressure = queue_size > threshold

    if is_backpressure:
        logger.warning(
            f"Backpressure detected: processor queue size={queue_size}, threshold={threshold}"
        )
        fetcher_backpressure_detected_total.inc()
    else:
        # Обновляем метрику даже если backpressure нет
        fetcher_processor_queue_size.set(queue_size)

    return is_backpressure


__all__ = ["check_backpressure", "get_processor_queue_size", "BackpressureError"]

