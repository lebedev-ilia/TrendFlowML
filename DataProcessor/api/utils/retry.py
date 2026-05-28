"""
Retry Utilities - утилиты для retry логики с exponential backoff

Реализует retry для transient errors:
- Triton timeout + retry (3 раза, timeout 30 сек)
- Storage retry с exponential backoff (1s, 2s, 4s, 8s, после 5 попыток → error)

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2241-2244)
"""

import asyncio
import logging
import time
from typing import Callable, TypeVar, Optional, List, Any
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryableError(Exception):
    """Базовый класс для ошибок, которые можно повторить."""
    pass


class TransientStorageError(RetryableError):
    """Transient ошибка Storage (500, network error, timeout)."""
    pass


class TritonTimeoutError(RetryableError):
    """Triton timeout ошибка."""
    pass


def is_transient_storage_error(error: Exception) -> bool:
    """
    Проверить, является ли ошибка Storage transient.
    
    Args:
        error: Исключение для проверки
        
    Returns:
        True если ошибка transient и можно повторить
    """
    # NotFoundError не transient - не нужно повторять
    from storage.base import NotFoundError, StorageError
    
    if isinstance(error, NotFoundError):
        return False
    
    # StorageError может быть transient (500, network, timeout)
    if isinstance(error, StorageError):
        return True
    
    # Проверить boto3 ClientError для S3
    try:
        from botocore.exceptions import ClientError
        if isinstance(error, ClientError):
            code = str(error.response.get("Error", {}).get("Code", ""))
            # 500, 503, 502 - transient errors
            if code in ("500", "502", "503", "InternalServerError", "ServiceUnavailable", "BadGateway"):
                return True
            # 429 - rate limit, можно повторить
            if code == "429":
                return True
            # 404, 403 - не transient
            return False
    except ImportError:
        pass
    
    # Network errors (ConnectionError, TimeoutError)
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    
    return False


def is_triton_timeout_error(error: Exception) -> bool:
    """
    Проверить, является ли ошибка Triton timeout.
    
    Args:
        error: Исключение для проверки
        
    Returns:
        True если ошибка timeout и можно повторить
    """
    # Проверить TritonError с timeout
    try:
        from dp_triton import TritonError
        if isinstance(error, TritonError):
            # Проверить код ошибки или сообщение
            error_code = getattr(error, "error_code", "")
            error_msg = str(error).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                return True
            if error_code in ("triton_timeout", "triton_unavailable"):
                return True
    except ImportError:
        pass
    
    # TimeoutError может быть от Triton
    if isinstance(error, TimeoutError):
        return True
    
    return False


async def retry_with_exponential_backoff(
    func: Callable[..., T],
    *args,
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    is_retryable: Optional[Callable[[Exception], bool]] = None,
    **kwargs
) -> T:
    """
    Выполнить функцию с retry и exponential backoff.
    
    Args:
        *args: Позиционные аргументы для func
        max_attempts: Максимальное количество попыток (по умолчанию 5)
        initial_delay: Начальная задержка в секундах (по умолчанию 1.0)
        max_delay: Максимальная задержка в секундах (по умолчанию 60.0)
        exponential_base: База для exponential backoff (по умолчанию 2.0)
        is_retryable: Функция для проверки, можно ли повторить ошибку
        **kwargs: Именованные аргументы для func
        
    Returns:
        Результат выполнения func
        
    Raises:
        Последнее исключение если все попытки исчерпаны
    """
    last_error = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_error = e
            
            # Проверить, можно ли повторить
            if is_retryable and not is_retryable(e):
                logger.debug(f"Error is not retryable: {e}")
                raise
            
            # Если это последняя попытка, не ждать
            if attempt >= max_attempts:
                logger.warning(
                    f"Max attempts ({max_attempts}) reached for {func.__name__}, "
                    f"last error: {e}"
                )
                raise
            
            # Вычислить задержку с exponential backoff
            delay = min(initial_delay * (exponential_base ** (attempt - 1)), max_delay)
            
            logger.info(
                f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                f"Retrying in {delay:.2f}s..."
            )
            
            await asyncio.sleep(delay)
    
    # Не должно быть достигнуто, но на всякий случай
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected error in retry_with_exponential_backoff")


async def retry_storage_operation(
    func: Callable[..., T],
    *args,
    **kwargs
) -> T:
    """
    Выполнить Storage операцию с retry и exponential backoff.
    
    Использует exponential backoff: 1s, 2s, 4s, 8s (после 5 попыток → error).
    
    Args:
        func: Storage операция для выполнения
        *args: Позиционные аргументы для func
        **kwargs: Именованные аргументы для func
        
    Returns:
        Результат выполнения func
        
    Raises:
        Последнее исключение если все попытки исчерпаны
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2242)
    """
    return await retry_with_exponential_backoff(
        func,
        *args,
        max_attempts=5,
        initial_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        is_retryable=is_transient_storage_error,
        **kwargs
    )


async def retry_triton_operation(
    func: Callable[..., T],
    timeout: float = 30.0,
    max_attempts: int = 3,
    *args,
    **kwargs
) -> T:
    """
    Выполнить Triton операцию с retry и exponential backoff.
    
    Использует timeout 30 сек и retry 3 раза с exponential backoff.
    
    Args:
        func: Triton операция для выполнения
        timeout: Timeout в секундах (по умолчанию 30.0)
        max_attempts: Максимальное количество попыток (по умолчанию 3)
        *args: Позиционные аргументы для func
        **kwargs: Именованные аргументы для func
        
    Returns:
        Результат выполнения func
        
    Raises:
        Последнее исключение если все попытки исчерпаны
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2241)
    """
    return await retry_with_exponential_backoff(
        func,
        *args,
        max_attempts=max_attempts,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        is_retryable=is_triton_timeout_error,
        **kwargs
    )


def with_storage_retry(func: Callable[..., T]) -> Callable[..., T]:
    """
    Декоратор для автоматического retry Storage операций.
    
    Используется для синхронных функций Storage.
    
    Args:
        func: Функция для обертки
        
    Returns:
        Обернутая функция с retry логикой
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await retry_storage_operation(func, *args, **kwargs)
    
    return wrapper

