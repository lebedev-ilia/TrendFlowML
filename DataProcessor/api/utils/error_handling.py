"""
Утилиты для обработки ошибок в фоновых задачах.

Этот модуль содержит вспомогательные функции для единообразной обработки ошибок
в фоновых задачах обработки видео.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2231-2416)
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from storage.base import StorageError, NotFoundError

from api.schemas.state import RunStatus
from api.services.redis_schema import release_run_lock

logger = logging.getLogger(__name__)


async def handle_background_error(
    run_id: str,
    error: Exception,
    error_message: str,
    task_manager,
    error_type: Optional[str] = None,
    update_metrics: bool = True
) -> None:
    """
    Обработать ошибку в фоновой задаче обработки.
    
    Выполняет:
    1. Логирование ошибки с контекстом
    2. Обновление статуса run'а на ERROR
    3. Обновление метрик ошибок (опционально)
    4. Освобождение Redis lock
    
    Args:
        run_id: UUID run'а
        error: Исключение, которое произошло
        error_message: Сообщение об ошибке для пользователя
        task_manager: TaskManager для обновления статуса
        error_type: Тип ошибки для метрик (если не указан, определяется автоматически)
        update_metrics: Обновлять ли метрики ошибок (по умолчанию True)
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2231-2245)
    """
    # Определить тип ошибки если не указан
    if not error_type:
        error_type = _determine_error_type(error)
    
    # Логирование ошибки с контекстом
    logger.exception(
        "Error in background processing",
        run_id=run_id,
        error=str(error),
        error_type=error_type,
        exception_type=type(error).__name__
    )
    
    # Обновить статус run'а на ERROR
    try:
        task_manager.update_run_status(
            run_id,
            RunStatus.ERROR,
            error=error_message,
            finished_at=asyncio.get_event_loop().time()
        )
    except Exception as e:
        logger.error(
            "Failed to update run status after error",
            run_id=run_id,
            error=str(e),
            original_error=str(error)
        )
    
    # Обновить метрики ошибок
    if update_metrics:
        _update_failure_metric(run_id, error_type)
    
    # Освободить Redis lock
    try:
        await release_run_lock(run_id)
    except Exception as e:
        logger.warning(
            "Failed to release run lock after error",
            run_id=run_id,
            error=str(e),
            original_error=str(error)
        )


def _determine_error_type(error: Exception) -> str:
    """
    Определить тип ошибки для метрик на основе исключения.
    
    Args:
        error: Исключение
        
    Returns:
        Строка с типом ошибки
    """
    if isinstance(error, (RedisError, ConnectionError, TimeoutError)):
        return "redis_error"
    elif isinstance(error, (StorageError, NotFoundError)):
        return "storage_error"
    elif isinstance(error, ValueError):
        return "validation_error"
    elif isinstance(error, FileNotFoundError):
        return "file_not_found"
    elif isinstance(error, PermissionError):
        return "permission_error"
    elif isinstance(error, TimeoutError):
        return "timeout_error"
    else:
        return "unknown_error"


def _update_failure_metric(run_id: str, error_type: str) -> None:
    """
    Обновить метрику ошибок обработки.
    
    Args:
        run_id: UUID run'а
        error_type: Тип ошибки
    """
    try:
        from api.services.metrics import failure_rate
        failure_rate.labels(
            processor="unknown",
            component="unknown",
            error_type=error_type
        ).inc()
    except (AttributeError, ImportError, ValueError) as e:
        logger.debug(
            "Failed to update failure_rate metric",
            run_id=run_id,
            error=str(e),
            error_type=type(e).__name__
        )
    except Exception as e:
        logger.debug(
            "Unexpected error updating failure_rate metric",
            run_id=run_id,
            error=str(e),
            error_type=type(e).__name__
        )


async def handle_processing_result(
    run_id: str,
    result: Dict[str, Any],
    task_manager
) -> None:
    """
    Обработать результат обработки (успешный или с ошибкой).
    
    Args:
        run_id: UUID run'а
        result: Результат обработки (словарь с полями success, error, error_type)
        task_manager: TaskManager для обновления статуса
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2231-2245)
    """
    if result.get("success"):
        # Успешное завершение
        task_manager.update_run_status(
            run_id,
            RunStatus.SUCCESS,
            finished_at=asyncio.get_event_loop().time()
        )
        logger.info(
            "Processing completed successfully",
            run_id=run_id
        )
    else:
        # Ошибка обработки
        error_msg = result.get("error", "Unknown error")
        error_type = result.get("error_type", "unknown")
        
        task_manager.update_run_status(
            run_id,
            RunStatus.ERROR,
            error=error_msg,
            finished_at=asyncio.get_event_loop().time()
        )
        
        logger.error(
            "Processing failed",
            run_id=run_id,
            error=error_msg,
            error_type=error_type
        )
        
        # Обновить метрику ошибок
        _update_failure_metric(run_id, error_type)

