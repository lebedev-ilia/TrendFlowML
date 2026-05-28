"""
Утилиты для структурированного логирования

Этот модуль предоставляет функции для структурированного логирования
с контекстом (run_id, video_id, platform_id).

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2524-2544)
"""

import logging
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)


class StructuredLogger:
    """
    Обертка над стандартным logger для структурированного логирования.
    
    Позволяет добавлять контекстные поля (run_id, video_id, platform_id) к логам.
    """
    
    def __init__(self, base_logger: logging.Logger):
        """
        Инициализация структурированного логгера.
        
        Args:
            base_logger: Базовый logger из logging.getLogger()
        """
        self.logger = base_logger
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        run_id: Optional[str] = None,
        video_id: Optional[str] = None,
        platform_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Логирование с контекстными полями.
        
        Args:
            level: Уровень логирования (logging.INFO, logging.ERROR, и т.д.)
            message: Сообщение для логирования
            run_id: UUID run'а (опционально)
            video_id: ID видео (опционально)
            platform_id: ID платформы (опционально)
            **kwargs: Дополнительные поля для логирования
        """
        extra = {}
        
        if run_id:
            extra["run_id"] = run_id
        if video_id:
            extra["video_id"] = video_id
        if platform_id:
            extra["platform_id"] = platform_id
        
        # Добавить дополнительные поля
        extra.update(kwargs)

        # Зарезервированные ключи LogRecord, которые нельзя переопределять через extra
        reserved_keys = {"exc_info", "stack_info"}
        for key in list(extra.keys()):
            if key in reserved_keys:
                extra.pop(key, None)

        self.logger.log(level, message, extra=extra)
    
    def info(
        self,
        message: str,
        run_id: Optional[str] = None,
        video_id: Optional[str] = None,
        platform_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование на уровне INFO с контекстом."""
        self._log_with_context(
            logging.INFO,
            message,
            run_id=run_id,
            video_id=video_id,
            platform_id=platform_id,
            **kwargs
        )
    
    def warning(
        self,
        message: str,
        run_id: Optional[str] = None,
        video_id: Optional[str] = None,
        platform_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование на уровне WARNING с контекстом."""
        self._log_with_context(
            logging.WARNING,
            message,
            run_id=run_id,
            video_id=video_id,
            platform_id=platform_id,
            **kwargs
        )
    
    def error(
        self,
        message: str,
        run_id: Optional[str] = None,
        video_id: Optional[str] = None,
        platform_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование на уровне ERROR с контекстом."""
        self._log_with_context(
            logging.ERROR,
            message,
            run_id=run_id,
            video_id=video_id,
            platform_id=platform_id,
            **kwargs
        )
    
    def debug(
        self,
        message: str,
        run_id: Optional[str] = None,
        video_id: Optional[str] = None,
        platform_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование на уровне DEBUG с контекстом."""
        self._log_with_context(
            logging.DEBUG,
            message,
            run_id=run_id,
            video_id=video_id,
            platform_id=platform_id,
            **kwargs
        )
    
    def exception(
        self,
        message: str,
        run_id: Optional[str] = None,
        video_id: Optional[str] = None,
        platform_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Логирование исключения с контекстом."""
        self._log_with_context(
            logging.ERROR,
            message,
            run_id=run_id,
            video_id=video_id,
            platform_id=platform_id,
            exc_info=True,
            **kwargs
        )


def get_logger(name: str) -> StructuredLogger:
    """
    Получить структурированный logger для модуля.
    
    Args:
        name: Имя модуля (обычно __name__)
        
    Returns:
        StructuredLogger для модуля
    """
    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger)


def log_with_context(
    run_id: Optional[str] = None,
    video_id: Optional[str] = None,
    platform_id: Optional[str] = None
):
    """
    Декоратор для автоматического добавления контекста к логам в функции.
    
    Args:
        run_id: UUID run'а (опционально)
        video_id: ID видео (опционально)
        platform_id: ID платформы (опционально)
        
    Usage:
        @log_with_context(run_id="...", video_id="...")
        def my_function():
            logger.info("Message")  # Автоматически добавит run_id и video_id
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Добавить контекст в kwargs для использования в функции
            if run_id:
                kwargs.setdefault("_log_run_id", run_id)
            if video_id:
                kwargs.setdefault("_log_video_id", video_id)
            if platform_id:
                kwargs.setdefault("_log_platform_id", platform_id)
            return func(*args, **kwargs)
        return wrapper
    return decorator

