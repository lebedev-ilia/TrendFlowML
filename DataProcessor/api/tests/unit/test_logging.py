"""
Unit тесты для Logging Utilities
"""

import pytest
import logging
from unittest.mock import patch, MagicMock

from api.utils.logging import (
    StructuredLogger,
    get_logger,
    log_with_context
)


class TestStructuredLogger:
    """Тесты для класса StructuredLogger."""
    
    def test_structured_logger_init(self):
        """Инициализация StructuredLogger."""
        base_logger = logging.getLogger("test")
        structured_logger = StructuredLogger(base_logger)
        
        assert structured_logger.logger == base_logger
    
    def test_structured_logger_info(self):
        """Логирование на уровне INFO."""
        base_logger = logging.getLogger("test")
        structured_logger = StructuredLogger(base_logger)
        
        with patch.object(base_logger, "log") as mock_log:
            structured_logger.info(
                "Test message",
                run_id="test-run-id",
                video_id="test_video"
            )
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.INFO
            assert call_args[0][1] == "Test message"
            assert call_args[1]["extra"]["run_id"] == "test-run-id"
            assert call_args[1]["extra"]["video_id"] == "test_video"
    
    def test_structured_logger_warning(self):
        """Логирование на уровне WARNING."""
        base_logger = logging.getLogger("test")
        structured_logger = StructuredLogger(base_logger)
        
        with patch.object(base_logger, "log") as mock_log:
            structured_logger.warning("Warning message", platform_id="youtube")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.WARNING
            assert call_args[1]["extra"]["platform_id"] == "youtube"
    
    def test_structured_logger_error(self):
        """Логирование на уровне ERROR."""
        base_logger = logging.getLogger("test")
        structured_logger = StructuredLogger(base_logger)
        
        with patch.object(base_logger, "log") as mock_log:
            structured_logger.error("Error message", run_id="test-run-id")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.ERROR
    
    def test_structured_logger_debug(self):
        """Логирование на уровне DEBUG."""
        base_logger = logging.getLogger("test")
        structured_logger = StructuredLogger(base_logger)
        
        with patch.object(base_logger, "log") as mock_log:
            structured_logger.debug("Debug message")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == logging.DEBUG
    
    def test_structured_logger_exception(self):
        """Логирование исключения."""
        base_logger = logging.getLogger("test")
        structured_logger = StructuredLogger(base_logger)
        
        with patch.object(base_logger, "log") as mock_log:
            try:
                raise ValueError("Test error")
            except ValueError:
                structured_logger.exception("Exception occurred", run_id="test-run-id")
            
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[1]["exc_info"] is True
    
    def test_structured_logger_additional_fields(self):
        """Логирование с дополнительными полями."""
        base_logger = logging.getLogger("test")
        structured_logger = StructuredLogger(base_logger)
        
        with patch.object(base_logger, "log") as mock_log:
            structured_logger.info(
                "Message",
                run_id="test-run-id",
                custom_field="custom_value"
            )
            
            call_args = mock_log.call_args
            assert call_args[1]["extra"]["custom_field"] == "custom_value"


class TestGetLogger:
    """Тесты для функции get_logger."""
    
    def test_get_logger(self):
        """Получение структурированного logger."""
        logger = get_logger("test_module")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test_module"
    
    def test_get_logger_different_modules(self):
        """Получение logger для разных модулей."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        
        assert logger1.logger.name == "module1"
        assert logger2.logger.name == "module2"


class TestLogWithContext:
    """Тесты для декоратора log_with_context."""
    
    def test_log_with_context_decorator(self):
        """Декоратор добавляет контекст в kwargs."""
        @log_with_context(run_id="test-run-id", video_id="test_video")
        def test_function(**kwargs):
            return kwargs
        
        result = test_function()
        
        assert result.get("_log_run_id") == "test-run-id"
        assert result.get("_log_video_id") == "test_video"
    
    def test_log_with_context_with_args(self):
        """Декоратор работает с аргументами функции."""
        @log_with_context(platform_id="youtube")
        def test_function(arg1, **kwargs):
            return arg1, kwargs
        
        result = test_function("test_arg")
        
        assert result[0] == "test_arg"
        assert result[1].get("_log_platform_id") == "youtube"
    
    def test_log_with_context_preserves_existing_kwargs(self):
        """Декоратор сохраняет существующие kwargs."""
        @log_with_context(run_id="test-run-id")
        def test_function(**kwargs):
            return kwargs
        
        result = test_function(existing_key="existing_value")
        
        assert result.get("existing_key") == "existing_value"
        assert result.get("_log_run_id") == "test-run-id"

