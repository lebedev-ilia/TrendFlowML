"""
Unit тесты для обработки ошибок
"""

import pytest
from fastapi import HTTPException
from api.utils.errors import (
    RunNotFoundError,
    BackpressureError,
    RunAlreadyExistsError,
    InvalidPayloadError,
    ProcessingError
)


class TestCustomErrors:
    """Тесты для кастомных исключений."""
    
    def test_run_not_found_error(self):
        """Тест RunNotFoundError."""
        error = RunNotFoundError("Run not found: test-run-id")
        assert str(error) == "Run not found: test-run-id"
        assert error.run_id is None
        
        error_with_id = RunNotFoundError("Run not found", run_id="test-run-id")
        assert error_with_id.run_id == "test-run-id"
    
    def test_backpressure_error(self):
        """Тест BackpressureError."""
        error = BackpressureError("Too many active runs", retry_after=60)
        assert str(error) == "Too many active runs"
        assert error.retry_after == 60
    
    def test_run_already_exists_error(self):
        """Тест RunAlreadyExistsError."""
        error = RunAlreadyExistsError("test-run-id")
        assert error.run_id == "test-run-id"
        assert "test-run-id" in str(error)
    
    def test_invalid_payload_error(self):
        """Тест InvalidPayloadError."""
        error = InvalidPayloadError("Invalid video path")
        assert str(error) == "Invalid video path"
    
    def test_processing_error(self):
        """Тест ProcessingError."""
        error = ProcessingError("Processing failed", run_id="test-run-id")
        assert str(error) == "Processing failed"
        assert error.run_id == "test-run-id"

