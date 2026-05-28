"""
Unit тесты для Custom Exceptions
"""

import pytest

from api.utils.errors import (
    DataProcessorAPIError,
    RunNotFoundError,
    InvalidPayloadError,
    ProcessingError,
    RunAlreadyExistsError,
    RateLimitError,
    BackpressureError
)


class TestDataProcessorAPIError:
    """Тесты для базового исключения."""
    
    def test_data_processor_api_error(self):
        """Создание базового исключения."""
        error = DataProcessorAPIError("Test error")
        
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestRunNotFoundError:
    """Тесты для RunNotFoundError."""
    
    def test_run_not_found_error(self):
        """Создание RunNotFoundError."""
        error = RunNotFoundError("Run not found", run_id="test-run-id")
        
        assert str(error) == "Run not found"
        assert error.run_id == "test-run-id"
        assert isinstance(error, DataProcessorAPIError)
    
    def test_run_not_found_error_no_run_id(self):
        """RunNotFoundError без run_id."""
        error = RunNotFoundError("Run not found")
        
        assert error.run_id is None


class TestInvalidPayloadError:
    """Тесты для InvalidPayloadError."""
    
    def test_invalid_payload_error(self):
        """Создание InvalidPayloadError."""
        details = {"field": "video_path", "value": "/invalid/path"}
        error = InvalidPayloadError("Invalid payload", details=details)
        
        assert str(error) == "Invalid payload"
        assert error.details == details
        assert isinstance(error, DataProcessorAPIError)
    
    def test_invalid_payload_error_no_details(self):
        """InvalidPayloadError без details."""
        error = InvalidPayloadError("Invalid payload")
        
        assert error.details == {}


class TestProcessingError:
    """Тесты для ProcessingError."""
    
    def test_processing_error(self):
        """Создание ProcessingError."""
        error = ProcessingError(
            "Processing failed",
            run_id="test-run-id",
            error_code="PROCESSING_ERROR"
        )
        
        assert str(error) == "Processing failed"
        assert error.run_id == "test-run-id"
        assert error.error_code == "PROCESSING_ERROR"
        assert isinstance(error, DataProcessorAPIError)
    
    def test_processing_error_minimal(self):
        """ProcessingError с минимальными параметрами."""
        error = ProcessingError("Processing failed")
        
        assert error.run_id is None
        assert error.error_code is None


class TestRunAlreadyExistsError:
    """Тесты для RunAlreadyExistsError."""
    
    def test_run_already_exists_error(self):
        """Создание RunAlreadyExistsError."""
        error = RunAlreadyExistsError("Run already exists", run_id="test-run-id")
        
        assert str(error) == "Run already exists"
        assert error.run_id == "test-run-id"
        assert isinstance(error, DataProcessorAPIError)


class TestRateLimitError:
    """Тесты для RateLimitError."""
    
    def test_rate_limit_error(self):
        """Создание RateLimitError."""
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 60
        assert isinstance(error, DataProcessorAPIError)
    
    def test_rate_limit_error_no_retry_after(self):
        """RateLimitError без retry_after."""
        error = RateLimitError("Rate limit exceeded")
        
        assert error.retry_after is None


class TestBackpressureError:
    """Тесты для BackpressureError."""
    
    def test_backpressure_error(self):
        """Создание BackpressureError."""
        error = BackpressureError("Service overloaded", retry_after=60)
        
        assert str(error) == "Service overloaded"
        assert error.retry_after == 60
        assert isinstance(error, DataProcessorAPIError)
    
    def test_backpressure_error_no_retry_after(self):
        """BackpressureError без retry_after."""
        error = BackpressureError("Service overloaded")
        
        assert error.retry_after is None

