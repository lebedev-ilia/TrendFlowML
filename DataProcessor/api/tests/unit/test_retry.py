"""
Unit тесты для Retry Utilities
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from api.utils.retry import (
    retry_with_exponential_backoff,
    retry_storage_operation,
    retry_triton_operation,
    is_transient_storage_error,
    is_triton_timeout_error,
    RetryableError,
    TransientStorageError,
    TritonTimeoutError
)


class TestIsTransientStorageError:
    """Тесты для функции is_transient_storage_error."""
    
    def test_is_transient_storage_error_storage_error(self):
        """StorageError является transient."""
        from storage.base import StorageError
        
        error = StorageError("Storage error")
        
        assert is_transient_storage_error(error) is True
    
    def test_is_transient_storage_error_not_found(self):
        """NotFoundError не является transient."""
        from storage.base import NotFoundError
        
        error = NotFoundError("Not found")
        
        assert is_transient_storage_error(error) is False
    
    def test_is_transient_storage_error_connection_error(self):
        """ConnectionError является transient."""
        error = ConnectionError("Connection failed")
        
        assert is_transient_storage_error(error) is True
    
    def test_is_transient_storage_error_timeout_error(self):
        """TimeoutError является transient."""
        error = TimeoutError("Timeout")
        
        assert is_transient_storage_error(error) is True
    
    def test_is_transient_storage_error_boto3_500(self):
        """Boto3 ClientError с кодом 500 является transient."""
        try:
            from botocore.exceptions import ClientError
            
            mock_response = MagicMock()
            mock_response.get = MagicMock(return_value={"Code": "500"})
            error = ClientError({"Error": {"Code": "500"}}, "operation")
            error.response = mock_response
            
            assert is_transient_storage_error(error) is True
        except ImportError:
            pytest.skip("botocore not available")
    
    def test_is_transient_storage_error_boto3_404(self):
        """Boto3 ClientError с кодом 404 не является transient."""
        try:
            from botocore.exceptions import ClientError
            
            mock_response = MagicMock()
            mock_response.get = MagicMock(return_value={"Code": "404"})
            error = ClientError({"Error": {"Code": "404"}}, "operation")
            error.response = mock_response
            
            assert is_transient_storage_error(error) is False
        except ImportError:
            pytest.skip("botocore not available")


class TestIsTritonTimeoutError:
    """Тесты для функции is_triton_timeout_error."""
    
    def test_is_triton_timeout_error_timeout_error(self):
        """TimeoutError является Triton timeout."""
        error = TimeoutError("Timeout")
        
        assert is_triton_timeout_error(error) is True
    
    def test_is_triton_timeout_error_triton_error(self):
        """TritonError с timeout является transient."""
        try:
            from dp_triton import TritonError
            
            error = TritonError("Triton timeout")
            error.error_code = "triton_timeout"
            
            assert is_triton_timeout_error(error) is True
        except ImportError:
            pytest.skip("dp_triton not available")
    
    def test_is_triton_timeout_error_not_timeout(self):
        """Обычное исключение не является Triton timeout."""
        error = ValueError("Not a timeout")
        
        assert is_triton_timeout_error(error) is False


class TestRetryWithExponentialBackoff:
    """Тесты для функции retry_with_exponential_backoff."""
    
    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Успешное выполнение с первой попытки."""
        async def success_func():
            return "success"
        
        result = await retry_with_exponential_backoff(success_func)
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_success_after_retries(self):
        """Успешное выполнение после нескольких попыток."""
        attempts = []
        
        async def retry_func():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("Transient error")
            return "success"
        
        result = await retry_with_exponential_backoff(
            retry_func,
            max_attempts=5,
            is_retryable=lambda e: isinstance(e, ConnectionError)
        )
        
        assert result == "success"
        assert len(attempts) == 3
    
    @pytest.mark.asyncio
    async def test_retry_max_attempts_exceeded(self):
        """Превышение максимального количества попыток."""
        async def failing_func():
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            await retry_with_exponential_backoff(
                failing_func,
                max_attempts=3,
                is_retryable=lambda e: isinstance(e, ConnectionError)
            )
    
    @pytest.mark.asyncio
    async def test_retry_not_retryable_error(self):
        """Ошибка которая не может быть повторена."""
        async def func():
            raise ValueError("Not retryable")
        
        with pytest.raises(ValueError):
            await retry_with_exponential_backoff(
                func,
                is_retryable=lambda e: isinstance(e, ConnectionError)
            )
    
    @pytest.mark.asyncio
    async def test_retry_exponential_backoff_delay(self):
        """Проверка exponential backoff задержки."""
        delays = []
        
        async def retry_func():
            if len(delays) == 0:
                delays.append(asyncio.get_event_loop().time())
                raise ConnectionError("First attempt")
            delays.append(asyncio.get_event_loop().time())
            return "success"
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await retry_with_exponential_backoff(
                retry_func,
                max_attempts=2,
                initial_delay=1.0,
                is_retryable=lambda e: isinstance(e, ConnectionError)
            )
            
            # Проверить что sleep был вызван с правильной задержкой
            mock_sleep.assert_called_once()
            call_args = mock_sleep.call_args[0]
            assert call_args[0] == 1.0  # initial_delay
    
    @pytest.mark.asyncio
    async def test_retry_sync_function(self):
        """Retry для синхронной функции."""
        def sync_func():
            return "sync success"
        
        result = await retry_with_exponential_backoff(sync_func)
        
        assert result == "sync success"
    
    @pytest.mark.asyncio
    async def test_retry_max_delay(self):
        """Проверка максимальной задержки."""
        async def retry_func():
            raise ConnectionError("Always fails")
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            try:
                await retry_with_exponential_backoff(
                    retry_func,
                    max_attempts=2,
                    initial_delay=1.0,
                    max_delay=5.0,
                    exponential_base=2.0,
                    is_retryable=lambda e: isinstance(e, ConnectionError)
                )
            except ConnectionError:
                pass
            
            # Проверить что задержка не превышает max_delay
            if mock_sleep.called:
                call_args = mock_sleep.call_args[0]
                assert call_args[0] <= 5.0


class TestRetryStorageOperation:
    """Тесты для функции retry_storage_operation."""
    
    @pytest.mark.asyncio
    async def test_retry_storage_operation_success(self):
        """Успешное выполнение Storage операции."""
        async def storage_func():
            return "storage success"
        
        result = await retry_storage_operation(storage_func)
        
        assert result == "storage success"
    
    @pytest.mark.asyncio
    async def test_retry_storage_operation_transient_error(self):
        """Retry при transient ошибке Storage."""
        attempts = []
        
        async def storage_func():
            attempts.append(1)
            if len(attempts) < 2:
                from storage.base import StorageError
                raise StorageError("Transient error")
            return "success"
        
        result = await retry_storage_operation(storage_func)
        
        assert result == "success"
        assert len(attempts) == 2


class TestRetryTritonOperation:
    """Тесты для функции retry_triton_operation."""
    
    @pytest.mark.asyncio
    async def test_retry_triton_operation_success(self):
        """Успешное выполнение Triton операции."""
        async def triton_func():
            return "triton success"
        
        result = await retry_triton_operation(triton_func)
        
        assert result == "triton success"
    
    @pytest.mark.asyncio
    async def test_retry_triton_operation_timeout(self):
        """Retry при Triton timeout."""
        attempts = []
        
        async def triton_func():
            attempts.append(1)
            if len(attempts) < 2:
                raise TimeoutError("Triton timeout")
            return "success"
        
        result = await retry_triton_operation(triton_func, max_attempts=3)
        
        assert result == "success"
        assert len(attempts) == 2

