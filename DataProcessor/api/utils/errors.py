"""
Кастомные исключения для DataProcessor API

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1387, 1631-1661)
"""


class DataProcessorAPIError(Exception):
    """Базовое исключение для DataProcessor API."""
    pass


class RunNotFoundError(DataProcessorAPIError):
    """
    Исключение когда run не найден.
    
    Используется в endpoints для возврата 404.
    """
    def __init__(self, message: str, run_id: str = None):
        super().__init__(message)
        self.run_id = run_id


class InvalidPayloadError(DataProcessorAPIError):
    """
    Исключение при невалидном payload.
    
    Используется в endpoints для возврата 400.
    """
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class ProcessingError(DataProcessorAPIError):
    """
    Исключение при ошибке обработки.
    
    Используется в endpoints для возврата 500.
    """
    def __init__(self, message: str, run_id: str = None, error_code: str = None):
        super().__init__(message)
        self.run_id = run_id
        self.error_code = error_code


class RunAlreadyExistsError(DataProcessorAPIError):
    """
    Исключение когда run с таким run_id уже существует.
    
    Используется в endpoints для возврата 409.
    """
    def __init__(self, message: str, run_id: str):
        super().__init__(message)
        self.run_id = run_id


class RateLimitError(DataProcessorAPIError):
    """
    Исключение при превышении лимита запросов.
    
    Используется в endpoints для возврата 429.
    """
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class BackpressureError(DataProcessorAPIError):
    """
    Исключение при перегрузке системы (backpressure).
    
    Используется в endpoints для возврата 503.
    """
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after

