"""
Request ID Middleware - добавление уникального ID для каждого запроса

Генерирует UUID для каждого HTTP запроса и добавляет его в headers ответа.
Также сохраняет request_id в request.state для использования в логах и audit log.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2776-2785)
"""

import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.utils.logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware для добавления Request ID к каждому запросу.
    
    Генерирует UUID для каждого запроса и добавляет его:
    - В request.state.request_id для использования в обработчиках
    - В заголовок X-Request-ID ответа
    
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2776-2785)
    """
    
    async def dispatch(self, request: Request, call_next):
        """
        Обработать запрос и добавить Request ID.
        
        Args:
            request: HTTP запрос
            call_next: Следующий middleware или endpoint handler
            
        Returns:
            HTTP ответ с заголовком X-Request-ID
        """
        # Генерировать или использовать существующий Request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Сохранить в request.state для использования в обработчиках
        request.state.request_id = request_id
        
        # Вызвать следующий middleware/handler
        response = await call_next(request)
        
        # Добавить Request ID в заголовок ответа
        response.headers["X-Request-ID"] = request_id
        
        return response

