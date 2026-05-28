"""
Unit тесты для Request ID Middleware
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.requests import Request
from starlette.responses import Response

from api.middleware.request_id import RequestIDMiddleware


class TestRequestIDMiddleware:
    """Тесты для RequestIDMiddleware."""
    
    @pytest.mark.asyncio
    async def test_generate_request_id(self):
        """Генерация нового Request ID если отсутствует."""
        middleware = RequestIDMiddleware(app=None)
        
        # Мокаем request без X-Request-ID header
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.state = MagicMock()
        
        # Мокаем response
        mock_response = MagicMock(spec=Response)
        mock_response.headers = {}
        
        # Мокаем call_next
        async def mock_call_next(request):
            return mock_response
        
        call_next = AsyncMock(side_effect=mock_call_next)
        
        # Вызвать middleware
        response = await middleware.dispatch(mock_request, call_next)
        
        # Проверить что Request ID был сгенерирован
        assert hasattr(mock_request.state, "request_id")
        assert mock_request.state.request_id is not None
        assert isinstance(mock_request.state.request_id, str)
        
        # Проверить что Request ID добавлен в headers ответа
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == mock_request.state.request_id
    
    @pytest.mark.asyncio
    async def test_use_existing_request_id(self):
        """Использование существующего Request ID из заголовка."""
        middleware = RequestIDMiddleware(app=None)
        
        existing_request_id = str(uuid.uuid4())
        
        # Мокаем request с X-Request-ID header
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Request-ID": existing_request_id}
        mock_request.state = MagicMock()
        
        # Мокаем response
        mock_response = MagicMock(spec=Response)
        mock_response.headers = {}
        
        # Мокаем call_next
        async def mock_call_next(request):
            return mock_response
        
        call_next = AsyncMock(side_effect=mock_call_next)
        
        # Вызвать middleware
        response = await middleware.dispatch(mock_request, call_next)
        
        # Проверить что использован существующий Request ID
        assert mock_request.state.request_id == existing_request_id
        assert response.headers["X-Request-ID"] == existing_request_id
    
    @pytest.mark.asyncio
    async def test_request_id_uniqueness(self):
        """Проверка уникальности Request ID."""
        middleware = RequestIDMiddleware(app=None)
        
        request_ids = set()
        
        # Генерируем несколько Request ID
        for _ in range(100):
            mock_request = MagicMock(spec=Request)
            mock_request.headers = {}
            mock_request.state = MagicMock()
            
            mock_response = MagicMock(spec=Response)
            mock_response.headers = {}
            
            async def mock_call_next(request):
                return mock_response
            
            call_next = AsyncMock(side_effect=mock_call_next)
            
            await middleware.dispatch(mock_request, call_next)
            
            request_id = mock_request.state.request_id
            request_ids.add(request_id)
        
        # Все Request ID должны быть уникальными
        assert len(request_ids) == 100
    
    @pytest.mark.asyncio
    async def test_request_id_in_state(self):
        """Request ID сохраняется в request.state."""
        middleware = RequestIDMiddleware(app=None)
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.state = MagicMock()
        
        mock_response = MagicMock(spec=Response)
        mock_response.headers = {}
        
        async def mock_call_next(request):
            # Проверить что request_id доступен в обработчике
            assert hasattr(request.state, "request_id")
            assert request.state.request_id is not None
            return mock_response
        
        call_next = AsyncMock(side_effect=mock_call_next)
        
        await middleware.dispatch(mock_request, call_next)
        
        # Проверить что request_id был установлен
        assert hasattr(mock_request.state, "request_id")
    
    @pytest.mark.asyncio
    async def test_request_id_in_response_header(self):
        """Request ID добавляется в заголовок ответа."""
        middleware = RequestIDMiddleware(app=None)
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.state = MagicMock()
        
        mock_response = MagicMock(spec=Response)
        mock_response.headers = {}
        
        async def mock_call_next(request):
            return mock_response
        
        call_next = AsyncMock(side_effect=mock_call_next)
        
        response = await middleware.dispatch(mock_request, call_next)
        
        # Проверить что Request ID в заголовке ответа
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == mock_request.state.request_id
    
    @pytest.mark.asyncio
    async def test_request_id_uuid_format(self):
        """Проверка формата Request ID (UUID)."""
        middleware = RequestIDMiddleware(app=None)
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.state = MagicMock()
        
        mock_response = MagicMock(spec=Response)
        mock_response.headers = {}
        
        async def mock_call_next(request):
            return mock_response
        
        call_next = AsyncMock(side_effect=mock_call_next)
        
        await middleware.dispatch(mock_request, call_next)
        
        request_id = mock_request.state.request_id
        
        # Проверить что это валидный UUID
        try:
            uuid.UUID(request_id)
        except ValueError:
            pytest.fail(f"Request ID {request_id} is not a valid UUID")


class TestRequestIDMiddlewareIntegration:
    """Integration тесты для Request ID Middleware."""
    
    def test_request_id_in_endpoint_response(self):
        """Request ID присутствует в ответе endpoint."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        response = client.get("/api/v1/health")
        
        # Проверить что Request ID в заголовке ответа
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] is not None
    
    def test_request_id_accessible_in_endpoint(self):
        """Request ID доступен в endpoint через request.state."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        # Создаем тестовый endpoint для проверки
        @app.get("/test-request-id")
        async def test_endpoint(request: Request):
            return {"request_id": getattr(request.state, "request_id", None)}
        
        client = TestClient(app)
        
        response = client.get("/test-request-id")
        
        # Проверить что request_id доступен в endpoint
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert data["request_id"] is not None

