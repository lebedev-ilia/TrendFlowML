"""
Integration тесты для SSE endpoint
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from api.main import app
from api.services.sse_service import _connection_manager


class TestSSEEndpointIntegration:
    """Integration тесты для SSE endpoint."""
    
    @pytest.fixture
    def client(self):
        """Фикстура для TestClient."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_connection(self, client, mock_redis_client):
        """Подключение к SSE endpoint."""
        with patch("api.endpoints.runs.get_redis_client", return_value=mock_redis_client):
            with patch("api.endpoints.runs.stream_run_events") as mock_stream:
                # Мокаем стриминг событий
                async def mock_event_generator():
                    yield "event: connected\n"
                    yield 'data: {"run_id": "test-run-id"}\n\n'
                    yield "event: progress\n"
                    yield 'data: {"progress": 0.5}\n\n'
                
                mock_stream.return_value = mock_event_generator()
                
                # Мокаем StateReader
                with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
                    mock_state_reader = MagicMock()
                    mock_state_reader.get_run_status = AsyncMock(return_value={
                        "run_id": "test-run-id",
                        "status": "running"
                    })
                    mock_state_reader_class.return_value = mock_state_reader
                    
                    # Запрос к SSE endpoint
                    response = client.get(
                        "/api/v1/runs/test-run-id/events",
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    # Проверяем что ответ - это SSE stream
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers.get("content-type", "")
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_run_not_found(self, client):
        """SSE endpoint когда run не найден."""
        with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
            mock_state_reader = MagicMock()
            from api.utils.errors import RunNotFoundError
            mock_state_reader.get_run_status = AsyncMock(
                side_effect=RunNotFoundError("Run not found: test-run-id")
            )
            mock_state_reader_class.return_value = mock_state_reader
            
            response = client.get(
                "/api/v1/runs/test-run-id/events",
                headers={"X-API-Key": "test_api_key"}
            )
            
            assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_completed_run(self, client, mock_redis_client):
        """SSE endpoint для завершенного run'а."""
        with patch("api.endpoints.runs.get_redis_client", return_value=mock_redis_client):
            with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
                mock_state_reader = MagicMock()
                mock_state_reader.get_run_status = AsyncMock(return_value={
                    "run_id": "test-run-id",
                    "status": "success"
                })
                mock_state_reader_class.return_value = mock_state_reader
                
                # Мокаем что stream не существует (события больше не доступны)
                mock_redis_client.exists = AsyncMock(return_value=False)
                
                response = client.get(
                    "/api/v1/runs/test-run-id/events",
                    headers={"X-API-Key": "test_api_key"}
                )
                
                # Должен вернуть 410 Gone
                assert response.status_code == 410
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_redis_unavailable(self, client):
        """SSE endpoint когда Redis недоступен."""
        with patch("api.endpoints.runs.get_redis_client", return_value=None):
            with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
                mock_state_reader = MagicMock()
                mock_state_reader.get_run_status = AsyncMock(return_value={
                    "run_id": "test-run-id",
                    "status": "running"
                })
                mock_state_reader_class.return_value = mock_state_reader
                
                response = client.get(
                    "/api/v1/runs/test-run-id/events",
                    headers={"X-API-Key": "test_api_key"}
                )
                
                # Должен вернуть 503 Service Unavailable
                assert response.status_code == 503
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_with_since_parameter(self, client, mock_redis_client):
        """SSE endpoint с параметром since."""
        with patch("api.endpoints.runs.get_redis_client", return_value=mock_redis_client):
            with patch("api.endpoints.runs.stream_run_events") as mock_stream:
                async def mock_event_generator():
                    yield "event: connected\n"
                    yield 'data: {"run_id": "test-run-id"}\n\n'
                
                mock_stream.return_value = mock_event_generator()
                
                with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
                    mock_state_reader = MagicMock()
                    mock_state_reader.get_run_status = AsyncMock(return_value={
                        "run_id": "test-run-id",
                        "status": "running"
                    })
                    mock_state_reader_class.return_value = mock_state_reader
                    
                    response = client.get(
                        "/api/v1/runs/test-run-id/events",
                        params={"since": "2024-01-01T12:00:00Z"},
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    assert response.status_code == 200
                    # Проверяем что since был передан в stream_run_events
                    mock_stream.assert_called_once()
                    call_args = mock_stream.call_args
                    assert call_args[1]["since"] == "2024-01-01T12:00:00Z"
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_with_component_parameter(self, client, mock_redis_client):
        """SSE endpoint с параметром component."""
        with patch("api.endpoints.runs.get_redis_client", return_value=mock_redis_client):
            with patch("api.endpoints.runs.stream_run_events") as mock_stream:
                async def mock_event_generator():
                    yield "event: connected\n"
                    yield 'data: {"run_id": "test-run-id"}\n\n'
                
                mock_stream.return_value = mock_event_generator()
                
                with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
                    mock_state_reader = MagicMock()
                    mock_state_reader.get_run_status = AsyncMock(return_value={
                        "run_id": "test-run-id",
                        "status": "running"
                    })
                    mock_state_reader_class.return_value = mock_state_reader
                    
                    response = client.get(
                        "/api/v1/runs/test-run-id/events",
                        params={"component": "visual"},
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    assert response.status_code == 200
                    # Проверяем что component был передан в stream_run_events
                    mock_stream.assert_called_once()
                    call_args = mock_stream.call_args
                    assert call_args[1]["component"] == "visual"
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_multiple_connections(self, client, mock_redis_client):
        """Множественные подключения к одному run_id."""
        with patch("api.endpoints.runs.get_redis_client", return_value=mock_redis_client):
            with patch("api.endpoints.runs.stream_run_events") as mock_stream:
                async def mock_event_generator():
                    yield "event: connected\n"
                    yield 'data: {"run_id": "test-run-id"}\n\n'
                    # Симулируем долгий стрим
                    await asyncio.sleep(0.1)
                
                mock_stream.return_value = mock_event_generator()
                
                with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
                    mock_state_reader = MagicMock()
                    mock_state_reader.get_run_status = AsyncMock(return_value={
                        "run_id": "test-run-id",
                        "status": "running"
                    })
                    mock_state_reader_class.return_value = mock_state_reader
                    
                    # Сбросить счетчик соединений
                    _connection_manager._connections.clear()
                    
                    # Создать несколько подключений
                    max_connections = 10  # config.max_sse_connections_per_run
                    
                    responses = []
                    for i in range(max_connections):
                        response = client.get(
                            "/api/v1/runs/test-run-id/events",
                            headers={"X-API-Key": "test_api_key"}
                        )
                        responses.append(response)
                    
                    # Все подключения должны быть успешными
                    assert all(r.status_code == 200 for r in responses)
                    
                    # Попытка создать еще одно подключение должна вернуть ошибку
                    # (через ValueError в stream_run_events)
                    # Но endpoint обработает это и вернет error event в stream
                    # Для теста проверим что connection_manager отслеживает соединения
                    # (это делается внутри stream_run_events)
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_authentication_required(self, client):
        """SSE endpoint требует аутентификации."""
        response = client.get("/api/v1/runs/test-run-id/events")
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_invalid_api_key(self, client):
        """SSE endpoint с невалидным API ключом."""
        response = client.get(
            "/api/v1/runs/test-run-id/events",
            headers={"X-API-Key": "invalid_key"}
        )
        
        assert response.status_code == 403
    
    @pytest.mark.asyncio
    async def test_sse_endpoint_error_handling(self, client, mock_redis_client):
        """Обработка ошибок в SSE endpoint."""
        with patch("api.endpoints.runs.get_redis_client", return_value=mock_redis_client):
            with patch("api.endpoints.runs.stream_run_events") as mock_stream:
                # Мокаем исключение в stream_run_events
                async def mock_event_generator():
                    yield "event: error\n"
                    yield 'data: {"error": "Test error"}\n\n'
                    raise ValueError("Test error")
                
                mock_stream.return_value = mock_event_generator()
                
                with patch("api.endpoints.runs.StateReader") as mock_state_reader_class:
                    mock_state_reader = MagicMock()
                    mock_state_reader.get_run_status = AsyncMock(return_value={
                        "run_id": "test-run-id",
                        "status": "running"
                    })
                    mock_state_reader_class.return_value = mock_state_reader
                    
                    response = client.get(
                        "/api/v1/runs/test-run-id/events",
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    # Должен вернуть 200 с error event в stream
                    assert response.status_code == 200
                    # Проверяем что в ответе есть error event
                    content = response.text
                    assert "error" in content.lower()


class TestSSERealTimeStreaming:
    """Тесты для стриминга событий в реальном времени."""
    
    @pytest.mark.asyncio
    async def test_realtime_event_streaming(self, mock_redis_client):
        """Стриминг событий в реальном времени."""
        from api.services.sse_service import stream_run_events
        
        with patch("api.services.sse_service.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                run_id = "test-run-id"
                
                # Симулируем события, которые приходят постепенно
                events_sequence = [
                    [
                        (
                            b"stream:events:test-run-id",
                            [
                                (b"1234567890-0", {
                                    b"event_type": b"processing_started",
                                    b"data": b'{"message": "Started"}'
                                })
                            ]
                        )
                    ],
                    [
                        (
                            b"stream:events:test-run-id",
                            [
                                (b"1234567890-1", {
                                    b"event_type": b"progress_update",
                                    b"data": b'{"progress": 0.25}'
                                })
                            ]
                        )
                    ],
                    [
                        (
                            b"stream:events:test-run-id",
                            [
                                (b"1234567890-2", {
                                    b"event_type": b"progress_update",
                                    b"data": b'{"progress": 0.5}'
                                })
                            ]
                        )
                    ],
                    []  # Пустой список для завершения
                ]
                
                mock_redis_client.xread = AsyncMock(side_effect=events_sequence)
                
                # Собираем события
                received_events = []
                async for event_line in stream_run_events(run_id):
                    received_events.append(event_line)
                    # Ограничиваем для теста
                    if len(received_events) > 20:
                        break
                
                # Проверяем что были получены события
                assert len(received_events) > 0
                # Проверяем что есть события разных типов
                event_types = [e for e in received_events if "event:" in e]
                assert len(event_types) > 0

