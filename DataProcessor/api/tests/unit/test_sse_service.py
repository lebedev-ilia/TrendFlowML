"""
Unit тесты для SSE Service
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from api.services.sse_service import (
    SSEConnectionManager,
    stream_run_events,
    _map_event_type_to_sse,
    _connection_manager
)
from api.config import config


class TestSSEConnectionManager:
    """Тесты для SSEConnectionManager."""
    
    @pytest.mark.asyncio
    async def test_acquire_connection_success(self):
        """Успешное получение соединения."""
        manager = SSEConnectionManager()
        
        result = await manager.acquire("test-run-id")
        
        assert result is True
        count = await manager.get_connection_count("test-run-id")
        assert count == 1
    
    @pytest.mark.asyncio
    async def test_acquire_connection_limit_exceeded(self):
        """Превышение лимита соединений."""
        manager = SSEConnectionManager()
        
        # Получить максимальное количество соединений
        max_connections = config.max_sse_connections_per_run
        
        for i in range(max_connections):
            result = await manager.acquire("test-run-id")
            assert result is True
        
        # Попытка получить еще одно соединение должна вернуть False
        result = await manager.acquire("test-run-id")
        assert result is False
        
        # Количество соединений не должно превысить лимит
        count = await manager.get_connection_count("test-run-id")
        assert count == max_connections
    
    @pytest.mark.asyncio
    async def test_release_connection(self):
        """Освобождение соединения."""
        manager = SSEConnectionManager()
        
        # Получить соединение
        await manager.acquire("test-run-id")
        assert await manager.get_connection_count("test-run-id") == 1
        
        # Освободить соединение
        await manager.release("test-run-id")
        assert await manager.get_connection_count("test-run-id") == 0
    
    @pytest.mark.asyncio
    async def test_release_multiple_connections(self):
        """Освобождение нескольких соединений."""
        manager = SSEConnectionManager()
        
        # Получить несколько соединений
        await manager.acquire("test-run-id")
        await manager.acquire("test-run-id")
        assert await manager.get_connection_count("test-run-id") == 2
        
        # Освободить одно соединение
        await manager.release("test-run-id")
        assert await manager.get_connection_count("test-run-id") == 1
        
        # Освободить еще одно
        await manager.release("test-run-id")
        assert await manager.get_connection_count("test-run-id") == 0
    
    @pytest.mark.asyncio
    async def test_release_nonexistent_connection(self):
        """Освобождение несуществующего соединения."""
        manager = SSEConnectionManager()
        
        # Не должно быть ошибки
        await manager.release("nonexistent-run-id")
        assert await manager.get_connection_count("nonexistent-run-id") == 0
    
    @pytest.mark.asyncio
    async def test_get_connection_count_empty(self):
        """Получение количества соединений когда их нет."""
        manager = SSEConnectionManager()
        
        count = await manager.get_connection_count("test-run-id")
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_multiple_runs_connections(self):
        """Управление соединениями для нескольких run'ов."""
        manager = SSEConnectionManager()
        
        # Получить соединения для разных run'ов
        await manager.acquire("run-1")
        await manager.acquire("run-1")
        await manager.acquire("run-2")
        
        assert await manager.get_connection_count("run-1") == 2
        assert await manager.get_connection_count("run-2") == 1
        
        # Освободить соединения
        await manager.release("run-1")
        assert await manager.get_connection_count("run-1") == 1
        assert await manager.get_connection_count("run-2") == 1


class TestStreamRunEvents:
    """Тесты для stream_run_events."""
    
    @pytest.mark.asyncio
    async def test_stream_events_success(self, mock_redis_client):
        """Успешный стриминг событий."""
        with patch("api.services.sse_service.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                run_id = "test-run-id"
                
                # Мокаем события из Redis Streams
                messages = [
                    (
                        b"stream:events:test-run-id",
                        [
                            (
                                b"1234567890-0",
                                {
                                    b"event_type": b"processing_started",
                                    b"data": b'{"run_id": "test-run-id", "message": "Started"}'
                                }
                            ),
                            (
                                b"1234567890-1",
                                {
                                    b"event_type": b"progress_update",
                                    b"data": b'{"progress": 0.5}'
                                }
                            )
                        ]
                    )
                ]
                
                # Первый вызов возвращает события, второй - пустой список (для завершения цикла)
                mock_redis_client.xread = AsyncMock(side_effect=[messages, []])
                
                # Собираем события из генератора
                events = []
                async for event_line in stream_run_events(run_id):
                    events.append(event_line)
                    # Ограничиваем количество итераций для теста
                    if len(events) > 20:
                        break
                
                # Проверяем что были получены события
                assert len(events) > 0
                # Проверяем формат SSE
                assert any("event: connected" in event for event in events)
                assert any("processing_started" in event or "progress" in event for event in events)
                
                # Проверяем что соединение было освобождено
                mock_manager.release.assert_called_once_with(run_id)
    
    @pytest.mark.asyncio
    async def test_stream_events_connection_limit(self):
        """Превышение лимита соединений."""
        with patch("api.services.sse_service._connection_manager") as mock_manager:
            mock_manager.acquire = AsyncMock(return_value=False)  # Лимит превышен
            
            run_id = "test-run-id"
            
            # Должно быть исключение ValueError
            with pytest.raises(ValueError, match="Max SSE connections"):
                async for _ in stream_run_events(run_id):
                    pass
    
    @pytest.mark.asyncio
    async def test_stream_events_redis_unavailable(self):
        """Стриминг когда Redis недоступен."""
        with patch("api.services.sse_service.get_redis_client", return_value=None):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                run_id = "test-run-id"
                
                # Должно быть исключение RuntimeError
                with pytest.raises(RuntimeError, match="Redis not available"):
                    async for _ in stream_run_events(run_id):
                        pass
                
                # Соединение должно быть освобождено
                mock_manager.release.assert_called_once_with(run_id)
    
    @pytest.mark.asyncio
    async def test_stream_events_filter_by_since(self, mock_redis_client):
        """Фильтрация событий по времени (since parameter)."""
        with patch("api.services.sse_service.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                run_id = "test-run-id"
                
                # Мокаем xrange для поиска начального ID
                mock_redis_client.xrange = AsyncMock(return_value=[
                    (b"1234567890-5", {})
                ])
                
                # Мокаем xread
                mock_redis_client.xread = AsyncMock(return_value=[])
                
                since = "2024-01-01T12:00:00Z"
                
                events = []
                async for event_line in stream_run_events(run_id, since=since):
                    events.append(event_line)
                    if len(events) > 10:
                        break
                
                # Проверяем что xrange был вызван для поиска начального ID
                mock_redis_client.xrange.assert_called()
    
    @pytest.mark.asyncio
    async def test_stream_events_filter_by_component(self, mock_redis_client):
        """Фильтрация событий по компоненту."""
        with patch("api.services.sse_service.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                run_id = "test-run-id"
                component = "visual"
                
                # Мокаем события с разными компонентами
                messages = [
                    (
                        b"stream:events:test-run-id",
                        [
                            (
                                b"1234567890-0",
                                {
                                    b"event_type": b"component_started",
                                    b"data": b'{"component": "visual", "message": "Started"}'
                                }
                            ),
                            (
                                b"1234567890-1",
                                {
                                    b"event_type": b"component_started",
                                    b"data": b'{"component": "audio", "message": "Started"}'
                                }
                            )
                        ]
                    )
                ]
                
                mock_redis_client.xread = AsyncMock(side_effect=[messages, []])
                
                events = []
                async for event_line in stream_run_events(run_id, component=component):
                    events.append(event_line)
                    if len(events) > 10:
                        break
                
                # Проверяем что были отфильтрованы события
                # Должны быть только события для компонента "visual"
                visual_events = [e for e in events if "visual" in e.lower()]
                assert len(visual_events) > 0
    
    @pytest.mark.asyncio
    async def test_stream_events_keepalive(self, mock_redis_client):
        """Keepalive сообщения при отсутствии событий."""
        with patch("api.services.sse_service.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                run_id = "test-run-id"
                
                # Мокаем пустые ответы от Redis (нет новых событий)
                mock_redis_client.xread = AsyncMock(return_value=[])
                
                # Создаем задачу для стриминга
                events = []
                task = asyncio.create_task(
                    self._collect_events(stream_run_events(run_id), events, max_events=5)
                )
                
                # Подождать немного
                await asyncio.sleep(0.1)
                
                # Отменить задачу
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                # Проверяем что были получены начальные сообщения
                assert len(events) > 0
    
    async def _collect_events(self, generator, events_list, max_events=10):
        """Вспомогательная функция для сбора событий."""
        count = 0
        async for event in generator:
            events_list.append(event)
            count += 1
            if count >= max_events:
                break
    
    @pytest.mark.asyncio
    async def test_stream_events_redis_error(self, mock_redis_client):
        """Обработка ошибки Redis при стриминге."""
        with patch("api.services.sse_service.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                from redis.exceptions import RedisError
                
                run_id = "test-run-id"
                
                # Мокаем ошибку Redis
                mock_redis_client.xread = AsyncMock(side_effect=RedisError("Connection lost"))
                
                events = []
                async for event_line in stream_run_events(run_id):
                    events.append(event_line)
                    # Ограничиваем для теста
                    if len(events) > 10:
                        break
                
                # Должно быть сообщение об ошибке
                error_events = [e for e in events if "error" in e.lower()]
                assert len(error_events) > 0
                
                # Соединение должно быть освобождено
                mock_manager.release.assert_called_once_with(run_id)
    
    @pytest.mark.asyncio
    async def test_stream_events_disconnect_message(self, mock_redis_client):
        """Отправка сообщения о завершении соединения."""
        with patch("api.services.sse_service.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.sse_service._connection_manager") as mock_manager:
                mock_manager.acquire = AsyncMock(return_value=True)
                mock_manager.release = AsyncMock()
                
                run_id = "test-run-id"
                
                # Мокаем пустые ответы для быстрого завершения
                mock_redis_client.xread = AsyncMock(return_value=[])
                
                events = []
                # Симулируем отмену через CancelledError
                try:
                    async for event_line in stream_run_events(run_id):
                        events.append(event_line)
                        # Прервать после первого события
                        break
                except Exception:
                    pass
                
                # Проверяем что было начальное сообщение
                assert len(events) > 0


class TestMapEventTypeToSSE:
    """Тесты для маппинга типов событий."""
    
    def test_map_processing_started(self):
        """Маппинг processing_started на stage."""
        result = _map_event_type_to_sse("processing_started")
        assert result == "stage"
    
    def test_map_processing_completed(self):
        """Маппинг processing_completed на complete."""
        result = _map_event_type_to_sse("processing_completed")
        assert result == "complete"
    
    def test_map_processing_failed(self):
        """Маппинг processing_failed на complete."""
        result = _map_event_type_to_sse("processing_failed")
        assert result == "complete"
    
    def test_map_progress_update(self):
        """Маппинг progress_update на progress."""
        result = _map_event_type_to_sse("progress_update")
        assert result == "progress"
    
    def test_map_component_started(self):
        """Маппинг component_started на component_start."""
        result = _map_event_type_to_sse("component_started")
        assert result == "component_start"
    
    def test_map_component_completed(self):
        """Маппинг component_completed на component_complete."""
        result = _map_event_type_to_sse("component_completed")
        assert result == "component_complete"
    
    def test_map_unknown_event_type(self):
        """Маппинг неизвестного типа события."""
        result = _map_event_type_to_sse("unknown_event_type")
        assert result == "unknown_event_type"
    
    def test_map_event_type_with_prefix(self):
        """Маппинг типа события с префиксом."""
        result = _map_event_type_to_sse("progress_custom")
        assert result == "progress"
        
        result = _map_event_type_to_sse("component_custom_start")
        assert result == "component_start"
        
        result = _map_event_type_to_sse("stage_custom")
        assert result == "stage"

