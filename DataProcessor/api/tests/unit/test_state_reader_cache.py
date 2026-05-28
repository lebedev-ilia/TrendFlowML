"""
Unit тесты для кэширования в StateReader
"""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock

from api.services.state_reader import StateReader
from api.utils.errors import RunNotFoundError


class TestStateReaderHotPath:
    """Тесты для hot path (чтение из Redis cache)."""
    
    @pytest.mark.asyncio
    async def test_get_run_status_from_redis_cache(self, mock_storage, mock_key_layout, mock_redis_client):
        """Получение статуса из Redis cache (hot path)."""
        with patch("api.services.state_reader.get_run_state_from_redis", new_callable=AsyncMock) as mock_get_state:
            redis_state = {
                "status": "running",
                "progress": 0.5,
                "updated_at": "2024-01-01T00:00:00"
            }
            mock_get_state.return_value = redis_state
            
            # Мокаем TaskManager для получения метаданных
            mock_task_manager = MagicMock()
            mock_task_manager.get_run = MagicMock(return_value={
                "platform_id": "youtube",
                "video_id": "test_video"
            })
            
            reader = StateReader(
                storage=mock_storage,
                key_layout=mock_key_layout,
                task_manager=mock_task_manager,
                redis_client=mock_redis_client
            )
            
            # Мокаем загрузку из Storage (не должна вызываться при наличии cache)
            with patch.object(reader, '_load_run_state', return_value=None):
                with patch.object(reader, '_load_processor_state', return_value=None):
                    status = await reader.get_run_status("test-run-id", include_components=False)
                    
                    # Должен использовать данные из Redis
                    assert status["run_id"] == "test-run-id"
                    mock_get_state.assert_called_once_with("test-run-id")
    
    @pytest.mark.asyncio
    async def test_get_run_status_cache_miss_fallback_to_storage(self, mock_storage, mock_key_layout, mock_redis_client):
        """Fallback на Storage когда cache не найден (cold path)."""
        with patch("api.services.state_reader.get_run_state_from_redis", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = None  # Cache miss
            
            mock_task_manager = MagicMock()
            mock_task_manager.get_run = MagicMock(return_value={
                "platform_id": "youtube",
                "video_id": "test_video"
            })
            
            reader = StateReader(
                storage=mock_storage,
                key_layout=mock_key_layout,
                task_manager=mock_task_manager,
                redis_client=mock_redis_client
            )
            
            # Мокаем загрузку из Storage
            run_state = {
                "run": {
                    "status": "running",
                    "started_at": "2024-01-01T00:00:00"
                },
                "updated_at": "2024-01-01T00:00:00"
            }
            
            with patch.object(reader, '_load_run_state', return_value=run_state):
                with patch.object(reader, '_load_processor_state', return_value=None):
                    with patch("api.services.state_reader.save_run_state_to_redis", new_callable=AsyncMock) as mock_save_state:
                        status = await reader.get_run_status("test-run-id", include_components=False)
                        
                        # Должен загрузить из Storage
                        assert status["status"] == "running"
                        # Должен обновить cache после чтения из Storage
                        mock_save_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_run_status_cache_update_after_storage_read(self, mock_storage, mock_key_layout, mock_redis_client):
        """Обновление cache после чтения из Storage."""
        with patch("api.services.state_reader.get_run_state_from_redis", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = None  # Cache miss
            
            mock_task_manager = MagicMock()
            mock_task_manager.get_run = MagicMock(return_value={
                "platform_id": "youtube",
                "video_id": "test_video"
            })
            
            reader = StateReader(
                storage=mock_storage,
                key_layout=mock_key_layout,
                task_manager=mock_task_manager,
                redis_client=mock_redis_client
            )
            
            run_state = {
                "run": {
                    "status": "success",
                    "started_at": "2024-01-01T00:00:00",
                    "finished_at": "2024-01-01T01:00:00"
                },
                "updated_at": "2024-01-01T01:00:00"
            }
            
            with patch.object(reader, '_load_run_state', return_value=run_state):
                with patch.object(reader, '_load_processor_state', return_value=None):
                    with patch("api.services.state_reader.save_run_state_to_redis", new_callable=AsyncMock) as mock_save_state:
                        await reader.get_run_status("test-run-id", include_components=False)
                        
                        # Проверяем что cache был обновлен
                        mock_save_state.assert_called_once()
                        call_args = mock_save_state.call_args
                        assert call_args[0][0] == "test-run-id"
                        saved_state = call_args[0][1]
                        assert saved_state["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_get_run_status_redis_unavailable_fallback(self, mock_storage, mock_key_layout):
        """Fallback на Storage когда Redis недоступен."""
        mock_task_manager = MagicMock()
        mock_task_manager.get_run = MagicMock(return_value={
            "platform_id": "youtube",
            "video_id": "test_video"
        })
        
        reader = StateReader(
            storage=mock_storage,
            key_layout=mock_key_layout,
            task_manager=mock_task_manager,
            redis_client=None  # Redis недоступен
        )
        
        run_state = {
            "run": {
                "status": "running"
            },
            "updated_at": "2024-01-01T00:00:00"
        }
        
        with patch.object(reader, '_load_run_state', return_value=run_state):
            with patch.object(reader, '_load_processor_state', return_value=None):
                status = await reader.get_run_status("test-run-id", include_components=False)
                
                # Должен работать без Redis
                assert status["status"] == "running"


class TestStateReaderEventsCache:
    """Тесты для кэширования событий."""
    
    @pytest.mark.asyncio
    async def test_get_events_from_redis_streams(self, mock_storage, mock_key_layout, mock_redis_client):
        """Получение событий из Redis Streams (hot path)."""
        mock_task_manager = MagicMock()
        
        reader = StateReader(
            storage=mock_storage,
            key_layout=mock_key_layout,
            task_manager=mock_task_manager,
            redis_client=mock_redis_client
        )
        
        redis_events = [
            {
                "id": "1234567890-0",
                "event_type": "processing_started",
                "timestamp": 1234567890.0,
                "data": {"message": "Started"}
            },
            {
                "id": "1234567890-1",
                "event_type": "progress",
                "timestamp": 1234567891.0,
                "data": {"progress": 0.5}
            }
        ]
        
        with patch("api.services.state_reader.get_run_events_from_redis", new_callable=AsyncMock) as mock_get_events:
            mock_get_events.return_value = redis_events
            
            events = await reader.get_events("test-run-id", limit=10)
            
            assert len(events) == 2
            assert events[0]["event_type"] == "processing_started"
            assert events[1]["event_type"] == "progress"
            mock_get_events.assert_called_once_with("test-run-id", count=10)
    
    @pytest.mark.asyncio
    async def test_get_events_fallback_to_storage(self, mock_storage, mock_key_layout, mock_redis_client):
        """Fallback на Storage когда Redis Streams пуст (cold path)."""
        mock_task_manager = MagicMock()
        mock_task_manager.get_run = MagicMock(return_value={
            "platform_id": "youtube",
            "video_id": "test_video"
        })
        
        reader = StateReader(
            storage=mock_storage,
            key_layout=mock_key_layout,
            task_manager=mock_task_manager,
            redis_client=mock_redis_client
        )
        
        # Redis Streams пуст
        with patch("api.services.state_reader.get_run_events_from_redis", new_callable=AsyncMock) as mock_get_events:
            mock_get_events.return_value = []
            
            # Мокаем чтение из Storage (state_events.jsonl)
            storage_events_data = b"""{"event_type": "processing_started", "timestamp": "2024-01-01T00:00:00", "data": {"message": "Started"}}\n{"event_type": "progress", "timestamp": "2024-01-01T00:01:00", "data": {"progress": 0.5}}\n"""
            
            mock_storage.exists = AsyncMock(return_value=True)
            mock_storage.read_bytes = AsyncMock(return_value=storage_events_data)
            
            # Мокаем путь к событиям
            mock_key_layout.state_run_prefix = MagicMock(return_value="state/youtube/test_video/test-run-id")
            
            events = await reader.get_events("test-run-id", limit=10)
            
            # Должен прочитать из Storage
            assert len(events) == 2
            assert events[0]["event_type"] == "processing_started"
            assert events[1]["event_type"] == "progress"
    
    @pytest.mark.asyncio
    async def test_get_events_filter_by_since(self, mock_storage, mock_key_layout, mock_redis_client):
        """Фильтрация событий по времени (since parameter)."""
        mock_task_manager = MagicMock()
        
        reader = StateReader(
            storage=mock_storage,
            key_layout=mock_key_layout,
            task_manager=mock_task_manager,
            redis_client=mock_redis_client
        )
        
        redis_events = [
            {
                "id": "1234567890-0",
                "event_type": "processing_started",
                "timestamp": 1234567890.0,
                "data": {}
            },
            {
                "id": "1234567890-1",
                "event_type": "progress",
                "timestamp": 1234567891.0,
                "data": {}
            }
        ]
        
        with patch("api.services.state_reader.get_run_events_from_redis", new_callable=AsyncMock) as mock_get_events:
            mock_get_events.return_value = redis_events
            
            # Фильтруем события после первого
            from datetime import datetime
            since = datetime.fromtimestamp(1234567890.5).isoformat()
            
            events = await reader.get_events("test-run-id", since=since, limit=10)
            
            # Должен вернуть только события после since
            assert len(events) == 1
            assert events[0]["event_type"] == "progress"
    
    @pytest.mark.asyncio
    async def test_get_events_limit(self, mock_storage, mock_key_layout, mock_redis_client):
        """Ограничение количества событий (limit parameter)."""
        mock_task_manager = MagicMock()
        
        reader = StateReader(
            storage=mock_storage,
            key_layout=mock_key_layout,
            task_manager=mock_task_manager,
            redis_client=mock_redis_client
        )
        
        # Создаем больше событий чем limit
        redis_events = [
            {
                "id": f"1234567890-{i}",
                "event_type": "progress",
                "timestamp": 1234567890.0 + i,
                "data": {}
            }
            for i in range(20)
        ]
        
        with patch("api.services.state_reader.get_run_events_from_redis", new_callable=AsyncMock) as mock_get_events:
            mock_get_events.return_value = redis_events
            
            events = await reader.get_events("test-run-id", limit=10)
            
            # Должен вернуть только 10 событий
            assert len(events) == 10


class TestStateReaderCacheTTL:
    """Тесты для TTL кэша."""
    
    @pytest.mark.asyncio
    async def test_cache_ttl_configuration(self, mock_storage, mock_key_layout, mock_redis_client):
        """Проверка конфигурации TTL для кэша."""
        reader = StateReader(
            storage=mock_storage,
            key_layout=mock_key_layout,
            redis_client=mock_redis_client
        )
        
        # Проверяем что TTL установлен (по умолчанию 300 секунд)
        assert reader.cache_ttl == 300
        
        # Можно изменить TTL
        reader.cache_ttl = 600
        assert reader.cache_ttl == 600
    
    @pytest.mark.asyncio
    async def test_save_state_to_redis_with_ttl(self, mock_storage, mock_key_layout, mock_redis_client):
        """Сохранение состояния в Redis с правильным TTL."""
        with patch("api.services.state_reader.get_run_state_from_redis", new_callable=AsyncMock, return_value=None):
            mock_task_manager = MagicMock()
            mock_task_manager.get_run = MagicMock(return_value={
                "platform_id": "youtube",
                "video_id": "test_video"
            })
            
            reader = StateReader(
                storage=mock_storage,
                key_layout=mock_key_layout,
                task_manager=mock_task_manager,
                redis_client=mock_redis_client
            )
            
            run_state = {
                "run": {"status": "running"},
                "updated_at": "2024-01-01T00:00:00"
            }
            
            with patch.object(reader, '_load_run_state', return_value=run_state):
                with patch.object(reader, '_load_processor_state', return_value=None):
                    with patch("api.services.state_reader.save_run_state_to_redis", new_callable=AsyncMock) as mock_save_state:
                        await reader.get_run_status("test-run-id", include_components=False)
                        
                        # Проверяем что save_run_state_to_redis был вызван
                        # TTL должен быть установлен в redis_schema (TTL_STATE = 1 день)
                        mock_save_state.assert_called_once()


class TestStateReaderCacheInvalidation:
    """Тесты для инвалидации кэша."""
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_on_status_change(self, mock_storage, mock_key_layout, mock_redis_client):
        """Инвалидация кэша при изменении статуса."""
        # Старое состояние в cache
        old_state = {
            "status": "running",
            "progress": 0.5,
            "updated_at": "2024-01-01T00:00:00"
        }
        
        with patch("api.services.state_reader.get_run_state_from_redis", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = old_state
            
            mock_task_manager = MagicMock()
            mock_task_manager.get_run = MagicMock(return_value={
                "platform_id": "youtube",
                "video_id": "test_video"
            })
            
            reader = StateReader(
                storage=mock_storage,
                key_layout=mock_key_layout,
                task_manager=mock_task_manager,
                redis_client=mock_redis_client
            )
            
            # Новое состояние в Storage (статус изменился)
            new_state = {
                "run": {
                    "status": "success",
                    "finished_at": "2024-01-01T01:00:00"
                },
                "updated_at": "2024-01-01T01:00:00"
            }
            
            with patch.object(reader, '_load_run_state', return_value=new_state):
                with patch.object(reader, '_load_processor_state', return_value=None):
                    with patch("api.services.state_reader.save_run_state_to_redis", new_callable=AsyncMock) as mock_save_state:
                        status = await reader.get_run_status("test-run-id", include_components=True)
                        
                        # Должен использовать новое состояние из Storage
                        assert status["status"] == "success"
                        # Должен обновить cache новым состоянием
                        mock_save_state.assert_called_once()
                        call_args = mock_save_state.call_args
                        saved_state = call_args[0][1]
                        assert saved_state["status"] == "success"


class TestStateReaderAggregatedRunState:
    @pytest.mark.asyncio
    async def test_aggregates_status_from_run_state_processors_and_profile(
        self,
        mock_storage,
        mock_key_layout,
        mock_redis_client,
    ):
        mock_task_manager = MagicMock()
        mock_task_manager.get_run = MagicMock(return_value={
            "platform_id": "youtube",
            "video_id": "test_video",
        })

        reader = StateReader(
            storage=mock_storage,
            key_layout=mock_key_layout,
            task_manager=mock_task_manager,
            redis_client=mock_redis_client,
        )

        run_state = {
            "run": {
                "platform_id": "youtube",
                "video_id": "test_video",
                "run_id": "test-run-id",
            },
            "processors": {
                "segmenter": {
                    "name": "segmenter",
                    "status": "success",
                    "started_at": "2024-01-01T00:00:00Z",
                    "finished_at": "2024-01-01T00:01:00Z",
                    "duration_ms": 60000,
                },
                "audio": {"name": "audio", "status": "waiting"},
                "text": {"name": "text", "status": "waiting"},
                "visual": {"name": "visual", "status": "waiting"},
            },
            "updated_at": "2024-01-01T00:01:00Z",
        }
        redis_meta = {
            "platform_id": "youtube",
            "video_id": "test_video",
            "profile_config": {
                "processors": {
                    "segmenter": {"enabled": True, "required": True},
                    "audio": {"enabled": False, "required": False},
                    "text": {"enabled": False, "required": False},
                    "visual": {"enabled": False, "required": False},
                }
            },
        }

        with patch("api.services.state_reader.get_run_state_from_redis", new_callable=AsyncMock, return_value=None):
            with patch("api.services.state_reader.get_run_metadata_from_redis", new_callable=AsyncMock, return_value=redis_meta):
                with patch.object(reader, "_load_run_state", return_value=run_state):
                    with patch.object(reader, "_load_processor_state", return_value=None):
                        with patch("api.services.state_reader.save_run_state_to_redis", new_callable=AsyncMock) as mock_save_state:
                            status = await reader.get_run_status("test-run-id", include_components=True)

        assert status["status"] == "success"
        assert status["started_at"] == "2024-01-01T00:00:00Z"
        assert status["finished_at"] == "2024-01-01T00:01:00Z"
        assert status["progress"]["overall"] == 1.0
        assert status["progress"]["components"]["segmenter"]["status"] == "success"
        assert status["progress"]["components"]["visual"]["status"] == "skipped"
        mock_save_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_not_used_when_include_components_true(self, mock_storage, mock_key_layout, mock_redis_client):
        """Кэш не используется полностью когда нужны детальные компоненты."""
        redis_state = {
            "status": "running",
            "progress": 0.5
        }
        
        with patch("api.services.state_reader.get_run_state_from_redis", new_callable=AsyncMock) as mock_get_state:
            mock_get_state.return_value = redis_state
            
            mock_task_manager = MagicMock()
            mock_task_manager.get_run = MagicMock(return_value={
                "platform_id": "youtube",
                "video_id": "test_video"
            })
            
            reader = StateReader(
                storage=mock_storage,
                key_layout=mock_key_layout,
                task_manager=mock_task_manager,
                redis_client=mock_redis_client
            )
            
            # Даже если есть cache, при include_components=True нужно загрузить из Storage
            run_state = {
                "run": {"status": "running"},
                "updated_at": "2024-01-01T00:00:00"
            }
            
            processor_state = {
                "processor": {
                    "status": "running",
                    "progress": 0.5
                }
            }
            
            with patch.object(reader, '_load_run_state', return_value=run_state):
                with patch.object(reader, '_load_processor_state', return_value=processor_state):
                    status = await reader.get_run_status("test-run-id", include_components=True)
                    
                    # Должен загрузить компоненты из Storage
                    assert "components" in status["progress"]
                    assert "visual" in status["progress"]["components"]

