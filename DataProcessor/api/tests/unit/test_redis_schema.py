"""
Unit тесты для Redis Schema Service
"""

import pytest
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from redis.exceptions import RedisError

from api.services.redis_schema import (
    save_run_metadata,
    get_run_metadata,
    save_run_state,
    get_run_state,
    update_run_heartbeat,
    get_run_heartbeat,
    is_run_alive,
    acquire_run_lock,
    release_run_lock,
    is_run_locked,
    save_run_priority,
    get_run_priority,
    add_run_event,
    get_run_events,
    set_cancel_flag,
    get_cancel_flag,
    clear_cancel_flag,
    delete_run_data,
    TTL_META,
    TTL_STATE,
    TTL_HEARTBEAT,
    TTL_LOCK,
    TTL_PRIORITY,
    TTL_EVENTS,
    TTL_CANCEL
)
from api.schemas.state import RunStatus


class TestRunMetadata:
    """Тесты для работы с метаданными run'а."""
    
    @pytest.mark.asyncio
    async def test_save_run_metadata_success(self, mock_redis_client):
        """Успешное сохранение метаданных."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.setex = AsyncMock(return_value=True)
            
            metadata = {
                "video_id": "test_video",
                "platform_id": "youtube",
                "profile_config": {"processors": {}}
            }
            result = await save_run_metadata("test-run-id", metadata)
            
            assert result is True
            mock_redis_client.setex.assert_called_once()
            call_args = mock_redis_client.setex.call_args
            assert call_args[0][0] == "run:meta:test-run-id"
            assert call_args[0][1] == TTL_META
            assert json.loads(call_args[0][2]) == metadata
    
    @pytest.mark.asyncio
    async def test_save_run_metadata_redis_unavailable(self):
        """Сохранение метаданных когда Redis недоступен."""
        with patch("api.services.redis_schema.get_redis_client", return_value=None):
            result = await save_run_metadata("test-run-id", {})
            assert result is False
    
    @pytest.mark.asyncio
    async def test_save_run_metadata_redis_error(self, mock_redis_client):
        """Обработка ошибки Redis при сохранении метаданных."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.setex = AsyncMock(side_effect=RedisError("Connection lost"))
            
            result = await save_run_metadata("test-run-id", {})
            assert result is False
    
    @pytest.mark.asyncio
    async def test_get_run_metadata_success(self, mock_redis_client):
        """Успешное получение метаданных."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            metadata = {"video_id": "test_video", "platform_id": "youtube"}
            mock_redis_client.get = AsyncMock(return_value=json.dumps(metadata).encode("utf-8"))
            
            result = await get_run_metadata("test-run-id")
            
            assert result == metadata
            mock_redis_client.get.assert_called_once_with("run:meta:test-run-id")
    
    @pytest.mark.asyncio
    async def test_get_run_metadata_not_found(self, mock_redis_client):
        """Получение метаданных когда они не найдены."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.get = AsyncMock(return_value=None)
            
            result = await get_run_metadata("test-run-id")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_run_metadata_redis_unavailable(self):
        """Получение метаданных когда Redis недоступен."""
        with patch("api.services.redis_schema.get_redis_client", return_value=None):
            result = await get_run_metadata("test-run-id")
            assert result is None


class TestRunState:
    """Тесты для работы с состоянием run'а."""
    
    @pytest.mark.asyncio
    async def test_save_run_state_success(self, mock_redis_client):
        """Успешное сохранение состояния."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.redis_schema.get_run_state", new_callable=AsyncMock, return_value=None):
                mock_redis_client.setex = AsyncMock(return_value=True)
                
                state = {"status": "running", "progress": 0.5}
                result = await save_run_state("test-run-id", state)
                
                assert result is True
                mock_redis_client.setex.assert_called_once()
                call_args = mock_redis_client.setex.call_args
                assert call_args[0][0] == "run:state:test-run-id"
                assert call_args[0][1] == TTL_STATE
                saved_state = json.loads(call_args[0][2])
                assert saved_state["status"] == "running"
                assert "updated_at" in saved_state
    
    @pytest.mark.asyncio
    async def test_save_run_state_with_status_transition_validation(self, mock_redis_client):
        """Сохранение состояния с валидацией перехода статуса."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            # Текущий статус - running
            current_state = {"status": "running"}
            with patch("api.services.redis_schema.get_run_state", new_callable=AsyncMock, return_value=current_state):
                mock_redis_client.setex = AsyncMock(return_value=True)
                
                # Валидный переход: running -> success
                new_state = {"status": "success"}
                result = await save_run_state("test-run-id", new_state, validate_status_transition=True)
                
                assert result is True
    
    @pytest.mark.asyncio
    async def test_save_run_state_invalid_transition(self, mock_redis_client):
        """Сохранение состояния с невалидным переходом статуса."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            # Текущий статус - success (финальное состояние)
            current_state = {"status": "success"}
            with patch("api.services.redis_schema.get_run_state", new_callable=AsyncMock, return_value=current_state):
                # Невалидный переход: success -> running
                new_state = {"status": "running"}
                result = await save_run_state("test-run-id", new_state, validate_status_transition=True)
                
                assert result is False
    
    @pytest.mark.asyncio
    async def test_save_run_state_without_validation(self, mock_redis_client):
        """Сохранение состояния без валидации перехода."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.setex = AsyncMock(return_value=True)
            
            state = {"status": "running"}
            result = await save_run_state("test-run-id", state, validate_status_transition=False)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_get_run_state_success(self, mock_redis_client):
        """Успешное получение состояния."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            state = {"status": "running", "progress": 0.5}
            mock_redis_client.get = AsyncMock(return_value=json.dumps(state).encode("utf-8"))
            
            result = await get_run_state("test-run-id")
            
            assert result == state
            mock_redis_client.get.assert_called_once_with("run:state:test-run-id")
    
    @pytest.mark.asyncio
    async def test_get_run_state_not_found(self, mock_redis_client):
        """Получение состояния когда оно не найдено."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.get = AsyncMock(return_value=None)
            
            result = await get_run_state("test-run-id")
            assert result is None


class TestRunHeartbeat:
    """Тесты для работы с heartbeat."""
    
    @pytest.mark.asyncio
    async def test_update_run_heartbeat_success(self, mock_redis_client):
        """Успешное обновление heartbeat."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.setex = AsyncMock(return_value=True)
            
            result = await update_run_heartbeat("test-run-id")
            
            assert result is True
            mock_redis_client.setex.assert_called_once()
            call_args = mock_redis_client.setex.call_args
            assert call_args[0][0] == "run:heartbeat:test-run-id"
            assert call_args[0][1] == TTL_HEARTBEAT
    
    @pytest.mark.asyncio
    async def test_get_run_heartbeat_success(self, mock_redis_client):
        """Успешное получение heartbeat."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            timestamp = str(time.time())
            mock_redis_client.get = AsyncMock(return_value=timestamp.encode("utf-8"))
            
            result = await get_run_heartbeat("test-run-id")
            
            assert result is not None
            assert isinstance(result, float)
            assert result == float(timestamp)
    
    @pytest.mark.asyncio
    async def test_get_run_heartbeat_not_found(self, mock_redis_client):
        """Получение heartbeat когда он не найден."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.get = AsyncMock(return_value=None)
            
            result = await get_run_heartbeat("test-run-id")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_is_run_alive_true(self, mock_redis_client):
        """Проверка что run жив (heartbeat существует)."""
        with patch("api.services.redis_schema.get_run_heartbeat", new_callable=AsyncMock, return_value=time.time()):
            result = await is_run_alive("test-run-id")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_is_run_alive_false(self):
        """Проверка что run не жив (heartbeat отсутствует)."""
        with patch("api.services.redis_schema.get_run_heartbeat", new_callable=AsyncMock, return_value=None):
            result = await is_run_alive("test-run-id")
            assert result is False


class TestRunLock:
    """Тесты для работы с idempotency lock."""
    
    @pytest.mark.asyncio
    async def test_acquire_run_lock_success(self, mock_redis_client):
        """Успешное получение блокировки."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.set = AsyncMock(return_value=True)
            
            result = await acquire_run_lock("test-run-id")
            
            assert result is True
            mock_redis_client.set.assert_called_once()
            call_args = mock_redis_client.set.call_args
            assert call_args[0][0] == "run:lock:test-run-id"
            assert call_args[0][1] == "locked"
            assert call_args[1]["ex"] == TTL_LOCK
            assert call_args[1]["nx"] is True
    
    @pytest.mark.asyncio
    async def test_acquire_run_lock_already_locked(self, mock_redis_client):
        """Попытка получить блокировку когда она уже существует."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.set = AsyncMock(return_value=False)  # NX вернул False
            
            result = await acquire_run_lock("test-run-id")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_acquire_run_lock_custom_timeout(self, mock_redis_client):
        """Получение блокировки с кастомным таймаутом."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.set = AsyncMock(return_value=True)
            
            custom_timeout = 7200
            result = await acquire_run_lock("test-run-id", timeout=custom_timeout)
            
            assert result is True
            call_args = mock_redis_client.set.call_args
            assert call_args[1]["ex"] == custom_timeout
    
    @pytest.mark.asyncio
    async def test_release_run_lock_success(self, mock_redis_client):
        """Успешное освобождение блокировки."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.delete = AsyncMock(return_value=1)
            
            result = await release_run_lock("test-run-id")
            
            assert result is True
            mock_redis_client.delete.assert_called_once_with("run:lock:test-run-id")
    
    @pytest.mark.asyncio
    async def test_is_run_locked_true(self, mock_redis_client):
        """Проверка что run заблокирован."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.exists = AsyncMock(return_value=1)
            
            result = await is_run_locked("test-run-id")
            
            assert result is True
            mock_redis_client.exists.assert_called_once_with("run:lock:test-run-id")
    
    @pytest.mark.asyncio
    async def test_is_run_locked_false(self, mock_redis_client):
        """Проверка что run не заблокирован."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.exists = AsyncMock(return_value=0)
            
            result = await is_run_locked("test-run-id")
            
            assert result is False


class TestRunPriority:
    """Тесты для работы с приоритетом run'а."""
    
    @pytest.mark.asyncio
    async def test_save_run_priority_success(self, mock_redis_client):
        """Успешное сохранение приоритета."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.setex = AsyncMock(return_value=True)
            
            result = await save_run_priority("test-run-id", "high")
            
            assert result is True
            mock_redis_client.setex.assert_called_once()
            call_args = mock_redis_client.setex.call_args
            assert call_args[0][0] == "run:priority:test-run-id"
            assert call_args[0][1] == TTL_PRIORITY
            assert call_args[0][2] == "high"
    
    @pytest.mark.asyncio
    async def test_get_run_priority_success(self, mock_redis_client):
        """Успешное получение приоритета."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.get = AsyncMock(return_value=b"high")
            
            result = await get_run_priority("test-run-id")
            
            assert result == "high"
            mock_redis_client.get.assert_called_once_with("run:priority:test-run-id")
    
    @pytest.mark.asyncio
    async def test_get_run_priority_not_found(self, mock_redis_client):
        """Получение приоритета когда он не найден."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.get = AsyncMock(return_value=None)
            
            result = await get_run_priority("test-run-id")
            assert result is None


class TestRunEvents:
    """Тесты для работы с событиями run'а."""
    
    @pytest.mark.asyncio
    async def test_add_run_event_success(self, mock_redis_client):
        """Успешное добавление события."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xadd = AsyncMock(return_value=b"1234567890-0")
            mock_redis_client.expire = AsyncMock(return_value=True)
            
            event_data = {"message": "Processing started"}
            message_id = await add_run_event("test-run-id", "processing_started", event_data)
            
            assert message_id == "1234567890-0"
            mock_redis_client.xadd.assert_called_once()
            call_args = mock_redis_client.xadd.call_args
            assert call_args[0][0] == "stream:events:test-run-id"
            assert call_args[1]["maxlen"] == 1000
            assert call_args[1]["approximate"] is True
            mock_redis_client.expire.assert_called_once_with("stream:events:test-run-id", TTL_EVENTS)
    
    @pytest.mark.asyncio
    async def test_get_run_events_success(self, mock_redis_client):
        """Успешное получение событий."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            # Симулируем ответ от xread
            messages = [
                (
                    b"stream:events:test-run-id",
                    [
                        (
                            b"1234567890-0",
                            {
                                b"event_type": b"processing_started",
                                b"timestamp": b"1234567890.0",
                                b"data": b'{"message": "Started"}'
                            }
                        )
                    ]
                )
            ]
            mock_redis_client.xread = AsyncMock(return_value=messages)
            
            events = await get_run_events("test-run-id", count=10)
            
            assert len(events) == 1
            assert events[0]["id"] == "1234567890-0"
            assert events[0]["event_type"] == "processing_started"
            assert events[0]["data"]["message"] == "Started"
    
    @pytest.mark.asyncio
    async def test_get_run_events_with_start_id(self, mock_redis_client):
        """Получение событий с указанием начального ID."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xread = AsyncMock(return_value=[])
            
            events = await get_run_events("test-run-id", count=10, start_id="1234567890-0")
            
            assert events == []
            mock_redis_client.xread.assert_called_once()
            call_args = mock_redis_client.xread.call_args
            assert call_args[0][0] == {"stream:events:test-run-id": "1234567890-0"}
            assert call_args[1]["count"] == 10
    
    @pytest.mark.asyncio
    async def test_get_run_events_empty(self, mock_redis_client):
        """Получение событий когда их нет."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xread = AsyncMock(return_value=[])
            
            events = await get_run_events("test-run-id")
            
            assert events == []


class TestCancelFlag:
    """Тесты для работы с флагом отмены."""
    
    @pytest.mark.asyncio
    async def test_set_cancel_flag_success(self, mock_redis_client):
        """Успешная установка флага отмены."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.setex = AsyncMock(return_value=True)
            
            result = await set_cancel_flag("test-run-id")
            
            assert result is True
            mock_redis_client.setex.assert_called_once()
            call_args = mock_redis_client.setex.call_args
            assert call_args[0][0] == "run:cancel:test-run-id"
            assert call_args[0][1] == TTL_CANCEL
            assert call_args[0][2] == "1"
    
    @pytest.mark.asyncio
    async def test_get_cancel_flag_true(self, mock_redis_client):
        """Проверка флага отмены когда он установлен."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.get = AsyncMock(return_value=b"1")
            
            result = await get_cancel_flag("test-run-id")
            
            assert result is True
            mock_redis_client.get.assert_called_once_with("run:cancel:test-run-id")
    
    @pytest.mark.asyncio
    async def test_get_cancel_flag_false(self, mock_redis_client):
        """Проверка флага отмены когда он не установлен."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.get = AsyncMock(return_value=None)
            
            result = await get_cancel_flag("test-run-id")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_clear_cancel_flag_success(self, mock_redis_client):
        """Успешная очистка флага отмены."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.delete = AsyncMock(return_value=1)
            
            result = await clear_cancel_flag("test-run-id")
            
            assert result is True
            mock_redis_client.delete.assert_called_once_with("run:cancel:test-run-id")


class TestDeleteRunData:
    """Тесты для удаления всех данных run'а."""
    
    @pytest.mark.asyncio
    async def test_delete_run_data_success(self, mock_redis_client):
        """Успешное удаление всех данных run'а."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.delete = AsyncMock(return_value=6)
            
            result = await delete_run_data("test-run-id")
            
            assert result is True
            mock_redis_client.delete.assert_called_once()
            # Проверяем что были переданы все ключи
            call_args = mock_redis_client.delete.call_args
            assert len(call_args[0]) == 6
            assert "run:meta:test-run-id" in call_args[0]
            assert "run:state:test-run-id" in call_args[0]
            assert "run:heartbeat:test-run-id" in call_args[0]
            assert "run:lock:test-run-id" in call_args[0]
            assert "run:priority:test-run-id" in call_args[0]
            assert "stream:events:test-run-id" in call_args[0]
    
    @pytest.mark.asyncio
    async def test_delete_run_data_redis_error(self, mock_redis_client):
        """Обработка ошибки Redis при удалении данных."""
        with patch("api.services.redis_schema.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.delete = AsyncMock(side_effect=RedisError("Connection lost"))
            
            result = await delete_run_data("test-run-id")
            
            assert result is False

