"""
Unit тесты для Redis Streams Queue Service
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from redis.exceptions import RedisError

from api.services.queue import (
    enqueue_run,
    get_queue_length,
    get_pending_count,
    get_total_queue_length,
    QUEUE_HIGH,
    QUEUE_NORMAL,
    QUEUE_LOW,
    MAX_STREAM_LENGTH
)


class TestEnqueueRun:
    """Тесты для функции enqueue_run."""
    
    @pytest.mark.asyncio
    async def test_enqueue_run_high_priority(self, mock_redis_client):
        """Добавление run с высоким приоритетом."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.queue.save_run_metadata", new_callable=AsyncMock) as mock_save_meta:
                with patch("api.services.queue.save_run_priority", new_callable=AsyncMock) as mock_save_priority:
                    mock_redis_client.xadd = AsyncMock(return_value=b"1234567890-0")
                    mock_redis_client.xlen = AsyncMock(return_value=5)
                    
                    metadata = {"video_id": "test_video", "platform_id": "youtube"}
                    message_id = await enqueue_run(
                        run_id="test-run-id",
                        priority="high",
                        metadata=metadata,
                        save_metadata_to_redis=True
                    )
                    
                    assert message_id == "1234567890-0"
                    mock_redis_client.xadd.assert_called_once()
                    call_args = mock_redis_client.xadd.call_args
                    assert call_args[0][0] == QUEUE_HIGH
                    assert call_args[1]["maxlen"] == MAX_STREAM_LENGTH
                    assert call_args[1]["approximate"] is True
                    mock_save_meta.assert_called_once_with("test-run-id", metadata)
                    mock_save_priority.assert_called_once_with("test-run-id", "high")
    
    @pytest.mark.asyncio
    async def test_enqueue_run_normal_priority(self, mock_redis_client):
        """Добавление run с нормальным приоритетом."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.queue.save_run_priority", new_callable=AsyncMock):
                mock_redis_client.xadd = AsyncMock(return_value=b"1234567890-1")
                mock_redis_client.xlen = AsyncMock(return_value=3)
                
                message_id = await enqueue_run(
                    run_id="test-run-id",
                    priority="normal"
                )
                
                assert message_id == "1234567890-1"
                call_args = mock_redis_client.xadd.call_args
                assert call_args[0][0] == QUEUE_NORMAL
    
    @pytest.mark.asyncio
    async def test_enqueue_run_low_priority(self, mock_redis_client):
        """Добавление run с низким приоритетом."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.queue.save_run_priority", new_callable=AsyncMock):
                mock_redis_client.xadd = AsyncMock(return_value=b"1234567890-2")
                mock_redis_client.xlen = AsyncMock(return_value=1)
                
                message_id = await enqueue_run(
                    run_id="test-run-id",
                    priority="low"
                )
                
                assert message_id == "1234567890-2"
                call_args = mock_redis_client.xadd.call_args
                assert call_args[0][0] == QUEUE_LOW
    
    @pytest.mark.asyncio
    async def test_enqueue_run_invalid_priority(self):
        """Добавление run с невалидным приоритетом."""
        with pytest.raises(ValueError, match="Invalid priority"):
            await enqueue_run(run_id="test-run-id", priority="invalid")
    
    @pytest.mark.asyncio
    async def test_enqueue_run_redis_unavailable(self):
        """Добавление run когда Redis недоступен."""
        with patch("api.services.queue.get_redis_client", return_value=None):
            result = await enqueue_run(run_id="test-run-id", priority="normal")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_enqueue_run_redis_error(self, mock_redis_client):
        """Обработка ошибки Redis при добавлении run."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.queue.save_run_priority", new_callable=AsyncMock):
                mock_redis_client.xadd = AsyncMock(side_effect=RedisError("Connection lost"))
                
                with pytest.raises(RedisError, match="Failed to enqueue run"):
                    await enqueue_run(run_id="test-run-id", priority="normal")
    
    @pytest.mark.asyncio
    async def test_enqueue_run_without_metadata_saving(self, mock_redis_client):
        """Добавление run без сохранения метаданных."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.queue.save_run_metadata", new_callable=AsyncMock) as mock_save_meta:
                with patch("api.services.queue.save_run_priority", new_callable=AsyncMock):
                    mock_redis_client.xadd = AsyncMock(return_value=b"1234567890-0")
                    mock_redis_client.xlen = AsyncMock(return_value=1)
                    
                    metadata = {"video_id": "test_video"}
                    await enqueue_run(
                        run_id="test-run-id",
                        priority="normal",
                        metadata=metadata,
                        save_metadata_to_redis=False
                    )
                    
                    mock_save_meta.assert_not_called()


class TestGetQueueLength:
    """Тесты для функции get_queue_length."""
    
    @pytest.mark.asyncio
    async def test_get_queue_length_all_priorities(self, mock_redis_client):
        """Получение длины всех очередей."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xlen = AsyncMock(side_effect=[10, 5, 2])
            
            lengths = await get_queue_length()
            
            assert lengths == {"high": 10, "normal": 5, "low": 2}
            assert mock_redis_client.xlen.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_queue_length_specific_priority(self, mock_redis_client):
        """Получение длины конкретной очереди."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xlen = AsyncMock(return_value=10)
            
            lengths = await get_queue_length(priority="high")
            
            assert lengths == {"high": 10}
            mock_redis_client.xlen.assert_called_once_with(QUEUE_HIGH)
    
    @pytest.mark.asyncio
    async def test_get_queue_length_redis_unavailable(self):
        """Получение длины очереди когда Redis недоступен."""
        with patch("api.services.queue.get_redis_client", return_value=None):
            lengths = await get_queue_length()
            assert lengths == {}
    
    @pytest.mark.asyncio
    async def test_get_queue_length_redis_error(self, mock_redis_client):
        """Обработка ошибки Redis при получении длины очереди."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xlen = AsyncMock(side_effect=RedisError("Connection lost"))
            
            lengths = await get_queue_length()
            # Должен вернуть пустой словарь при ошибке
            assert lengths == {}


class TestGetPendingCount:
    """Тесты для функции get_pending_count."""
    
    @pytest.mark.asyncio
    async def test_get_pending_count_all_priorities(self, mock_redis_client):
        """Получение количества pending сообщений для всех очередей."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            # xpending возвращает кортеж: (total_pending, min_id, max_id, consumers)
            mock_redis_client.xpending = AsyncMock(side_effect=[
                (5, b"0-0", b"0-4", []),
                (3, b"0-0", b"0-2", []),
                (1, b"0-0", b"0-0", [])
            ])
            
            pending_counts = await get_pending_count()
            
            assert pending_counts == {"high": 5, "normal": 3, "low": 1}
            assert mock_redis_client.xpending.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_pending_count_specific_priority(self, mock_redis_client):
        """Получение количества pending сообщений для конкретной очереди."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xpending = AsyncMock(return_value=(5, b"0-0", b"0-4", []))
            
            pending_counts = await get_pending_count(priority="high", group_name="workers")
            
            assert pending_counts == {"high": 5}
            mock_redis_client.xpending.assert_called_once_with(QUEUE_HIGH, "workers")
    
    @pytest.mark.asyncio
    async def test_get_pending_count_no_pending(self, mock_redis_client):
        """Получение количества pending когда их нет."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xpending = AsyncMock(return_value=None)
            
            pending_counts = await get_pending_count()
            
            assert pending_counts == {"high": 0, "normal": 0, "low": 0}
    
    @pytest.mark.asyncio
    async def test_get_pending_count_redis_unavailable(self):
        """Получение количества pending когда Redis недоступен."""
        with patch("api.services.queue.get_redis_client", return_value=None):
            pending_counts = await get_pending_count()
            assert pending_counts == {}
    
    @pytest.mark.asyncio
    async def test_get_pending_count_redis_error(self, mock_redis_client):
        """Обработка ошибки Redis при получении pending count."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xpending = AsyncMock(side_effect=RedisError("Connection lost"))
            
            pending_counts = await get_pending_count()
            # Должен вернуть пустой словарь при ошибке
            assert pending_counts == {}


class TestGetTotalQueueLength:
    """Тесты для функции get_total_queue_length."""
    
    @pytest.mark.asyncio
    async def test_get_total_queue_length_success(self, mock_redis_client):
        """Успешное получение общей длины очередей."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.queue.queue_length") as mock_metric:
                mock_redis_client.xlen = AsyncMock(side_effect=[10, 5, 2])
                mock_metric.labels = MagicMock(return_value=mock_metric)
                mock_metric.set = MagicMock()
                
                total = await get_total_queue_length()
                
                assert total == 17  # 10 + 5 + 2
                assert mock_redis_client.xlen.call_count == 3
                assert mock_metric.set.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_total_queue_length_redis_unavailable(self):
        """Получение общей длины когда Redis недоступен."""
        with patch("api.services.queue.get_redis_client", return_value=None):
            total = await get_total_queue_length()
            assert total == 0
    
    @pytest.mark.asyncio
    async def test_get_total_queue_length_redis_error(self, mock_redis_client):
        """Обработка ошибки Redis при получении общей длины."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            mock_redis_client.xlen = AsyncMock(side_effect=RedisError("Connection lost"))
            
            total = await get_total_queue_length()
            # Должен вернуть 0 при ошибке, чтобы не блокировать обработку
            assert total == 0
    
    @pytest.mark.asyncio
    async def test_get_total_queue_length_metric_error(self, mock_redis_client):
        """Обработка ошибки при обновлении метрики."""
        with patch("api.services.queue.get_redis_client", return_value=mock_redis_client):
            with patch("api.services.queue.queue_length") as mock_metric:
                mock_redis_client.xlen = AsyncMock(side_effect=[10, 5, 2])
                mock_metric.labels = MagicMock(side_effect=Exception("Metric error"))
                
                # Должен вернуть результат даже если метрика не обновилась
                total = await get_total_queue_length()
                assert total == 17

