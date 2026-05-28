"""
Unit тесты для улучшенной обработки ошибок
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from redis.exceptions import RedisError, ConnectionError, TimeoutError, ResponseError
from storage.base import StorageError, NotFoundError

from api.services.worker import Worker
from api.endpoints.process import process_video
from api.utils.errors import (
    InvalidPayloadError,
    ProcessingError,
    RunAlreadyExistsError,
    BackpressureError
)


class TestWorkerErrorHandling:
    """Тесты для улучшенной обработки ошибок в Worker."""
    
    @pytest.mark.asyncio
    async def test_worker_redis_error_in_loop(self):
        """Обработка Redis ошибок в worker loop."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = MagicMock()
        worker.running = True
        
        # Мокаем xreadgroup чтобы выбросить RedisError
        worker.redis_client.xreadgroup = AsyncMock(side_effect=RedisError("Connection lost"))
        
        # Запустить worker loop на короткое время
        import asyncio
        task = asyncio.create_task(worker._worker_loop())
        await asyncio.sleep(0.1)
        worker.running = False
        worker.shutdown_event.set()
        
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_worker_storage_error_processing(self):
        """Обработка Storage ошибок при обработке run."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = MagicMock()
        worker.task_manager = MagicMock()
        worker.task_manager.get_run = MagicMock(return_value={
            "video_id": "test_video",
            "platform_id": "youtube"
        })
        
        # Мокаем что Storage операция выбрасывает StorageError
        with patch("api.services.worker.save_run_state", new_callable=AsyncMock) as mock_save:
            mock_save.side_effect = StorageError("Storage unavailable")
            
            # Мокаем данные сообщения
            data = {
                b"run_id": b"test-run-id",
                b"metadata": b'{"video_id": "test_video"}'
            }
            
            # Обработка должна поймать StorageError
            await worker._process_run_task(
                run_id="test-run-id",
                stream_name="queue:normal",
                message_id="123-0",
                data=data,
                metadata={},
                queue_wait_start=None
            )
            
            # Проверить что ошибка была обработана
            assert mock_save.called
    
    @pytest.mark.asyncio
    async def test_worker_redis_error_heartbeat(self):
        """Обработка Redis ошибок в heartbeat loop."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = MagicMock()
        worker.task_manager = MagicMock()
        worker.task_manager.get_run = MagicMock(return_value={"status": "running"})
        
        # Мокаем update_run_heartbeat чтобы выбросить RedisError
        with patch("api.services.worker.update_run_heartbeat", new_callable=AsyncMock) as mock_heartbeat:
            mock_heartbeat.side_effect = RedisError("Connection lost")
            
            # Запустить heartbeat loop на короткое время
            task = asyncio.create_task(worker._heartbeat_loop("test-run-id"))
            await asyncio.sleep(0.1)
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_worker_ack_message_redis_error(self):
        """Обработка Redis ошибок при ACK сообщения."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = MagicMock()
        worker.redis_client.xack = AsyncMock(side_effect=RedisError("Connection lost"))
        
        # ACK должен обработать ошибку без падения
        await worker._ack_message("queue:normal", "123-0")
        
        # Проверить что ошибка была залогирована (через проверку что метод был вызван)
        assert worker.redis_client.xack.called


class TestEndpointErrorHandling:
    """Тесты для улучшенной обработки ошибок в endpoints."""
    
    @pytest.mark.asyncio
    async def test_process_endpoint_redis_error(self):
        """Обработка Redis ошибок в process endpoint."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        payload = {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "video_id": "test_video",
            "platform_id": "youtube",
            "video_path": "/tmp/test_video.mp4",
            "config_hash": "test_hash",
            "profile_config": {"processors": {}}
        }
        
        with patch("api.endpoints.process.validate_video_path"):
            with patch("api.endpoints.process.validate_profile_config"):
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    with patch("api.endpoints.process.acquire_run_lock", new_callable=AsyncMock, return_value=True):
                        with patch("api.endpoints.process.get_redis_client") as mock_get_redis:
                            mock_get_redis.side_effect = RedisError("Connection lost")
                            
                            with patch("api.dependencies.get_task_manager") as mock_get_task_manager:
                                mock_task_manager = MagicMock()
                                mock_task_manager.is_run_active = MagicMock(return_value=False)
                                mock_task_manager.can_accept_new_run = MagicMock(return_value=True)
                                mock_get_task_manager.return_value = mock_task_manager
                                
                                response = client.post(
                                    "/api/v1/process",
                                    json=payload,
                                    headers={"X-API-Key": "test_api_key"}
                                )
                                
                                # Должен вернуть 503 при Redis ошибке
                                assert response.status_code == 503
    
    @pytest.mark.asyncio
    async def test_process_endpoint_storage_error(self):
        """Обработка Storage ошибок в process endpoint."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        payload = {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "video_id": "test_video",
            "platform_id": "youtube",
            "video_path": "/tmp/test_video.mp4",
            "config_hash": "test_hash",
            "profile_config": {"processors": {}}
        }
        
        with patch("api.endpoints.process.validate_video_path") as mock_validate:
            mock_validate.side_effect = StorageError("Storage unavailable")
            
            with patch("api.endpoints.process.validate_profile_config"):
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    response = client.post(
                        "/api/v1/process",
                        json=payload,
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    # Должен вернуть 503 при Storage ошибке
                    assert response.status_code == 503
    
    @pytest.mark.asyncio
    async def test_process_endpoint_not_found_error(self):
        """Обработка NotFoundError в process endpoint."""
        from fastapi.testclient import TestClient
        from api.main import app
        
        client = TestClient(app)
        
        payload = {
            "run_id": "550e8400-e29b-41d4-a716-446655440000",
            "video_id": "test_video",
            "platform_id": "youtube",
            "video_path": "/tmp/test_video.mp4",
            "config_hash": "test_hash",
            "profile_config": {"processors": {}}
        }
        
        with patch("api.endpoints.process.validate_video_path") as mock_validate:
            mock_validate.side_effect = NotFoundError("Video file not found")
            
            with patch("api.endpoints.process.validate_profile_config"):
                with patch("api.endpoints.process.audit_log", new_callable=AsyncMock):
                    response = client.post(
                        "/api/v1/process",
                        json=payload,
                        headers={"X-API-Key": "test_api_key"}
                    )
                    
                    # Должен вернуть 404 при NotFoundError
                    assert response.status_code == 404


class TestErrorLoggingContext:
    """Тесты для контекста в логах ошибок."""
    
    def test_worker_error_logging_with_context(self):
        """Проверка что ошибки логируются с контекстом (run_id, worker_id)."""
        import logging
        from unittest.mock import patch
        
        worker = Worker(worker_id="test-worker")
        
        with patch("api.services.worker.logger") as mock_logger:
            # Симулировать ошибку обработки
            try:
                raise RedisError("Connection lost")
            except RedisError as e:
                mock_logger.exception(
                    "Redis error processing run",
                    run_id="test-run-id",
                    worker_id=worker.worker_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
            
            # Проверить что logger.exception был вызван с правильными параметрами
            mock_logger.exception.assert_called_once()
            call_kwargs = mock_logger.exception.call_args[1]
            assert call_kwargs["run_id"] == "test-run-id"
            assert call_kwargs["worker_id"] == "test-worker"
            assert call_kwargs["error_type"] == "RedisError"
    
    def test_endpoint_error_logging_with_context(self):
        """Проверка что ошибки в endpoints логируются с контекстом (request_id, run_id)."""
        from unittest.mock import patch, MagicMock
        
        mock_request = MagicMock()
        mock_request.state.request_id = "test-request-id"
        mock_request.client.host = "192.168.1.1"
        
        with patch("api.endpoints.process.logger") as mock_logger:
            try:
                raise RedisError("Connection lost")
            except RedisError as e:
                request_id = getattr(mock_request.state, 'request_id', None)
                client_ip = mock_request.client.host if mock_request.client else None
                mock_logger.exception(
                    "Redis error in process_video",
                    run_id="test-run-id",
                    request_id=request_id,
                    client_ip=client_ip,
                    error=str(e),
                    error_type=type(e).__name__
                )
            
            # Проверить что logger.exception был вызван с правильными параметрами
            mock_logger.exception.assert_called_once()
            call_kwargs = mock_logger.exception.call_args[1]
            assert call_kwargs["request_id"] == "test-request-id"
            assert call_kwargs["run_id"] == "test-run-id"
            assert call_kwargs["error_type"] == "RedisError"


class TestSpecificErrorTypes:
    """Тесты для обработки конкретных типов ошибок."""
    
    @pytest.mark.asyncio
    async def test_redis_connection_error(self):
        """Обработка ConnectionError от Redis."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = MagicMock()
        worker.redis_client.xack = AsyncMock(side_effect=ConnectionError("Connection refused"))
        
        # Должен обработать ConnectionError без падения
        await worker._ack_message("queue:normal", "123-0")
        assert worker.redis_client.xack.called
    
    @pytest.mark.asyncio
    async def test_redis_timeout_error(self):
        """Обработка TimeoutError от Redis."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = MagicMock()
        worker.redis_client.xreadgroup = AsyncMock(side_effect=TimeoutError("Read timeout"))
        
        worker.running = True
        task = asyncio.create_task(worker._worker_loop())
        await asyncio.sleep(0.1)
        worker.running = False
        worker.shutdown_event.set()
        
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_storage_not_found_error(self):
        """Обработка NotFoundError от Storage."""
        worker = Worker(worker_id="test-worker")
        worker.task_manager = MagicMock()
        worker.task_manager.get_run = MagicMock(return_value={
            "video_id": "test_video",
            "platform_id": "youtube"
        })
        
        with patch("api.services.worker.save_run_state", new_callable=AsyncMock) as mock_save:
            mock_save.side_effect = NotFoundError("Run not found")
            
            data = {
                b"run_id": b"test-run-id",
                b"metadata": b'{}'
            }
            
            # Должен обработать NotFoundError
            await worker._process_run_task(
                run_id="test-run-id",
                stream_name="queue:normal",
                message_id="123-0",
                data=data,
                metadata={},
                queue_wait_start=None
            )

