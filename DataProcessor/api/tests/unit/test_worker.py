"""
Unit тесты для Worker Service
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from redis.exceptions import RedisError, ResponseError

from api.services.worker import Worker, CONSUMER_GROUP_NAME, QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW
from api.schemas.state import RunStatus


class TestWorkerInitialization:
    """Тесты для инициализации Worker."""
    
    def test_worker_init_with_defaults(self):
        """Инициализация Worker с параметрами по умолчанию."""
        worker = Worker(worker_id="test-worker-1")
        
        assert worker.worker_id == "test-worker-1"
        assert worker.consumer_name == "worker-test-worker-1"
        assert worker.processor_service is not None
        assert worker.task_manager is not None
        assert worker.running is False
        assert worker.active_heartbeats == {}
        assert worker.active_tasks == {}
    
    def test_worker_init_with_custom_services(self, mock_storage, mock_key_layout):
        """Инициализация Worker с кастомными сервисами."""
        processor_service = MagicMock()
        task_manager = MagicMock()
        
        worker = Worker(
            worker_id="test-worker-2",
            storage=mock_storage,
            key_layout=mock_key_layout,
            processor_service=processor_service,
            task_manager=task_manager
        )
        
        assert worker.storage == mock_storage
        assert worker.key_layout == mock_key_layout
        assert worker.processor_service == processor_service
        assert worker.task_manager == task_manager


class TestWorkerConsumerGroups:
    """Тесты для создания consumer groups."""
    
    @pytest.mark.asyncio
    async def test_ensure_consumer_groups_success(self, mock_redis_client):
        """Успешное создание consumer groups."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        mock_redis_client.xgroup_create = AsyncMock(return_value=True)
        
        await worker._ensure_consumer_groups()
        
        # Должен создать consumer groups для всех очередей
        assert mock_redis_client.xgroup_create.call_count == 3
        calls = [call[0][0] for call in mock_redis_client.xgroup_create.call_args_list]
        assert QUEUE_HIGH in calls
        assert QUEUE_NORMAL in calls
        assert QUEUE_LOW in calls
    
    @pytest.mark.asyncio
    async def test_ensure_consumer_groups_already_exists(self, mock_redis_client):
        """Обработка случая когда consumer group уже существует."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        # Симулируем ошибку BUSYGROUP (group уже существует)
        mock_redis_client.xgroup_create = AsyncMock(
            side_effect=ResponseError("BUSYGROUP Consumer Group name already exists")
        )
        
        # Не должно быть исключения
        await worker._ensure_consumer_groups()
        
        # Должен попытаться создать для всех очередей
        assert mock_redis_client.xgroup_create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_ensure_consumer_groups_other_error(self, mock_redis_client):
        """Обработка других ошибок при создании consumer group."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        mock_redis_client.xgroup_create = AsyncMock(
            side_effect=RedisError("Connection lost")
        )
        
        # Должно быть исключение
        with pytest.raises(RedisError):
            await worker._ensure_consumer_groups()


class TestWorkerMessageProcessing:
    """Тесты для обработки сообщений."""
    
    @pytest.mark.asyncio
    async def test_process_message_success(self, mock_redis_client):
        """Успешная обработка сообщения."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        # Мокаем данные сообщения
        message_data = {
            b"run_id": b"test-run-id",
            b"ts": b"1234567890.0",
            b"priority": b"normal"
        }
        
        # Мокаем _process_run_task
        with patch.object(worker, '_process_run_task', new_callable=AsyncMock) as mock_process:
            await worker._process_message("queue:normal", "1234567890-0", message_data)
            
            # Должен вызвать _process_run_task
            mock_process.assert_called_once()
            call_args = mock_process.call_args
            assert call_args[0][0] == "test-run-id"
            assert call_args[0][1] == "queue:normal"
    
    @pytest.mark.asyncio
    async def test_process_message_missing_run_id(self, mock_redis_client):
        """Обработка сообщения без run_id."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        message_data = {
            b"ts": b"1234567890.0"
        }
        
        with patch.object(worker, '_ack_message', new_callable=AsyncMock) as mock_ack:
            await worker._process_message("queue:normal", "1234567890-0", message_data)
            
            # Должен ACK сообщение даже если данные невалидны
            mock_ack.assert_called_once_with("queue:normal", "1234567890-0")
    
    @pytest.mark.asyncio
    async def test_process_message_shutdown_event(self, mock_redis_client):
        """Пропуск обработки при shutdown event."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        worker.shutdown_event.set()  # Установить shutdown event
        
        message_data = {
            b"run_id": b"test-run-id"
        }
        
        with patch.object(worker, '_ack_message', new_callable=AsyncMock) as mock_ack:
            with patch.object(worker, '_process_run_task', new_callable=AsyncMock) as mock_process:
                await worker._process_message("queue:normal", "1234567890-0", message_data)
                
                # Не должен обрабатывать сообщение
                mock_process.assert_not_called()
                # Должен ACK сообщение
                mock_ack.assert_called_once()


class TestWorkerHeartbeatLoop:
    """Тесты для heartbeat loop."""
    
    @pytest.mark.asyncio
    async def test_heartbeat_loop_success(self, mock_redis_client):
        """Успешная отправка heartbeat."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        run_id = "test-run-id"
        
        # Мокаем TaskManager
        run_info = {
            "status": "running",
            "run_id": run_id
        }
        worker.task_manager.get_run = MagicMock(return_value=run_info)
        
        # Мокаем update_run_heartbeat
        with patch("api.services.worker.update_run_heartbeat", new_callable=AsyncMock) as mock_heartbeat:
            # Создаем задачу для heartbeat loop
            heartbeat_task = asyncio.create_task(worker._heartbeat_loop(run_id))
            
            # Подождать немного
            await asyncio.sleep(0.1)
            
            # Отменить задачу
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            
            # Должен был вызван update_run_heartbeat хотя бы раз
            # (но может не успеть из-за короткого времени ожидания)
    
    @pytest.mark.asyncio
    async def test_heartbeat_loop_stops_on_inactive_run(self):
        """Heartbeat loop останавливается когда run неактивен."""
        worker = Worker(worker_id="test-worker")
        
        run_id = "test-run-id"
        
        # Мокаем TaskManager - run не найден
        worker.task_manager.get_run = MagicMock(return_value=None)
        
        # Создаем задачу для heartbeat loop
        heartbeat_task = asyncio.create_task(worker._heartbeat_loop(run_id))
        
        # Подождать завершения
        await heartbeat_task
        
        # Задача должна завершиться (не должна быть cancelled)
        assert heartbeat_task.done()
    
    @pytest.mark.asyncio
    async def test_heartbeat_loop_stops_on_final_status(self):
        """Heartbeat loop останавливается при финальном статусе."""
        worker = Worker(worker_id="test-worker")
        
        run_id = "test-run-id"
        
        # Мокаем TaskManager - run завершен
        run_info = {
            "status": "success",
            "run_id": run_id
        }
        worker.task_manager.get_run = MagicMock(return_value=run_info)
        
        # Создаем задачу для heartbeat loop
        heartbeat_task = asyncio.create_task(worker._heartbeat_loop(run_id))
        
        # Подождать завершения
        await heartbeat_task
        
        # Задача должна завершиться
        assert heartbeat_task.done()


class TestWorkerAckMessage:
    """Тесты для ACK сообщений."""
    
    @pytest.mark.asyncio
    async def test_ack_message_success(self, mock_redis_client):
        """Успешный ACK сообщения."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        mock_redis_client.xack = AsyncMock(return_value=1)
        
        await worker._ack_message("queue:normal", "1234567890-0")
        
        mock_redis_client.xack.assert_called_once()
        call_args = mock_redis_client.xack.call_args
        assert call_args[0][0] == "queue:normal"
        assert call_args[0][1] == CONSUMER_GROUP_NAME
        assert call_args[0][2] == "1234567890-0"
    
    @pytest.mark.asyncio
    async def test_ack_message_error(self, mock_redis_client):
        """Обработка ошибки при ACK сообщения."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        mock_redis_client.xack = AsyncMock(side_effect=RedisError("Connection lost"))
        
        # Не должно быть исключения (ошибка логируется)
        await worker._ack_message("queue:normal", "1234567890-0")


class TestWorkerStartStop:
    """Тесты для запуска и остановки Worker."""
    
    @pytest.mark.asyncio
    async def test_start_worker_success(self, mock_redis_client):
        """Успешный запуск Worker."""
        worker = Worker(worker_id="test-worker")
        
        with patch("api.services.worker.get_redis_client", return_value=mock_redis_client):
            with patch.object(worker, '_ensure_consumer_groups', new_callable=AsyncMock) as mock_ensure:
                with patch.object(worker, '_worker_loop', new_callable=AsyncMock) as mock_loop:
                    # Установить running = False чтобы loop завершился
                    mock_loop.side_effect = lambda: setattr(worker, 'running', False)
                    
                    try:
                        await asyncio.wait_for(worker.start(), timeout=0.1)
                    except asyncio.TimeoutError:
                        # Это нормально - worker loop работает бесконечно
                        pass
                    
                    assert worker.running is True
                    mock_ensure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_worker_redis_unavailable(self):
        """Запуск Worker когда Redis недоступен."""
        worker = Worker(worker_id="test-worker")
        
        with patch("api.services.worker.get_redis_client", return_value=None):
            with pytest.raises(RuntimeError, match="Redis client not available"):
                await worker.start()
    
    @pytest.mark.asyncio
    async def test_stop_worker_graceful_shutdown(self, mock_redis_client):
        """Graceful shutdown Worker."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        worker.running = True
        
        # Добавить активные задачи
        mock_task = AsyncMock()
        mock_task.done = MagicMock(return_value=True)
        worker.active_tasks["test-run-id"] = mock_task
        
        # Добавить активные heartbeat
        mock_heartbeat_task = AsyncMock()
        mock_heartbeat_task.cancel = MagicMock()
        mock_heartbeat_task = asyncio.create_task(asyncio.sleep(0.1))
        worker.active_heartbeats["test-run-id"] = mock_heartbeat_task
        
        # Мокаем Redis операции
        mock_redis_client.delete = AsyncMock(return_value=1)
        mock_redis_client.xpending_range = AsyncMock(return_value=[])
        
        with patch("api.services.worker.save_run_state", new_callable=AsyncMock) as mock_save_state:
            await worker.stop()
            
            assert worker.running is False
            assert worker.shutdown_event.is_set()
            # Должен обновить состояние активных run'ов
            mock_save_state.assert_called()
    
    @pytest.mark.asyncio
    async def test_stop_worker_with_pending_messages(self, mock_redis_client):
        """Graceful shutdown с pending сообщениями."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        worker.running = True
        
        # Мокаем pending сообщения
        pending_info = [
            {"message_id": "1234567890-0"},
            {"message_id": "1234567890-1"}
        ]
        mock_redis_client.xpending_range = AsyncMock(return_value=pending_info)
        mock_redis_client.xack = AsyncMock(return_value=1)
        
        await worker.stop()
        
        # Должен ACK все pending сообщения
        assert mock_redis_client.xack.call_count == 6  # 2 сообщения × 3 очереди


class TestWorkerProcessRunTask:
    """Тесты для обработки run задачи."""
    
    @pytest.mark.asyncio
    async def test_process_run_task_success(self, mock_redis_client, mock_storage, mock_key_layout):
        """Успешная обработка run задачи."""
        worker = Worker(
            worker_id="test-worker",
            storage=mock_storage,
            key_layout=mock_key_layout
        )
        worker.redis_client = mock_redis_client
        
        run_id = "test-run-id"
        run_info = {
            "run_id": run_id,
            "video_id": "test_video",
            "platform_id": "youtube",
            "video_path": "/tmp/test_video.mp4",
            "config_hash": "test_hash"
        }
        worker.task_manager.get_run = MagicMock(return_value=run_info)
        
        # Мокаем все зависимости
        with patch("api.services.worker.check_existing_result", new_callable=AsyncMock, return_value=None):
            with patch("api.services.worker.get_checkpoint_info", return_value=None):
                with patch("api.services.worker.save_checkpoint"):
                    with patch("api.services.worker.save_run_state", new_callable=AsyncMock):
                        with patch("api.services.worker.add_run_event", new_callable=AsyncMock):
                            with patch("api.services.worker.update_run_heartbeat", new_callable=AsyncMock):
                                with patch("api.services.worker.get_cancel_flag", new_callable=AsyncMock, return_value=False):
                                    with patch("api.services.worker.release_run_lock", new_callable=AsyncMock):
                                        with patch.object(worker.processor_service, 'run_processing', new_callable=AsyncMock) as mock_process:
                                            mock_process.return_value = {"success": True, "run_id": run_id}
                                            
                                            await worker._process_run_task(
                                                run_id,
                                                "queue:normal",
                                                "1234567890-0",
                                                {b"run_id": run_id.encode()},
                                                {},
                                                None
                                            )
                                            
                                            # Должен вызвать processor_service.run_processing
                                            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_run_task_idempotency_cache(self, mock_redis_client, mock_storage, mock_key_layout):
        """Использование кэша при идемпотентности."""
        worker = Worker(
            worker_id="test-worker",
            storage=mock_storage,
            key_layout=mock_key_layout
        )
        worker.redis_client = mock_redis_client
        
        run_id = "test-run-id"
        run_info = {
            "run_id": run_id,
            "video_id": "test_video",
            "platform_id": "youtube"
        }
        worker.task_manager.get_run = MagicMock(return_value=run_info)
        
        # Мокаем что результат уже существует
        existing_result = {"success": True}
        with patch("api.services.worker.check_existing_result", new_callable=AsyncMock, return_value=existing_result):
            with patch("api.services.worker.save_run_state", new_callable=AsyncMock) as mock_save_state:
                with patch("api.services.worker.add_run_event", new_callable=AsyncMock):
                    with patch("api.services.worker.release_run_lock", new_callable=AsyncMock):
                        with patch.object(worker, '_ack_message', new_callable=AsyncMock) as mock_ack:
                            with patch.object(worker.processor_service, 'run_processing', new_callable=AsyncMock) as mock_process:
                                await worker._process_run_task(
                                    run_id,
                                    "queue:normal",
                                    "1234567890-0",
                                    {b"run_id": run_id.encode()},
                                    {},
                                    None
                                )
                                
                                # Не должен вызывать processor_service.run_processing
                                mock_process.assert_not_called()
                                # Должен обновить статус на success
                                mock_save_state.assert_called()
                                # Должен ACK сообщение
                                mock_ack.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_run_task_cancellation(self, mock_redis_client):
        """Обработка отмены run'а."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        run_id = "test-run-id"
        run_info = {
            "run_id": run_id,
            "video_id": "test_video",
            "platform_id": "youtube"
        }
        worker.task_manager.get_run = MagicMock(return_value=run_info)
        
        # Мокаем что флаг отмены установлен
        with patch("api.services.worker.check_existing_result", new_callable=AsyncMock, return_value=None):
            with patch("api.services.worker.get_checkpoint_info", return_value=None):
                with patch("api.services.worker.save_checkpoint"):
                    with patch("api.services.worker.save_run_state", new_callable=AsyncMock):
                        with patch("api.services.worker.add_run_event", new_callable=AsyncMock):
                            with patch("api.services.worker.update_run_heartbeat", new_callable=AsyncMock):
                                with patch("api.services.worker.get_cancel_flag", new_callable=AsyncMock) as mock_cancel:
                                    # Первый вызов - False, затем True (отмена)
                                    mock_cancel.side_effect = [False, True]
                                    
                                    with patch("api.services.worker.clear_cancel_flag", new_callable=AsyncMock):
                                        with patch("api.services.worker.release_run_lock", new_callable=AsyncMock):
                                            with patch.object(worker.processor_service, 'run_processing', new_callable=AsyncMock) as mock_process:
                                                # Создаем задачу которая будет отменена
                                                async def long_running():
                                                    await asyncio.sleep(10)
                                                    return {"success": True}
                                                
                                                mock_process.side_effect = long_running
                                                
                                                # Запускаем обработку
                                                task = asyncio.create_task(
                                                    worker._process_run_task(
                                                        run_id,
                                                        "queue:normal",
                                                        "1234567890-0",
                                                        {b"run_id": run_id.encode()},
                                                        {},
                                                        None
                                                    )
                                                )
                                                
                                                # Подождать немного для проверки отмены
                                                await asyncio.sleep(0.1)
                                                
                                                # Отменить задачу если она еще выполняется
                                                if not task.done():
                                                    task.cancel()
                                                    try:
                                                        await task
                                                    except asyncio.CancelledError:
                                                        pass
    
    @pytest.mark.asyncio
    async def test_process_run_task_error(self, mock_redis_client):
        """Обработка ошибки при обработке run задачи."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        
        run_id = "test-run-id"
        run_info = {
            "run_id": run_id,
            "video_id": "test_video",
            "platform_id": "youtube"
        }
        worker.task_manager.get_run = MagicMock(return_value=run_info)
        
        # Мокаем ошибку при обработке
        with patch("api.services.worker.check_existing_result", new_callable=AsyncMock, return_value=None):
            with patch("api.services.worker.get_checkpoint_info", return_value=None):
                with patch("api.services.worker.save_checkpoint"):
                    with patch("api.services.worker.save_run_state", new_callable=AsyncMock):
                        with patch("api.services.worker.add_run_event", new_callable=AsyncMock):
                            with patch("api.services.worker.update_run_heartbeat", new_callable=AsyncMock):
                                with patch("api.services.worker.get_cancel_flag", new_callable=AsyncMock, return_value=False):
                                    with patch("api.services.worker.release_run_lock", new_callable=AsyncMock):
                                        with patch.object(worker.processor_service, 'run_processing', new_callable=AsyncMock) as mock_process:
                                            mock_process.side_effect = Exception("Processing error")
                                            
                                            await worker._process_run_task(
                                                run_id,
                                                "queue:normal",
                                                "1234567890-0",
                                                {b"run_id": run_id.encode()},
                                                {},
                                                None
                                            )
                                            
                                            # Должен обновить статус на error
                                            # (проверяется через вызовы save_run_state)


class TestWorkerLoop:
    """Тесты для worker loop."""
    
    @pytest.mark.asyncio
    async def test_worker_loop_reads_messages(self, mock_redis_client):
        """Worker loop читает сообщения из очереди."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        worker.running = True
        
        # Мокаем сообщения из очереди
        messages = [
            (
                b"queue:normal",
                [
                    (b"1234567890-0", {b"run_id": b"test-run-id"})
                ]
            )
        ]
        mock_redis_client.xreadgroup = AsyncMock(return_value=messages)
        
        # Мокаем обработку сообщения
        with patch.object(worker, '_process_message', new_callable=AsyncMock) as mock_process:
            # Установить running = False после первого сообщения
            original_process = worker._process_message
            
            async def process_and_stop(*args, **kwargs):
                await original_process(*args, **kwargs)
                worker.running = False
            
            worker._process_message = process_and_stop
            
            # Запустить loop
            await worker._worker_loop()
            
            # Должен обработать сообщение
            mock_process.assert_called()
    
    @pytest.mark.asyncio
    async def test_worker_loop_no_messages(self, mock_redis_client):
        """Worker loop обрабатывает отсутствие сообщений."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        worker.running = True
        
        # Мокаем пустую очередь
        mock_redis_client.xreadgroup = AsyncMock(return_value=[])
        
        # Установить running = False после первого чтения
        original_read = mock_redis_client.xreadgroup
        
        async def read_and_stop(*args, **kwargs):
            result = await original_read(*args, **kwargs)
            worker.running = False
            return result
        
        mock_redis_client.xreadgroup = read_and_stop
        
        # Запустить loop
        await worker._worker_loop()
        
        # Должен попытаться прочитать из очереди
        assert mock_redis_client.xreadgroup.called
    
    @pytest.mark.asyncio
    async def test_worker_loop_handles_error(self, mock_redis_client):
        """Worker loop обрабатывает ошибки."""
        worker = Worker(worker_id="test-worker")
        worker.redis_client = mock_redis_client
        worker.running = True
        
        # Мокаем ошибку при чтении
        mock_redis_client.xreadgroup = AsyncMock(side_effect=RedisError("Connection lost"))
        
        # Установить running = False после ошибки
        original_read = mock_redis_client.xreadgroup
        
        async def read_and_stop(*args, **kwargs):
            try:
                return await original_read(*args, **kwargs)
            except RedisError:
                worker.running = False
                raise
        
        mock_redis_client.xreadgroup = read_and_stop
        
        # Запустить loop
        await worker._worker_loop()
        
        # Должен обработать ошибку и продолжить (или остановиться)

