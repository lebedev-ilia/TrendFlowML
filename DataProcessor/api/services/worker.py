"""
Worker Service - обработка задач из Redis Streams queue

Этот модуль реализует worker loop для обработки задач из Redis Streams.
Использует consumer groups для масштабирования и ACK механизм для надежности.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 686-728)
"""

import logging
import asyncio
import time
import json
from typing import Optional, Dict, Any
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from api.services.redis_client import get_redis_client
from api.services.processor import ProcessorService
from api.services.task_manager import TaskManager
from api.dependencies import get_task_manager
from api.services.redis_schema import (
    update_run_heartbeat,
    save_run_state,
    add_run_event,
    release_run_lock
)
from api.services.checkpoint import (
    save_checkpoint,
    get_checkpoint_info_async,
)
from api.schemas.requests import ProcessRequest
from api.schemas.state import RunStatus
from storage.base import Storage, StorageError, NotFoundError
from storage.paths import KeyLayout
from api.utils.logging import get_logger

logger = get_logger(__name__)

PROCESS_REQUEST_FIELDS = (
    "video_id",
    "platform_id",
    "video_path",
    "video_url",
    "config_hash",
    "profile_config",
    "profile_version",
    "feature_schema_version",
    "pipeline_version",
    "sampling_policy_version",
    "dataprocessor_version",
    "analysis_fps",
    "analysis_width",
    "analysis_height",
    "chunk_size",
    "visual_cfg_path",
    "dag_path",
    "dag_stage",
    "rs_base",
    "output",
    "run_audio",
    "run_text",
    "global_config_path",
)

# Имена очередей
QUEUE_HIGH = "queue:high"
QUEUE_NORMAL = "queue:normal"
QUEUE_LOW = "queue:low"

# Consumer group имя
CONSUMER_GROUP_NAME = "workers"

# Таймаут для чтения из очереди (мс)
STREAM_READ_TIMEOUT = 5000  # 5 секунд

# Максимальное количество сообщений за один read
STREAM_READ_COUNT = 1


class Worker:
    """
    Worker для обработки задач из Redis Streams queue.
    
    Использует consumer groups для масштабирования и ACK механизм для надежности.
    Поддерживает heartbeat для обнаружения crashed run'ов и recovery механизм.
    
    Attributes:
        worker_id: Уникальный ID worker'а
        consumer_name: Имя consumer'а для Redis consumer groups
        processor_service: Сервис для запуска обработки
        task_manager: Менеджер задач для отслеживания состояния
        storage: Storage для checkpoint support (опционально)
        key_layout: KeyLayout для checkpoint support (опционально)
        redis_client: Redis клиент для работы с очередями
        running: Флаг работы worker'а
        shutdown_event: Event для graceful shutdown
        active_heartbeats: Словарь активных heartbeat задач {run_id: task}
        active_tasks: Словарь активных задач обработки {run_id: task}
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 686-728)
    
    Example:
        ```python
        worker = Worker(
            worker_id="worker-1",
            storage=storage,
            key_layout=key_layout
        )
        await worker.start()  # Запустить worker loop
        # ... worker обрабатывает задачи ...
        await worker.stop()  # Graceful shutdown
        ```
    """
    
    def __init__(
        self,
        worker_id: str,
        storage: Optional[Storage] = None,
        key_layout: Optional[KeyLayout] = None,
        processor_service: Optional[ProcessorService] = None,
        task_manager: Optional[TaskManager] = None
    ):
        """
        Инициализация Worker.
        
        Args:
            worker_id: Уникальный ID worker'а (например, "worker-1")
            storage: Storage для checkpoint support (опционально)
            key_layout: KeyLayout для checkpoint support (опционально)
            processor_service: Сервис для запуска обработки (опционально, создается по умолчанию)
            task_manager: Менеджер задач для отслеживания состояния (опционально, создается по умолчанию)
            
        Note:
            Если storage и key_layout не предоставлены, checkpoint функциональность будет недоступна.
        """
        self.worker_id = worker_id
        self.consumer_name = f"worker-{worker_id}"
        self.processor_service = processor_service or ProcessorService()
        # Используем общий TaskManager из dependencies, чтобы разделять состояние run'ов
        self.task_manager = task_manager or get_task_manager()
        self.storage = storage
        self.key_layout = key_layout
        self.running = False
        self.shutdown_event = asyncio.Event()  # Event для graceful shutdown
        self.redis_client: Optional[Redis] = None
        self.active_heartbeats: Dict[str, asyncio.Task] = {}  # run_id -> heartbeat task
        self.active_tasks: Dict[str, asyncio.Task] = {}  # run_id -> processing task

    @staticmethod
    def _build_process_request_data(run_id: str, *sources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge queue/redis/task-manager metadata into a ProcessRequest payload."""
        request_data: Dict[str, Any] = {"run_id": run_id}
        for source in sources:
            if not source or not isinstance(source, dict):
                continue
            for field in PROCESS_REQUEST_FIELDS:
                value = source.get(field)
                if value is not None:
                    request_data[field] = value
        return request_data
    
    async def start(self) -> None:
        """
        Запустить worker loop.
        
        Инициализирует Redis клиент, создает consumer groups для всех очередей
        и запускает цикл обработки задач из Redis Streams.
        
        Raises:
            RuntimeError: Если Redis клиент недоступен
            
        Note:
            Метод блокирующий - будет работать до вызова stop() или получения сигнала shutdown.
        """
        self.redis_client = get_redis_client()
        if not self.redis_client:
            raise RuntimeError("Redis client not available, cannot start worker")
        
        # Создать consumer groups для всех очередей
        await self._ensure_consumer_groups()
        
        self.running = True
        logger.info(f"Worker {self.worker_id} started")
        
        # Запустить worker loop
        await self._worker_loop()
    
    async def stop(self) -> None:
        """
        Остановить worker loop с graceful shutdown.
        
        Выполняет следующие шаги:
        1. Stop accepting new tasks
        2. Finish current tasks
        3. Update state (пометить активные run'ы как recovering)
        4. Remove heartbeat
        5. ACK queue (вернуть pending задачи в queue)
        
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2287-2317)
        """
        logger.info(f"Worker {self.worker_id} initiating graceful shutdown...")
        
        # 1. Stop accepting new tasks
        self.running = False
        self.shutdown_event.set()
        logger.info("Stopped accepting new tasks")
        
        # 2. Finish current tasks
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to finish...")
            # Подождать завершения активных задач с таймаутом
            wait_tasks = list(self.active_tasks.values())
            try:
                await asyncio.wait_for(
                    asyncio.gather(*wait_tasks, return_exceptions=True),
                    timeout=300  # Максимум 5 минут на завершение задач
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks to finish, cancelling remaining tasks")
                for task in wait_tasks:
                    if not task.done():
                        task.cancel()
            logger.info("All active tasks finished")
        
        # 3. Update state - пометить активные run'ы как recovering
        if self.redis_client:
            active_run_ids = list(self.active_tasks.keys())
            if active_run_ids:
                logger.info(f"Updating state for {len(active_run_ids)} active runs to 'recovering'")
                from api.services.redis_schema import save_run_state
                for run_id in active_run_ids:
                    try:
                        await save_run_state(run_id, {
                            "status": RunStatus.RECOVERING.value,
                            "updated_at": time.time(),
                            "shutdown_reason": "worker_shutdown"
                        })
                    except (RedisError, ConnectionError, TimeoutError) as e:
                        logger.warning(
                            "Failed to update state for run during shutdown",
                            run_id=run_id,
                            worker_id=self.worker_id,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                    except Exception as e:
                        logger.warning(
                            "Unexpected error updating state for run during shutdown",
                            run_id=run_id,
                            worker_id=self.worker_id,
                            error=str(e),
                            error_type=type(e).__name__
                        )
        
        # 4. Remove heartbeat
        if self.active_heartbeats:
            logger.info(f"Stopping {len(self.active_heartbeats)} heartbeat loops...")
            for run_id, heartbeat_task in self.active_heartbeats.items():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Удалить heartbeat записи из Redis
            if self.redis_client:
                from api.services.redis_schema import KEY_PREFIX_HEARTBEAT
                for run_id in self.active_heartbeats.keys():
                    try:
                        await self.redis_client.delete(f"{KEY_PREFIX_HEARTBEAT}{run_id}")
                    except (RedisError, ConnectionError, TimeoutError) as e:
                        logger.warning(
                            "Failed to remove heartbeat during shutdown",
                            run_id=run_id,
                            worker_id=self.worker_id,
                            error=str(e),
                            error_type=type(e).__name__
                        )
                    except Exception as e:
                        logger.warning(
                            "Unexpected error removing heartbeat during shutdown",
                            run_id=run_id,
                            worker_id=self.worker_id,
                            error=str(e),
                            error_type=type(e).__name__
                        )
            
            self.active_heartbeats.clear()
            logger.info("All heartbeats stopped")
        
        # 5. ACK queue (вернуть pending задачи в queue)
        if self.redis_client:
            queues = [QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW]
            for queue_name in queues:
                try:
                    # Получить pending сообщения для этого consumer
                    pending_info = await self.redis_client.xpending_range(
                        queue_name,
                        CONSUMER_GROUP_NAME,
                        min="-",
                        max="+",
                        count=100,
                        consumername=self.consumer_name
                    )
                    
                    if pending_info:
                        logger.info(f"ACKing {len(pending_info)} pending messages from {queue_name}")
                        for msg_info in pending_info:
                            msg_id = msg_info.get("message_id")
                            if msg_id:
                                try:
                                    # ACK сообщение - оно вернется в queue для обработки другим worker'ом
                                    await self.redis_client.xack(
                                        queue_name,
                                        CONSUMER_GROUP_NAME,
                                        msg_id
                                    )
                                except (RedisError, ResponseError, ConnectionError, TimeoutError) as e:
                                    logger.warning(
                                        "Failed to ACK message during shutdown",
                                        worker_id=self.worker_id,
                                        queue_name=queue_name,
                                        message_id=msg_id,
                                        error=str(e),
                                        error_type=type(e).__name__
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Unexpected error ACKing message during shutdown",
                                        worker_id=self.worker_id,
                                        queue_name=queue_name,
                                        message_id=msg_id,
                                        error=str(e),
                                        error_type=type(e).__name__
                                    )
                except (RedisError, ResponseError, ConnectionError, TimeoutError) as e:
                    logger.warning(
                        "Error processing pending messages during shutdown",
                        worker_id=self.worker_id,
                        queue_name=queue_name,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                except Exception as e:
                    logger.warning(
                        "Unexpected error processing pending messages during shutdown",
                        worker_id=self.worker_id,
                        queue_name=queue_name,
                        error=str(e),
                        error_type=type(e).__name__
                    )
        
        logger.info(f"Worker {self.worker_id} stopped gracefully")
    
    async def _ensure_consumer_groups(self) -> None:
        """
        Создать consumer groups для всех приоритетных очередей.
        
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 691-696)
        """
        queues = [QUEUE_HIGH, QUEUE_NORMAL, QUEUE_LOW]
        
        for queue_name in queues:
            try:
                # Создать consumer group с начальным ID "0" (читать все сообщения)
                await self.redis_client.xgroup_create(
                    queue_name,
                    CONSUMER_GROUP_NAME,
                    id="0",
                    mkstream=True  # Создать stream если не существует
                )
                logger.info(f"Created consumer group '{CONSUMER_GROUP_NAME}' for {queue_name}")
            except ResponseError as e:
                # Group уже существует - это нормально
                if "BUSYGROUP" in str(e):
                    logger.debug(
                        "Consumer group already exists",
                        worker_id=self.worker_id,
                        queue_name=queue_name,
                        group_name=CONSUMER_GROUP_NAME
                    )
                else:
                    logger.warning(
                        "Error creating consumer group",
                        worker_id=self.worker_id,
                        queue_name=queue_name,
                        group_name=CONSUMER_GROUP_NAME,
                        error=str(e),
                        error_type=type(e).__name__
                    )
            except (RedisError, ConnectionError, TimeoutError) as e:
                logger.error(
                    "Redis error creating consumer group",
                    worker_id=self.worker_id,
                    queue_name=queue_name,
                    group_name=CONSUMER_GROUP_NAME,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
            except Exception as e:
                logger.exception(
                    "Unexpected error creating consumer group",
                    worker_id=self.worker_id,
                    queue_name=queue_name,
                    group_name=CONSUMER_GROUP_NAME,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
    
    async def _worker_loop(self) -> None:
        """
        Основной цикл worker'а для обработки задач из очереди.
        
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 698-728)
        """
        while self.running and not self.shutdown_event.is_set():
            try:
                # Читать из всех приоритетных очередей
                # Используем ">" для чтения только новых сообщений
                streams = {
                    QUEUE_HIGH: ">",
                    QUEUE_NORMAL: ">",
                    QUEUE_LOW: ">"
                }
                
                # Чтение из consumer group
                messages = await self.redis_client.xreadgroup(
                    CONSUMER_GROUP_NAME,
                    self.consumer_name,
                    streams,
                    count=STREAM_READ_COUNT,
                    block=STREAM_READ_TIMEOUT
                )
                
                # Обработать полученные сообщения
                if messages:
                    for stream_name, messages_list in messages:
                        stream_name_str = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
                        
                        for msg_id, data in messages_list:
                            msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                            
                            try:
                                await self._process_message(stream_name_str, msg_id_str, data)
                            except (RedisError, ResponseError, ConnectionError, TimeoutError) as e:
                                logger.exception(
                                    "Redis error processing message",
                                    worker_id=self.worker_id,
                                    stream_name=stream_name_str,
                                    message_id=msg_id_str,
                                    error=str(e),
                                    error_type=type(e).__name__
                                )
                                # Сообщение останется в pending для retry
                            except Exception as e:
                                logger.exception(
                                    "Unexpected error processing message",
                                    worker_id=self.worker_id,
                                    stream_name=stream_name_str,
                                    message_id=msg_id_str,
                                    error=str(e),
                                    error_type=type(e).__name__
                                )
                                # Сообщение останется в pending для retry
                
            except asyncio.CancelledError:
                logger.info(
                    "Worker cancelled",
                    worker_id=self.worker_id,
                )
                break
            except (RedisError, ConnectionError, TimeoutError) as e:
                # Таймауты и временные ошибки Redis считаем восстановимыми:
                # логируем warning и продолжаем цикл, чтобы worker не падал.
                logger.warning(
                    "Redis error in worker loop",
                    worker_id=self.worker_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Небольшая задержка перед повтором при ошибке
                await asyncio.sleep(1)
            except Exception as e:
                logger.exception(
                    "Unexpected error in worker loop",
                    worker_id=self.worker_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
                # Небольшая задержка перед повтором при ошибке
                await asyncio.sleep(1)
    
    async def _process_message(
        self,
        stream_name: str,
        message_id: str,
        data: Dict[bytes, bytes]
    ) -> None:
        """
        Обработать одно сообщение из очереди.
        
        Args:
            stream_name: Имя stream'а
            message_id: ID сообщения
            data: Данные сообщения
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 714-728)
        """
        # Извлечь run_id из данных
        run_id_bytes = data.get(b"run_id")
        if not run_id_bytes:
            logger.error(f"Message {message_id} from {stream_name} missing run_id")
            # ACK даже если данные невалидны, чтобы не застрять
            await self._ack_message(stream_name, message_id)
            return
        
        run_id = run_id_bytes.decode("utf-8")
        
        # Извлечь метаданные если есть
        metadata = {}
        if b"metadata" in data:
            try:
                metadata = json.loads(data[b"metadata"].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as e:
                logger.warning(
                    "Failed to parse metadata for run",
                    run_id=run_id,
                    worker_id=self.worker_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
            except Exception as e:
                logger.warning(
                    "Unexpected error parsing metadata for run",
                    run_id=run_id,
                    worker_id=self.worker_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
        
        message_id_str = message_id.decode() if isinstance(message_id, bytes) else str(message_id)
        stream_name_str = stream_name.decode() if isinstance(stream_name, bytes) else str(stream_name)
        
        logger.info(
            "Processing run from queue",
            run_id=run_id,
            stream_name=stream_name_str,
            message_id=message_id_str
        )
        
        # Проверить shutdown event перед обработкой
        if self.shutdown_event.is_set():
            logger.info(
                "Shutdown event set, skipping run",
                run_id=run_id
            )
            await self._ack_message(stream_name, message_id)
            return
        
        # Измерить время ожидания в очереди (если есть timestamp в сообщении)
        queue_wait_start = None
        if b"ts" in data:
            try:
                queue_wait_start = float(data[b"ts"].decode("utf-8"))
            except (ValueError, KeyError, UnicodeDecodeError):
                # Игнорируем ошибки парсинга timestamp - это не критично
                pass
        
        # Создать задачу для обработки run'а
        processing_task = asyncio.create_task(self._process_run_task(
            run_id, stream_name, message_id, data, metadata, queue_wait_start
        ))
        self.active_tasks[run_id] = processing_task
        
        try:
            await processing_task
        finally:
            # Удалить задачу из активных после завершения
            self.active_tasks.pop(run_id, None)
    
    async def _process_run_task(
        self,
        run_id: str,
        stream_name: str,
        message_id: str,
        data: Dict[bytes, bytes],
        metadata: Dict[str, Any],
        queue_wait_start: Optional[float]
    ) -> None:
        """
        Обработать run задачу.
        
        Вынесено в отдельный метод для отслеживания активных задач.
        
        Args:
            run_id: UUID run'а
            stream_name: Имя stream'а
            message_id: ID сообщения
            data: Данные сообщения
            metadata: Метаданные из сообщения
            queue_wait_start: Timestamp начала ожидания в очереди
        """
        # Инициализировать переменные для контекста логирования
        video_id = None
        platform_id = None
        
        try:
            # Получить информацию о run из TaskManager.
            # Если API и worker работают в разных процессах, run может отсутствовать
            # в локальном in-memory TaskManager worker'а.
            run_info = self.task_manager.get_run(run_id)
            if not run_info:
                # Попробовать восстановить метаданные run'а из Redis (run:meta:{run_id})
                redis_metadata: Dict[str, Any] | None = None
                try:
                    from api.services.redis_schema import get_run_metadata

                    redis_metadata = await get_run_metadata(run_id) or None
                except Exception as e:
                    logger.warning(
                        "Failed to load run metadata from Redis",
                        run_id=run_id,
                        worker_id=self.worker_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                # Объединить метаданные из Redis и из сообщения очереди
                combined_metadata: Dict[str, Any] = {}
                if redis_metadata and isinstance(redis_metadata, dict):
                    combined_metadata.update(redis_metadata)
                if metadata and isinstance(metadata, dict):
                    combined_metadata.update(metadata)

                reconstructed_metadata = self._build_process_request_data(
                    run_id,
                    combined_metadata,
                )
                reconstructed_metadata.pop("run_id", None)

                if reconstructed_metadata:
                    self.task_manager.register_run(
                        run_id,
                        reconstructed_metadata,
                        initial_status=RunStatus.QUEUED,
                    )
                    run_info = self.task_manager.get_run(run_id)
                    logger.info(
                        "Run registered in TaskManager from Redis/queue metadata",
                        run_id=run_id,
                        worker_id=self.worker_id,
                    )
                else:
                    logger.warning(
                        "Run not found in TaskManager and cannot reconstruct metadata, skipping",
                        run_id=run_id,
                        worker_id=self.worker_id,
                    )
                    await self._ack_message(stream_name, message_id)
                    return

            
            # Извлечь контекст для логирования
            metadata = metadata or {}
            video_id = run_info.get("video_id") or metadata.get("video_id")
            platform_id = run_info.get("platform_id") or metadata.get("platform_id")
            
            # Подготовить данные для ProcessRequest
            # ProcessRequest будет создан в processor_service, но нам нужны данные
            request_data = self._build_process_request_data(run_id, run_info, metadata)
            
            # Проверить существующий результат для идемпотентности
            # Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2257-2268)
            if self.storage and self.key_layout and platform_id and video_id:
                from api.services.idempotency import check_existing_result
                
                existing_result = await check_existing_result(
                    self.storage,
                    self.key_layout,
                    platform_id,
                    video_id,
                    run_id
                )
                
                if existing_result and existing_result.get("success"):
                    # Run уже полностью обработан - использовать кэш (идемпотентность)
                    logger.info(
                        "Run already completed, using cached result (idempotency)",
                        run_id=run_id,
                        video_id=video_id,
                        platform_id=platform_id
                    )
                    
                    # Обновить статус на success
                    finished_at = time.time()
                    self.task_manager.update_run_status(
                        run_id,
                        RunStatus.SUCCESS,
                        finished_at=finished_at
                    )
                    await save_run_state(run_id, {
                        "status": RunStatus.SUCCESS.value,
                        "finished_at": finished_at,
                        "updated_at": finished_at,
                        "from_cache": True
                    })
                    await add_run_event(run_id, "processing_completed_from_cache", {
                        "run_id": run_id,
                        "timestamp": finished_at,
                        "message": "Run already completed, using cached result"
                    })
                    
                    # ACK сообщение и завершить обработку
                    await self._ack_message(stream_name, message_id)
                    return
            
            # Проверить checkpoint перед запуском
            is_resume = False
            last_processor = None
            
            if self.storage and self.key_layout and platform_id and video_id:
                checkpoint_info = await get_checkpoint_info_async(
                    self.storage,
                    self.key_layout,
                    platform_id,
                    video_id,
                    run_id
                )
                
                if checkpoint_info and checkpoint_info.get("can_resume"):
                    is_resume = True
                    last_processor = checkpoint_info.get("last_processor")
                    logger.info(
                        "Resuming run from checkpoint",
                        run_id=run_id,
                        video_id=video_id,
                        platform_id=platform_id,
                        last_processor=last_processor
                    )
                    await add_run_event(run_id, "resume_from_checkpoint", {
                        "run_id": run_id,
                        "last_processor": last_processor,
                        "timestamp": time.time()
                    })
                elif checkpoint_info:
                    # Есть checkpoint, но нельзя resume (например, завершен)
                    logger.debug(f"Checkpoint exists for run {run_id}, but cannot resume")
            
            # Создать ProcessRequest из данных
            from api.schemas.requests import ProcessRequest
            request = ProcessRequest(**request_data)
            
            # Обновить статус на "running"
            started_at = time.time()
            self.task_manager.update_run_status(run_id, RunStatus.RUNNING, started_at=started_at)
            
            # Обновить метрику времени ожидания в очереди
            if queue_wait_start:
                try:
                    from api.services.metrics import queue_wait_time
                    wait_time = started_at - queue_wait_start
                    queue_wait_time.observe(wait_time)
                except (AttributeError, ImportError, ValueError) as e:
                    logger.debug(
                        "Failed to update queue_wait_time metric",
                        run_id=run_id,
                        worker_id=self.worker_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                except Exception as e:
                    logger.debug(
                        "Unexpected error updating queue_wait_time metric",
                        run_id=run_id,
                        worker_id=self.worker_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
            
            # Сохранить checkpoint перед запуском (если есть storage)
            if self.storage and self.key_layout and platform_id and video_id:
                await save_checkpoint(
                    self.storage,
                    self.key_layout,
                    platform_id,
                    video_id,
                    run_id,
                    last_processor=last_processor,
                    status=RunStatus.RUNNING.value
                )
            
            # Сохранить состояние в Redis
            await save_run_state(run_id, {
                "status": RunStatus.RUNNING.value,
                "started_at": started_at,
                "updated_at": time.time()
            })
            
            # Добавить событие начала обработки
            await add_run_event(run_id, "processing_started", {
                "run_id": run_id,
                "timestamp": started_at
            })
            
            # Обновить heartbeat (начальное обновление)
            await update_run_heartbeat(run_id)
            
            # Запустить heartbeat loop для этого run'а
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(run_id)
            )
            self.active_heartbeats[run_id] = heartbeat_task
            
            # Запустить обработку с измерением времени и проверкой отмены
            processing_start = time.time()
            
            # Создать задачу для обработки с возможностью отмены
            processing_task = asyncio.create_task(
                self.processor_service.run_processing(request),
                name=f"process_{run_id}"
            )
            
            # Периодически проверять флаг отмены во время обработки
            cancel_check_interval = 5.0  # Проверять каждые 5 секунд
            result = None
            cancelled = False
            
            while not processing_task.done():
                # Проверить флаг отмены
                from api.services.redis_schema import get_cancel_flag
                if await get_cancel_flag(run_id):
                    logger.info(
                        "Run cancelled, stopping processing",
                        run_id=run_id,
                        video_id=video_id,
                        platform_id=platform_id
                    )
                    cancelled = True
                    
                    # Отменить задачу обработки
                    processing_task.cancel()
                    try:
                        await processing_task
                    except asyncio.CancelledError:
                        pass
                    
                    # Обновить статус на cancelled
                    finished_at = time.time()
                    self.task_manager.update_run_status(run_id, RunStatus.CANCELLED, finished_at=finished_at)
                    await save_run_state(run_id, {
                        "status": RunStatus.CANCELLED.value,
                        "finished_at": finished_at,
                        "updated_at": finished_at,
                        "cancelled_at": finished_at
                    })
                    await add_run_event(run_id, "processing_cancelled", {
                        "run_id": run_id,
                        "timestamp": finished_at
                    })
                    
                    # Очистить флаг отмены
                    from api.services.redis_schema import clear_cancel_flag
                    await clear_cancel_flag(run_id)
                    
                    logger.info(
                        "Run cancelled successfully",
                        run_id=run_id,
                        video_id=video_id,
                        platform_id=platform_id
                    )
                    break
                
                # Подождать перед следующей проверкой
                await asyncio.sleep(cancel_check_interval)
            
            # Если не была отмена, получить результат
            if not cancelled:
                result = await processing_task
            else:
                # Создать результат для отмененного run'а
                result = {
                    "success": False,
                    "run_id": run_id,
                    "error": "Run was cancelled",
                    "error_type": "cancelled"
                }
            
            processing_end = time.time()
            processing_duration = processing_end - processing_start
            
            # Обновить метрику времени обработки
            try:
                from api.services.metrics import processing_time
                # Определить процессор и компонент из результата или метаданных
                processor = result.get("processor") or "unknown"
                component = result.get("component") or "unknown"
                processing_time.labels(processor=processor, component=component).observe(processing_duration)
            except (AttributeError, ImportError, ValueError) as e:
                logger.debug(
                    "Failed to update processing_time metric",
                    run_id=run_id,
                    worker_id=self.worker_id,
                    video_id=video_id,
                    platform_id=platform_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
            except Exception as e:
                logger.debug(
                    "Unexpected error updating processing_time metric",
                    run_id=run_id,
                    worker_id=self.worker_id,
                    video_id=video_id,
                    platform_id=platform_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
            
            # Остановить heartbeat loop после завершения обработки
            if run_id in self.active_heartbeats:
                heartbeat_task = self.active_heartbeats.pop(run_id)
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Обновить статус на основе результата (если не была отмена)
            if not cancelled:
                finished_at = time.time()
                if result.get("success"):
                    self.task_manager.update_run_status(
                        run_id,
                        RunStatus.SUCCESS,
                        finished_at=finished_at
                    )
                    await save_run_state(run_id, {
                        "status": RunStatus.SUCCESS.value,
                        "finished_at": finished_at,
                        "updated_at": finished_at
                    })
                    await add_run_event(run_id, "processing_completed", {
                        "run_id": run_id,
                        "timestamp": finished_at
                    })
                    logger.info(
                        "Run processed successfully",
                        run_id=run_id,
                        video_id=video_id,
                        platform_id=platform_id
                    )
                else:
                    error_msg = result.get("error", "Unknown error")
                    error_type = result.get("error_type", "unknown")
                    
                    # Обработка OOM: возврат в queue с lower priority
                    if error_type == "killed_by_memory_limit":
                        logger.warning(
                            "Run failed due to OOM, re-enqueuing with lower priority",
                            run_id=run_id,
                            video_id=video_id,
                            platform_id=platform_id
                        )
                        
                        # Обновить статус на "recovering" для повторной попытки
                        await save_run_state(run_id, {
                            "status": RunStatus.RECOVERING.value,
                            "error": error_msg,
                            "error_type": error_type,
                            "updated_at": finished_at,
                            "recovery_reason": "oom_retry"
                        })
                        
                        # Re-enqueue с lower priority
                        try:
                            from api.services.queue import enqueue_run
                            from api.services.redis_schema import get_run_metadata
                            
                            # Получить метаданные для re-enqueue
                            metadata = await get_run_metadata(run_id)
                            if metadata:
                                # Re-enqueue с priority "low" для OOM случаев
                                await enqueue_run(
                                    run_id=run_id,
                                    priority="low",  # Lower priority для OOM retry
                                    metadata=metadata,
                                    save_metadata_to_redis=False  # Метаданные уже сохранены
                                )
                                
                                await add_run_event(run_id, "oom_retry_enqueued", {
                                    "run_id": run_id,
                                    "timestamp": finished_at,
                                    "new_priority": "low"
                                })
                                
                                logger.info(
                                    "Run re-enqueued with lower priority after OOM",
                                    run_id=run_id,
                                    video_id=video_id,
                                    platform_id=platform_id,
                                    new_priority="low"
                                )
                            else:
                                # Если метаданные не найдены, пометить как error
                                logger.error(
                                    "Cannot re-enqueue OOM run: metadata not found",
                                    run_id=run_id
                                )
                                await save_run_state(run_id, {
                                    "status": RunStatus.ERROR.value,
                                    "error": f"{error_msg} (OOM, metadata not found for retry)",
                                    "error_type": error_type,
                                    "finished_at": finished_at,
                                    "updated_at": finished_at
                                })
                                self.task_manager.update_run_status(
                                    run_id,
                                    RunStatus.ERROR,
                                    error=f"{error_msg} (OOM)",
                                    finished_at=finished_at
                                )
                        except (RedisError, ConnectionError, TimeoutError) as e:
                            logger.error(
                                "Redis error re-enqueuing OOM run",
                                run_id=run_id,
                                worker_id=self.worker_id,
                                video_id=video_id,
                                platform_id=platform_id,
                                error=str(e),
                                error_type=type(e).__name__
                            )
                        except Exception as e:
                            logger.exception(
                                "Unexpected error re-enqueuing OOM run",
                                run_id=run_id,
                                worker_id=self.worker_id,
                                video_id=video_id,
                                platform_id=platform_id,
                                error=str(e),
                                error_type=type(e).__name__
                            )
                            # Fallback: пометить как error
                            await save_run_state(run_id, {
                                "status": RunStatus.ERROR.value,
                                "error": f"{error_msg} (OOM, re-enqueue failed: {e})",
                                "error_type": error_type,
                                "finished_at": finished_at,
                                "updated_at": finished_at
                            })
                            self.task_manager.update_run_status(
                                run_id,
                                RunStatus.ERROR,
                                error=f"{error_msg} (OOM)",
                                finished_at=finished_at
                            )
                    else:
                        # Обычная ошибка - пометить как error
                        self.task_manager.update_run_status(
                            run_id,
                            RunStatus.ERROR,
                            error=error_msg,
                            finished_at=finished_at
                        )
                        await save_run_state(run_id, {
                            "status": RunStatus.ERROR.value,
                            "error": error_msg,
                            "error_type": error_type,
                            "finished_at": finished_at,
                            "updated_at": finished_at
                        })
                        await add_run_event(run_id, "processing_failed", {
                            "run_id": run_id,
                            "error": error_msg,
                            "error_type": error_type,
                            "timestamp": finished_at
                        })
                        logger.error(
                            "Run processing failed",
                            run_id=run_id,
                            video_id=video_id,
                            platform_id=platform_id,
                            error=error_msg,
                            error_type=error_type
                        )
                
                # Обновить метрику ошибок
                try:
                    from api.services.metrics import failure_rate
                    processor = result.get("processor") or "unknown"
                    component = result.get("component") or "unknown"
                    error_type = result.get("error_type") or "unknown"
                    failure_rate.labels(processor=processor, component=component, error_type=error_type).inc()
                except (AttributeError, ImportError, ValueError) as e:
                    logger.debug(
                        "Failed to update failure_rate metric",
                        run_id=run_id,
                        worker_id=self.worker_id,
                        video_id=video_id,
                        platform_id=platform_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                except Exception as e:
                    logger.debug(
                        "Unexpected error updating failure_rate metric",
                        run_id=run_id,
                        worker_id=self.worker_id,
                        video_id=video_id,
                        platform_id=platform_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
            
            # Освободить блокировку
            await release_run_lock(run_id)
            
            # ACK сообщение после успешной обработки
            await self._ack_message(stream_name, message_id)
            
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.exception(
                "Redis error processing run",
                run_id=run_id,
                worker_id=self.worker_id,
                video_id=video_id if 'video_id' in locals() else None,
                platform_id=platform_id if 'platform_id' in locals() else None,
                error=str(e),
                error_type=type(e).__name__
            )
        except (StorageError, NotFoundError) as e:
            logger.exception(
                "Storage error processing run",
                run_id=run_id,
                worker_id=self.worker_id,
                video_id=video_id if 'video_id' in locals() else None,
                platform_id=platform_id if 'platform_id' in locals() else None,
                error=str(e),
                error_type=type(e).__name__
            )
        except Exception as e:
            logger.exception(
                "Unexpected error processing run",
                run_id=run_id,
                worker_id=self.worker_id,
                video_id=video_id if 'video_id' in locals() else None,
                platform_id=platform_id if 'platform_id' in locals() else None,
                error=str(e),
                error_type=type(e).__name__
            )
            # Обновить статус на error
            error_time = time.time()
            self.task_manager.update_run_status(
                run_id,
                RunStatus.ERROR,
                error=str(e),
                finished_at=error_time
            )
            await save_run_state(run_id, {
                "status": RunStatus.ERROR.value,
                "error": str(e),
                "finished_at": error_time,
                "updated_at": error_time
            })
            await add_run_event(run_id, "processing_error", {
                "run_id": run_id,
                "error": str(e),
                "timestamp": error_time
            })
            
            # Обновить метрику ошибок
            try:
                from api.services.metrics import failure_rate
                failure_rate.labels(processor="unknown", component="unknown", error_type="exception").inc()
            except (AttributeError, ImportError, ValueError) as e_metric:
                logger.debug(
                    "Failed to update failure_rate metric",
                    run_id=run_id,
                    worker_id=self.worker_id,
                    error=str(e_metric),
                    error_type=type(e_metric).__name__
                )
            except Exception as e_metric:
                logger.debug(
                    "Unexpected error updating failure_rate metric",
                    run_id=run_id,
                    worker_id=self.worker_id,
                    error=str(e_metric),
                    error_type=type(e_metric).__name__
                )
            # Остановить heartbeat loop при ошибке
            if run_id in self.active_heartbeats:
                heartbeat_task = self.active_heartbeats.pop(run_id)
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Освободить блокировку даже при ошибке
            await release_run_lock(run_id)
            # НЕ ACK сообщение - оно останется в pending для retry
            # В будущем можно добавить счетчик retry и после N попыток переместить в DLQ
    
    async def _heartbeat_loop(self, run_id: str) -> None:
        """
        Heartbeat loop для отправки heartbeat каждые 30 секунд.
        
        Args:
            run_id: UUID run'а
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 806-813)
        """
        heartbeat_interval = 30  # Отправлять каждые 30 секунд
        
        try:
            while True:
                await asyncio.sleep(heartbeat_interval)
                
                # Проверить что run еще активен
                run_info = self.task_manager.get_run(run_id)
                if not run_info:
                    logger.debug(f"Run {run_id} no longer active, stopping heartbeat")
                    break
                
                status = run_info.get("status", "unknown")
                if status not in ("running", "recovering"):
                    logger.debug(f"Run {run_id} status is {status}, stopping heartbeat")
                    break
                
                # Обновить heartbeat
                await update_run_heartbeat(run_id)
                logger.debug(f"Sent heartbeat for run {run_id}")
                
        except asyncio.CancelledError:
            logger.debug(
                "Heartbeat loop cancelled",
                run_id=run_id,
                worker_id=self.worker_id
            )
        except (RedisError, ConnectionError, TimeoutError) as e:
            logger.error(
                "Redis error in heartbeat loop",
                run_id=run_id,
                worker_id=self.worker_id,
                error=str(e),
                error_type=type(e).__name__
            )
        except Exception as e:
            logger.error(
                "Unexpected error in heartbeat loop",
                run_id=run_id,
                worker_id=self.worker_id,
                error=str(e),
                error_type=type(e).__name__
            )
    
    async def _ack_message(self, stream_name: str, message_id: str) -> None:
        """
        Подтвердить обработку сообщения (ACK).
        
        Args:
            stream_name: Имя stream'а
            message_id: ID сообщения
            
        Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 722-723)
        """
        try:
            await self.redis_client.xack(
                stream_name,
                CONSUMER_GROUP_NAME,
                message_id
            )
            logger.debug(
                "ACKed message",
                worker_id=self.worker_id,
                stream_name=stream_name,
                message_id=message_id
            )
        except (RedisError, ResponseError, ConnectionError, TimeoutError) as e:
            logger.error(
                "Redis error ACKing message",
                worker_id=self.worker_id,
                stream_name=stream_name,
                message_id=message_id,
                error=str(e),
                error_type=type(e).__name__
            )
        except Exception as e:
            logger.error(
                "Unexpected error ACKing message",
                worker_id=self.worker_id,
                stream_name=stream_name,
                message_id=message_id,
                error=str(e),
                error_type=type(e).__name__
            )


async def start_worker(worker_id: str) -> None:
    """
    Запустить worker с указанным ID.
    
    Args:
        worker_id: Уникальный ID worker'а
    """
    worker = Worker(worker_id=worker_id)
    await worker.start()

