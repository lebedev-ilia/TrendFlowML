"""
Redis Streams Queue Service - управление очередью задач

Этот модуль предоставляет функции для работы с Redis Streams queue:
- Добавление задач в очередь с приоритетами
- Управление приоритетными очередями (high, normal, low)

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 661-728)
"""

import logging
import time
from typing import Optional, Dict, Any
from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.services.redis_client import get_redis_client
from api.services.redis_schema import (
    save_run_metadata,
    save_run_priority
)

logger = logging.getLogger(__name__)

# Имена очередей по приоритетам
QUEUE_HIGH = "queue:high"
QUEUE_NORMAL = "queue:normal"
QUEUE_LOW = "queue:low"

# Максимальный размер stream (для предотвращения переполнения)
MAX_STREAM_LENGTH = 10000


async def enqueue_run(
    run_id: str,
    priority: str = "normal",
    metadata: Optional[Dict[str, Any]] = None,
    save_metadata_to_redis: bool = True
) -> Optional[str]:
    """
    Добавить run в очередь через Redis Streams.
    
    Добавляет задачу в Redis Stream с указанным приоритетом.
    Сохраняет метаданные в Redis для быстрого доступа.
    Обновляет метрики длины очереди.
    
    Args:
        run_id: UUID run'а (обязательно)
        priority: Приоритет задачи (по умолчанию "normal")
            - "high": Высокий приоритет (queue:high)
            - "normal": Обычный приоритет (queue:normal)
            - "low": Низкий приоритет (queue:low)
        metadata: Дополнительные метаданные для сохранения в Redis (опционально)
            Пример: {"video_id": "video-123", "platform_id": "youtube"}
        save_metadata_to_redis: Сохранять ли метаданные в Redis (по умолчанию True)
        
    Returns:
        str: Message ID из Redis Stream (например, "1234567890-0")
        None: Если Redis не доступен
        
    Raises:
        ValueError: Если приоритет невалидный (не "high", "normal" или "low")
        RedisError: Если произошла ошибка при работе с Redis
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 677-683)
    
    Example:
        ```python
        message_id = await enqueue_run(
            run_id="550e8400-e29b-41d4-a716-446655440000",
            priority="high",
            metadata={
                "video_id": "video-123",
                "platform_id": "youtube",
                "config_hash": "abc123..."
            }
        )
        if message_id:
            print(f"Run enqueued with message_id: {message_id}")
        ```
    """
    # Валидация приоритета
    valid_priorities = ["high", "normal", "low"]
    if priority not in valid_priorities:
        raise ValueError(f"Invalid priority: {priority}. Must be one of {valid_priorities}")
    
    # Получить Redis клиент
    redis_client = get_redis_client()
    if not redis_client:
        logger.warning("Redis not available, cannot enqueue run")
        return None
    
    # Определить имя очереди по приоритету
    queue_name = f"queue:{priority}"
    
    # Подготовить данные для записи в stream
    stream_data = {
        "run_id": run_id,
        "ts": str(time.time()),
        "priority": priority
    }
    
    # Сохранить метаданные в Redis для быстрого доступа
    if save_metadata_to_redis and metadata:
        await save_run_metadata(run_id, metadata)
    
    # Добавить метаданные в stream данные если есть
    if metadata:
        import json
        stream_data["metadata"] = json.dumps(metadata)
    
    try:
        # Сохранить приоритет run'а в Redis
        await save_run_priority(run_id, priority)
        
        # Добавить в stream с ограничением размера
        message_id = await redis_client.xadd(
            queue_name,
            stream_data,
            maxlen=MAX_STREAM_LENGTH,
            approximate=True  # Приблизительное ограничение для производительности
        )
        
        logger.info(f"Enqueued run {run_id} to {queue_name} with message_id {message_id}")
        
        # Обновить метрику длины очереди
        try:
            from api.services.metrics import queue_length
            queue_length.labels(priority=priority).set(await redis_client.xlen(queue_name))
        except Exception as e:
            logger.debug(f"Failed to update queue_length metric: {e}")
        
        return message_id.decode() if isinstance(message_id, bytes) else str(message_id)
        
    except RedisError as e:
        logger.error(f"Failed to enqueue run {run_id} to Redis: {e}")
        raise RedisError(f"Failed to enqueue run: {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected error enqueueing run {run_id}: {e}")
        raise RedisError(f"Unexpected error enqueueing run: {e}") from e


async def get_queue_length(priority: Optional[str] = None) -> Dict[str, int]:
    """
    Получить длину очереди (количество сообщений в stream).
    
    Возвращает количество сообщений в указанной очереди или во всех очередях.
    
    Args:
        priority: Приоритет для проверки (опционально)
            - None: Вернуть длины всех очередей
            - "high", "normal", "low": Вернуть длину указанной очереди
            
    Returns:
        Dict[str, int]: Словарь с длинами очередей
            Пример: {"high": 10, "normal": 5, "low": 2}
            Если priority указан: {"high": 10} или {"normal": 5}
            Если Redis недоступен: {}
            
    Example:
        ```python
        # Получить длины всех очередей
        lengths = await get_queue_length()
        print(f"High priority queue: {lengths.get('high', 0)} tasks")
        
        # Получить длину конкретной очереди
        normal_length = await get_queue_length(priority="normal")
        print(f"Normal queue: {normal_length.get('normal', 0)} tasks")
        ```
    """
    redis_client = get_redis_client()
    if not redis_client:
        return {}
    
    priorities = [priority] if priority else ["high", "normal", "low"]
    lengths = {}
    
    try:
        for p in priorities:
            queue_name = f"queue:{p}"
            length = await redis_client.xlen(queue_name)
            lengths[p] = length
            
    except Exception as e:
        logger.warning(f"Failed to get queue lengths: {e}")
    
    return lengths


async def get_pending_count(priority: Optional[str] = None, group_name: str = "workers") -> Dict[str, int]:
    """
    Получить количество pending сообщений (не обработанных).
    
    Args:
        priority: Приоритет для проверки (None = все очереди)
        group_name: Имя consumer group
        
    Returns:
        Словарь с количеством pending сообщений
    """
    redis_client = get_redis_client()
    if not redis_client:
        return {}
    
    priorities = [priority] if priority else ["high", "normal", "low"]
    pending_counts = {}
    
    try:
        for p in priorities:
            queue_name = f"queue:{p}"
            # Получить информацию о pending сообщениях
            pending_info = await redis_client.xpending(queue_name, group_name)
            if pending_info:
                # pending_info это кортеж: (total_pending, min_id, max_id, consumers)
                pending_counts[p] = pending_info[0] if isinstance(pending_info, tuple) else 0
            else:
                pending_counts[p] = 0
                
    except Exception as e:
        logger.warning(f"Failed to get pending counts: {e}")
    
    return pending_counts


async def get_total_queue_length() -> int:
    """
    Получить общую длину всех очередей (high + normal + low).
    
    Также обновляет метрики Prometheus для каждой очереди.
    
    Returns:
        Общее количество сообщений во всех очередях
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 954-969)
    """
    redis_client = get_redis_client()
    if not redis_client:
        # Если Redis не доступен, вернуть 0 (не блокируем обработку)
        logger.debug("Redis not available, returning 0 for queue length")
        return 0
    
    try:
        total = 0
        from api.services.metrics import queue_length
        
        for priority in ["high", "normal", "low"]:
            queue_name = f"queue:{priority}"
            length = await redis_client.xlen(queue_name)
            total += length
            
            # Обновить метрику Prometheus
            try:
                queue_length.labels(priority=priority).set(length)
            except Exception as e:
                logger.debug(f"Failed to update queue_length metric for {priority}: {e}")
        
        logger.debug(f"Total queue length: {total}")
        return total
        
    except Exception as e:
        logger.warning(f"Failed to get total queue length: {e}")
        # В случае ошибки возвращаем 0, чтобы не блокировать обработку
        return 0

