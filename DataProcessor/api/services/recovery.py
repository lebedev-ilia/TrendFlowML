"""
Recovery Service - восстановление crashed run'ов

Этот модуль реализует логику восстановления run'ов, которые упали во время обработки.
Проверяет heartbeat и возвращает crashed run'ы обратно в очередь.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 830-839)
"""

import logging
import time
from typing import Optional, Dict, Any
from redis.asyncio import Redis

from api.services.redis_client import get_redis_client
from api.services.redis_schema import (
    get_run_heartbeat,
    get_run_state,
    save_run_state,
    get_run_priority,
    get_run_metadata,
    add_run_event
)
from api.services.queue import enqueue_run
from api.schemas.state import RunStatus
from api.services.state_machine import parse_status

logger = logging.getLogger(__name__)


async def check_and_recover_run(run_id: str) -> bool:
    """
    Проверить heartbeat run'а и восстановить если необходимо.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если run был восстановлен, False если не требуется восстановление
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 830-839)
    """
    redis_client = get_redis_client()
    if not redis_client:
        logger.warning("Redis not available, cannot check heartbeat")
        return False
    
    # Получить состояние run'а
    run_state = await get_run_state(run_id)
    if not run_state:
        logger.debug(f"Run {run_id} not found in Redis state")
        return False
    
    status_str = run_state.get("status", "unknown")
    try:
        status = parse_status(status_str) if status_str else None
    except ValueError:
        logger.warning(f"Invalid status '{status_str}' for run {run_id}")
        return False
    
    # Проверять heartbeat только для running run'ов
    if status != RunStatus.RUNNING:
        return False
    
    # Проверить heartbeat
    heartbeat = await get_run_heartbeat(run_id)
    
    if heartbeat is None:
        # Heartbeat отсутствует - run crashed
        logger.warning(f"Run {run_id} has no heartbeat, initiating recovery")
        
        # Обновить метрику crashed runs
        try:
            from api.services.metrics import crashed_runs
            crashed_runs.inc()
        except Exception as e:
            logger.debug(f"Failed to update crashed_runs metric: {e}")
        
        return await recover_run(run_id)
    
    return False


async def recover_run(run_id: str) -> bool:
    """
    Восстановить crashed run - вернуть в очередь.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если успешно восстановлено, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 830-839)
    """
    redis_client = get_redis_client()
    if not redis_client:
        logger.warning("Redis not available, cannot recover run")
        return False
    
    try:
        # Получить приоритет run'а
        priority = await get_run_priority(run_id) or "normal"
        
        # Получить метаданные для восстановления
        metadata = await get_run_metadata(run_id)
        
        if not metadata:
            logger.error(f"Cannot recover run {run_id}: metadata not found")
            return False
        
        # Обновить статус на "recovering"
        recovery_time = time.time()
        await save_run_state(run_id, {
            "status": "recovering",
            "recovery_started_at": recovery_time,
            "updated_at": recovery_time
        })
        
        # Добавить событие восстановления
        await add_run_event(run_id, "recovery_started", {
            "run_id": run_id,
            "timestamp": recovery_time,
            "reason": "missing_heartbeat"
        })
        
        # Вернуть в queue
        await enqueue_run(
            run_id,
            priority=priority,
            metadata=metadata,
            save_metadata_to_redis=False  # Метаданные уже в Redis
        )
        
        logger.info(f"Recovered run {run_id} and re-enqueued with priority {priority}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error recovering run {run_id}: {e}")
        return False


async def recover_all_crashed_runs() -> int:
    """
    Найти и восстановить все crashed run'ы.
    
    Проверяет все run'ы со статусом "running" и восстанавливает те, у которых нет heartbeat.
    
    Returns:
        Количество восстановленных run'ов
    """
    redis_client = get_redis_client()
    if not redis_client:
        logger.warning("Redis not available, cannot recover crashed runs")
        return 0
    
    recovered_count = 0
    
    try:
        # Найти все run:state:* ключи
        pattern = "run:state:*"
        keys = []
        async for key in redis_client.scan_iter(match=pattern):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            keys.append(key)
        
        # Проверить каждый run
        for key in keys:
            # Извлечь run_id из ключа
            run_id = key.replace("run:state:", "")
            
            # Проверить и восстановить если необходимо
            if await check_and_recover_run(run_id):
                recovered_count += 1
        
        if recovered_count > 0:
            logger.info(f"Recovered {recovered_count} crashed run(s)")
        
        return recovered_count
        
    except Exception as e:
        logger.exception(f"Error recovering crashed runs: {e}")
        return recovered_count

