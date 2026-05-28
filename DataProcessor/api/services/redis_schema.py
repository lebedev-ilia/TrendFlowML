"""
Redis Schema Service - управление структурой данных в Redis

Этот модуль предоставляет функции для работы со схемой данных Redis:
- run:meta:{run_id} - метаданные run'а (TTL 7 дней)
- run:state:{run_id} - кэш состояния (TTL 1 день)
- run:heartbeat:{run_id} - heartbeat (TTL 60 сек)
- run:lock:{run_id} - idempotency lock (TTL 3600 сек)
- run:priority:{run_id} - приоритет (TTL 7 дней)
- stream:events:{run_id} - события (TTL 1 день)

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 972-1016)
"""

import logging
import json
import time
from typing import Any, Dict, Optional
from datetime import datetime
from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.services.redis_client import get_redis_client
from api.schemas.state import RunStatus
from api.services.state_machine import validate_transition, parse_status

logger = logging.getLogger(__name__)

# TTL константы (в секундах)
TTL_META = 7 * 24 * 3600  # 7 дней
TTL_STATE = 24 * 3600  # 1 день
TTL_HEARTBEAT = 60  # 60 секунд
TTL_LOCK = 3600  # 1 час
TTL_PRIORITY = 7 * 24 * 3600  # 7 дней
TTL_EVENTS = 24 * 3600  # 1 день
TTL_CANCEL = 3600  # 1 час (флаг отмены)

# Префиксы ключей
KEY_PREFIX_META = "run:meta:"
KEY_PREFIX_STATE = "run:state:"
KEY_PREFIX_HEARTBEAT = "run:heartbeat:"
KEY_PREFIX_LOCK = "run:lock:"
KEY_PREFIX_PRIORITY = "run:priority:"
KEY_PREFIX_EVENTS = "stream:events:"
KEY_PREFIX_CANCEL = "run:cancel:"


# ============================================================================
# run:meta:{run_id} - Метаданные run'а
# ============================================================================

async def save_run_metadata(
    run_id: str,
    metadata: Dict[str, Any]
) -> bool:
    """
    Сохранить метаданные run'а в Redis.
    
    Args:
        run_id: UUID run'а
        metadata: Метаданные для сохранения
        
    Returns:
        True если успешно сохранено, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 976-986)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_META}{run_id}"
        value = json.dumps(metadata)
        
        await redis_client.setex(key, TTL_META, value)
        logger.debug(f"Saved metadata for run {run_id} to Redis")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save metadata for run {run_id}: {e}")
        return False


async def get_run_metadata(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Получить метаданные run'а из Redis.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        Словарь с метаданными или None если не найдено
    """
    redis_client = get_redis_client()
    if not redis_client:
        return None
    
    try:
        key = f"{KEY_PREFIX_META}{run_id}"
        value = await redis_client.get(key)
        
        if value:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return json.loads(value)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get metadata for run {run_id}: {e}")
        return None


# ============================================================================
# run:state:{run_id} - Кэш состояния (hot path)
# ============================================================================

async def save_run_state(
    run_id: str,
    state: Dict[str, Any],
    validate_status_transition: bool = True
) -> bool:
    """
    Сохранить состояние run'а в Redis (hot path cache).
    
    Args:
        run_id: UUID run'а
        state: Состояние для сохранения
        validate_status_transition: Валидировать переход статуса (по умолчанию True)
        
    Returns:
        True если успешно сохранено, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 988-995)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        # Валидация перехода статуса если включена
        if validate_status_transition and "status" in state:
            new_status_str = state.get("status")
            try:
                new_status = parse_status(new_status_str)
                
                # Получить текущий статус из Redis
                current_state = await get_run_state(run_id)
                current_status = None
                if current_state and "status" in current_state:
                    try:
                        current_status = parse_status(current_state["status"])
                    except ValueError:
                        logger.warning(f"Invalid current status '{current_state['status']}' for run {run_id}")

                # Повторная запись того же статуса допустима: это обычный случай
                # для hot-path refresh после успешного завершения.
                if current_status == new_status:
                    validate_status_transition = False
                
                # Валидировать переход
                if validate_status_transition:
                    validate_transition(current_status, new_status, run_id)
            except ValueError as e:
                logger.error(f"Invalid status transition for run {run_id}: {e}")
                return False
        
        key = f"{KEY_PREFIX_STATE}{run_id}"
        
        # Добавить updated_at если его нет
        if "updated_at" not in state:
            state["updated_at"] = datetime.now().isoformat()
        
        value = json.dumps(state)
        await redis_client.setex(key, TTL_STATE, value)
        logger.debug(f"Saved state for run {run_id} to Redis")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save state for run {run_id}: {e}")
        return False


async def get_run_state(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Получить состояние run'а из Redis (hot path).
    
    Args:
        run_id: UUID run'а
        
    Returns:
        Словарь с состоянием или None если не найдено
    """
    redis_client = get_redis_client()
    if not redis_client:
        return None
    
    try:
        key = f"{KEY_PREFIX_STATE}{run_id}"
        value = await redis_client.get(key)
        
        if value:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return json.loads(value)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get state for run {run_id}: {e}")
        return None


# ============================================================================
# run:heartbeat:{run_id} - Heartbeat
# ============================================================================

async def update_run_heartbeat(run_id: str) -> bool:
    """
    Обновить heartbeat run'а в Redis.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если успешно обновлено, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 998-999)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_HEARTBEAT}{run_id}"
        timestamp = str(time.time())
        
        await redis_client.setex(key, TTL_HEARTBEAT, timestamp)
        logger.debug(f"Updated heartbeat for run {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update heartbeat for run {run_id}: {e}")
        return False


async def get_run_heartbeat(run_id: str) -> Optional[float]:
    """
    Получить последний heartbeat run'а из Redis.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        Timestamp последнего heartbeat или None если не найдено
    """
    redis_client = get_redis_client()
    if not redis_client:
        return None
    
    try:
        key = f"{KEY_PREFIX_HEARTBEAT}{run_id}"
        value = await redis_client.get(key)
        
        if value:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return float(value)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get heartbeat for run {run_id}: {e}")
        return None


async def is_run_alive(run_id: str) -> bool:
    """
    Проверить, жив ли run (heartbeat не истек).
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если run жив, False иначе
    """
    heartbeat = await get_run_heartbeat(run_id)
    return heartbeat is not None


# ============================================================================
# run:lock:{run_id} - Idempotency lock
# ============================================================================

async def acquire_run_lock(
    run_id: str,
    timeout: Optional[int] = None
) -> bool:
    """
    Получить блокировку для run'а (idempotency lock).
    
    Args:
        run_id: UUID run'а
        timeout: Таймаут блокировки в секундах (по умолчанию TTL_LOCK)
        
    Returns:
        True если блокировка получена, False если уже заблокировано
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1002-1003)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_LOCK}{run_id}"
        ttl = timeout or TTL_LOCK
        
        # Используем SET с NX (только если не существует) для атомарной блокировки
        result = await redis_client.set(key, "locked", ex=ttl, nx=True)
        
        if result:
            logger.debug(f"Acquired lock for run {run_id}")
            return True
        else:
            logger.debug(f"Lock already exists for run {run_id}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to acquire lock for run {run_id}: {e}")
        return False


async def release_run_lock(run_id: str) -> bool:
    """
    Освободить блокировку для run'а.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если блокировка освобождена, False иначе
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_LOCK}{run_id}"
        await redis_client.delete(key)
        logger.debug(f"Released lock for run {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to release lock for run {run_id}: {e}")
        return False


async def is_run_locked(run_id: str) -> bool:
    """
    Проверить, заблокирован ли run.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если run заблокирован, False иначе
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_LOCK}{run_id}"
        exists = await redis_client.exists(key)
        return bool(exists)
        
    except Exception as e:
        logger.error(f"Failed to check lock for run {run_id}: {e}")
        return False


# ============================================================================
# run:priority:{run_id} - Приоритет
# ============================================================================

async def save_run_priority(
    run_id: str,
    priority: str
) -> bool:
    """
    Сохранить приоритет run'а в Redis.
    
    Args:
        run_id: UUID run'а
        priority: Приоритет ("high", "normal", "low")
        
    Returns:
        True если успешно сохранено, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1006)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_PRIORITY}{run_id}"
        await redis_client.setex(key, TTL_PRIORITY, priority)
        logger.debug(f"Saved priority for run {run_id}: {priority}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save priority for run {run_id}: {e}")
        return False


async def get_run_priority(run_id: str) -> Optional[str]:
    """
    Получить приоритет run'а из Redis.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        Приоритет или None если не найдено
    """
    redis_client = get_redis_client()
    if not redis_client:
        return None
    
    try:
        key = f"{KEY_PREFIX_PRIORITY}{run_id}"
        value = await redis_client.get(key)
        
        if value:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get priority for run {run_id}: {e}")
        return None


# ============================================================================
# stream:events:{run_id} - События (Redis Streams)
# ============================================================================

async def add_run_event(
    run_id: str,
    event_type: str,
    event_data: Dict[str, Any]
) -> Optional[str]:
    """
    Добавить событие для run'а в Redis Stream.
    
    Args:
        run_id: UUID run'а
        event_type: Тип события
        event_data: Данные события
        
    Returns:
        Message ID из Redis Stream или None если Redis не доступен
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1014-1015)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return None
    
    try:
        stream_name = f"{KEY_PREFIX_EVENTS}{run_id}"
        
        # Подготовить данные события
        stream_data = {
            "event_type": event_type,
            "timestamp": str(time.time()),
            "data": json.dumps(event_data)
        }
        
        # Добавить в stream
        message_id = await redis_client.xadd(
            stream_name,
            stream_data,
            maxlen=1000,  # Ограничение размера stream
            approximate=True
        )
        
        # Установить TTL для stream (через EXPIRE)
        await redis_client.expire(stream_name, TTL_EVENTS)
        
        logger.debug(f"Added event {event_type} for run {run_id}")
        return message_id.decode() if isinstance(message_id, bytes) else str(message_id)
        
    except Exception as e:
        logger.error(f"Failed to add event for run {run_id}: {e}")
        return None


async def get_run_events(
    run_id: str,
    count: int = 100,
    start_id: Optional[str] = None
) -> list:
    """
    Получить события для run'а из Redis Stream.
    
    Args:
        run_id: UUID run'а
        count: Количество событий для получения
        start_id: Начальный ID сообщения (None = с начала)
        
    Returns:
        Список событий
    """
    redis_client = get_redis_client()
    if not redis_client:
        return []
    
    try:
        stream_name = f"{KEY_PREFIX_EVENTS}{run_id}"
        
        # Читать из stream
        if start_id:
            messages = await redis_client.xread({stream_name: start_id}, count=count)
        else:
            messages = await redis_client.xread({stream_name: "0"}, count=count)
        
        events = []
        if messages:
            for stream, messages_list in messages:
                for msg_id, data in messages_list:
                    event = {
                        "id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                        "event_type": data.get(b"event_type", b"").decode("utf-8") if isinstance(data.get(b"event_type"), bytes) else str(data.get("event_type", "")),
                        "timestamp": float(data.get(b"timestamp", b"0").decode("utf-8")) if isinstance(data.get(b"timestamp"), bytes) else float(data.get("timestamp", 0)),
                        "data": json.loads(data.get(b"data", b"{}").decode("utf-8")) if isinstance(data.get(b"data"), bytes) else json.loads(data.get("data", "{}"))
                    }
                    events.append(event)
        
        return events
        
    except Exception as e:
        logger.error(f"Failed to get events for run {run_id}: {e}")
        return []


# ============================================================================
# Вспомогательные функции
# ============================================================================

# ============================================================================
# run:cancel:{run_id} - Флаг отмены run'а
# ============================================================================

async def set_cancel_flag(run_id: str) -> bool:
    """
    Установить флаг отмены для run'а.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если успешно установлен, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2733)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_CANCEL}{run_id}"
        await redis_client.setex(key, TTL_CANCEL, "1")
        logger.debug(f"Set cancel flag for run {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to set cancel flag for run {run_id}: {e}")
        return False


async def get_cancel_flag(run_id: str) -> bool:
    """
    Проверить флаг отмены для run'а.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если флаг установлен, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2746)
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_CANCEL}{run_id}"
        value = await redis_client.get(key)
        return value is not None and value.decode("utf-8") == "1"
        
    except Exception as e:
        logger.error(f"Failed to get cancel flag for run {run_id}: {e}")
        return False


async def clear_cancel_flag(run_id: str) -> bool:
    """
    Удалить флаг отмены для run'а.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если успешно удален, False иначе
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        key = f"{KEY_PREFIX_CANCEL}{run_id}"
        await redis_client.delete(key)
        logger.debug(f"Cleared cancel flag for run {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear cancel flag for run {run_id}: {e}")
        return False


async def delete_run_data(run_id: str) -> bool:
    """
    Удалить все данные run'а из Redis.
    
    Args:
        run_id: UUID run'а
        
    Returns:
        True если успешно удалено, False иначе
    """
    redis_client = get_redis_client()
    if not redis_client:
        return False
    
    try:
        keys = [
            f"{KEY_PREFIX_META}{run_id}",
            f"{KEY_PREFIX_STATE}{run_id}",
            f"{KEY_PREFIX_HEARTBEAT}{run_id}",
            f"{KEY_PREFIX_LOCK}{run_id}",
            f"{KEY_PREFIX_PRIORITY}{run_id}",
            f"{KEY_PREFIX_EVENTS}{run_id}"
        ]
        
        await redis_client.delete(*keys)
        logger.debug(f"Deleted all Redis data for run {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete Redis data for run {run_id}: {e}")
        return False


# Статусы, которые занимают слот max_concurrent_runs (согласовано с TaskManager).
_ACTIVE_RUN_STATUSES_CAP = frozenset({"pending", "queued", "running", "recovering"})


async def count_active_runs_from_redis_state() -> Optional[int]:
    """
    Число «активных» run по ключам ``run:state:*`` в Redis.

    Worker обновляет состояние здесь; API-процесс держит отдельный in-memory TaskManager,
    поэтому для /health и backpressure нужен общий источник — Redis (см. process.py register + worker).
    """
    redis_client = get_redis_client()
    if not redis_client:
        return None
    try:
        total = 0
        cursor: int | str = 0
        prefix = KEY_PREFIX_STATE
        while True:
            cursor, keys = await redis_client.scan(
                cursor=cursor,
                match=f"{prefix}*",
                count=200,
            )
            if keys:
                pipe = redis_client.pipeline(transaction=False)
                for k in keys:
                    pipe.get(k)
                values = await pipe.execute()
                for raw in values:
                    if not raw:
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        data = json.loads(raw)
                        st = str(data.get("status") or "").strip().lower()
                        if st in _ACTIVE_RUN_STATUSES_CAP:
                            total += 1
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        continue
            if cursor == 0:
                break
        return total
    except Exception as e:
        logger.warning("count_active_runs_from_redis_state failed: %s", e)
        return None


async def get_effective_active_runs_count(task_manager: Any) -> int:
    """Активные run: Redis при доступности, иначе fallback на TaskManager (один процесс / без Redis)."""
    n = await count_active_runs_from_redis_state()
    if n is not None:
        return n
    return task_manager.get_active_runs_count() if task_manager else 0


async def effective_can_accept_new_run(task_manager: Any) -> bool:
    from api.config import config

    active = await get_effective_active_runs_count(task_manager)
    return active < config.max_concurrent_runs



