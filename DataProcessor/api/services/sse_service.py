"""
SSE Service - управление SSE соединениями и стриминг событий

Этот модуль реализует функциональность для Server-Sent Events (SSE) стриминга
событий обработки в реальном времени.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1191-1220)
"""

import logging
import asyncio
import json
import time
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime

from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.services.redis_client import get_redis_client
from api.services.redis_schema import KEY_PREFIX_EVENTS
from api.config import config

logger = logging.getLogger(__name__)


class SSEConnectionManager:
    """
    Менеджер для отслеживания активных SSE соединений.
    
    Ограничивает количество одновременных соединений на run_id.
    """
    
    def __init__(self):
        self._connections: Dict[str, int] = {}  # run_id -> количество соединений
        self._lock = asyncio.Lock()
    
    async def acquire(self, run_id: str) -> bool:
        """
        Попытаться получить соединение для run_id.
        
        Args:
            run_id: UUID run'а
            
        Returns:
            True если соединение получено, False если превышен лимит
        """
        async with self._lock:
            current_count = self._connections.get(run_id, 0)
            max_connections = config.max_sse_connections_per_run
            
            if current_count >= max_connections:
                logger.warning(
                    f"Max SSE connections ({max_connections}) reached for run {run_id}"
                )
                return False
            
            self._connections[run_id] = current_count + 1
            logger.debug(f"SSE connection acquired for run {run_id} ({current_count + 1}/{max_connections})")
            return True
    
    async def release(self, run_id: str) -> None:
        """
        Освободить соединение для run_id.
        
        Args:
            run_id: UUID run'а
        """
        async with self._lock:
            if run_id in self._connections:
                self._connections[run_id] = max(0, self._connections[run_id] - 1)
                if self._connections[run_id] == 0:
                    del self._connections[run_id]
                logger.debug(f"SSE connection released for run {run_id}")
    
    async def get_connection_count(self, run_id: str) -> int:
        """
        Получить количество активных соединений для run_id.
        
        Args:
            run_id: UUID run'а
            
        Returns:
            Количество активных соединений
        """
        async with self._lock:
            return self._connections.get(run_id, 0)


# Глобальный экземпляр менеджера соединений
_connection_manager = SSEConnectionManager()


async def stream_run_events(
    run_id: str,
    since: Optional[str] = None,
    component: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Стриминг событий для run'а через SSE.
    
    Args:
        run_id: UUID run'а
        since: ISO 8601 timestamp для фильтрации событий (опционально)
        component: Имя компонента для фильтрации (опционально)
        
    Yields:
        SSE форматированные строки событий
        
    Raises:
        ValueError: Если превышен лимит соединений
        RuntimeError: Если Redis не доступен
    """
    # Проверить лимит соединений
    if not await _connection_manager.acquire(run_id):
        raise ValueError(
            f"Max SSE connections ({config.max_sse_connections_per_run}) "
            f"reached for run {run_id}"
        )
    
    redis_client = get_redis_client()
    if not redis_client:
        await _connection_manager.release(run_id)
        raise RuntimeError("Redis not available, cannot stream events")
    
    try:
        # Отправить начальное сообщение
        yield "event: connected\n"
        yield f"data: {{\"run_id\": \"{run_id}\", \"timestamp\": \"{datetime.utcnow().isoformat()}Z\"}}\n\n"
        
        stream_name = f"{KEY_PREFIX_EVENTS}{run_id}"
        
        # Определить начальный ID для чтения
        start_id = "0"  # Начать с начала stream
        
        # Если указан since, найти соответствующий ID
        if since:
            try:
                # Конвертировать ISO 8601 в timestamp
                since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
                since_ts = since_dt.timestamp()
                
                # Найти первое сообщение после since timestamp
                # Используем XRANGE для поиска
                messages = await redis_client.xrange(
                    stream_name,
                    min=f"{int(since_ts * 1000)}-0",
                    count=1
                )
                if messages:
                    start_id = messages[0][0].decode() if isinstance(messages[0][0], bytes) else str(messages[0][0])
            except Exception as e:
                logger.warning(f"Failed to parse 'since' parameter: {e}, starting from beginning")
        
        # Читать события в реальном времени
        last_id = start_id
        
        while True:
            try:
                # Читать новые сообщения из stream
                messages = await redis_client.xread(
                    {stream_name: last_id},
                    count=10,
                    block=config.sse_stream_read_timeout
                )
                
                if messages:
                    for stream, messages_list in messages:
                        for msg_id, data in messages_list:
                            msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                            
                            # Парсить событие
                            event_type = (
                                data.get(b"event_type", b"").decode("utf-8")
                                if isinstance(data.get(b"event_type"), bytes)
                                else str(data.get("event_type", ""))
                            )
                            
                            event_data_str = (
                                data.get(b"data", b"{}").decode("utf-8")
                                if isinstance(data.get(b"data"), bytes)
                                else str(data.get("data", "{}"))
                            )
                            
                            try:
                                event_data = json.loads(event_data_str)
                            except Exception:
                                event_data = {}
                            
                            # Фильтрация по компоненту
                            if component:
                                event_component = event_data.get("component")
                                if event_component != component:
                                    last_id = msg_id_str
                                    continue
                            
                            # Форматировать событие в SSE формат
                            # Маппинг типов событий на SSE event types
                            sse_event_type = _map_event_type_to_sse(event_type)
                            
                            # Добавить timestamp если его нет
                            if "ts" not in event_data and "timestamp" not in event_data:
                                event_data["ts"] = datetime.utcnow().isoformat() + "Z"
                            
                            # Отправить событие
                            yield f"event: {sse_event_type}\n"
                            yield f"data: {json.dumps(event_data)}\n\n"
                            
                            last_id = msg_id_str
                
                # Отправить keepalive каждые 30 секунд если нет событий
                await asyncio.sleep(0.1)  # Небольшая задержка для предотвращения busy loop
                
            except asyncio.CancelledError:
                logger.debug(f"SSE stream cancelled for run {run_id}")
                break
            except RedisError as e:
                logger.error(f"Redis error while streaming events for run {run_id}: {e}")
                yield f"event: error\n"
                yield f"data: {{\"error\": \"Redis error: {str(e)}\"}}\n\n"
                break
            except Exception as e:
                logger.exception(f"Unexpected error while streaming events for run {run_id}: {e}")
                yield f"event: error\n"
                yield f"data: {{\"error\": \"Unexpected error: {str(e)}\"}}\n\n"
                break
        
        # Отправить сообщение о завершении
        yield "event: disconnected\n"
        yield f"data: {{\"run_id\": \"{run_id}\", \"timestamp\": \"{datetime.utcnow().isoformat()}Z\"}}\n\n"
        
    finally:
        await _connection_manager.release(run_id)


def _map_event_type_to_sse(event_type: str) -> str:
    """
    Маппинг типов событий на SSE event types.
    
    Args:
        event_type: Тип события из Redis Stream
        
    Returns:
        SSE event type
    """
    # Маппинг типов событий
    mapping = {
        "processing_started": "stage",
        "processing_completed": "complete",
        "processing_failed": "complete",
        "processing_error": "error",
        "component_started": "component_start",
        "component_completed": "component_complete",
        "component_failed": "component_complete",
        "progress_update": "progress",
        "recovery_started": "stage",
        "resume_from_checkpoint": "stage",
    }
    
    # Если есть точное совпадение, использовать его
    if event_type in mapping:
        return mapping[event_type]
    
    # Проверить префиксы
    if event_type.startswith("progress"):
        return "progress"
    elif event_type.startswith("component"):
        if "start" in event_type:
            return "component_start"
        elif "complete" in event_type or "finish" in event_type:
            return "component_complete"
    elif event_type.startswith("stage") or event_type.startswith("processor"):
        return "stage"
    
    # По умолчанию использовать исходный тип
    return event_type

