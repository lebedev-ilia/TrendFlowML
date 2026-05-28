"""
Redis Client Service - управление подключением к Redis

Этот модуль предоставляет async Redis клиент для использования в API.
Используется для:
- Очереди задач (Redis Streams)
- Кэширование состояния (hot path)
- Координация между worker'ами
- Блокировки

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 179, 1826-1844)
"""

import logging
from typing import Optional
import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from api.config import config

logger = logging.getLogger(__name__)

# Blocking Redis Streams reads (XREAD/XREADGROUP) are used by both
# worker queue consumption and SSE event streaming. Keep socket timeout
# comfortably above the per-command BLOCK timeout to avoid spurious
# TimeoutError noise during normal idle waiting.
REDIS_SOCKET_TIMEOUT_SECONDS = 30

# Глобальный экземпляр Redis клиента
_redis_client: Optional[Redis] = None


def get_redis_client() -> Optional[Redis]:
    """
    Получить глобальный экземпляр Redis клиента.
    
    Returns:
        Redis клиент или None если Redis не настроен
    """
    return _redis_client


async def init_redis_client() -> Optional[Redis]:
    """
    Инициализировать async Redis клиент.
    
    Подключение происходит из переменных окружения:
    - REDIS_URL (приоритет) или
    - REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
    
    Returns:
        Redis клиент или None если Redis не настроен
        
    Raises:
        RedisError: Если не удалось подключиться к Redis
    """
    global _redis_client
    
    # Проверить, настроен ли Redis
    redis_url = getattr(config, 'redis_url', None)
    redis_host = getattr(config, 'redis_host', None)
    
    if not redis_url and not redis_host:
        logger.warning("Redis not configured, skipping initialization")
        return None
    
    try:
        # Использовать REDIS_URL если указан, иначе собрать из компонентов
        if redis_url:
            client = aioredis.from_url(
                redis_url,
                decode_responses=False,  # Работаем с bytes для совместимости
                socket_connect_timeout=5,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                retry_on_timeout=True,
                health_check_interval=30
            )
        else:
            redis_port = getattr(config, 'redis_port', 6379)
            redis_db = getattr(config, 'redis_db', 0)
            redis_password = getattr(config, 'redis_password', None)
            
            client = aioredis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                retry_on_timeout=True,
                health_check_interval=30
            )
        
        # Проверить подключение
        await client.ping()
        logger.info(f"Redis client initialized successfully: {redis_host or redis_url}")
        
        _redis_client = client
        return client
        
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Failed to connect to Redis: {e}")
        _redis_client = None
        raise RedisError(f"Redis connection failed: {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected error initializing Redis: {e}")
        _redis_client = None
        raise RedisError(f"Redis initialization failed: {e}") from e


async def close_redis_client() -> None:
    """
    Закрыть подключение к Redis.
    """
    global _redis_client
    
    if _redis_client:
        try:
            await _redis_client.aclose()
            logger.info("Redis client closed")
        except Exception as e:
            logger.warning(f"Error closing Redis client: {e}")
        finally:
            _redis_client = None


async def check_redis_health() -> dict:
    """
    Проверить здоровье Redis подключения.
    
    Returns:
        Словарь с информацией о статусе Redis:
        {
            "status": "healthy" | "unhealthy" | "not_configured",
            "error": str (если есть)
        }
    """
    if not _redis_client:
        return {
            "status": "not_configured",
            "error": "Redis client not initialized"
        }
    
    try:
        # Проверка подключения
        await _redis_client.ping()
        
        # Дополнительная проверка: получить информацию о сервере
        info = await _redis_client.info("server")
        
        return {
            "status": "healthy",
            "redis_version": info.get("redis_version", "unknown") if isinstance(info, dict) else "unknown"
        }
        
    except (ConnectionError, TimeoutError) as e:
        return {
            "status": "unhealthy",
            "error": f"Connection error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": f"Health check failed: {str(e)}"
        }


# redis_url уже есть в config через pydantic-settings, хак больше не нужен

