"""
Audit Log Service - логирование всех действий для аудита

Логирует все действия пользователей и системы в Redis для последующего анализа.
Включает информацию о действиях, run_id, пользователе, timestamp и детали.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2798-2806)
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from api.services.redis_client import get_redis_client
from api.config import config
from api.utils.logging import get_logger

logger = get_logger(__name__)

# Redis ключ для audit log
AUDIT_LOG_KEY = "audit:log"
# Максимальная длина списка audit log (для ограничения размера)
AUDIT_LOG_MAX_LENGTH = 10000


async def audit_log(
    action: str,
    run_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    ip_address: Optional[str] = None
) -> None:
    """
    Записать запись в audit log.
    
    Логирует действие в Redis для последующего анализа и аудита.
    
    Args:
        action: Тип действия (например, "process_started", "process_completed", "run_cancelled")
        run_id: UUID run'а (если применимо)
        details: Дополнительные детали действия
        user_id: ID пользователя (если доступно)
        request_id: Request ID из middleware
        ip_address: IP адрес клиента
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 2798-2806)
    """
    # Проверить, включен ли audit log
    if not getattr(config, 'audit_log_enabled', True):
        return
    
    redis_client = get_redis_client()
    if not redis_client:
        # Если Redis недоступен, логируем только в обычный лог
        logger.warning(
            "Audit log: Redis not available, logging to standard log only",
            action=action,
            run_id=run_id,
            request_id=request_id
        )
        logger.info(
            f"Audit: {action}",
            run_id=run_id,
            request_id=request_id,
            user_id=user_id,
            ip_address=ip_address,
            details=details
        )
        return
    
    try:
        # Создать запись audit log
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "run_id": run_id,
            "user_id": user_id,
            "request_id": request_id,
            "ip_address": ip_address,
            "details": details or {}
        }
        
        # Сериализовать в JSON
        log_entry_json = json.dumps(log_entry)
        
        # Добавить в Redis список (LPUSH для добавления в начало)
        await redis_client.lpush(AUDIT_LOG_KEY, log_entry_json)
        
        # Установить TTL для ключа
        audit_log_ttl = getattr(config, 'audit_log_ttl', 30 * 24 * 3600)  # 30 дней по умолчанию
        await redis_client.expire(AUDIT_LOG_KEY, audit_log_ttl)
        
        # Ограничить размер списка (удалить старые записи)
        await redis_client.ltrim(AUDIT_LOG_KEY, 0, AUDIT_LOG_MAX_LENGTH - 1)
        
        logger.debug(
            "Audit log entry created",
            action=action,
            run_id=run_id,
            request_id=request_id
        )
        
    except Exception as e:
        # Не прерывать выполнение при ошибке audit log
        logger.error(
            "Failed to write audit log entry",
            action=action,
            run_id=run_id,
            request_id=request_id,
            error=str(e)
        )


async def get_audit_logs(
    limit: int = 100,
    action: Optional[str] = None,
    run_id: Optional[str] = None
) -> list[Dict[str, Any]]:
    """
    Получить записи из audit log.
    
    Args:
        limit: Максимальное количество записей для возврата
        action: Фильтр по типу действия (опционально)
        run_id: Фильтр по run_id (опционально)
        
    Returns:
        Список записей audit log
    """
    redis_client = get_redis_client()
    if not redis_client:
        return []
    
    try:
        # Получить записи из Redis (LRANGE для чтения из начала списка)
        entries_json = await redis_client.lrange(AUDIT_LOG_KEY, 0, limit - 1)
        
        # Десериализовать JSON записи
        entries = []
        for entry_json in entries_json:
            try:
                entry = json.loads(entry_json)
                
                # Применить фильтры
                if action and entry.get("action") != action:
                    continue
                if run_id and entry.get("run_id") != run_id:
                    continue
                
                entries.append(entry)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse audit log entry: {entry_json}")
                continue
        
        return entries
        
    except Exception as e:
        logger.error(f"Failed to read audit log: {e}")
        return []

