"""
Task Manager - управление задачами

Для MVP реализует in-memory registry активных run'ов.
В будущем (Этап 2) будет заменён на Redis-based registry.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 1383, 429, 1458)
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from api.config import config
from api.schemas.state import RunStatus
from api.services.state_machine import validate_transition, parse_status

logger = logging.getLogger(__name__)


class TaskManager:
    """
    Менеджер задач для отслеживания активных run'ов.
    
    Для MVP использует in-memory registry.
    В Этапе 2 будет заменён на Redis-based registry.
    """
    
    def __init__(self):
        """Инициализация TaskManager."""
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        self.semaphore = asyncio.Semaphore(config.max_concurrent_runs)
    
    def register_run(self, run_id: str, metadata: Dict[str, Any], initial_status: RunStatus = RunStatus.QUEUED) -> None:
        """
        Зарегистрировать новый run.
        
        Args:
            run_id: UUID run'а
            metadata: Метаданные run'а
            initial_status: Начальный статус (по умолчанию QUEUED)
        """
        # Валидировать начальный статус
        validate_transition(None, initial_status, run_id)
        
        self.active_runs[run_id] = {
            **metadata,
            "status": initial_status.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        logger.info(f"Registered run: {run_id} with status: {initial_status.value}")
        
        # Обновить метрику активных run'ов
        try:
            from api.services.metrics import active_runs
            active_runs.set(len([r for r in self.active_runs.values() 
                                if r.get("status") in ("queued", "running", "recovering")]))
        except Exception as e:
            logger.debug(f"Failed to update active_runs metric: {e}")
    
    def update_run_status(self, run_id: str, status: RunStatus, **kwargs) -> None:
        """
        Обновить статус run'а с валидацией перехода.
        
        Args:
            run_id: UUID run'а
            status: Новый статус (RunStatus enum)
            **kwargs: Дополнительные поля для обновления
            
        Raises:
            ValueError: Если переход не разрешен
        """
        if run_id not in self.active_runs:
            logger.warning(f"Attempted to update non-existent run: {run_id}")
            return
        
        # Получить текущий статус
        current_status_str = self.active_runs[run_id].get("status")
        current_status = None
        if current_status_str:
            try:
                current_status = parse_status(current_status_str)
            except ValueError:
                logger.warning(f"Invalid current status '{current_status_str}' for run {run_id}, allowing transition")
        
        # Валидировать переход
        validate_transition(current_status, status, run_id)
        
        # Обновить статус
        self.active_runs[run_id].update({
            "status": status.value,
            "updated_at": datetime.now().isoformat(),
            **kwargs
        })
        logger.debug(f"Updated run {run_id} status: {current_status.value if current_status else 'None'} → {status.value}")
        
        # Обновить метрику активных run'ов
        try:
            from api.services.metrics import active_runs
            active_runs.set(len([r for r in self.active_runs.values() 
                                if r.get("status") in ("queued", "running", "recovering")]))
        except Exception as e:
            logger.debug(f"Failed to update active_runs metric: {e}")
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию о run'е.
        
        Args:
            run_id: UUID run'а
            
        Returns:
            Словарь с информацией о run'е или None если не найден
        """
        return self.active_runs.get(run_id)
    
    def is_run_active(self, run_id: str) -> bool:
        """
        Проверить, активен ли run.
        
        Args:
            run_id: UUID run'а
            
        Returns:
            True если run активен, False иначе
        """
        run = self.active_runs.get(run_id)
        if not run:
            return False
        
        status_str = run.get("status", "unknown")
        try:
            status = parse_status(status_str)
            return status in (RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.RECOVERING)
        except ValueError:
            logger.warning(f"Invalid status '{status_str}' for run {run_id}")
            return False
    
    def can_accept_new_run(self) -> bool:
        """
        Проверить, можно ли принять новый run.
        
        Returns:
            True если можно принять новый run, False иначе
        """
        active_count = sum(
            1 for run in self.active_runs.values()
            if run.get("status") in ("queued", "running", "recovering")
        )
        return active_count < config.max_concurrent_runs
    
    def get_active_runs_count(self) -> int:
        """
        Получить количество активных run'ов.
        
        Returns:
            Количество активных run'ов
        """
        return sum(
            1 for run in self.active_runs.values()
            if run.get("status") in ("queued", "running", "recovering")
        )
    
    async def acquire_slot(self):
        """
        Получить слот для выполнения задачи (через semaphore).
        
        Semaphore уже является async context manager, поэтому возвращаем его напрямую.
        
        Использование:
            async with await task_manager.acquire_slot():
                # Выполнение задачи
        """
        return self.semaphore

