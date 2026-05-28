"""
State Machine Service - управление переходами статусов run'ов

Этот модуль реализует строгую state machine для управления статусами run'ов,
предотвращая недопустимые переходы между статусами.

Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 903-944)
"""

import logging
from typing import Dict, List, Optional
from api.schemas.state import RunStatus

logger = logging.getLogger(__name__)

# Таблица разрешенных переходов между статусами
# Ключ - текущий статус, значение - список разрешенных следующих статусов
ALLOWED_TRANSITIONS: Dict[RunStatus, List[RunStatus]] = {
    RunStatus.PENDING: [RunStatus.QUEUED, RunStatus.CANCELLED],
    RunStatus.QUEUED: [RunStatus.RUNNING, RunStatus.CANCELLED],
    RunStatus.RUNNING: [
        RunStatus.SUCCESS,
        RunStatus.ERROR,
        RunStatus.RECOVERING,
        RunStatus.CANCELLED
    ],
    RunStatus.RECOVERING: [RunStatus.RUNNING, RunStatus.ERROR],
    RunStatus.SUCCESS: [],  # Финальное состояние
    RunStatus.ERROR: [],    # Финальное состояние
    RunStatus.CANCELLED: [] # Финальное состояние
}


def can_transition(from_status: RunStatus, to_status: RunStatus) -> bool:
    """
    Проверить, разрешен ли переход между статусами.
    
    Args:
        from_status: Текущий статус
        to_status: Целевой статус
        
    Returns:
        True если переход разрешен, False иначе
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 932-933)
    """
    allowed = ALLOWED_TRANSITIONS.get(from_status, [])
    return to_status in allowed


def validate_transition(
    from_status: Optional[RunStatus],
    to_status: RunStatus,
    run_id: Optional[str] = None
) -> None:
    """
    Валидировать переход между статусами.
    
    Вызывает ValueError если переход не разрешен.
    
    Args:
        from_status: Текущий статус (может быть None для новых run'ов)
        to_status: Целевой статус
        run_id: UUID run'а для логирования (опционально)
        
    Raises:
        ValueError: Если переход не разрешен
        
    Ссылка: DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md (строка 935-943)
    """
    # Если статус не установлен (новый run), разрешаем только PENDING или QUEUED
    if from_status is None:
        if to_status not in (RunStatus.PENDING, RunStatus.QUEUED):
            raise ValueError(
                f"Invalid initial status: {to_status.value}. "
                f"New runs must start with 'pending' or 'queued'"
            )
        return
    
    # Проверить разрешен ли переход
    if not can_transition(from_status, to_status):
        error_msg = (
            f"Invalid status transition: {from_status.value} → {to_status.value}. "
            f"Allowed transitions from {from_status.value}: "
            f"{[s.value for s in ALLOWED_TRANSITIONS.get(from_status, [])]}"
        )
        if run_id:
            error_msg = f"Run {run_id}: {error_msg}"
        
        logger.warning(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(
        f"Status transition validated: {from_status.value} → {to_status.value}"
        + (f" (run_id: {run_id})" if run_id else "")
    )


def get_allowed_transitions(from_status: RunStatus) -> List[RunStatus]:
    """
    Получить список разрешенных переходов из текущего статуса.
    
    Args:
        from_status: Текущий статус
        
    Returns:
        Список разрешенных следующих статусов
    """
    return ALLOWED_TRANSITIONS.get(from_status, [])


def is_final_status(status: RunStatus) -> bool:
    """
    Проверить, является ли статус финальным (не может перейти в другой статус).
    
    Args:
        status: Статус для проверки
        
    Returns:
        True если статус финальный, False иначе
    """
    return len(ALLOWED_TRANSITIONS.get(status, [])) == 0


def parse_status(status: str) -> RunStatus:
    """
    Парсить строку статуса в RunStatus enum.
    
    Args:
        status: Строка статуса
        
    Returns:
        RunStatus enum
        
    Raises:
        ValueError: Если статус не распознан
    """
    try:
        # Попробовать найти по значению
        for run_status in RunStatus:
            if run_status.value == status.lower():
                return run_status
        
        # Если не найдено, попробовать по имени (case-insensitive)
        status_upper = status.upper()
        for run_status in RunStatus:
            if run_status.name == status_upper:
                return run_status
        
        raise ValueError(f"Unknown status: {status}")
    except Exception as e:
        logger.error(f"Failed to parse status '{status}': {e}")
        raise ValueError(f"Invalid status: {status}") from e

