"""State Machine для Fetcher - управление переходами статусов run'ов.

Этот модуль реализует строгую state machine для управления статусами run'ов,
предотвращая недопустимые переходы между статусами (event ordering correctness).

Соответствует Quality Assurance Checklist (Event ordering correctness).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Статусы Fetcher (из BACKEND_CONTRACTS.md и PIPELINE_ORCHESTRATION.md)
# Используем строки для совместимости с существующим кодом
RUN_STATUS_PENDING = "PENDING"
RUN_STATUS_NORMALIZING_SOURCE = "NORMALIZING_SOURCE"
RUN_STATUS_CHECKING_CACHE = "CHECKING_CACHE"
RUN_STATUS_FETCHING_METADATA = "FETCHING_METADATA"
RUN_STATUS_FETCHING_CHANNEL = "FETCHING_CHANNEL"
RUN_STATUS_FETCHING_COMMENTS = "FETCHING_COMMENTS"
RUN_STATUS_DOWNLOADING_VIDEO = "DOWNLOADING_VIDEO"
RUN_STATUS_UPLOADING_ARTIFACTS = "UPLOADING_ARTIFACTS"
RUN_STATUS_FINALIZING = "FINALIZING"
RUN_STATUS_COMPLETED = "COMPLETED"
RUN_STATUS_FAILED = "FAILED"

# Таблица разрешенных переходов между статусами
# Ключ - текущий статус, значение - список разрешенных следующих статусов
ALLOWED_TRANSITIONS: Dict[str, List[str]] = {
    RUN_STATUS_PENDING: [RUN_STATUS_NORMALIZING_SOURCE, RUN_STATUS_FAILED],
    RUN_STATUS_NORMALIZING_SOURCE: [RUN_STATUS_CHECKING_CACHE, RUN_STATUS_FAILED],
    RUN_STATUS_CHECKING_CACHE: [
        RUN_STATUS_FETCHING_METADATA,
        RUN_STATUS_FINALIZING,  # Cache hit
        RUN_STATUS_FAILED,
    ],
    RUN_STATUS_FETCHING_METADATA: [
        RUN_STATUS_FETCHING_CHANNEL,
        RUN_STATUS_FETCHING_COMMENTS,  # Если channel не нужен
        RUN_STATUS_DOWNLOADING_VIDEO,  # Параллельные задачи
        RUN_STATUS_UPLOADING_ARTIFACTS,  # Если всё готово
        # Параллельный ingest: все артефакты готовы → finalize (см. _maybe_enqueue_finalize_after_cache_miss)
        RUN_STATUS_FINALIZING,
        RUN_STATUS_FAILED,
    ],
    RUN_STATUS_FETCHING_CHANNEL: [
        RUN_STATUS_FETCHING_COMMENTS,
        RUN_STATUS_DOWNLOADING_VIDEO,  # Параллельные задачи
        RUN_STATUS_UPLOADING_ARTIFACTS,  # Если всё готово
        RUN_STATUS_FINALIZING,
        RUN_STATUS_FAILED,
    ],
    RUN_STATUS_FETCHING_COMMENTS: [
        RUN_STATUS_DOWNLOADING_VIDEO,  # Параллельные задачи
        RUN_STATUS_UPLOADING_ARTIFACTS,  # Если всё готово
        RUN_STATUS_FINALIZING,
        RUN_STATUS_FAILED,
    ],
    RUN_STATUS_DOWNLOADING_VIDEO: [
        RUN_STATUS_UPLOADING_ARTIFACTS,
        RUN_STATUS_FINALIZING,
        RUN_STATUS_FAILED,
    ],
    RUN_STATUS_UPLOADING_ARTIFACTS: [
        RUN_STATUS_FINALIZING,
        RUN_STATUS_FAILED,
    ],
    RUN_STATUS_FINALIZING: [
        RUN_STATUS_COMPLETED,
        RUN_STATUS_FAILED,
    ],
    RUN_STATUS_COMPLETED: [],  # Финальное состояние
    RUN_STATUS_FAILED: [],  # Финальное состояние
}


def can_transition(from_status: str, to_status: str) -> bool:
    """Проверить, разрешен ли переход между статусами.

    Args:
        from_status: Текущий статус
        to_status: Целевой статус

    Returns:
        True если переход разрешен, False иначе
    """
    allowed = ALLOWED_TRANSITIONS.get(from_status, [])
    return to_status in allowed


def validate_transition(
    from_status: Optional[str],
    to_status: str,
    run_id: Optional[str] = None,
) -> None:
    """Валидировать переход между статусами.

    Вызывает ValueError если переход не разрешен.

    Args:
        from_status: Текущий статус (может быть None для новых run'ов)
        to_status: Целевой статус
        run_id: UUID run'а для логирования (опционально)

    Raises:
        ValueError: Если переход не разрешен
    """
    # Если статус не установлен (новый run), разрешаем только PENDING
    if from_status is None:
        if to_status != RUN_STATUS_PENDING:
            raise ValueError(
                f"Invalid initial status: {to_status}. "
                f"New runs must start with '{RUN_STATUS_PENDING}'"
            )
        return

    # Нормализуем статусы (uppercase)
    from_status_upper = from_status.upper() if from_status else None
    to_status_upper = to_status.upper()

    # Проверить разрешен ли переход
    if not can_transition(from_status_upper, to_status_upper):
        allowed = ALLOWED_TRANSITIONS.get(from_status_upper, [])
        error_msg = (
            f"Invalid status transition: {from_status_upper} → {to_status_upper}. "
            f"Allowed transitions from {from_status_upper}: {allowed}"
        )
        if run_id:
            error_msg = f"Run {run_id}: {error_msg}"

        logger.warning(error_msg)
        raise ValueError(error_msg)

    logger.debug(
        f"Status transition validated: {from_status_upper} → {to_status_upper}"
        + (f" (run_id: {run_id})" if run_id else "")
    )


def get_allowed_transitions(from_status: str) -> List[str]:
    """Получить список разрешенных переходов из текущего статуса.

    Args:
        from_status: Текущий статус

    Returns:
        Список разрешенных следующих статусов
    """
    from_status_upper = from_status.upper() if from_status else None
    return ALLOWED_TRANSITIONS.get(from_status_upper, [])


def is_final_status(status: str) -> bool:
    """Проверить, является ли статус финальным.

    Args:
        status: Статус для проверки

    Returns:
        True если статус финальный (COMPLETED или FAILED), False иначе
    """
    status_upper = status.upper()
    return status_upper in (RUN_STATUS_COMPLETED, RUN_STATUS_FAILED)


__all__ = [
    "RUN_STATUS_PENDING",
    "RUN_STATUS_NORMALIZING_SOURCE",
    "RUN_STATUS_CHECKING_CACHE",
    "RUN_STATUS_FETCHING_METADATA",
    "RUN_STATUS_FETCHING_CHANNEL",
    "RUN_STATUS_FETCHING_COMMENTS",
    "RUN_STATUS_DOWNLOADING_VIDEO",
    "RUN_STATUS_UPLOADING_ARTIFACTS",
    "RUN_STATUS_FINALIZING",
    "RUN_STATUS_COMPLETED",
    "RUN_STATUS_FAILED",
    "can_transition",
    "validate_transition",
    "get_allowed_transitions",
    "is_final_status",
]

