"""Модуль для публикации событий Fetcher.

Поддерживает публикацию событий в Kafka (если включено) или только логирование.
Интегрируется с state machine и workers для автоматической публикации событий.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from schemas.events import (
    FetcherEvent,
    FetcherJobFailedEvent,
    FetcherJobFinishedEvent,
    FetcherJobStartedEvent,
    FetcherRunStageChangedEvent,
    FetcherRunStatusChangedEvent,
)
from .config import settings
from .kafka_producer import get_producer, publish_event as kafka_publish_event

logger = logging.getLogger(__name__)


def publish_status_changed(
    run_id: UUID | str,
    old_status: Optional[str],
    new_status: str,
    platform: Optional[str] = None,
    platform_video_id: Optional[str] = None,
    stage: Optional[str] = None,
    reason: Optional[str] = None,
    error_code: Optional[str] = None,
) -> None:
    """Опубликовать событие изменения статуса run'а.

    Args:
        run_id: UUID run'а
        old_status: Предыдущий статус
        new_status: Новый статус
        platform: Платформа (опционально)
        platform_video_id: ID видео на платформе (опционально)
        stage: Текущий stage (опционально)
        reason: Причина изменения (опционально)
        error_code: Код ошибки (опционально)
    """
    event = FetcherRunStatusChangedEvent(
        event_version="1.0",
        source="fetcher",
        run_id=UUID(str(run_id)),
        type="run.status_changed",
        ts=datetime.now(timezone.utc),
        platform=platform,
        platform_video_id=platform_video_id,
        stage=stage,
        status=new_status,
        payload={
            "old_status": old_status,
            "new_status": new_status,
            "reason": reason,
            "error_code": error_code,
        },
    )

    # Публикуем в Kafka, если включено
    if settings.kafka_enabled:
        kafka_publish_event(event)

    logger.debug(f"Published status_changed event: run_id={run_id}, {old_status} → {new_status}")


def publish_stage_changed(
    run_id: UUID | str,
    old_stage: Optional[str],
    new_stage: str,
    platform: Optional[str] = None,
    platform_video_id: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    """Опубликовать событие изменения stage run'а.

    Args:
        run_id: UUID run'а
        old_stage: Предыдущий stage
        new_stage: Новый stage
        platform: Платформа (опционально)
        platform_video_id: ID видео на платформе (опционально)
        status: Текущий статус (опционально)
    """
    event = FetcherRunStageChangedEvent(
        event_version="1.0",
        source="fetcher",
        run_id=UUID(str(run_id)),
        type="run.stage_changed",
        ts=datetime.now(timezone.utc),
        platform=platform,
        platform_video_id=platform_video_id,
        stage=new_stage,
        status=status,
        payload={
            "old_stage": old_stage,
            "new_stage": new_stage,
        },
    )

    # Публикуем в Kafka, если включено
    if settings.kafka_enabled:
        kafka_publish_event(event)

    logger.debug(f"Published stage_changed event: run_id={run_id}, {old_stage} → {new_stage}")


def publish_job_started(
    run_id: UUID | str,
    job_type: str,
    job_id: UUID | str,
    platform: Optional[str] = None,
    platform_video_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    """Опубликовать событие начала job'а.

    Args:
        run_id: UUID run'а
        job_type: Тип job'а (fetch_metadata, download_video, etc.)
        job_id: UUID job'а
        platform: Платформа (опционально)
        platform_video_id: ID видео на платформе (опционально)
        stage: Текущий stage (опционально)
    """
    event = FetcherJobStartedEvent(
        event_version="1.0",
        source="fetcher",
        run_id=UUID(str(run_id)),
        type="job.started",
        ts=datetime.now(timezone.utc),
        platform=platform,
        platform_video_id=platform_video_id,
        stage=stage,
        payload={
            "job_type": job_type,
            "job_id": UUID(str(job_id)),
        },
    )

    # Публикуем в Kafka, если включено
    if settings.kafka_enabled:
        kafka_publish_event(event)

    logger.debug(f"Published job_started event: run_id={run_id}, job_type={job_type}, job_id={job_id}")


def publish_job_finished(
    run_id: UUID | str,
    job_type: str,
    job_id: UUID | str,
    duration_ms: Optional[int] = None,
    platform: Optional[str] = None,
    platform_video_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    """Опубликовать событие завершения job'а.

    Args:
        run_id: UUID run'а
        job_type: Тип job'а
        job_id: UUID job'а
        duration_ms: Длительность выполнения в миллисекундах (опционально)
        platform: Платформа (опционально)
        platform_video_id: ID видео на платформе (опционально)
        stage: Текущий stage (опционально)
    """
    event = FetcherJobFinishedEvent(
        event_version="1.0",
        source="fetcher",
        run_id=UUID(str(run_id)),
        type="job.finished",
        ts=datetime.now(timezone.utc),
        platform=platform,
        platform_video_id=platform_video_id,
        stage=stage,
        payload={
            "job_type": job_type,
            "job_id": UUID(str(job_id)),
            "duration_ms": duration_ms,
        },
    )

    # Публикуем в Kafka, если включено
    if settings.kafka_enabled:
        kafka_publish_event(event)

    logger.debug(f"Published job_finished event: run_id={run_id}, job_type={job_type}, job_id={job_id}")


def publish_job_failed(
    run_id: UUID | str,
    job_type: str,
    job_id: UUID | str,
    error_code: str,
    error_message: Optional[str] = None,
    platform: Optional[str] = None,
    platform_video_id: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    """Опубликовать событие ошибки job'а.

    Args:
        run_id: UUID run'а
        job_type: Тип job'а
        job_id: UUID job'а
        error_code: Код ошибки
        error_message: Сообщение об ошибке (опционально)
        platform: Платформа (опционально)
        platform_video_id: ID видео на платформе (опционально)
        stage: Текущий stage (опционально)
    """
    event = FetcherJobFailedEvent(
        event_version="1.0",
        source="fetcher",
        run_id=UUID(str(run_id)),
        type="job.failed",
        ts=datetime.now(timezone.utc),
        platform=platform,
        platform_video_id=platform_video_id,
        stage=stage,
        status="error",
        payload={
            "job_type": job_type,
            "job_id": UUID(str(job_id)),
            "error_code": error_code,
            "error_message": error_message,
        },
    )

    # Публикуем в Kafka, если включено
    if settings.kafka_enabled:
        kafka_publish_event(event)

    logger.debug(f"Published job_failed event: run_id={run_id}, job_type={job_type}, error_code={error_code}")


__all__ = [
    "publish_status_changed",
    "publish_stage_changed",
    "publish_job_started",
    "publish_job_finished",
    "publish_job_failed",
]

