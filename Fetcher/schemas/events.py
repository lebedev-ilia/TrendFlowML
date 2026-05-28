from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


FetcherEventType = Literal[
    "run.status_changed",
    "run.stage_changed",
    "job.started",
    "job.finished",
    "job.failed",
    "log.line",
]


class FetcherEventBase(BaseModel):
    """Базовый envelope события Fetcher.

    Поля соответствуют разделу "2. Схема событий pipeline" в BACKEND_CONTRACTS.md.
    """

    event_version: str = Field(
        "1.0",
        description="Версия схемы события. Мажорное изменение требует новой версии.",
    )
    source: Literal["fetcher"] = Field(
        "fetcher",
        description='Источник события. Для Fetcher всегда "fetcher".',
    )
    run_id: UUID = Field(
        ...,
        description="run_id, к которому относится событие.",
    )
    type: FetcherEventType = Field(
        ...,
        description="Тип события (run.status_changed, job.failed, log.line, ...).",
    )
    ts: datetime = Field(
        ...,
        description="Момент генерации события в формате UTC (ISO 8601).",
    )

    # Рекомендуемые поля (могут быть пустыми для некоторых типов)
    platform: Optional[str] = Field(
        None,
        description="Платформа видео (youtube, tiktok, ...), если известна на момент события.",
    )
    platform_video_id: Optional[str] = Field(
        None,
        description="Нормализованный идентификатор видео на платформе.",
    )
    stage: Optional[str] = Field(
        None,
        description="Текущий шаг state machine Fetcher (FETCHING_METADATA, DOWNLOADING_VIDEO, ...).",
    )
    status: Optional[str] = Field(
        None,
        description="Агрегированный статус на момент события (running, success, error, ...).",
    )


class FetcherRunStatusChangedPayload(BaseModel):
    """Детали события run.status_changed."""

    old_status: Optional[str] = Field(
        None,
        description="Предыдущий статус run'а в Fetcher (может быть None для первого статуса).",
    )
    new_status: str = Field(
        ...,
        description="Новый статус run'а в Fetcher.",
    )
    reason: Optional[str] = Field(
        None,
        description="Человекочитаемое объяснение (особенно для FAILED).",
    )
    error_code: Optional[str] = Field(
        None,
        description="Машинно-читаемый код ошибки (YOUTUBE_429, VIDEO_NOT_FOUND, DOWNLOAD_TIMEOUT, ...).",
    )


class FetcherRunStageChangedPayload(BaseModel):
    """Детали события run.stage_changed."""

    old_stage: Optional[str] = Field(
        None,
        description="Предыдущий шаг state machine (может быть None для первого).",
    )
    new_stage: str = Field(
        ...,
        description="Новый шаг state machine Fetcher.",
    )


class FetcherJobStartedPayload(BaseModel):
    """Детали события job.started."""

    job_type: str = Field(
        ...,
        description="Тип job'а (fetch_metadata, download_video, fetch_comments, finalize, ...).",
    )
    job_id: UUID = Field(
        ...,
        description="Идентификатор записи в таблице fetch_jobs.",
    )


class FetcherJobFinishedPayload(BaseModel):
    """Детали события job.finished."""

    job_type: str = Field(
        ...,
        description="Тип job'а (fetch_metadata, download_video, fetch_comments, finalize, ...).",
    )
    job_id: UUID = Field(
        ...,
        description="Идентификатор записи в таблице fetch_jobs.",
    )
    duration_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Длительность выполнения job'а в миллисекундах (если измеряется).",
    )


class FetcherJobFailedPayload(BaseModel):
    """Детали события job.failed."""

    job_type: str = Field(
        ...,
        description="Тип job'а.",
    )
    job_id: UUID = Field(
        ...,
        description="Идентификатор записи в таблице fetch_jobs.",
    )
    error_code: str = Field(
        ...,
        description="Нормализованный код ошибки.",
    )
    error_message: Optional[str] = Field(
        None,
        description="Человекочитаемое описание ошибки.",
    )


class FetcherLogLinePayload(BaseModel):
    """Детали события log.line."""

    level: Literal["debug", "info", "warning", "error"] = Field(
        "info",
        description="Уровень лог-сообщения.",
    )
    message: str = Field(
        ...,
        description="Текст лог-сообщения.",
    )


class FetcherRunStatusChangedEvent(FetcherEventBase):
    """Событие типа run.status_changed."""

    type: Literal["run.status_changed"] = "run.status_changed"
    payload: FetcherRunStatusChangedPayload


class FetcherRunStageChangedEvent(FetcherEventBase):
    """Событие типа run.stage_changed."""

    type: Literal["run.stage_changed"] = "run.stage_changed"
    payload: FetcherRunStageChangedPayload


class FetcherJobStartedEvent(FetcherEventBase):
    """Событие типа job.started."""

    type: Literal["job.started"] = "job.started"
    payload: FetcherJobStartedPayload


class FetcherJobFinishedEvent(FetcherEventBase):
    """Событие типа job.finished."""

    type: Literal["job.finished"] = "job.finished"
    payload: FetcherJobFinishedPayload


class FetcherJobFailedEvent(FetcherEventBase):
    """Событие типа job.failed."""

    type: Literal["job.failed"] = "job.failed"
    payload: FetcherJobFailedPayload


class FetcherLogLineEvent(FetcherEventBase):
    """Событие типа log.line."""

    type: Literal["log.line"] = "log.line"
    payload: FetcherLogLinePayload


FetcherEvent = Union[
    FetcherRunStatusChangedEvent,
    FetcherRunStageChangedEvent,
    FetcherJobStartedEvent,
    FetcherJobFinishedEvent,
    FetcherJobFailedEvent,
    FetcherLogLineEvent,
]


__all__ = [
    "FetcherEventType",
    "FetcherEventBase",
    "FetcherRunStatusChangedPayload",
    "FetcherRunStageChangedPayload",
    "FetcherJobStartedPayload",
    "FetcherJobFinishedPayload",
    "FetcherJobFailedPayload",
    "FetcherLogLinePayload",
    "FetcherRunStatusChangedEvent",
    "FetcherRunStageChangedEvent",
    "FetcherJobStartedEvent",
    "FetcherJobFinishedEvent",
    "FetcherJobFailedEvent",
    "FetcherLogLineEvent",
    "FetcherEvent",
]


