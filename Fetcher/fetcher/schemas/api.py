"""Pydantic схемы для Fetcher REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, validator


# ============================================================================
# Request Schemas
# ============================================================================


class CreateRunRequest(BaseModel):
    """Request schema для POST /api/v1/runs."""

    run_id: UUID = Field(..., description="UUID run'а от Backend")
    source_url: HttpUrl = Field(..., description="URL видео для ingestion")
    platform: Optional[str] = Field(
        None,
        description="Платформа видео (youtube, tiktok, etc.). Если не указана, определяется автоматически",
    )
    priority: Optional[str] = Field(
        "normal",
        description="Приоритет run'а (low, normal, high). Определяет очередь",
    )
    webhook_url: Optional[HttpUrl] = Field(
        None,
        description="URL для webhook уведомлений при завершении run'а",
    )
    max_run_duration_seconds: Optional[int] = Field(
        7200,
        description="Максимальная длительность run'а в секундах (default: 2 часа). Watchdog отменит run если превышен",
        ge=60,
        le=86400,
    )

    @validator("priority")
    @classmethod
    def validate_priority(cls, v: Optional[str]) -> str:
        """Валидация приоритета."""
        if v is None:
            return "normal"
        if v not in ("low", "normal", "high"):
            raise ValueError("priority must be one of: low, normal, high")
        return v


# ============================================================================
# Response Schemas
# ============================================================================


class RunProgress(BaseModel):
    """Прогресс выполнения run'а."""

    stage: Optional[str] = Field(None, description="Текущая stage")
    completed_stages: list[str] = Field(default_factory=list, description="Завершённые stages")
    total_stages: int = Field(7, description="Общее количество stages")


class RunArtifactsInfo(BaseModel):
    """Информация об артефактах run'а."""

    video_file: Optional[str] = None
    meta_file: Optional[str] = None
    comments_file: Optional[str] = None
    manifest_file: Optional[str] = None


class RunResponse(BaseModel):
    """Response schema для GET /api/v1/runs/{run_id}."""

    run_id: UUID = Field(..., description="UUID run'а")
    status: str = Field(..., description="Статус run'а")
    source_url: str = Field(..., description="Исходный URL")
    platform: Optional[str] = Field(None, description="Платформа видео")
    platform_video_id: Optional[str] = Field(None, description="ID видео на платформе")
    created_at: datetime = Field(..., description="Время создания")
    started_at: Optional[datetime] = Field(None, description="Время начала выполнения")
    finished_at: Optional[datetime] = Field(None, description="Время завершения")
    error: Optional[str] = Field(None, description="Сообщение об ошибке (если есть)")
    error_code: Optional[str] = Field(None, description="Код ошибки (если есть)")
    cancel_requested: Optional[bool] = Field(
        None, description="Запрошена ли отмена run'а (будет отменён на следующем checkpoint)",
    )
    video_id: Optional[UUID] = Field(None, description="UUID видео в Fetcher БД")
    artifacts: Optional[RunArtifactsInfo] = Field(None, description="Информация об артефактах")
    progress: Optional[RunProgress] = Field(None, description="Прогресс выполнения")


class CreateRunResponse(BaseModel):
    """Response schema для POST /api/v1/runs."""

    run_id: UUID = Field(..., description="UUID run'а")
    status: str = Field(..., description="Статус run'а (обычно PENDING)")
    source_url: str = Field(..., description="Исходный URL")
    platform: Optional[str] = Field(None, description="Платформа видео")
    created_at: datetime = Field(..., description="Время создания")
    message: str = Field(..., description="Сообщение о результате создания")
    existing_run_id: Optional[UUID] = Field(
        None,
        description="UUID существующего run'а (если найден duplicate)",
    )


class ManifestResponse(BaseModel):
    """Response schema для GET /api/v1/runs/{run_id}/manifest."""

    manifest_version: str
    run_id: UUID
    video_id: str
    platform: str
    duration_seconds: float
    storage_layout_version: str
    artifacts: dict[str, Any]
    created_at: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Стандартизированный формат ошибок."""

    error: dict[str, Any] = Field(
        ...,
        description="Информация об ошибке",
        examples=[
            {
                "code": "RUN_NOT_FOUND",
                "message": "Run with id 550e8400-... not found",
                "details": {"run_id": "550e8400-e29b-41d4-a716-446655440000"},
            }
        ],
    )


# ============================================================================
# Phase 2 Schemas
# ============================================================================


class RunListItem(BaseModel):
    """Элемент списка runs для GET /api/v1/runs."""

    run_id: UUID = Field(..., description="UUID run'а")
    status: str = Field(..., description="Статус run'а")
    platform: Optional[str] = Field(None, description="Платформа видео")
    created_at: datetime = Field(..., description="Время создания")
    finished_at: Optional[datetime] = Field(None, description="Время завершения")


class RunListResponse(BaseModel):
    """Response schema для GET /api/v1/runs."""

    runs: list[RunListItem] = Field(..., description="Список runs")
    pagination: dict[str, Any] = Field(
        ...,
        description="Информация о пагинации",
        examples=[
            {
                "limit": 50,
                "has_more": True,
                "next_cursor": "eyJjcmVhdGVkX2F0IjoiMjAyNi0wMy0wNVQxMjowNTowMFoiLCJydW5faWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAifQ==",
            }
        ],
    )


class ArtifactItem(BaseModel):
    """Элемент списка артефактов."""

    artifact_type: str = Field(..., description="Тип артефакта (video_file, meta_file, etc.)")
    download_url: Optional[str] = Field(
        None,
        description="Signed URL для скачивания (null если artifact_status != READY)",
    )
    download_url_expires_at: Optional[datetime] = Field(
        None,
        description="Время истечения signed URL",
    )
    size_bytes: Optional[int] = Field(None, description="Размер артефакта в байтах")
    checksum: Optional[str] = Field(None, description="SHA256 checksum артефакта")
    artifact_status: str = Field(
        ...,
        description="Статус артефакта (PENDING, READY, FAILED)",
    )
    status: str = Field(..., description="Статус обработки (COMPLETED, IN_PROGRESS, etc.)")
    created_at: Optional[datetime] = Field(None, description="Время создания артефакта")


class ArtifactsResponse(BaseModel):
    """Response schema для GET /api/v1/runs/{run_id}/artifacts."""

    run_id: UUID = Field(..., description="UUID run'а")
    artifacts: list[ArtifactItem] = Field(..., description="Список артефактов")


class LogsUrlResponse(BaseModel):
    """Response schema для GET /api/v1/runs/{run_id}/logs_url."""

    run_id: UUID = Field(..., description="UUID run'а")
    logs_url: Optional[str] = Field(
        None,
        description="URL для доступа к логам через Grafana/Loki/Elasticsearch",
    )
    logs_backend: Optional[str] = Field(
        None,
        description="Backend для логов (loki, elasticsearch, cloudwatch)",
    )
    message: str = Field(..., description="Сообщение о доступности логов")


class RetryRunResponse(BaseModel):
    """Response schema для POST /api/v1/runs/{run_id}/retry."""

    run_id: UUID = Field(..., description="UUID run'а")
    status: str = Field(..., description="Новый статус run'а (обычно PENDING)")
    message: str = Field(..., description="Сообщение о результате перезапуска")


class UpdateRunRequest(BaseModel):
    """Request schema для PATCH /api/v1/runs/{run_id}."""

    cancel_requested: Optional[bool] = Field(
        None,
        description="Запросить отмену выполнения run'а",
    )


class UpdateRunResponse(BaseModel):
    """Response schema для PATCH /api/v1/runs/{run_id}."""

    run_id: UUID = Field(..., description="UUID run'а")
    status: str = Field(..., description="Статус run'а")
    cancel_requested: Optional[bool] = Field(
        None,
        description="Флаг запроса отмены",
    )
    message: str = Field(..., description="Сообщение о результате обновления")


# ============================================================================
# Phase 3 Schemas
# ============================================================================


class BulkCreateRunItem(BaseModel):
    """Элемент для bulk ingestion."""

    run_id: UUID = Field(..., description="UUID run'а от Backend")
    source_url: HttpUrl = Field(..., description="URL видео для ingestion")
    platform: Optional[str] = Field(
        None,
        description="Платформа видео (youtube, tiktok, etc.). Если не указана, определяется автоматически",
    )
    priority: Optional[str] = Field(
        "normal",
        description="Приоритет run'а (low, normal, high)",
    )


class BulkCreateRunsRequest(BaseModel):
    """Request schema для POST /api/v1/runs/bulk."""

    runs: list[BulkCreateRunItem] = Field(
        ...,
        description="Список runs для создания (1–100 элементов)",
    )


class BulkCreateRunsResponse(BaseModel):
    """Response schema для POST /api/v1/runs/bulk."""

    created: list[CreateRunResponse] = Field(..., description="Успешно созданные runs")
    duplicates: list[dict[str, Any]] = Field(
        ...,
        description="Дубликаты (existing_run_id, requested_run_id)",
    )
    errors: list[dict[str, Any]] = Field(
        ...,
        description="Ошибки при создании (run_id, error)",
    )
    total_requested: int = Field(..., description="Общее количество запрошенных runs")
    total_created: int = Field(..., description="Количество успешно созданных runs")


class RunEventItem(BaseModel):
    """Элемент истории событий run'а."""

    event_type: str = Field(..., description="Тип события (status_changed, stage_changed, etc.)")
    timestamp: datetime = Field(..., description="Время события")
    stage: Optional[str] = Field(None, description="Stage на момент события")
    status: Optional[str] = Field(None, description="Status на момент события")
    message: Optional[str] = Field(None, description="Сообщение события")
    level: Optional[str] = Field(None, description="Уровень лога (info, warning, error)")


class RunEventsResponse(BaseModel):
    """Response schema для GET /api/v1/runs/{run_id}/events."""

    run_id: UUID = Field(..., description="UUID run'а")
    events: list[RunEventItem] = Field(..., description="История событий")
    total: int = Field(..., description="Общее количество событий")


# ============================================================================
# Phase 4 Schemas
# ============================================================================


class QueueInfo(BaseModel):
    """Информация об очереди."""

    queue_name: str = Field(..., description="Имя очереди")
    pending: int = Field(..., description="Количество задач в очереди")
    running: int = Field(..., description="Количество выполняющихся задач")
    retry: int = Field(..., description="Количество задач на retry")


class QueueResponse(BaseModel):
    """Response schema для GET /api/v1/queue."""

    queues: list[QueueInfo] = Field(..., description="Информация о каждой очереди")
    total_pending: int = Field(..., description="Общее количество pending задач")
    total_running: int = Field(..., description="Общее количество running задач")
    total_retry: int = Field(..., description="Общее количество retry задач")
    kafka_consumer_lag: Optional[int] = Field(
        None,
        description="Kafka consumer lag (если Kafka включен)",
    )


class LimitsResponse(BaseModel):
    """Response schema для GET /api/v1/limits."""

    rate_limits: dict[str, Any] = Field(
        ...,
        description="Rate limits (runs_per_minute, runs_per_hour, runs_per_day)",
    )
    resource_limits: dict[str, Any] = Field(
        ...,
        description="Resource limits (max_video_size_mb, max_video_duration_seconds, max_comments_per_video)",
    )
    platform_limits: dict[str, dict[str, Any]] = Field(
        ...,
        description="Platform limits (max_requests_per_minute для каждой платформы)",
    )
    current_usage: dict[str, Any] = Field(
        ...,
        description="Current usage (runs_today, runs_this_hour)",
    )


class VideoCacheResponse(BaseModel):
    """Response schema для GET /api/v1/videos/{platform}/{video_id} — информация о видео из кеша."""

    video_id: UUID = Field(..., description="UUID видео в Fetcher БД")
    platform: str = Field(..., description="Платформа (youtube, tiktok, ...)")
    platform_video_id: str = Field(..., description="ID видео на платформе")
    artifacts_available: bool = Field(
        ...,
        description="Есть ли собранные артефакты (video, meta, comments) в storage",
    )
    snapshots_count: int = Field(0, description="Количество временных снэпшотов")
    comments_count: int = Field(0, description="Количество комментариев в БД")


class StatsResponse(BaseModel):
    """Response schema для GET /api/v1/stats."""

    period: str = Field(..., description="Период статистики (1h, 24h, 7d, 30d)")
    runs: dict[str, int] = Field(
        ...,
        description="Статистика по runs (total, completed, failed, running)",
    )
    throughput: dict[str, float] = Field(
        ...,
        description="Throughput (videos_per_hour, videos_per_day)",
    )
    cache: dict[str, Any] = Field(
        ...,
        description="Cache statistics (hit_rate, hits, misses)",
    )
    platforms: dict[str, int] = Field(
        ...,
        description="Статистика по платформам",
    )
    errors: dict[str, int] = Field(
        ...,
        description="Статистика по ошибкам",
    )


__all__ = [
    "CreateRunRequest",
    "CreateRunResponse",
    "RunResponse",
    "RunProgress",
    "RunArtifactsInfo",
    "ManifestResponse",
    "ErrorResponse",
    "RunListItem",
    "RunListResponse",
    "ArtifactItem",
    "ArtifactsResponse",
    "LogsUrlResponse",
    "RetryRunResponse",
    "UpdateRunRequest",
    "UpdateRunResponse",
    "VideoCacheResponse",
]

