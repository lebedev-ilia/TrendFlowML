from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field

from .dbv2.enums import AnalysisStatus, SourceType, SubscriptionStatus, VideoType, WorkspaceRole


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=72)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(max_length=72)


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    email_verified: bool
    created_at: datetime
    updated_at: datetime


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    slug: Optional[str] = Field(default=None, min_length=1, max_length=200)


class WorkspaceOut(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    owner_user_id: uuid.UUID
    archived_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class WorkspaceMemberAdd(BaseModel):
    user_email: EmailStr
    role: WorkspaceRole = WorkspaceRole.viewer


class WorkspaceMemberOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: uuid.UUID
    role: WorkspaceRole
    invited_by: Optional[uuid.UUID]
    joined_at: datetime
    archived_at: Optional[datetime]


class SubscriptionPlanOut(BaseModel):
    id: int
    name: str
    max_videos_per_month: int
    max_analyses_per_month: int
    max_channels: int
    max_storage_gb: int
    has_api_access: bool
    has_advanced_explainability: bool
    price: float


class SubscriptionCreate(BaseModel):
    plan_id: int
    status: SubscriptionStatus = SubscriptionStatus.active
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False


class SubscriptionOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    plan_id: int
    status: SubscriptionStatus
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    created_at: datetime
    updated_at: datetime


class ChannelCreate(BaseModel):
    platform: str = Field(min_length=1, max_length=50)
    external_channel_id: Optional[str] = None
    channel_name: str = Field(min_length=1, max_length=200)
    connected_oauth_id: Optional[uuid.UUID] = None


class ChannelOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    platform: str
    external_channel_id: Optional[str]
    channel_name: str
    connected_oauth_id: Optional[uuid.UUID]
    archived_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Processing configs (конфигурации анализа)
# ---------------------------------------------------------------------------


class ProcessingConfigCreate(BaseModel):
    """Состав конфигурации задаёт клиент; валидацией компонентов занимается
    DataProcessor при запуске обработки."""

    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    payload: dict = Field(default_factory=dict)
    estimated_cost_units: Optional[int] = Field(default=None, ge=0)
    estimated_minutes: Optional[int] = Field(default=None, ge=0)


class ProcessingConfigUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    payload: Optional[dict] = None
    estimated_cost_units: Optional[int] = Field(default=None, ge=0)
    estimated_minutes: Optional[int] = Field(default=None, ge=0)


class ProcessingConfigOut(BaseModel):
    id: uuid.UUID
    workspace_id: Optional[uuid.UUID]
    created_by_user_id: Optional[uuid.UUID]
    name: str
    description: Optional[str]
    is_system: bool
    payload: dict
    estimated_cost_units: Optional[int]
    estimated_minutes: Optional[int]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Billing (внутренние единицы)
# ---------------------------------------------------------------------------


class BalanceOut(BaseModel):
    workspace_id: uuid.UUID
    balance_units: int


class CreditTransactionOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    kind: str
    amount_units: int
    balance_after: int
    analysis_job_id: Optional[uuid.UUID]
    description: Optional[str]
    amount_rub: Optional[float]
    created_at: datetime


class TopUpRequest(BaseModel):
    """Пополнение баланса.

    TODO(платежи): сейчас единицы начисляются напрямую. При подключении
    платёжной системы этот эндпоинт должен вызываться только по подтверждению
    провайдера, а не по запросу клиента.
    """

    amount_units: int = Field(gt=0, le=1_000_000)
    amount_rub: Optional[float] = Field(default=None, ge=0)
    idempotency_key: Optional[str] = Field(default=None, max_length=200)


class VideoCreate(BaseModel):
    external_video_id: Optional[str] = None
    title: str = Field(min_length=1, max_length=500)
    description: Optional[str] = None
    duration_seconds: int = Field(ge=0)
    video_type: VideoType
    source_type: SourceType
    source_url: Optional[str] = None
    storage_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    checksum: Optional[str] = None


class VideoUpdate(BaseModel):
    """Частичное обновление видео: передаются только изменяемые поля."""

    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    description: Optional[str] = None
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    video_type: Optional[VideoType] = None
    source_url: Optional[str] = None
    storage_path: Optional[str] = None
    file_size_mb: Optional[float] = None


class VideoOut(BaseModel):
    id: uuid.UUID
    channel_id: uuid.UUID
    external_video_id: Optional[str]
    title: str
    description: Optional[str]
    duration_seconds: int
    video_type: VideoType
    source_type: SourceType
    source_url: Optional[str]
    storage_path: Optional[str]
    file_size_mb: Optional[float]
    checksum: Optional[str]
    archived_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class AnalysisJobCreate(BaseModel):
    processing_config_id: uuid.UUID
    model_version_id: str = Field(min_length=1, max_length=200)


class AnalysisJobOut(BaseModel):
    id: uuid.UUID
    workspace_id: uuid.UUID
    video_id: uuid.UUID
    triggered_by_user_id: uuid.UUID
    processing_config_id: uuid.UUID
    model_version_id: str
    status: AnalysisStatus
    retry_count: int
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class PredictionCreate(BaseModel):
    horizon_days: int = Field(ge=1)
    predicted_views: float
    predicted_likes: float
    percentile_score: float
    confidence_lower: float
    confidence_upper: float
    model_version_id: str = Field(min_length=1, max_length=200)


class PredictionOut(BaseModel):
    id: uuid.UUID
    analysis_job_id: uuid.UUID
    horizon_days: int
    predicted_views: float
    predicted_likes: float
    percentile_score: float
    confidence_lower: float
    confidence_upper: float
    model_version_id: str
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# Ingestion runs (Backend ↔ Fetcher, Phase 1). См. backend/docs/FETCHER_INTEGRATION.md
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    """Запрос на создание run по URL (YouTube и др.). Fetcher выполняет ingestion."""

    source_url: str = Field(..., min_length=1, description="URL видео (например YouTube)")
    workspace_id: Optional[uuid.UUID] = Field(
        None,
        description="Опционально: workspace для привязки run (проверяется доступ пользователя)",
    )


class IngestionRunOut(BaseModel):
    """Ответ: созданный или существующий run ингестиции (Phase 4: поля из Fetcher)."""

    run_id: uuid.UUID
    source_url: str
    workspace_id: Optional[uuid.UUID]
    ingestion_status: str
    created_at: datetime
    updated_at: datetime
    message: Optional[str] = Field(
        None,
        description="Сообщение от Fetcher или Backend (например при idempotency)",
    )
    fetcher_stage: Optional[str] = Field(
        None,
        description="Текущая стадия в Fetcher (Phase 4: metadata, video, comments, finalize, …)",
    )
    fetcher_error_code: Optional[str] = Field(
        None,
        description="Код ошибки от Fetcher при status=FAILED",
    )
    fetcher_error_message: Optional[str] = Field(
        None,
        description="Сообщение об ошибке от Fetcher",
    )


