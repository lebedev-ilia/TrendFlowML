from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Базовый DeclarativeBase для моделей Fetcher."""


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class Run(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Таблица runs.

    См. `Fetcher/docs/DATABASE.md` — раздел 2.1.
    """

    __tablename__ = "runs"

    source_type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    video_sources: Mapped[list["VideoSource"]] = relationship(back_populates="run")
    jobs: Mapped[list["FetchJob"]] = relationship(back_populates="run")
    logs: Mapped[list["FetchLog"]] = relationship(back_populates="run")


class VideoSource(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Таблица video_sources.

    См. `Fetcher/docs/DATABASE.md` — раздел 2.1.
    """

    __tablename__ = "video_sources"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    platform: Mapped[str] = mapped_column(String(20), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_video_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    run: Mapped[Run] = relationship(back_populates="video_sources")


class Video(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Таблица videos (глобальный кеш по платформе и video_id)."""

    __tablename__ = "videos"
    __table_args__ = (
        UniqueConstraint("platform", "platform_video_id", name="uq_videos_platform_vid"),
    )

    platform: Mapped[str] = mapped_column(String(20), nullable=False)
    platform_video_id: Mapped[str] = mapped_column(String(100), nullable=False)
    channel_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    video_metadata: Mapped[Optional["VideoMetadata"]] = relationship(
        back_populates="video", uselist=False
    )
    channel_metadata: Mapped[Optional["ChannelMetadata"]] = relationship(
        back_populates="video", uselist=False
    )
    snapshots: Mapped[list["VideoSnapshot"]] = relationship(back_populates="video")
    comments: Mapped[list["Comment"]] = relationship(back_populates="video")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="video")


class VideoMetadata(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Таблица video_metadata."""

    __tablename__ = "video_metadata"

    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False
    )
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    video: Mapped[Video] = relationship(back_populates="video_metadata")


class ChannelMetadata(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Таблица channel_metadata."""

    __tablename__ = "channel_metadata"

    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False
    )
    channel_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    channel_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subscriber_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    video_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    view_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    video: Mapped[Video] = relationship(back_populates="channel_metadata")


class VideoSnapshot(Base, UUIDPrimaryKeyMixin):
    """Таблица video_snapshots."""

    __tablename__ = "video_snapshots"

    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False
    )
    snapshot_index: Mapped[int] = mapped_column(Integer, nullable=False)
    view_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    like_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    comment_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    subscriber_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    collected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    video: Mapped[Video] = relationship(back_populates="snapshots")


class Comment(Base, UUIDPrimaryKeyMixin):
    """Таблица comments."""

    __tablename__ = "comments"

    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False
    )
    author: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    like_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reply_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    video: Mapped[Video] = relationship(back_populates="comments")


class Artifact(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Таблица artifacts (артефакты в S3/MinIO)."""

    __tablename__ = "artifacts"

    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False
    )
    artifact_type: Mapped[str] = mapped_column(String(30), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    checksum: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    video: Mapped[Video] = relationship(back_populates="artifacts")


class FetchJob(Base, UUIDPrimaryKeyMixin):
    """Таблица fetch_jobs (отдельные шаги pipeline)."""

    __tablename__ = "fetch_jobs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    job_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="jobs")


class FetchLog(Base):
    """Таблица fetch_logs (event‑лог pipeline)."""

    __tablename__ = "fetch_logs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, nullable=False
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    stage: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    run: Mapped[Run] = relationship(back_populates="logs")


class Proxy(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Таблица proxies (конфигурация прокси-серверов).

    Используется для управления пулом прокси с поддержкой geographic rotation.
    """

    __tablename__ = "proxies"

    url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    country: Mapped[Optional[str]] = mapped_column(String(2), nullable=True)  # ISO 3166-1 alpha-2 код
    enabled: Mapped[bool] = mapped_column(default=True, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    usage_logs: Mapped[list["ProxyUsage"]] = relationship(back_populates="proxy")


class ProxyUsage(Base, UUIDPrimaryKeyMixin):
    """Таблица proxy_usage (детальный учёт использования прокси).

    Логирует каждое использование прокси для анализа и мониторинга.
    """

    __tablename__ = "proxy_usage"

    proxy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("proxies.id"), nullable=False
    )
    operation: Mapped[str] = mapped_column(String(50), nullable=False)  # metadata, download, comments
    success: Mapped[bool] = mapped_column(nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Время ответа в миллисекундах
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    proxy: Mapped[Proxy] = relationship(back_populates="usage_logs")


__all__ = [
    "Base",
    "Run",
    "VideoSource",
    "Video",
    "VideoMetadata",
    "ChannelMetadata",
    "VideoSnapshot",
    "Comment",
    "Artifact",
    "FetchJob",
    "FetchLog",
]


