from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(Text, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    role = Column(Text, nullable=False, default="user")
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow)

    profiles = relationship("AnalysisProfile", back_populates="user")


class Video(Base):
    __tablename__ = "videos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    platform_id = Column(Text, nullable=False)
    video_id = Column(Text, nullable=False)
    source_type = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    language = Column(Text, nullable=True)
    category = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    sources = relationship("VideoSource", back_populates="video")


class VideoFile(Base):
    __tablename__ = "video_files"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    sha256_hex = Column(Text, unique=True, nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    mime_type = Column(Text, nullable=True)
    object_key = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    retention_until = Column(DateTime, nullable=True)


class VideoSource(Base):
    __tablename__ = "video_sources"

    video_id = Column(String(36), ForeignKey("videos.id"), primary_key=True)
    youtube_url = Column(Text, nullable=True)
    uploaded_file_id = Column(String(36), ForeignKey("video_files.id"), nullable=True)
    fetched_at = Column(DateTime, nullable=True)
    duration_sec = Column(Integer, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    video = relationship("Video", back_populates="sources")


class UserVideoLink(Base):
    __tablename__ = "user_video_links"

    user_id = Column(String(36), ForeignKey("users.id"), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), primary_key=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class AnalysisProfile(Base):
    __tablename__ = "analysis_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, nullable=False, default=False)
    config_json = Column(JSON, nullable=False, default=dict)
    config_hash = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow)

    user = relationship("User", back_populates="profiles")


class ProfileComponent(Base):
    __tablename__ = "profile_components"

    profile_id = Column(String(36), ForeignKey("analysis_profiles.id"), primary_key=True)
    component_name = Column(Text, primary_key=True)
    enabled = Column(Boolean, nullable=False, default=True)
    required = Column(Boolean, nullable=False, default=True)
    component_params = Column(JSON, nullable=False, default=dict)
    cost_units = Column(BigInteger, nullable=False, default=0)


class Run(Base):
    __tablename__ = "runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    profile_id = Column(String(36), ForeignKey("analysis_profiles.id"), nullable=True)
    config_hash = Column(Text, nullable=False, default="")
    status = Column(Text, nullable=False, default="queued")
    stage = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    cancel_requested_at = Column(DateTime, nullable=True)
    error_code = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    estimated_cost_units = Column(BigInteger, nullable=False, default=0)
    actual_cost_units = Column(BigInteger, nullable=False, default=0)


class RunComponent(Base):
    __tablename__ = "run_components"

    run_id = Column(String(36), ForeignKey("runs.id"), primary_key=True)
    component_name = Column(Text, primary_key=True)
    status = Column(Text, nullable=False, default="queued")
    schema_version = Column(Text, nullable=True)
    producer_version = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    device_used = Column(Text, nullable=True)
    empty_reason = Column(Text, nullable=True)
    error_code = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    cost_units = Column(BigInteger, nullable=False, default=0)


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    component_name = Column(Text, nullable=False)
    kind = Column(Text, nullable=False)
    object_key = Column(Text, nullable=False)
    size_bytes = Column(BigInteger, nullable=True)
    sha256_hex = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class RunLog(Base):
    __tablename__ = "run_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    ts = Column(DateTime, nullable=False, default=_utcnow)
    level = Column(Text, nullable=False)
    message = Column(Text, nullable=False)


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    status = Column(Text, nullable=False, default="init")
    temp_path = Column(Text, nullable=True)
    filename = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow)

