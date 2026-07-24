from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..config import Settings
from ..dbv2.models import Channel, Video, Workspace, WorkspaceMember
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import VideoCreate, VideoOut, VideoUpdate

# Ограничения загрузки (SITE_SPECIFICATION.md §5.6). Максимальный размер файла
# определяется экспериментально; 2 ГБ — безопасный потолок для видео до 20 минут.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

router = APIRouter(prefix="/api/channels/{channel_id}/videos", tags=["videos"])

# Операции над конкретным видео живут вне канала: клиент знает только video_id
# (например, из AnalysisJobOut). Контракт описан в docs/API.md §4.
video_router = APIRouter(prefix="/api/videos", tags=["videos"])


def _require_channel_access(db: Session, channel_id: uuid.UUID, user_id: uuid.UUID) -> Channel:
    ch = db.query(Channel).filter(Channel.id == channel_id).first()
    if not ch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Channel not found")
    ws = db.query(Workspace).filter(Workspace.id == ch.workspace_id).first()
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    m = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == ws.id, WorkspaceMember.user_id == user_id)
        .first()
    )
    if not m:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workspace access denied")
    return ch


@router.post("", response_model=VideoOut)
def create_video(
    channel_id: uuid.UUID,
    payload: VideoCreate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_channel_access(db, channel_id, user.id)
    v = Video(
        channel_id=channel_id,
        external_video_id=payload.external_video_id,
        title=payload.title,
        description=payload.description,
        duration_seconds=payload.duration_seconds,
        video_type=payload.video_type,
        source_type=payload.source_type,
        source_url=payload.source_url,
        storage_path=payload.storage_path,
        file_size_mb=payload.file_size_mb,
        checksum=payload.checksum,
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


@router.get("", response_model=List[VideoOut])
def list_videos(
    channel_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_channel_access(db, channel_id, user.id)
    return (
        db.query(Video)
        .filter(Video.channel_id == channel_id, Video.archived_at.is_(None))
        .order_by(Video.created_at.desc())
        .all()
    )


def _require_video_access(db: Session, video_id: uuid.UUID, user_id: uuid.UUID) -> Video:
    """Видео доступно, если пользователь состоит в workspace его канала."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    _require_channel_access(db, video.channel_id, user_id)
    return video


@video_router.get("/{video_id}", response_model=VideoOut)
def get_video(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    return _require_video_access(db, video_id, user.id)


@video_router.put("/{video_id}", response_model=VideoOut)
def update_video(
    video_id: uuid.UUID,
    payload: VideoUpdate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    video = _require_video_access(db, video_id, user.id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(video, field, value)
    db.commit()
    db.refresh(video)
    return video


@video_router.post("/{video_id}/upload", response_model=VideoOut)
def upload_video_file(
    video_id: uuid.UUID,
    file: UploadFile,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Загрузка видеофайла для уже зарегистрированного видео.

    Файл сохраняется в raw_uploads_dir, путь пишется в Video.storage_path —
    оттуда его берёт DataProcessor при обработке.

    Загрузка идёт потоково на диск: файл не буферизуется в память целиком,
    иначе большое видео исчерпало бы память процесса.

    TODO(прод): для гигабайтных файлов индустриальный путь — presigned URL
    напрямую в объектное хранилище (S3/MinIO), а не приём через API. Здесь
    прямой upload на диск, потому что объектного хранилища в локальной среде нет.
    """
    video = _require_video_access(db, video_id, user.id)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video format: {suffix or 'unknown'}",
        )

    paths = Settings().resolve_paths()
    target_dir = paths.raw_uploads_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{video.id}{suffix}"

    size = 0
    try:
        with target_path.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    out.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Video file is too large",
                    )
                out.write(chunk)
    finally:
        file.file.close()

    video.storage_path = str(target_path)
    video.file_size_mb = round(size / (1024 * 1024), 2)
    db.commit()
    db.refresh(video)
    return video


@video_router.delete("/{video_id}")
def delete_video(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Мягкое удаление: видео скрывается из списков, история анализов остаётся."""
    video = _require_video_access(db, video_id, user.id)
    video.archived_at = datetime.utcnow()
    db.commit()
    return {"status": "ok"}


