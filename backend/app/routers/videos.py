from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..dbv2.models import Channel, Video, Workspace, WorkspaceMember
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import VideoCreate, VideoOut

router = APIRouter(prefix="/api/channels/{channel_id}/videos", tags=["videos"])


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


