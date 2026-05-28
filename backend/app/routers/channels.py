from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..dbv2.models import Channel, Workspace, WorkspaceMember
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import ChannelCreate, ChannelOut

router = APIRouter(prefix="/api/workspaces/{workspace_id}/channels", tags=["channels"])


def _require_member(db: Session, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    m = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id)
        .first()
    )
    if not m:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workspace access denied")


@router.post("", response_model=ChannelOut)
def create_channel(
    workspace_id: uuid.UUID,
    payload: ChannelCreate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_member(db, workspace_id, user.id)
    ch = Channel(
        workspace_id=workspace_id,
        platform=payload.platform,
        external_channel_id=payload.external_channel_id,
        channel_name=payload.channel_name,
        connected_oauth_id=payload.connected_oauth_id,
    )
    db.add(ch)
    db.commit()
    db.refresh(ch)
    return ch


@router.get("", response_model=List[ChannelOut])
def list_channels(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_member(db, workspace_id, user.id)
    return (
        db.query(Channel)
        .filter(Channel.workspace_id == workspace_id, Channel.archived_at.is_(None))
        .order_by(Channel.created_at.desc())
        .all()
    )


