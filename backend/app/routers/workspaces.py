from __future__ import annotations

import re
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..dbv2.enums import WorkspaceRole
from ..dbv2.models import User as CoreUser
from ..dbv2.models import Workspace, WorkspaceMember
from ..deps import get_current_user, get_db
from ..schemas import (
    WorkspaceCreate,
    WorkspaceMemberAdd,
    WorkspaceMemberOut,
    WorkspaceOut,
)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "workspace"


def _require_owner(db: Session, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    if ws.owner_user_id == user_id:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner only")


@router.post("", response_model=WorkspaceOut)
def create_workspace(
    payload: WorkspaceCreate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    slug = (payload.slug or _slugify(payload.name)).strip().lower()
    existing = db.query(Workspace).filter(Workspace.slug == slug).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Slug already exists")

    ws = Workspace(name=payload.name, slug=slug, owner_user_id=user.id)
    db.add(ws)
    db.flush()

    # Auto-add owner as member
    member = WorkspaceMember(
        workspace_id=ws.id,
        user_id=user.id,
        role=WorkspaceRole.owner,
        invited_by=None,
    )
    db.add(member)
    db.commit()
    db.refresh(ws)
    return ws


@router.get("", response_model=List[WorkspaceOut])
def list_workspaces(db: Session = Depends(get_db), user: CoreUser = Depends(get_current_user)):
    # Workspaces where user is a member
    q = (
        db.query(Workspace)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
        .filter(WorkspaceMember.user_id == user.id)
        .order_by(Workspace.created_at.desc())
    )
    return q.all()


@router.get("/{workspace_id}", response_model=WorkspaceOut)
def get_workspace(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    ws = (
        db.query(Workspace)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
        .filter(Workspace.id == workspace_id, WorkspaceMember.user_id == user.id)
        .first()
    )
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    return ws


@router.post("/{workspace_id}/members", response_model=WorkspaceMemberOut)
def add_member(
    workspace_id: uuid.UUID,
    payload: WorkspaceMemberAdd,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_owner(db, workspace_id, user.id)

    target = db.query(CoreUser).filter(CoreUser.email == payload.user_email).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    exists = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == target.id)
        .first()
    )
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Already a member")

    m = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=target.id,
        role=payload.role,
        invited_by=user.id,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


