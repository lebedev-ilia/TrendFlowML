from __future__ import annotations

import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..dbv2.models import ProcessingConfig, Workspace, WorkspaceMember
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import ProcessingConfigCreate, ProcessingConfigOut, ProcessingConfigUpdate

"""Конфигурации анализа: какие компоненты включены и с какими параметрами.

Системные пресеты (`is_system=True`) не привязаны к workspace и доступны всем
на чтение; изменять и удалять можно только свои конфигурации.
"""

router = APIRouter(prefix="/api/workspaces/{workspace_id}/processing-configs", tags=["configs"])
config_router = APIRouter(prefix="/api/processing-configs", tags=["configs"])


def _require_workspace_access(
    db: Session, workspace_id: uuid.UUID, user_id: uuid.UUID
) -> Workspace:
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    member = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        .first()
    )
    if not member:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workspace access denied")
    return ws


@router.get("", response_model=List[ProcessingConfigOut])
def list_processing_configs(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Системные пресеты и конфигурации рабочего пространства одним списком."""
    _require_workspace_access(db, workspace_id, user.id)
    return (
        db.query(ProcessingConfig)
        .filter(
            ProcessingConfig.archived_at.is_(None),
            or_(
                ProcessingConfig.is_system.is_(True),
                ProcessingConfig.workspace_id == workspace_id,
            ),
        )
        .order_by(ProcessingConfig.is_system.desc(), ProcessingConfig.created_at.desc())
        .all()
    )


@router.post("", response_model=ProcessingConfigOut, status_code=status.HTTP_201_CREATED)
def create_processing_config(
    workspace_id: uuid.UUID,
    payload: ProcessingConfigCreate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_workspace_access(db, workspace_id, user.id)
    config = ProcessingConfig(
        workspace_id=workspace_id,
        created_by_user_id=user.id,
        name=payload.name,
        description=payload.description,
        is_system=False,
        payload=payload.payload,
        estimated_cost_units=payload.estimated_cost_units,
        estimated_minutes=payload.estimated_minutes,
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def _require_config_access(
    db: Session, config_id: uuid.UUID, user_id: uuid.UUID, *, for_write: bool
) -> ProcessingConfig:
    config = db.query(ProcessingConfig).filter(ProcessingConfig.id == config_id).first()
    if not config or config.archived_at is not None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Processing config not found"
        )

    if config.is_system:
        if for_write:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="System config is read-only",
            )
        return config

    if config.workspace_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Processing config not found"
        )

    _require_workspace_access(db, config.workspace_id, user_id)
    return config


@config_router.get("/{config_id}", response_model=ProcessingConfigOut)
def get_processing_config(
    config_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    return _require_config_access(db, config_id, user.id, for_write=False)


@config_router.put("/{config_id}", response_model=ProcessingConfigOut)
def update_processing_config(
    config_id: uuid.UUID,
    payload: ProcessingConfigUpdate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    config = _require_config_access(db, config_id, user.id, for_write=True)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(config, field, value)
    db.commit()
    db.refresh(config)
    return config


@config_router.delete("/{config_id}")
def delete_processing_config(
    config_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Мягкое удаление: анализы, запущенные с этой конфигурацией, остаются валидными."""
    config = _require_config_access(db, config_id, user.id, for_write=True)
    config.archived_at = datetime.utcnow()
    db.commit()
    return {"status": "ok"}
