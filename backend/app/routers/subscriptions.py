from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..dbv2.enums import SubscriptionStatus
from ..dbv2.models import Subscription, SubscriptionPlan, Workspace, WorkspaceMember
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import SubscriptionCreate, SubscriptionOut, SubscriptionPlanOut

"""Тарифы и подписки (docs/API.md §7).

Эндпоинты были описаны в контракте, но роутер отсутствовал. Схемы и таблицы
(core.subscription_plans, core.subscriptions) уже существовали.
"""

router = APIRouter(prefix="/api", tags=["subscriptions"])

# Один платёжный период — 30 дней (биллинг помесячный).
_PERIOD_DAYS = 30


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


@router.get("/subscription-plans", response_model=List[SubscriptionPlanOut])
def list_plans(db: Session = Depends(get_db)):
    """Доступные тарифы. Публичный список — нужен и на странице цен."""
    return db.query(SubscriptionPlan).order_by(SubscriptionPlan.price).all()


@router.get(
    "/workspaces/{workspace_id}/subscriptions",
    response_model=List[SubscriptionOut],
)
def list_subscriptions(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_workspace_access(db, workspace_id, user.id)
    return (
        db.query(Subscription)
        .filter(Subscription.workspace_id == workspace_id)
        .order_by(Subscription.created_at.desc())
        .all()
    )


@router.post(
    "/workspaces/{workspace_id}/subscriptions",
    response_model=SubscriptionOut,
    status_code=status.HTTP_201_CREATED,
)
def create_subscription(
    workspace_id: uuid.UUID,
    payload: SubscriptionCreate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Оформление тарифа.

    Активная подписка в рабочем пространстве одна: прежнюю помечаем canceled,
    чтобы не держать две активные одновременно.

    TODO(платежи): смена тарифа происходит без оплаты — платёжная система не
    подключена (как и пополнение баланса).
    """
    _require_workspace_access(db, workspace_id, user.id)

    plan = db.query(SubscriptionPlan).filter(SubscriptionPlan.id == payload.plan_id).first()
    if not plan:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plan not found")

    db.query(Subscription).filter(
        Subscription.workspace_id == workspace_id,
        Subscription.status == SubscriptionStatus.active,
    ).update({Subscription.status: SubscriptionStatus.canceled})

    now = datetime.utcnow()
    subscription = Subscription(
        workspace_id=workspace_id,
        plan_id=payload.plan_id,
        status=SubscriptionStatus.active,
        current_period_start=now,
        current_period_end=now + timedelta(days=_PERIOD_DAYS),
        cancel_at_period_end=False,
    )
    db.add(subscription)
    db.commit()
    db.refresh(subscription)
    return subscription


def _require_subscription(
    db: Session, subscription_id: uuid.UUID, user_id: uuid.UUID
) -> Subscription:
    sub = db.query(Subscription).filter(Subscription.id == subscription_id).first()
    if not sub:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")
    _require_workspace_access(db, sub.workspace_id, user_id)
    return sub


@router.put("/subscriptions/{subscription_id}", response_model=SubscriptionOut)
def cancel_at_period_end(
    subscription_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Отмена в конце периода: доступ сохраняется до конца оплаченного срока."""
    sub = _require_subscription(db, subscription_id, user.id)
    sub.cancel_at_period_end = True
    db.commit()
    db.refresh(sub)
    return sub


@router.delete("/subscriptions/{subscription_id}")
def delete_subscription(
    subscription_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Немедленная отмена подписки."""
    sub = _require_subscription(db, subscription_id, user.id)
    sub.status = SubscriptionStatus.canceled
    db.commit()
    return {"status": "ok"}
