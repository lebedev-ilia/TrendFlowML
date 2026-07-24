from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..dbv2.models import CreditTransaction, Workspace, WorkspaceMember
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import BalanceOut, CreditTransactionOut, TopUpRequest
from ..services import billing

"""Баланс и история операций во внутренних единицах.

Баланс вычисляется по журналу `core.credit_transactions`, отдельного поля с
балансом не существует — см. app/services/billing.py.
"""

router = APIRouter(prefix="/api/workspaces/{workspace_id}", tags=["billing"])


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


@router.get("/balance", response_model=BalanceOut)
def get_balance(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_workspace_access(db, workspace_id, user.id)
    return BalanceOut(
        workspace_id=workspace_id,
        balance_units=billing.get_balance(db, workspace_id),
    )


@router.get("/transactions", response_model=List[CreditTransactionOut])
def list_transactions(
    workspace_id: uuid.UUID,
    limit: int = 50,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_workspace_access(db, workspace_id, user.id)
    return (
        db.query(CreditTransaction)
        .filter(CreditTransaction.workspace_id == workspace_id)
        .order_by(CreditTransaction.created_at.desc())
        .limit(min(limit, 500))
        .all()
    )


@router.post(
    "/transactions/top-up",
    response_model=CreditTransactionOut,
    status_code=status.HTTP_201_CREATED,
)
def top_up(
    workspace_id: uuid.UUID,
    payload: TopUpRequest,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Начисление единиц.

    TODO(платежи): вызывается напрямую клиентом. После подключения платёжной
    системы начисление должно происходить только по её подтверждению.
    """
    _require_workspace_access(db, workspace_id, user.id)
    transaction = billing.record_transaction(
        db,
        workspace_id=workspace_id,
        kind="topup",
        amount_units=payload.amount_units,
        user_id=user.id,
        description="Пополнение баланса",
        idempotency_key=payload.idempotency_key,
        amount_rub=payload.amount_rub,
    )
    db.commit()
    db.refresh(transaction)
    return transaction
