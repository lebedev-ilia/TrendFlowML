"""Учёт внутренних единиц: баланс, списания, возвраты.

Модель данных — неизменяемый журнал (`core.credit_transactions`). Баланс равен
сумме движений, отдельного изменяемого поля нет: так баланс невозможно
рассинхронизировать с историей операций.

Одновременные списания защищены блокировкой строки рабочего пространства
(`SELECT ... FOR UPDATE`): без неё два параллельных запроса могли бы прочитать
один и тот же баланс и увести его в минус.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..dbv2.models import CreditTransaction, Workspace


class InsufficientFunds(Exception):
    """Недостаточно единиц для операции."""

    def __init__(self, balance: int, required: int) -> None:
        super().__init__(f"Insufficient funds: balance={balance}, required={required}")
        self.balance = balance
        self.required = required


def get_balance(db: Session, workspace_id: uuid.UUID) -> int:
    """Текущий баланс — сумма всех движений по рабочему пространству."""
    total = db.execute(
        select(func.coalesce(func.sum(CreditTransaction.amount_units), 0)).where(
            CreditTransaction.workspace_id == workspace_id
        )
    ).scalar_one()
    return int(total)


def _find_by_idempotency_key(
    db: Session, workspace_id: uuid.UUID, key: Optional[str]
) -> Optional[CreditTransaction]:
    if not key:
        return None
    return (
        db.query(CreditTransaction)
        .filter(
            CreditTransaction.workspace_id == workspace_id,
            CreditTransaction.idempotency_key == key,
        )
        .first()
    )


def record_transaction(
    db: Session,
    *,
    workspace_id: uuid.UUID,
    kind: str,
    amount_units: int,
    user_id: Optional[uuid.UUID] = None,
    analysis_job_id: Optional[uuid.UUID] = None,
    description: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    amount_rub: Optional[float] = None,
    allow_negative: bool = False,
) -> CreditTransaction:
    """Записывает движение единиц.

    Списание передаётся отрицательным `amount_units`. При нехватке средств
    поднимается `InsufficientFunds` — кроме случая `allow_negative=True`,
    который нужен для служебных корректировок.

    Повторный вызов с тем же `idempotency_key` возвращает существующую запись,
    а не создаёт новую: сетевой ретрай не должен списывать дважды.
    """
    existing = _find_by_idempotency_key(db, workspace_id, idempotency_key)
    if existing:
        return existing

    # Блокируем рабочее пространство на время расчёта, чтобы параллельные
    # списания не прочитали один и тот же баланс.
    db.execute(select(Workspace.id).where(Workspace.id == workspace_id).with_for_update())

    balance = get_balance(db, workspace_id)
    new_balance = balance + amount_units

    if new_balance < 0 and not allow_negative:
        raise InsufficientFunds(balance=balance, required=abs(amount_units))

    transaction = CreditTransaction(
        workspace_id=workspace_id,
        user_id=user_id,
        kind=kind,
        amount_units=amount_units,
        balance_after=new_balance,
        analysis_job_id=analysis_job_id,
        description=description,
        idempotency_key=idempotency_key,
        amount_rub=amount_rub,
    )
    db.add(transaction)
    db.flush()
    return transaction


def charge_for_analysis(
    db: Session,
    *,
    workspace_id: uuid.UUID,
    user_id: uuid.UUID,
    analysis_job_id: uuid.UUID,
    cost_units: int,
    description: Optional[str] = None,
) -> CreditTransaction:
    """Списание за запуск анализа.

    Ключ идемпотентности привязан к задаче: повторная обработка одного и того
    же анализа не спишет средства второй раз.
    """
    return record_transaction(
        db,
        workspace_id=workspace_id,
        kind="charge",
        amount_units=-abs(cost_units),
        user_id=user_id,
        analysis_job_id=analysis_job_id,
        description=description or "Запуск анализа",
        idempotency_key=f"charge:{analysis_job_id}",
    )


def refund_for_analysis(
    db: Session,
    *,
    workspace_id: uuid.UUID,
    analysis_job_id: uuid.UUID,
    description: Optional[str] = None,
) -> Optional[CreditTransaction]:
    """Возврат за прерванный анализ.

    Возвращается вся списанная сумма: если обработка не дошла до результата,
    пользователь не должен за неё платить. Частичное списание за выполненные
    этапы появится, когда DataProcessor начнёт сообщать их стоимость
    (см. DataProcessor/docs/architecture/BILLING_AND_PRICING.md).
    """
    charge = (
        db.query(CreditTransaction)
        .filter(
            CreditTransaction.analysis_job_id == analysis_job_id,
            CreditTransaction.kind == "charge",
        )
        .first()
    )
    if not charge:
        return None

    return record_transaction(
        db,
        workspace_id=workspace_id,
        kind="refund",
        amount_units=abs(charge.amount_units),
        analysis_job_id=analysis_job_id,
        description=description or "Возврат за прерванный анализ",
        idempotency_key=f"refund:{analysis_job_id}",
    )
