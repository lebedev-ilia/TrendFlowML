"""Add core.credit_transactions — журнал движения внутренних единиц.

Баланс не хранится отдельным полем: он равен сумме amount_units по рабочему
пространству. Журнал неизменяемый (append-only), поэтому баланс и история не
могут разойтись.

Уникальность (workspace_id, idempotency_key) защищает от повторного списания
при ретраях — это требование к финансовым операциям, а не оптимизация.

Revision ID: 0006_credit_transactions
Revises: 0005_processing_configs
Create Date: 2026-07-24

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_credit_transactions"
down_revision: Union[str, None] = "0005_processing_configs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "credit_transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("amount_units", sa.Integer(), nullable=False),
        sa.Column("balance_after", sa.Integer(), nullable=False),
        sa.Column("analysis_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=True),
        sa.Column("amount_rub", sa.Float(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["core.workspaces.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["core.users.id"]),
        sa.ForeignKeyConstraint(
            ["analysis_job_id"], ["core.analysis_jobs.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint(
            "workspace_id", "idempotency_key", name="uq_credit_tx_idempotency"
        ),
        schema="core",
    )
    op.create_index(
        "ix_credit_tx_workspace_created",
        "credit_transactions",
        ["workspace_id", "created_at"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_credit_tx_workspace_created",
        table_name="credit_transactions",
        schema="core",
    )
    op.drop_table("credit_transactions", schema="core")
