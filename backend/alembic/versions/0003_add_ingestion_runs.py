"""Add core.ingestion_runs for Backend ↔ Fetcher integration (Phase 1).

Run по YouTube URL: Backend создаёт запись с run_id и передаёт его в Fetcher.
См. docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md, backend/docs/FETCHER_INTEGRATION.md.

Revision ID: 0003_ingestion_runs
Revises: 0002_legacy_init
Create Date: 2026-03-10

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_ingestion_runs"
down_revision: Union[str, None] = "0002_legacy_init"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ingestion_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_url", sa.String(), nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ingestion_status", sa.String(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("idempotency_key", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["core.users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["core.workspaces.id"], ondelete="SET NULL"),
        schema="core",
    )
    op.create_index(
        "ix_ingestion_runs_user",
        "ingestion_runs",
        ["user_id"],
        schema="core",
    )
    op.create_index(
        "ix_ingestion_runs_workspace",
        "ingestion_runs",
        ["workspace_id"],
        schema="core",
    )
    op.create_unique_constraint(
        "uq_ingestion_runs_idempotency_key",
        "ingestion_runs",
        ["idempotency_key"],
        schema="core",
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_ingestion_runs_idempotency_key",
        "ingestion_runs",
        schema="core",
        type_="unique",
    )
    op.drop_index("ix_ingestion_runs_workspace", table_name="ingestion_runs", schema="core")
    op.drop_index("ix_ingestion_runs_user", table_name="ingestion_runs", schema="core")
    op.drop_table("ingestion_runs", schema="core")
