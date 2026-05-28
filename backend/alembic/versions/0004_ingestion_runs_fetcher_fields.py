"""Add fetcher_stage, fetcher_error_code, fetcher_error_message to core.ingestion_runs (Phase 4).

Backend синхронизирует статус и стадию из Fetcher через polling; поля для отображения в UI.
См. docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md (Фаза 4), backend/docs/FETCHER_INTEGRATION.md.

Revision ID: 0004_fetcher_fields
Revises: 0003_ingestion_runs
Create Date: 2026-03-10

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_fetcher_fields"
down_revision: Union[str, None] = "0003_ingestion_runs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ingestion_runs",
        sa.Column("fetcher_stage", sa.String(), nullable=True),
        schema="core",
    )
    op.add_column(
        "ingestion_runs",
        sa.Column("fetcher_error_code", sa.String(), nullable=True),
        schema="core",
    )
    op.add_column(
        "ingestion_runs",
        sa.Column("fetcher_error_message", sa.Text(), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("ingestion_runs", "fetcher_error_message", schema="core")
    op.drop_column("ingestion_runs", "fetcher_error_code", schema="core")
    op.drop_column("ingestion_runs", "fetcher_stage", schema="core")
