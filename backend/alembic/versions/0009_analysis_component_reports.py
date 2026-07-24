"""Add core.analysis_component_reports — разбор по компонентам (manifest → UI).

JSONB-хранилище модальностей/групп/метрик. Формат задаёт DataProcessor
(см. AnalysisComponentReport). Одна запись на анализ.

Revision ID: 0009_analysis_component_reports
Revises: 0008_analysis_snapshots
Create Date: 2026-07-25

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_analysis_component_reports"
down_revision: Union[str, None] = "0008_analysis_snapshots"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_component_reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("analysis_job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "modalities",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.ForeignKeyConstraint(
            ["analysis_job_id"], ["core.analysis_jobs.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "analysis_job_id", name="uq_analysis_component_report_job"
        ),
        schema="core",
    )


def downgrade() -> None:
    op.drop_table("analysis_component_reports", schema="core")
