"""Add core.analysis_snapshots — snapshot_0 (состояние на момент анализа).

Поля соответствуют контракту Models (TARGETS_SPLITS_METRICS.md, snapshot_0):
эти значения подаются в модель как вход. Одна запись на анализ; заполняется
Fetcher/DataProcessor на момент сбора метрик.

Revision ID: 0008_analysis_snapshots
Revises: 0007_analysis_share_token
Create Date: 2026-07-24

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_analysis_snapshots"
down_revision: Union[str, None] = "0007_analysis_share_token"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("analysis_job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("views_0", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("likes_0", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("comments_0", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "channel_subscribers_0", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "channel_total_views_0", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column(
            "channel_total_videos_0", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "captured_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.ForeignKeyConstraint(
            ["analysis_job_id"], ["core.analysis_jobs.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("analysis_job_id", name="uq_analysis_snapshot_job"),
        schema="core",
    )


def downgrade() -> None:
    op.drop_table("analysis_snapshots", schema="core")
