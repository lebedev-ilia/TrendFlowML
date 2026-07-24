"""Add core.analysis_jobs.share_token — публичная ссылка на отчёт.

Токен генерируется при шаринге отчёта и обнуляется при отзыве. Публичный
эндпоинт находит анализ по токену и отдаёт top-line данные (прогноз) без
чувствительного контекста.

Revision ID: 0007_analysis_share_token
Revises: 0006_credit_transactions
Create Date: 2026-07-24

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007_analysis_share_token"
down_revision: Union[str, None] = "0006_credit_transactions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("share_token", sa.String(), nullable=True),
        schema="core",
    )
    op.create_index(
        "ix_analysis_jobs_share_token",
        "analysis_jobs",
        ["share_token"],
        unique=True,
        schema="core",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_analysis_jobs_share_token", table_name="analysis_jobs", schema="core"
    )
    op.drop_column("analysis_jobs", "share_token", schema="core")
