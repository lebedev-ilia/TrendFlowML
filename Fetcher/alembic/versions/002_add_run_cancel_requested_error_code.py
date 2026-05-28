"""Add cancel_requested and error_code to runs (уже в 001).

Revision ID: 002
Revises: 001
Create Date: 2026-03-09

Поля cancel_requested и error_code включены в 001_initial_schema.
Миграция оставлена для совместимости цепочки; upgrade/downgrade — no-op.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Уже есть в 001
    pass


def downgrade() -> None:
    # Не откатываем — схема 001 уже содержит эти поля
    pass
