"""Add core.processing_configs — хранение конфигураций анализа.

До этой миграции конфигураций не существовало вовсе: analysis_jobs хранил
processing_config_id как UUID без таблицы и без внешнего ключа, а конструктор
конфигураций на сайте сохранять состав было некуда.

Состав компонентов лежит в JSONB, потому что каталог принадлежит DataProcessor
(configs/global_config.yaml) и меняется независимо от схемы БД.

Revision ID: 0005_processing_configs
Revises: 0004_fetcher_fields
Create Date: 2026-07-23

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_processing_configs"
down_revision: Union[str, None] = "0004_fetcher_fields"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "processing_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("workspace_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("estimated_cost_units", sa.Integer(), nullable=True),
        sa.Column("estimated_minutes", sa.Integer(), nullable=True),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.Column(
            "updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["core.workspaces.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["core.users.id"]),
        schema="core",
    )
    op.create_index(
        "ix_processing_config_workspace",
        "processing_configs",
        ["workspace_id"],
        schema="core",
    )

    # Внешний ключ на analysis_jobs.processing_config_id намеренно НЕ добавляется:
    # в таблице уже есть строки со сгенерированными идентификаторами, а перенос
    # исторических данных выходит за рамки этой миграции.


def downgrade() -> None:
    op.drop_index(
        "ix_processing_config_workspace", table_name="processing_configs", schema="core"
    )
    op.drop_table("processing_configs", schema="core")
