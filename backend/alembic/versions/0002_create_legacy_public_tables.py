"""Create legacy public tables (idempotent).

This migration captures the current `backend/app/models.py` schema so the backend
can move away from `Base.metadata.create_all()` and use Alembic instead.

Revision ID: 0002_legacy_init
Revises: 0001_core_init
Create Date: 2026-03-03
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision: str = "0002_legacy_init"
down_revision: str | None = "0001_core_init"
branch_labels = None
depends_on = None


def _has_table(bind, name: str) -> bool:
    insp = sa.inspect(bind)
    return insp.has_table(name)


def upgrade() -> None:
    bind = op.get_bind()

    if not _has_table(bind, "users"):
        op.create_table(
            "users",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("email", sa.Text(), nullable=False, unique=True),
            sa.Column("password_hash", sa.Text(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False, server_default=sa.text("'user'")),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        )

    if not _has_table(bind, "videos"):
        op.create_table(
            "videos",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("platform_id", sa.Text(), nullable=False),
            sa.Column("video_id", sa.Text(), nullable=False),
            sa.Column("source_type", sa.Text(), nullable=False),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("language", sa.Text(), nullable=True),
            sa.Column("category", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        )

    if not _has_table(bind, "video_files"):
        op.create_table(
            "video_files",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("sha256_hex", sa.Text(), nullable=False, unique=True),
            sa.Column("size_bytes", sa.BigInteger(), nullable=False),
            sa.Column("mime_type", sa.Text(), nullable=True),
            sa.Column("object_key", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("retention_until", sa.DateTime(), nullable=True),
        )

    if not _has_table(bind, "video_sources"):
        op.create_table(
            "video_sources",
            sa.Column("video_id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("youtube_url", sa.Text(), nullable=True),
            sa.Column("uploaded_file_id", sa.String(length=36), nullable=True),
            sa.Column("fetched_at", sa.DateTime(), nullable=True),
            sa.Column("duration_sec", sa.Integer(), nullable=True),
            sa.Column("width", sa.Integer(), nullable=True),
            sa.Column("height", sa.Integer(), nullable=True),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
            sa.ForeignKeyConstraint(["uploaded_file_id"], ["video_files.id"]),
        )

    if not _has_table(bind, "user_video_links"):
        op.create_table(
            "user_video_links",
            sa.Column("user_id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("video_id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
        )

    if not _has_table(bind, "analysis_profiles"):
        op.create_table(
            "analysis_profiles",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=True),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("is_public", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("config_json", sa.dialects.postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("config_hash", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        )

    if not _has_table(bind, "profile_components"):
        op.create_table(
            "profile_components",
            sa.Column("profile_id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("component_name", sa.Text(), primary_key=True, nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column("required", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column(
                "component_params",
                sa.dialects.postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("cost_units", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
            sa.ForeignKeyConstraint(["profile_id"], ["analysis_profiles.id"]),
        )

    if not _has_table(bind, "runs"):
        op.create_table(
            "runs",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("video_id", sa.String(length=36), nullable=False),
            sa.Column("profile_id", sa.String(length=36), nullable=True),
            sa.Column("config_hash", sa.Text(), nullable=False, server_default=sa.text("''")),
            sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'queued'")),
            sa.Column("stage", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.Column("cancel_requested_at", sa.DateTime(), nullable=True),
            sa.Column("error_code", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("estimated_cost_units", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
            sa.Column("actual_cost_units", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
            sa.ForeignKeyConstraint(["profile_id"], ["analysis_profiles.id"]),
        )

    if not _has_table(bind, "run_components"):
        op.create_table(
            "run_components",
            sa.Column("run_id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("component_name", sa.Text(), primary_key=True, nullable=False),
            sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'queued'")),
            sa.Column("schema_version", sa.Text(), nullable=True),
            sa.Column("producer_version", sa.Text(), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.Column("duration_ms", sa.Integer(), nullable=True),
            sa.Column("device_used", sa.Text(), nullable=True),
            sa.Column("empty_reason", sa.Text(), nullable=True),
            sa.Column("error_code", sa.Text(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("cost_units", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
            sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        )

    if not _has_table(bind, "artifacts"):
        op.create_table(
            "artifacts",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("component_name", sa.Text(), nullable=False),
            sa.Column("kind", sa.Text(), nullable=False),
            sa.Column("object_key", sa.Text(), nullable=False),
            sa.Column("size_bytes", sa.BigInteger(), nullable=True),
            sa.Column("sha256_hex", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        )

    if not _has_table(bind, "run_logs"):
        op.create_table(
            "run_logs",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=36), nullable=False),
            sa.Column("ts", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("level", sa.Text(), nullable=False),
            sa.Column("message", sa.Text(), nullable=False),
            sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        )

    if not _has_table(bind, "uploads"):
        op.create_table(
            "uploads",
            sa.Column("id", sa.String(length=36), primary_key=True, nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("video_id", sa.String(length=36), nullable=False),
            sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'init'")),
            sa.Column("temp_path", sa.Text(), nullable=True),
            sa.Column("filename", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
        )


def downgrade() -> None:
    # Best-effort reverse; if tables exist, drop them.
    bind = op.get_bind()
    insp = sa.inspect(bind)
    for name in [
        "uploads",
        "run_logs",
        "artifacts",
        "run_components",
        "runs",
        "profile_components",
        "analysis_profiles",
        "user_video_links",
        "video_sources",
        "video_files",
        "videos",
        "users",
    ]:
        if insp.has_table(name):
            op.drop_table(name)


