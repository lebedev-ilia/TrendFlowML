"""Create core schema and initial tables (v2).

Revision ID: 0001_core_init
Revises:
Create Date: 2026-03-03
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_core_init"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS core")

    # Enum types (in schema core to avoid polluting public)
    workspace_role = sa.Enum(
        "owner",
        "admin",
        "editor",
        "viewer",
        name="workspace_role",
        schema="core",
    )
    subscription_status = sa.Enum(
        "active",
        "canceled",
        "expired",
        name="subscription_status",
        schema="core",
    )
    video_type = sa.Enum("shorts", "video", name="video_type", schema="core")
    source_type = sa.Enum("upload", "link", name="source_type", schema="core")
    analysis_status = sa.Enum(
        "queued",
        "processing",
        "completed",
        "failed",
        "canceled",
        name="analysis_status",
        schema="core",
    )

    workspace_role.create(op.get_bind(), checkfirst=True)
    subscription_status.create(op.get_bind(), checkfirst=True)
    video_type.create(op.get_bind(), checkfirst=True)
    source_type.create(op.get_bind(), checkfirst=True)
    analysis_status.create(op.get_bind(), checkfirst=True)

    # Типы для колонок: create_type=False, чтобы при create_table не выполнять CREATE TYPE повторно
    pg_enum = sa.dialects.postgresql.ENUM
    workspace_role_col = pg_enum(
        "owner", "admin", "editor", "viewer",
        name="workspace_role", schema="core", create_type=False,
    )
    subscription_status_col = pg_enum(
        "active", "canceled", "expired",
        name="subscription_status", schema="core", create_type=False,
    )
    video_type_col = pg_enum("shorts", "video", name="video_type", schema="core", create_type=False)
    source_type_col = pg_enum("upload", "link", name="source_type", schema="core", create_type=False)
    analysis_status_col = pg_enum(
        "queued", "processing", "completed", "failed", "canceled",
        name="analysis_status", schema="core", create_type=False,
    )

    op.create_table(
        "users",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("email", sa.String(), nullable=False, unique=True),
        sa.Column("email_verified", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("password_hash", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        schema="core",
    )

    op.create_table(
        "user_oauth_accounts",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("provider_user_id", sa.String(), nullable=False),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["core.users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_user"),
        schema="core",
    )

    op.create_table(
        "user_security",
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("two_factor_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("two_factor_secret", sa.String(), nullable=True),
        sa.Column("password_reset_token", sa.String(), nullable=True),
        sa.Column("password_reset_expires_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["core.users.id"], ondelete="CASCADE"),
        schema="core",
    )

    op.create_table(
        "workspaces",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False, unique=True),
        sa.Column("owner_user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["owner_user_id"], ["core.users.id"]),
        schema="core",
    )

    op.create_table(
        "workspace_members",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("workspace_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", workspace_role_col, nullable=False),
        sa.Column("invited_by", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("joined_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["workspace_id"], ["core.workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["core.users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["invited_by"], ["core.users.id"]),
        sa.UniqueConstraint("workspace_id", "user_id", name="uq_workspace_user"),
        schema="core",
    )

    op.create_table(
        "subscription_plans",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("max_videos_per_month", sa.Integer(), nullable=False),
        sa.Column("max_analyses_per_month", sa.Integer(), nullable=False),
        sa.Column("max_channels", sa.Integer(), nullable=False),
        sa.Column("max_storage_gb", sa.Integer(), nullable=False),
        sa.Column("has_api_access", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("has_advanced_explainability", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("price", sa.Float(), nullable=False),
        schema="core",
    )

    op.create_table(
        "subscriptions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("workspace_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("plan_id", sa.Integer(), nullable=False),
        sa.Column("status", subscription_status_col, nullable=False),
        sa.Column("current_period_start", sa.DateTime(), nullable=False),
        sa.Column("current_period_end", sa.DateTime(), nullable=False),
        sa.Column("cancel_at_period_end", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["workspace_id"], ["core.workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["plan_id"], ["core.subscription_plans.id"]),
        schema="core",
    )

    op.create_table(
        "channels",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("workspace_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("external_channel_id", sa.String(), nullable=True),
        sa.Column("channel_name", sa.String(), nullable=False),
        sa.Column("connected_oauth_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["workspace_id"], ["core.workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["connected_oauth_id"], ["core.user_oauth_accounts.id"]),
        schema="core",
    )
    op.create_index(
        "ix_platform_external",
        "channels",
        ["platform", "external_channel_id"],
        unique=False,
        schema="core",
    )

    op.create_table(
        "videos",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("channel_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("external_video_id", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=False),
        sa.Column("video_type", video_type_col, nullable=False),
        sa.Column("source_type", source_type_col, nullable=False),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("storage_path", sa.String(), nullable=True),
        sa.Column("file_size_mb", sa.Float(), nullable=True),
        sa.Column("checksum", sa.String(), nullable=True),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["channel_id"], ["core.channels.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("channel_id", "external_video_id", name="uq_channel_external_video"),
        schema="core",
    )

    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("workspace_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("video_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("triggered_by_user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("processing_config_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_version_id", sa.String(), nullable=False),
        sa.Column("status", analysis_status_col, nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["workspace_id"], ["core.workspaces.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["video_id"], ["core.videos.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["triggered_by_user_id"], ["core.users.id"]),
        schema="core",
    )
    op.create_index("ix_analysis_workspace", "analysis_jobs", ["workspace_id"], schema="core")
    op.create_index("ix_analysis_video", "analysis_jobs", ["video_id"], schema="core")

    op.create_table(
        "predictions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("analysis_job_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("predicted_views", sa.Float(), nullable=False),
        sa.Column("predicted_likes", sa.Float(), nullable=False),
        sa.Column("percentile_score", sa.Float(), nullable=False),
        sa.Column("confidence_lower", sa.Float(), nullable=False),
        sa.Column("confidence_upper", sa.Float(), nullable=False),
        sa.Column("model_version_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["analysis_job_id"], ["core.analysis_jobs.id"], ondelete="CASCADE"),
        schema="core",
    )


def downgrade() -> None:
    op.drop_table("predictions", schema="core")
    op.drop_index("ix_analysis_video", table_name="analysis_jobs", schema="core")
    op.drop_index("ix_analysis_workspace", table_name="analysis_jobs", schema="core")
    op.drop_table("analysis_jobs", schema="core")
    op.drop_table("videos", schema="core")
    op.drop_index("ix_platform_external", table_name="channels", schema="core")
    op.drop_table("channels", schema="core")
    op.drop_table("subscriptions", schema="core")
    op.drop_table("subscription_plans", schema="core")
    op.drop_table("workspace_members", schema="core")
    op.drop_table("workspaces", schema="core")
    op.drop_table("user_security", schema="core")
    op.drop_table("user_oauth_accounts", schema="core")
    op.drop_table("users", schema="core")

    # Drop enum types last
    sa.Enum(name="analysis_status", schema="core").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="source_type", schema="core").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="video_type", schema="core").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="subscription_status", schema="core").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="workspace_role", schema="core").drop(op.get_bind(), checkfirst=True)


