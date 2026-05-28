"""Initial schema (совпадает с fetcher.models).

Revision ID: 001
Revises:
Create Date: 2026-03-05 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # runs
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # video_sources
    op.create_table(
        "video_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("platform", sa.String(20), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("normalized_video_id", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
    )

    # videos
    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("platform", sa.String(20), nullable=False),
        sa.Column("platform_video_id", sa.String(100), nullable=False),
        sa.Column("channel_id", sa.String(100), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("platform", "platform_video_id", name="uq_videos_platform_vid"),
    )

    # video_metadata
    op.create_table(
        "video_metadata",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )

    # channel_metadata
    op.create_table(
        "channel_metadata",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("channel_id", sa.String(100), nullable=True),
        sa.Column("channel_title", sa.Text(), nullable=True),
        sa.Column("subscriber_count", sa.BigInteger(), nullable=True),
        sa.Column("video_count", sa.Integer(), nullable=True),
        sa.Column("view_count", sa.BigInteger(), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )

    # video_snapshots
    op.create_table(
        "video_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("snapshot_index", sa.Integer(), nullable=False),
        sa.Column("view_count", sa.BigInteger(), nullable=True),
        sa.Column("like_count", sa.BigInteger(), nullable=True),
        sa.Column("comment_count", sa.BigInteger(), nullable=True),
        sa.Column("subscriber_count", sa.BigInteger(), nullable=True),
        sa.Column("collected_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )

    # comments
    op.create_table(
        "comments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("like_count", sa.Integer(), nullable=True),
        sa.Column("reply_count", sa.Integer(), nullable=True),
        sa.Column("published_at", sa.DateTime(), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )

    # artifacts
    op.create_table(
        "artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artifact_type", sa.String(30), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("checksum", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"]),
    )

    # fetch_jobs
    op.create_table(
        "fetch_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("retries", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
    )

    # fetch_logs (id — integer, autoincrement)
    op.create_table(
        "fetch_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, primary_key=True, nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stage", sa.String(50), nullable=True),
        sa.Column("level", sa.String(20), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
    )

    # proxies
    op.create_table(
        "proxies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("country", sa.String(2), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("url", name="uq_proxies_url"),
    )

    # proxy_usage
    op.create_table(
        "proxy_usage",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("proxy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("operation", sa.String(50), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["proxy_id"], ["proxies.id"]),
    )

    # Индексы
    op.create_index("ix_video_sources_run_id", "video_sources", ["run_id"])
    op.create_index("ix_video_metadata_video_id", "video_metadata", ["video_id"])
    op.create_index("ix_channel_metadata_video_id", "channel_metadata", ["video_id"])
    op.create_index("ix_video_snapshots_video_id", "video_snapshots", ["video_id"])
    op.create_index("ix_comments_video_id", "comments", ["video_id"])
    op.create_index("ix_artifacts_video_id", "artifacts", ["video_id"])
    op.create_index("ix_fetch_jobs_run_id", "fetch_jobs", ["run_id"])
    op.create_index("ix_fetch_logs_run_id", "fetch_logs", ["run_id"])
    op.create_index("ix_fetch_logs_created_at", "fetch_logs", ["created_at"])
    op.create_index("ix_proxy_usage_proxy_id", "proxy_usage", ["proxy_id"])


def downgrade() -> None:
    op.drop_index("ix_proxy_usage_proxy_id", table_name="proxy_usage")
    op.drop_index("ix_fetch_logs_created_at", table_name="fetch_logs")
    op.drop_index("ix_fetch_logs_run_id", table_name="fetch_logs")
    op.drop_index("ix_fetch_jobs_run_id", table_name="fetch_jobs")
    op.drop_index("ix_artifacts_video_id", table_name="artifacts")
    op.drop_index("ix_comments_video_id", table_name="comments")
    op.drop_index("ix_video_snapshots_video_id", table_name="video_snapshots")
    op.drop_index("ix_channel_metadata_video_id", table_name="channel_metadata")
    op.drop_index("ix_video_metadata_video_id", table_name="video_metadata")
    op.drop_index("ix_video_sources_run_id", table_name="video_sources")

    op.drop_table("proxy_usage")
    op.drop_table("proxies")
    op.drop_table("fetch_logs")
    op.drop_table("fetch_jobs")
    op.drop_table("artifacts")
    op.drop_table("comments")
    op.drop_table("video_snapshots")
    op.drop_table("channel_metadata")
    op.drop_table("video_metadata")
    op.drop_table("videos")
    op.drop_table("video_sources")
    op.drop_table("runs")
