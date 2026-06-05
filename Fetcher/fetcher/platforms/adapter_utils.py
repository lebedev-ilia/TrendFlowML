from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.exc import IntegrityError

from fetcher.checksums import compute_sha256
from fetcher.config import settings
from fetcher.db import session_scope
from fetcher.models import Artifact, ChannelMetadata, Video, VideoMetadata
from fetcher.schemas.platform_video import PlatformVideoDto
from fetcher.snapshots import create_initial_snapshot_from_info
from fetcher.storage import storage_client


def persist_metadata(
    *,
    platform: str,
    dto: PlatformVideoDto,
    run_id: str | None = None,
) -> str:
    """Сохранить метаданные в БД, storage и optional snapshot. Returns platform_video_id."""
    platform_video_id = dto.video_id
    info = dto.to_info_dict()

    with session_scope() as db:
        video: Optional[Video] = (
            db.query(Video)
            .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
            .first()
        )
        if video is None:
            video = Video(platform=platform, platform_video_id=platform_video_id)
            db.add(video)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                video = (
                    db.query(Video)
                    .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
                    .one()
                )

        video.channel_id = dto.channel_id or video.channel_id
        if isinstance(dto.duration_seconds, int):
            video.duration_seconds = dto.duration_seconds

        vm: Optional[VideoMetadata] = (
            db.query(VideoMetadata).filter(VideoMetadata.video_id == video.id).first()
        )
        if vm is None:
            vm = VideoMetadata(video_id=video.id)
            db.add(vm)

        vm.title = dto.title
        vm.description = dto.description
        vm.language = dto.language
        vm.duration_seconds = dto.duration_seconds
        vm.published_at = dto.published_at
        vm.raw_json = dto.raw_json if settings.retain_raw_meta else None

        cm: Optional[ChannelMetadata] = (
            db.query(ChannelMetadata).filter(ChannelMetadata.video_id == video.id).first()
        )
        if cm is None:
            cm = ChannelMetadata(video_id=video.id)
            db.add(cm)
        cm.channel_id = dto.channel_id or cm.channel_id
        cm.channel_title = dto.channel_title or cm.channel_title
        db.flush()

    if settings.enable_snapshots:
        create_initial_snapshot_from_info(platform, platform_video_id, info)

    today = datetime.now(timezone.utc)
    date_prefix = today.strftime("%Y/%m/%d")
    storage_key = f"raw/{platform}/{date_prefix}/{platform_video_id}/meta.json"

    meta_to_save: dict[str, Any]
    if settings.retain_raw_meta:
        meta_to_save = {**info, "source_provider": dto.source_provider}
    else:
        meta_to_save = {
            "id": dto.video_id,
            "title": dto.title,
            "duration": dto.duration_seconds,
            "channel_id": dto.channel_id,
            "channel": dto.channel_title,
            "view_count": dto.view_count,
            "like_count": dto.like_count,
            "comment_count": dto.comment_count,
            "source_provider": dto.source_provider,
        }

    tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_meta"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{platform_video_id}_meta.json"
    tmp_path.write_text(json.dumps(meta_to_save, ensure_ascii=False), encoding="utf-8")
    size_bytes = tmp_path.stat().st_size
    checksum = f"sha256:{compute_sha256(tmp_path)}"

    try:
        storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    with session_scope() as db:
        video = (
            db.query(Video)
            .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
            .one()
        )
        artifact: Optional[Artifact] = (
            db.query(Artifact)
            .filter(Artifact.video_id == video.id, Artifact.artifact_type == "metadata_file")
            .order_by(Artifact.created_at.desc())
            .first()
        )
        if artifact is None:
            artifact = Artifact(
                video_id=video.id,
                artifact_type="metadata_file",
                storage_path=storage_key,
                status="COMPLETED",
                size_bytes=size_bytes,
                checksum=checksum,
            )
            db.add(artifact)
        else:
            artifact.storage_path = storage_key
            artifact.status = "COMPLETED"
            artifact.size_bytes = size_bytes
            artifact.checksum = checksum
        db.flush()

    return platform_video_id


def persist_empty_comments(*, platform: str, platform_video_id: str) -> None:
    """Записать пустой comments.json (платформы без API комментариев)."""
    with session_scope() as db:
        video: Optional[Video] = (
            db.query(Video)
            .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
            .one_or_none()
        )
        if video is None:
            video = Video(platform=platform, platform_video_id=platform_video_id)
            db.add(video)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                video = (
                    db.query(Video)
                    .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
                    .one()
                )

    today = datetime.now(timezone.utc)
    date_prefix = today.strftime("%Y/%m/%d")
    storage_key = f"raw/{platform}/{date_prefix}/{platform_video_id}/comments.json"
    tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_comments"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{platform_video_id}_comments.json"
    tmp_path.write_text("[]", encoding="utf-8")
    size_bytes = tmp_path.stat().st_size
    checksum = f"sha256:{compute_sha256(tmp_path)}"
    try:
        storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    with session_scope() as db:
        video = (
            db.query(Video)
            .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
            .one()
        )
        artifact: Optional[Artifact] = (
            db.query(Artifact)
            .filter(Artifact.video_id == video.id, Artifact.artifact_type == "comments_file")
            .order_by(Artifact.created_at.desc())
            .first()
        )
        if artifact is None:
            db.add(
                Artifact(
                    video_id=video.id,
                    artifact_type="comments_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
            )
        else:
            artifact.storage_path = storage_key
            artifact.status = "COMPLETED"
            artifact.size_bytes = size_bytes
            artifact.checksum = checksum
        db.flush()


__all__ = ["persist_empty_comments", "persist_metadata"]
