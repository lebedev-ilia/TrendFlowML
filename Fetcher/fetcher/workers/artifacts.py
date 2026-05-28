from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from ..db import session_scope
from ..manifest_validator import validate_manifest, validate_manifest_artifacts_exist
from ..models import Artifact, Comment, Run, Video, VideoSource
from schemas.manifest import ArtifactInfo, FetcherManifest, ManifestArtifacts
from ..storage import storage_client


def _get_video_for_run(db: Session, run_id: uuid.UUID) -> tuple[Video, str]:
    """Найти Video и platform для указанного run_id.

    Логика:
    - берём первый VideoSource для run;
    - ищем Video по (platform, normalized_video_id);
    """

    vs: Optional[VideoSource] = (
        db.query(VideoSource)
        .filter(VideoSource.run_id == run_id)
        .order_by(VideoSource.created_at)
        .first()
    )
    if vs is None or not vs.platform or not vs.normalized_video_id:
        raise ValueError(f"VideoSource with platform/normalized_video_id not found for run_id={run_id}")

    video: Optional[Video] = (
        db.query(Video)
        .filter(
            Video.platform == vs.platform,
            Video.platform_video_id == vs.normalized_video_id,
        )
        .order_by(Video.created_at.desc())
        .first()
    )
    if video is None:
        raise ValueError(
            f"Video record not found for platform={vs.platform}, video_id={vs.normalized_video_id}"
        )
    return video, vs.platform


def run_artifact_builder(run_id: str) -> None:
    """Собрать manifest.json для указанного run'а и зарегистрировать артефакт.

    Использует FetcherManifest из `Fetcher/schemas/manifest.py` и layout из STORAGE_LAYOUT.md.
    """

    run_uuid = uuid.UUID(run_id)

    with session_scope() as db:
        run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).one_or_none()
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        video, platform = _get_video_for_run(db, run_uuid)
        video_id = video.platform_video_id

        # Ищем артефакты (может быть несколько записей одного типа — берём последнюю по created_at)
        video_art: Optional[Artifact] = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type == "video_file",
                Artifact.status == "COMPLETED",
            )
            .order_by(Artifact.created_at.desc())
            .first()
        )
        meta_art: Optional[Artifact] = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type == "metadata_file",
                Artifact.status == "COMPLETED",
            )
            .order_by(Artifact.created_at.desc())
            .first()
        )
        comments_art: Optional[Artifact] = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type == "comments_file",
                Artifact.status == "COMPLETED",
            )
            .order_by(Artifact.created_at.desc())
            .first()
        )

        if video_art is None or meta_art is None:
            raise ValueError("Required artifacts (video_file, metadata_file) are not ready")

        # Подсчитываем количество комментариев, если возможно
        comment_count: Optional[int] = None
        if comments_art is not None:
            comment_count = (
                db.query(Comment).filter(Comment.video_id == video.id).count()
            )

        artifacts = ManifestArtifacts(
            video_file=ArtifactInfo(
                path=video_art.storage_path,
                checksum=video_art.checksum,
                size_bytes=video_art.size_bytes,
            ),
            meta_file=ArtifactInfo(
                path=meta_art.storage_path,
                checksum=meta_art.checksum,
                size_bytes=meta_art.size_bytes,
            ),
            comments_file=ArtifactInfo(
                path=comments_art.storage_path,
                checksum=comments_art.checksum,
                size_bytes=comments_art.size_bytes,
                comment_count=comment_count,
            )
            if comments_art is not None
            else None,
        )

        manifest = FetcherManifest(
            run_id=run_uuid,
            video_id=video_id,
            platform=platform,
            duration_seconds=float(video.duration_seconds or 0),
            artifacts=artifacts,
        )

    # Валидируем manifest перед сохранением
    is_valid, error_msg = validate_manifest(manifest)
    if not is_valid:
        raise ValueError(f"Manifest validation failed: {error_msg}")

    # Сериализуем manifest и пишем в storage
    today = datetime.now(timezone.utc)
    date_prefix = today.strftime("%Y/%m/%d")
    storage_key = f"raw/{platform}/{date_prefix}/{video_id}/manifest.json"

    tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_manifest"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{video_id}_manifest.json"

    # default=str для UUID/datetime в manifest.dict() (JSON не умеет их без конвертации)
    tmp_path.write_text(
        json.dumps(manifest.dict(), ensure_ascii=False, default=str), encoding="utf-8"
    )
    try:
        storage_client.upload_file(tmp_path, bucket="video-analytics-raw", key=storage_key)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    # Регистрируем/обновляем артефакт manifest в БД
    with session_scope() as db:
        video = (
            db.query(Video)
            .filter(
                Video.platform == platform,
                Video.platform_video_id == video_id,
            )
            .order_by(Video.created_at.desc())
            .first()
        )
        if video is None:
            raise ValueError(f"Video not found for platform={platform}, video_id={video_id}")
        artifact: Optional[Artifact] = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type == "manifest",
            )
            .order_by(Artifact.created_at.desc())
            .first()
        )
        if artifact is None:
            artifact = Artifact(
                video_id=video.id,
                artifact_type="manifest",
                storage_path=storage_key,
                status="COMPLETED",
            )
            db.add(artifact)
        else:
            artifact.storage_path = storage_key
            artifact.status = "COMPLETED"
        db.flush()


__all__ = ["run_artifact_builder"]


