"""Утилиты для Fetcher.

Вспомогательные функции для проверки состояния runs и других операций.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from .db import session_scope
from .models import Artifact, Run, Video, VideoSource
from .state_machine import RUN_STATUS_FAILED, validate_transition
from .config import settings as fetcher_settings

logger = logging.getLogger(__name__)


def check_cancel_requested(run_id: str | UUID) -> bool:
    """Проверить, запрошена ли отмена run'а.

    Args:
        run_id: UUID run'а

    Returns:
        True если отмена запрошена, False иначе
    """
    if isinstance(run_id, str):
        run_uuid = UUID(run_id)
    else:
        run_uuid = run_id

    with session_scope() as db:
        run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).first()
        if not run:
            return False

        # Проверяем флаг cancel_requested (поле или legacy в error)
        if getattr(run, "cancel_requested", False):
            return True
        if run.error and "CANCELLATION_REQUESTED" in run.error:
            return True

    return False


def cancel_run_if_requested(run_id: str | UUID) -> bool:
    """Отменить run если запрошена отмена.

    Устанавливает статус FAILED и добавляет "CANCELLED" в error.

    Args:
        run_id: UUID run'а

    Returns:
        True если run был отменён, False иначе
    """
    if isinstance(run_id, str):
        run_uuid = UUID(run_id)
    else:
        run_uuid = run_id

    with session_scope() as db:
        run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).first()
        if not run:
            return False

        # Проверяем флаг cancel_requested (поле или legacy в error)
        cancel_requested = getattr(run, "cancel_requested", False) or (
            run.error and "CANCELLATION_REQUESTED" in run.error
        )
        if cancel_requested:
            try:
                from datetime import datetime

                validate_transition(run.status, RUN_STATUS_FAILED, run_id=str(run_id))
                run.status = RUN_STATUS_FAILED
                run.error = (run.error or "").replace("CANCELLATION_REQUESTED", "").strip()
                if run.error:
                    run.error = run.error + " CANCELLED"
                else:
                    run.error = "CANCELLED"
                run.finished_at = datetime.utcnow()
                if hasattr(run, "cancel_requested"):
                    run.cancel_requested = False
                db.commit()
                logger.info(f"Run {run_id} cancelled due to cancellation request")
                return True
            except Exception as e:
                logger.warning(f"Failed to cancel run {run_id}: {e}")
                return False

    return False


def all_artifacts_ready(run_id: str | UUID) -> bool:
    """Проверить, готовы ли все обязательные артефакты для run'а.

    Обязательными считаются:
    - metadata_file
    - video_file
    - comments_file
    """
    if isinstance(run_id, str):
        run_uuid = UUID(run_id)
    else:
        run_uuid = run_id

    with session_scope() as db:
        # Находим video_source для run'а
        video_source = (
            db.query(VideoSource)
            .filter(VideoSource.run_id == run_uuid)
            .order_by(VideoSource.created_at)
            .first()
        )
        if not video_source or not video_source.normalized_video_id:
            return False

        # Находим соответствующее видео (при дубликатах берём последнее по created_at)
        video: Optional[Video] = (
            db.query(Video)
            .filter(
                Video.platform == video_source.platform,
                Video.platform_video_id == video_source.normalized_video_id,
            )
            .order_by(Video.created_at.desc())
            .first()
        )
        if not video:
            return False

        required_types = {"metadata_file", "video_file", "comments_file"}
        if fetcher_settings.allow_finalize_without_comments:
            required_types = {"metadata_file", "video_file"}
        artifacts = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type.in_(required_types),
            )
            .all()
        )
        completed_types = {
            a.artifact_type for a in artifacts if a.status == "COMPLETED"
        }
        return required_types.issubset(completed_types)


def get_missing_artifact_types(run_id: str | UUID) -> tuple[Optional[set[str]], Optional[str]]:
    """Вернуть недостающие типы артефактов или причину (no_video_source / no_video).

    Returns:
        (missing_types, None) если video и video_source есть; missing_types — подмножество
        {"metadata_file", "video_file", "comments_file"}.
        (None, "no_video_source") если нет VideoSource или normalized_video_id.
        (None, "no_video") если нет Video по platform_video_id.
    """
    if isinstance(run_id, str):
        run_uuid = UUID(run_id)
    else:
        run_uuid = run_id

    with session_scope() as db:
        video_source = (
            db.query(VideoSource)
            .filter(VideoSource.run_id == run_uuid)
            .order_by(VideoSource.created_at)
            .first()
        )
        if not video_source or not video_source.normalized_video_id:
            return None, "no_video_source"

        video: Optional[Video] = (
            db.query(Video)
            .filter(
                Video.platform == video_source.platform,
                Video.platform_video_id == video_source.normalized_video_id,
            )
            .order_by(Video.created_at.desc())
            .first()
        )
        if not video:
            return None, "no_video"

        required_types = {"metadata_file", "video_file", "comments_file"}
        if fetcher_settings.allow_finalize_without_comments:
            required_types = {"metadata_file", "video_file"}
        artifacts = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type.in_(required_types),
            )
            .all()
        )
        completed_types = {
            a.artifact_type for a in artifacts if a.status == "COMPLETED"
        }
        missing = required_types - completed_types
        return missing, None


__all__ = [
    "check_cancel_requested",
    "cancel_run_if_requested",
    "all_artifacts_ready",
    "get_missing_artifact_types",
]
