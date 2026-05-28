"""Resume после сбоя для Fetcher.

Обеспечивает возможность продолжения pipeline после сбоя worker'а.
Соответствует Quality Assurance Checklist (Resume after worker crash).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .db import session_scope
from .models import Artifact, Run, Video, VideoSource

logger = logging.getLogger(__name__)


def get_incomplete_runs(statuses: Optional[List[str]] = None) -> List[Run]:
    """Получить список незавершённых run'ов для resume.

    Args:
        statuses: Список статусов для фильтрации (по умолчанию все незавершённые)

    Returns:
        Список Run объектов, которые можно продолжить
    """
    if statuses is None:
        # Статусы, которые можно продолжить (не FINAL и не FAILED)
        statuses = [
            "PENDING",
            "NORMALIZING_SOURCE",
            "CHECKING_CACHE",
            "FETCHING_METADATA",
            "FETCHING_CHANNEL",
            "FETCHING_COMMENTS",
            "DOWNLOADING_VIDEO",
            "UPLOADING_ARTIFACTS",
            "FINALIZING",
        ]

    with session_scope() as db:
        runs: List[Run] = (
            db.query(Run)
            .filter(Run.status.in_(statuses))
            .filter(Run.finished_at.is_(None))  # Только незавершённые
            .all()
        )
        return runs


def get_missing_artifacts_for_run(run_id: str) -> List[str]:
    """Получить список отсутствующих артефактов для run'а.

    Args:
        run_id: UUID run'а

    Returns:
        Список типов артефактов, которые отсутствуют (video_file, metadata_file, comments_file)
    """
    # В unit‑тестах и вспомогательных сценариях run_id может быть произвольной строкой,
    # поэтому не навязываем строгую UUID‑валидацию на этом уровне.
    run_key = run_id
    required_artifact_types = ["video_file", "metadata_file", "comments_file"]

    with session_scope() as db:
        # Получаем video_id через run → video_source → video
        video_source: Optional[VideoSource] = (
            db.query(VideoSource)
            .filter(VideoSource.run_id == run_key)
            .order_by(VideoSource.created_at)
            .first()
        )

        if video_source is None or video_source.normalized_video_id is None:
            # Если нет normalized_video_id, все артефакты отсутствуют
            return required_artifact_types

        video: Optional[Video] = (
            db.query(Video)
            .filter(
                Video.platform == video_source.platform,
                Video.platform_video_id == video_source.normalized_video_id,
            )
            .first()
        )

        if video is None:
            return required_artifact_types

        # Проверяем наличие артефактов
        artifacts = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type.in_(required_artifact_types),
                Artifact.status == "COMPLETED",
            )
            .all()
        )

        artifact_types = {a.artifact_type for a in artifacts}
        missing = [atype for atype in required_artifact_types if atype not in artifact_types]
        return missing


def determine_next_stage(run_id: str) -> Optional[str]:
    """Определить следующую stage для resume.

    Args:
        run_id: UUID run'а

    Returns:
        Название следующей stage (metadata, download, comments, finalize) или None если всё готово
    """
    missing_artifacts = get_missing_artifacts_for_run(run_id)

    if not missing_artifacts:
        # Все артефакты готовы, можно finalize
        return "finalize"

    # Определяем следующую stage на основе отсутствующих артефактов
    # Приоритет: metadata → download → comments
    if "metadata_file" in missing_artifacts:
        return "metadata"
    if "video_file" in missing_artifacts:
        return "download"
    if "comments_file" in missing_artifacts:
        return "comments"

    return None


__all__ = [
    "get_incomplete_runs",
    "get_missing_artifacts_for_run",
    "determine_next_stage",
]

