"""Lifecycle storage policies для Fetcher.

Реализует автоматическую очистку старых артефактов согласно retention policy.
Соответствует Phase 7 чеклиста (Lifecycle storage policies).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .config import settings
from .db import session_scope
from .models import Artifact, Comment, FetchJob, FetchLog, Run, VideoSource
from .storage import storage_client

logger = logging.getLogger(__name__)


def cleanup_old_raw_videos(
    retention_days: int = 30,
    bucket: Optional[str] = None,
) -> Dict[str, int]:
    """Очистить raw видео старше retention_days дней.

    Args:
        retention_days: Количество дней для хранения raw видео (по умолчанию 30)
        bucket: Bucket для очистки (по умолчанию settings.bucket_raw)

    Returns:
        Dict с результатами: {"checked": int, "deleted": int, "errors": int}
    """
    if bucket is None:
        bucket = settings.bucket_raw

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_prefix = cutoff_date.strftime("%Y/%m/%d")

    checked = 0
    deleted = 0
    errors = 0

    try:
        # Получаем все артефакты video_file из БД, которые старше cutoff_date
        with session_scope() as db:
            # Ищем артефакты video_file, у которых storage_path содержит дату до cutoff
            # Формат пути: raw/{platform}/{YYYY}/{MM}/{DD}/{video_id}/video.mp4
            artifacts: List[Artifact] = (
                db.query(Artifact)
                .filter(Artifact.artifact_type == "video_file")
                .filter(Artifact.status == "COMPLETED")
                .all()
            )

            for artifact in artifacts:
                checked += 1
                try:
                    # Извлекаем дату из storage_path
                    # Формат: raw/youtube/2026/03/05/video_id/video.mp4
                    path_parts = artifact.storage_path.split("/")
                    if len(path_parts) >= 5:
                        try:
                            year = int(path_parts[2])
                            month = int(path_parts[3])
                            day = int(path_parts[4])
                            artifact_date = datetime(year, month, day, tzinfo=timezone.utc)

                            if artifact_date < cutoff_date:
                                # Удаляем артефакт из storage
                                try:
                                    storage_client.delete_object(bucket, artifact.storage_path)
                                    logger.info(
                                        f"Deleted old video artifact: {artifact.storage_path} "
                                        f"(date: {artifact_date}, cutoff: {cutoff_date})"
                                    )
                                    deleted += 1
                                except Exception as e:
                                    errors += 1
                                    logger.error(
                                        f"Failed to delete artifact {artifact.storage_path}: {e}"
                                    )
                        except (ValueError, IndexError) as e:
                            # Не удалось извлечь дату из пути
                            logger.warning(
                                f"Could not parse date from path {artifact.storage_path}: {e}"
                            )
                            continue

                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing artifact {artifact.id}: {e}")

    except Exception as e:
        logger.exception(f"Error in cleanup_old_raw_videos: {e}")
        errors += 1

    return {"checked": checked, "deleted": deleted, "errors": errors}


def cleanup_old_raw_comments(
    retention_days: int = 30,
    hard_cap_days: int = 60,
    bucket: Optional[str] = None,
) -> Dict[str, int]:
    """Очистить raw комментарии старше retention_days дней (с учётом hard_cap).

    Args:
        retention_days: Количество дней для хранения raw комментариев (по умолчанию 30)
        hard_cap_days: Hard cap для хранения (максимум, даже если retention_days больше, по умолчанию 60)
        bucket: Bucket для очистки comments.json (по умолчанию settings.bucket_raw)

    Returns:
        Dict с результатами: {"checked": int, "deleted": int, "errors": int}
    """
    if bucket is None:
        bucket = settings.bucket_raw

    # Применяем hard cap
    effective_retention_days = min(retention_days, hard_cap_days)
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=effective_retention_days)

    checked = 0
    deleted = 0
    errors = 0

    try:
        with session_scope() as db:
            # Находим комментарии старше cutoff_date
            old_comments: List[Comment] = (
                db.query(Comment)
                .filter(Comment.published_at < cutoff_date)
                .all()
            )

            checked = len(old_comments)

            for comment in old_comments:
                try:
                    # Удаляем комментарий из БД
                    db.delete(comment)
                    deleted += 1

                    # Также удаляем comments.json из storage, если существует
                    # Формат пути: raw/{platform}/{YYYY}/{MM}/{DD}/{video_id}/comments.json
                    # Нужно найти artifact comments_file для этого video_id
                    artifact: Optional[Artifact] = (
                        db.query(Artifact)
                        .filter(Artifact.video_id == comment.video_id)
                        .filter(Artifact.artifact_type == "comments_file")
                        .filter(Artifact.status == "COMPLETED")
                        .first()
                    )

                    if artifact and artifact.storage_path:
                        try:
                            storage_client.delete_object(bucket, artifact.storage_path)
                            logger.info(
                                f"Deleted old comments artifact: {artifact.storage_path} "
                                f"(cutoff: {cutoff_date})"
                            )
                        except Exception as e:
                            # Логируем, но не считаем критичной ошибкой
                            logger.warning(
                                f"Failed to delete comments artifact {artifact.storage_path}: {e}"
                            )

                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing comment {comment.id}: {e}")

            db.commit()

    except Exception as e:
        logger.exception(f"Error in cleanup_old_raw_comments: {e}")
        errors += 1

    return {"checked": checked, "deleted": deleted, "errors": errors}


def cleanup_old_temp_files(
    retention_days: int = 7,
    bucket: Optional[str] = None,
) -> Dict[str, int]:
    """Очистить временные файлы старше retention_days дней.

    Args:
        retention_days: Количество дней для хранения temp файлов (по умолчанию 7)
        bucket: Bucket для очистки (по умолчанию settings.bucket_temp)

    Returns:
        Dict с результатами: {"checked": int, "deleted": int, "errors": int}
    """
    if bucket is None:
        bucket = settings.bucket_temp

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

    checked = 0
    deleted = 0
    errors = 0

    try:
        logger.info(
            f"Temp files cleanup: retention_days={retention_days}, "
            f"cutoff_date={cutoff_date}, bucket={bucket}"
        )
        objects = storage_client.list_objects(bucket, prefix="", max_keys=5000)
        for key, last_modified in objects:
            if last_modified.tzinfo is None:
                last_modified = last_modified.replace(tzinfo=timezone.utc)
            if last_modified < cutoff_date:
                try:
                    storage_client.delete_object(bucket, key)
                    deleted += 1
                    logger.debug(f"Deleted temp object: {key} (last_modified: {last_modified})")
                except Exception as e:
                    errors += 1
                    logger.warning(f"Failed to delete temp object {key}: {e}")
            checked += 1

    except Exception as e:
        logger.exception(f"Error in cleanup_old_temp_files: {e}")
        errors += 1

    return {"checked": checked, "deleted": deleted, "errors": errors}


def cleanup_old_failed_runs(
    retention_days: int = 7,
) -> Dict[str, int]:
    """Очистить записи о failed run'ах старше retention_days дней.

    Args:
        retention_days: Количество дней для хранения failed runs (по умолчанию 7)

    Returns:
        Dict с результатами: {"checked": int, "deleted": int, "errors": int}
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

    checked = 0
    deleted = 0
    errors = 0

    try:
        with session_scope() as db:
            # Находим failed runs старше cutoff_date (учитываем оба варианта статуса)
            failed_runs: List[Run] = (
                db.query(Run)
                .filter(Run.status.in_(["FAILED", "failed"]))
                .filter(Run.finished_at < cutoff_date)
                .all()
            )

            checked = len(failed_runs)

            for run in failed_runs:
                try:
                    run_id = run.id
                    db.query(FetchLog).filter(FetchLog.run_id == run_id).delete()
                    db.query(FetchJob).filter(FetchJob.run_id == run_id).delete()
                    db.query(VideoSource).filter(VideoSource.run_id == run_id).delete()
                    db.delete(run)
                    db.flush()
                    logger.info(
                        f"Deleted old failed run: {run_id} "
                        f"(finished_at: {run.finished_at})"
                    )
                    deleted += 1
                except Exception as e:
                    errors += 1
                    logger.error(f"Error processing failed run {run.id}: {e}")

    except Exception as e:
        logger.exception(f"Error in cleanup_old_failed_runs: {e}")
        errors += 1

    return {"checked": checked, "deleted": deleted, "errors": errors}


def run_lifecycle_cleanup(
    raw_video_retention_days: int = 30,
    raw_comments_retention_days: int = 30,
    raw_comments_hard_cap_days: int = 60,
    temp_files_retention_days: int = 7,
    failed_runs_retention_days: int = 7,
) -> Dict[str, Dict[str, int]]:
    """Запустить полную очистку согласно lifecycle policies.

    Args:
        raw_video_retention_days: Retention для raw видео (по умолчанию 30 дней)
        raw_comments_retention_days: Retention для raw комментариев (по умолчанию 30 дней)
        raw_comments_hard_cap_days: Hard cap для raw комментариев (по умолчанию 60 дней)
        temp_files_retention_days: Retention для temp файлов (по умолчанию 7 дней)
        failed_runs_retention_days: Retention для failed runs (по умолчанию 7 дней)

    Returns:
        Dict с результатами по каждому типу очистки
    """
    logger.info("Starting lifecycle cleanup")

    results = {
        "raw_videos": cleanup_old_raw_videos(retention_days=raw_video_retention_days),
        "raw_comments": cleanup_old_raw_comments(
            retention_days=raw_comments_retention_days,
            hard_cap_days=raw_comments_hard_cap_days,
        ),
        "temp_files": cleanup_old_temp_files(retention_days=temp_files_retention_days),
        "failed_runs": cleanup_old_failed_runs(retention_days=failed_runs_retention_days),
    }

    total_checked = sum(r["checked"] for r in results.values())
    total_deleted = sum(r["deleted"] for r in results.values())
    total_errors = sum(r["errors"] for r in results.values())

    logger.info(
        f"Lifecycle cleanup completed: checked={total_checked}, "
        f"deleted={total_deleted}, errors={total_errors}"
    )

    return results


__all__ = [
    "cleanup_old_raw_videos",
    "cleanup_old_raw_comments",
    "cleanup_old_temp_files",
    "cleanup_old_failed_runs",
    "run_lifecycle_cleanup",
]

