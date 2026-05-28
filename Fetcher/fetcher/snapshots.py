from __future__ import annotations

"""Утилиты для записи временных снэпшотов (video_snapshots).

Каркас для Phase 5 (Snapshot ingestion) из чеклиста.
На данном этапе реализуем простую функцию для записи snapshot=0
на основе текущих метрик из info_dict YouTube.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from .db import session_scope
from .models import Video, VideoSnapshot
from .config import settings


def create_initial_snapshot_from_info(
    platform: str,
    platform_video_id: str,
    info: dict,
) -> None:
    """Создать snapshot_index=0 по данным из info_dict (views/likes/comments/subs).

    Если snapshot уже существует для данного video_id и snapshot_index=0, функция ничего не делает.
    """
    with session_scope() as db:
        video: Optional[Video] = (
            db.query(Video)
            .filter(
                Video.platform == platform,
                Video.platform_video_id == platform_video_id,
            )
            .one_or_none()
        )
        if video is None:
            # Без записи Video снэпшот не имеет смысла.
            return

        existing = (
            db.query(VideoSnapshot)
            .filter(
                VideoSnapshot.video_id == video.id,
                VideoSnapshot.snapshot_index == 0,
            )
            .one_or_none()
        )
        if existing is not None:
            return

        snapshot = VideoSnapshot(
            video_id=video.id,
            snapshot_index=0,
            view_count=info.get("view_count"),
            like_count=info.get("like_count"),
            comment_count=info.get("comment_count"),
            subscriber_count=info.get("channel_follower_count"),
            collected_at=datetime.now(timezone.utc),
        )
        db.add(snapshot)
        db.flush()


def create_periodic_snapshot(
    platform: str,
    platform_video_id: str,
    info: dict,
    snapshot_index: int,
) -> None:
    """Создать периодический snapshot с указанным индексом.

    Args:
        platform: Платформа (youtube, tiktok, etc.)
        platform_video_id: ID видео на платформе
        info: info_dict от yt-dlp или другой платформы
        snapshot_index: Индекс snapshot'а (1, 2, 3, ... для периодических)
    """
    with session_scope() as db:
        video: Optional[Video] = (
            db.query(Video)
            .filter(
                Video.platform == platform,
                Video.platform_video_id == platform_video_id,
            )
            .one_or_none()
        )
        if video is None:
            return

        # Проверяем, не существует ли уже snapshot с таким индексом
        existing = (
            db.query(VideoSnapshot)
            .filter(
                VideoSnapshot.video_id == video.id,
                VideoSnapshot.snapshot_index == snapshot_index,
            )
            .one_or_none()
        )
        if existing is not None:
            return

        snapshot = VideoSnapshot(
            video_id=video.id,
            snapshot_index=snapshot_index,
            view_count=info.get("view_count"),
            like_count=info.get("like_count"),
            comment_count=info.get("comment_count"),
            subscriber_count=info.get("channel_follower_count"),
            collected_at=datetime.now(timezone.utc),
        )
        db.add(snapshot)
        db.flush()


def get_videos_needing_snapshot(
    schedule_days: Optional[List[int]] = None,
    batch_size: int = 100,
) -> List[tuple[str, str, int]]:
    """Получить список видео, которым нужен новый snapshot.

    Args:
        schedule_days: Расписание в днях (например, [0, 7, 14, 21]). Если None, берётся из settings.
        batch_size: Максимальное количество видео для обработки за раз

    Returns:
        List of tuples (platform, platform_video_id, next_snapshot_index)
    """
    if schedule_days is None:
        schedule_days = settings.snapshot_schedule_days

    if not schedule_days:
        return []

    with session_scope() as db:
        # Получаем видео с начальным snapshot (snapshot_index=0)
        videos_with_initial_snapshot = (
            db.query(Video, VideoSnapshot)
            .join(VideoSnapshot, VideoSnapshot.video_id == Video.id)
            .filter(VideoSnapshot.snapshot_index == 0)
            .all()
        )

        result: List[tuple[str, str, int]] = []

        for video, initial_snapshot in videos_with_initial_snapshot:
            # Вычисляем, сколько дней прошло с первого snapshot
            days_since_first = (datetime.now(timezone.utc) - initial_snapshot.collected_at).days

            # Определяем, какой snapshot должен быть следующим
            next_snapshot_index = None
            for i, day in enumerate(schedule_days):
                if day > days_since_first:
                    next_snapshot_index = i
                    break

            # Если прошло больше максимального дня в schedule, используем последний индекс
            if next_snapshot_index is None:
                next_snapshot_index = len(schedule_days) - 1

            # Проверяем, существует ли уже snapshot с таким индексом
            existing = (
                db.query(VideoSnapshot)
                .filter(
                    VideoSnapshot.video_id == video.id,
                    VideoSnapshot.snapshot_index == next_snapshot_index,
                )
                .one_or_none()
            )

            if existing is None:
                result.append((video.platform, video.platform_video_id, next_snapshot_index))

            if len(result) >= batch_size:
                break

        return result


def create_snapshots_for_videos(
    videos: List[tuple[str, str, int]],
) -> dict[str, int]:
    """Создать snapshots для списка видео.

    Args:
        videos: List of tuples (platform, platform_video_id, snapshot_index)

    Returns:
        Dict с результатами: {"created": N, "failed": M, "skipped": K}
    """
    import logging

    logger = logging.getLogger(__name__)
    results = {"created": 0, "failed": 0, "skipped": 0}

    for platform, platform_video_id, snapshot_index in videos:
        try:
            # Получаем info_dict через yt-dlp (для YouTube)
            if platform == "youtube":
                # Формируем URL из platform_video_id
                url = f"https://www.youtube.com/watch?v={platform_video_id}"

                try:
                    import yt_dlp
                    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)

                    # Создаём snapshot
                    create_periodic_snapshot(platform, platform_video_id, info, snapshot_index)
                    results["created"] += 1
                except Exception as e:
                    # Логируем ошибку, но продолжаем
                    logger.warning(f"Failed to create snapshot for {platform}/{platform_video_id}: {e}")
                    results["failed"] += 1
            else:
                # Другие платформы пока не поддерживаются
                logger.debug(f"Skipping snapshot for unsupported platform: {platform}")
                results["skipped"] += 1
        except Exception as e:
            logger.error(f"Error processing snapshot for {platform}/{platform_video_id}: {e}")
            results["failed"] += 1

    return results


__all__ = [
    "create_initial_snapshot_from_info",
    "create_periodic_snapshot",
    "get_videos_needing_snapshot",
    "create_snapshots_for_videos",
]


