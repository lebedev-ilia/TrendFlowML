"""Оркестратор Fetcher.

Главная точка входа для запуска ingestion pipeline для конкретного run_id.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from .db import session_scope
from .config import settings
from .events import publish_status_changed, publish_stage_changed
from .logging import get_logger, log_with_context
from .metrics import fetcher_cache_hits_total, fetcher_cache_miss_total
from .models import Artifact, Run, Video, VideoSource
from .state_machine import (
    RUN_STATUS_CHECKING_CACHE,
    RUN_STATUS_FAILED,
    RUN_STATUS_FETCHING_METADATA,
    RUN_STATUS_FINALIZING,
    RUN_STATUS_NORMALIZING_SOURCE,
    validate_transition,
)
from .tasks import (
    download_video_task,
    fetch_comments_task,
    fetch_metadata_task,
    finalize_task,
)

logger = get_logger(__name__)


def normalize_source(url: str) -> tuple[str, str]:
    """Нормализовать URL в platform_video_id.

    Определяет платформу по URL и извлекает platform_video_id.
    Поддерживаемые платформы:
    - YouTube: youtube.com, youtu.be
    - TikTok: tiktok.com (TODO: реализовать)
    - Instagram: instagram.com (TODO: реализовать)

    Args:
        url: Исходный URL (например, https://www.youtube.com/watch?v=abc123)

    Returns:
        Tuple (platform, platform_video_id)
        Например: ("youtube", "abc123")

    Raises:
        ValueError: Если URL не может быть нормализован или платформа не поддерживается
    """
    url_lower = url.lower()

    # Определение платформы по домену
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        platform = "youtube"
    elif "tiktok.com" in url_lower:
        platform = "tiktok"
    elif "instagram.com" in url_lower:
        platform = "instagram"
        # TODO: Реализовать нормализацию для Instagram
        raise ValueError("Instagram platform is not yet implemented")
    else:
        raise ValueError(f"Unsupported platform for URL: {url}")

    # Нормализация для YouTube
    if platform == "youtube":
        # В проде по умолчанию используем yt-dlp (более надёжный путь).
        # Для локальной разработки/CI без доступа в интернет можно отключить через
        # FETCHER_YOUTUBE_USE_YT_DLP=false и использовать лёгкий парсер URL.
        if settings.youtube_use_yt_dlp:
            logger.info(
                "normalize_source: calling yt-dlp for URL (network request to YouTube, may take up to ~20s or timeout)",
                extra={"url": url[:80]},
            )
            try:
                import yt_dlp
            except ImportError:
                raise ValueError("yt-dlp not installed")

            try:
                ydl_opts = {"quiet": True, "no_warnings": True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_id = info.get("id")
                    if video_id:
                        return (platform, video_id)
                    else:
                        raise ValueError(f"yt-dlp did not return video id for URL: {url}")
            except Exception as e:
                logger.exception(f"Failed to normalize YouTube URL: {url}")
                raise ValueError(f"Failed to normalize YouTube URL: {e}") from e
        else:
            # Без сетевых запросов: вытаскиваем video_id из URL.
            logger.info("normalize_source: parsing YouTube URL without yt-dlp (no network)")
            parsed = urllib.parse.urlparse(url)
            host = parsed.netloc.lower()

            # youtu.be/<id>
            if "youtu.be" in host:
                video_id = parsed.path.lstrip("/").split("/")[0]
                if video_id:
                    return (platform, video_id)

            if "youtube.com" in host:
                # Формат shorts: /shorts/<id>
                path_parts = parsed.path.lstrip("/").split("/")
                if len(path_parts) >= 2 and path_parts[0] == "shorts" and path_parts[1]:
                    return (platform, path_parts[1])

                # Классический формат: /watch?v=<id>
                query = urllib.parse.parse_qs(parsed.query)
                v_vals = query.get("v") or []
                if v_vals and v_vals[0]:
                    return (platform, v_vals[0])

            raise ValueError(f"Failed to parse YouTube video id from URL without yt-dlp: {url}")

    # Нормализация для TikTok
    if platform == "tiktok":
        parsed = urllib.parse.urlparse(url)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""

        # Common format: /@user/video/<id>
        m = re.search(r"/@[^/]+/video/(?P<vid>\d+)", path)
        if m:
            return (platform, m.group("vid"))

        # Legacy mobile format: /v/<id>.html
        m = re.search(r"/v/(?P<vid>\d+)", path)
        if m:
            return (platform, m.group("vid"))

        # Short links (tiktok.com/t/<code>) usually require resolution via network.
        if settings.tiktok_use_yt_dlp:
            logger.info(
                "normalize_source: calling yt-dlp for TikTok URL (network request may be required)",
                extra={"url": url[:80]},
            )
            try:
                import yt_dlp
            except ImportError:
                raise ValueError("yt-dlp not installed")
            try:
                ydl_opts = {"quiet": True, "no_warnings": True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_id = info.get("id") if isinstance(info, dict) else None
                    if video_id:
                        return (platform, str(video_id))
                raise ValueError(f"yt-dlp did not return video id for URL: {url}")
            except Exception as e:
                logger.exception(f"Failed to normalize TikTok URL: {url}")
                raise ValueError(f"Failed to normalize TikTok URL: {e}") from e

        raise ValueError(
            f"Failed to parse TikTok video id from URL without yt-dlp: {url} "
            f"(expected /@user/video/<id>)"
        )

    raise ValueError(f"Unsupported URL format: {url}")


def check_cache(platform: str, platform_video_id: str, db: Session) -> bool:
    """Проверить глобальный кеш по (platform, platform_video_id).

    Args:
        platform: Платформа (например, "youtube")
        platform_video_id: ID видео на платформе
        db: SQLAlchemy session

    Returns:
        True если видео уже скачано и все артефакты готовы, False иначе
    """
    # Проверяем наличие видео в кеше
    video: Optional[Video] = (
        db.query(Video)
        .filter(
            Video.platform == platform,
            Video.platform_video_id == platform_video_id,
        )
        .first()
    )

    if video is None:
        return False

    # Проверяем наличие всех обязательных артефактов (см. utils.all_artifacts_ready)
    required_artifact_types = ["video_file", "metadata_file", "comments_file"]
    if settings.allow_finalize_without_comments:
        required_artifact_types = ["video_file", "metadata_file"]
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
    return all(atype in artifact_types for atype in required_artifact_types)


def fetch_video(run_id: str) -> None:
    """Главная функция оркестратора.

    Логика:
    1. Нормализация source (URL → platform_video_id)
    2. Проверка глобального кеша
    3. Если cache hit — сразу finalize
    4. Если cache miss — постановка задач в очереди

    Args:
        run_id: UUID run'а
    """
    run_uuid = uuid.UUID(run_id)

    with session_scope() as db:
        run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).first()
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        # Обновляем статус на NORMALIZING_SOURCE с валидацией перехода
        old_status = run.status
        validate_transition(old_status, RUN_STATUS_NORMALIZING_SOURCE, run_id=run_id)
        run.status = RUN_STATUS_NORMALIZING_SOURCE
        db.commit()

        # Публикуем событие изменения статуса
        publish_status_changed(
            run_id=run_id,
            old_status=old_status,
            new_status=RUN_STATUS_NORMALIZING_SOURCE,
            stage="normalize_source",
        )

        # Получаем source URL
        video_source: Optional[VideoSource] = (
            db.query(VideoSource)
            .filter(VideoSource.run_id == run_uuid)
            .order_by(VideoSource.created_at)
            .first()
        )

        if video_source is None:
            raise ValueError(f"No video_source found for run_id={run_id}")

        source_url = video_source.url

    try:
        log_with_context(
            logger,
            logging.INFO,
            f"Starting orchestrator for run_id={run_id}, url={source_url}",
            run_id=run_id,
            stage="normalize_source",
        )

        # 1. Нормализация source
        platform, platform_video_id = normalize_source(source_url)

        with session_scope() as db:
            # Обновляем video_source в текущей сессии (объект из предыдущего session_scope detached)
            vs = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )
            if vs:
                vs.normalized_video_id = platform_video_id
                vs.platform = platform

            # Обновляем статус run на CHECKING_CACHE с валидацией перехода
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if not run:
                raise ValueError(f"Run {run_id} not found")
            old_status = RUN_STATUS_NORMALIZING_SOURCE
            validate_transition(old_status, RUN_STATUS_CHECKING_CACHE, run_id=run_id)
            run.status = RUN_STATUS_CHECKING_CACHE
            db.commit()

            # Публикуем события изменения статуса и stage
            publish_status_changed(
                run_id=run_id,
                old_status=old_status,
                new_status=RUN_STATUS_CHECKING_CACHE,
                platform=platform,
                platform_video_id=platform_video_id,
                stage="check_cache",
            )
            publish_stage_changed(
                run_id=run_id,
                old_stage="normalize_source",
                new_stage="check_cache",
                platform=platform,
                platform_video_id=platform_video_id,
            )

        log_with_context(
            logger,
            logging.INFO,
            f"Normalized source: platform={platform}, video_id={platform_video_id}",
            run_id=run_id,
            stage="check_cache",
            platform=platform,
            platform_video_id=platform_video_id,
        )

        # 2. Проверка кеша
        with session_scope() as db:
            cache_hit = check_cache(platform, platform_video_id, db)

        if cache_hit:
            # Обновляем метрику cache hit
            fetcher_cache_hits_total.labels(platform=platform).inc()

            log_with_context(
                logger,
                logging.INFO,
                f"Cache hit for run_id={run_id}, skipping download",
                run_id=run_id,
                stage="cache_hit",
                platform=platform,
                platform_video_id=platform_video_id,
            )

            # Cache hit — сразу finalize с валидацией перехода
            with session_scope() as db:
                run = db.query(Run).filter(Run.id == run_uuid).first()
                old_status = RUN_STATUS_CHECKING_CACHE
                validate_transition(old_status, RUN_STATUS_FINALIZING, run_id=run_id)
                run.status = RUN_STATUS_FINALIZING
                db.commit()

                # Публикуем события
                publish_status_changed(
                    run_id=run_id,
                    old_status=old_status,
                    new_status=RUN_STATUS_FINALIZING,
                    platform=platform,
                    platform_video_id=platform_video_id,
                    stage="finalize",
                )
                publish_stage_changed(
                    run_id=run_id,
                    old_stage="check_cache",
                    new_stage="finalize",
                    platform=platform,
                    platform_video_id=platform_video_id,
                )

            finalize_task.delay(run_id)
        else:
            # Обновляем метрику cache miss
            fetcher_cache_miss_total.labels(platform=platform).inc()

            log_with_context(
                logger,
                logging.INFO,
                f"Cache miss for run_id={run_id}, enqueueing tasks",
                run_id=run_id,
                stage="cache_miss",
                platform=platform,
                platform_video_id=platform_video_id,
            )

            # Cache miss — ставим задачи в очереди с валидацией перехода
            with session_scope() as db:
                run = db.query(Run).filter(Run.id == run_uuid).first()
                old_status = RUN_STATUS_CHECKING_CACHE
                validate_transition(old_status, RUN_STATUS_FETCHING_METADATA, run_id=run_id)
                run.status = RUN_STATUS_FETCHING_METADATA
                db.commit()

                # Публикуем события
                publish_status_changed(
                    run_id=run_id,
                    old_status=old_status,
                    new_status=RUN_STATUS_FETCHING_METADATA,
                    platform=platform,
                    platform_video_id=platform_video_id,
                    stage="fetch_metadata",
                )
                publish_stage_changed(
                    run_id=run_id,
                    old_stage="check_cache",
                    new_stage="fetch_metadata",
                    platform=platform,
                    platform_video_id=platform_video_id,
                )

            # Параллельно запускаем metadata, video, comments
            fetch_metadata_task.delay(run_id)
            download_video_task.delay(run_id)
            fetch_comments_task.delay(run_id)

            # Finalize будет запущен после завершения всех задач
            # (через polling в finalize_task или через callback)

    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"Orchestrator failed for run_id={run_id}: {e}",
            run_id=run_id,
            stage="orchestrator",
            exception=str(e),
        )

        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if run:
                # FAILED может быть достигнут из любого промежуточного статуса
                old_status = run.status
                try:
                    validate_transition(old_status, RUN_STATUS_FAILED, run_id=run_id)
                except ValueError:
                    # Если переход не разрешен, логируем предупреждение, но всё равно переходим в FAILED
                    logger.warning(
                        f"Invalid transition to FAILED from {old_status}, "
                        f"but allowing it due to error: {e}"
                    )
                run.status = RUN_STATUS_FAILED
                run.error = str(e)
                db.commit()

        raise


__all__ = ["fetch_video", "normalize_source", "check_cache"]

