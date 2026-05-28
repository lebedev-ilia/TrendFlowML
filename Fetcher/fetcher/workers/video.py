from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from ..db import session_scope
from ..idempotency import is_stage_idempotent
from ..logging import get_logger, log_with_context
from ..metrics import (
    fetcher_download_latency_seconds,
    fetcher_videos_downloaded_total,
    fetcher_videos_failed_total,
)
from ..models import Run, VideoSource
from ..platforms.registry import get_adapter

logger = get_logger(__name__)


def _get_source_for_run(db: Session, run_id: uuid.UUID) -> str:
    """Получить source (URL или normalized_video_id) для указанного run_id.

    Логика соответствует metadata worker'у.
    """

    vs: Optional[VideoSource] = (
        db.query(VideoSource)
        .filter(VideoSource.run_id == run_id)
        .order_by(VideoSource.created_at)
        .first()
    )
    if vs is None:
        raise ValueError(f"No video_source found for run_id={run_id}")
    return vs.normalized_video_id or vs.url


def run_video_download_worker(run_id: str) -> None:
    """Синхронный video download worker для Fetcher (MVP).

    - проверяет наличие run'а;
    - получает source по `video_sources`;
    - вызывает `YouTubeAdapter.download_video(...)`.

    Метрики:
    - `fetcher_download_latency_seconds` (histogram) — время скачивания и загрузки.
    - `fetcher_videos_downloaded_total` (counter) — успешные скачивания.
    """
    start_time = time.time()
    
    # Определяем платформу из run или из video_source
    run_uuid = uuid.UUID(run_id)
    with session_scope() as db:
        run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).one_or_none()
        if run:
            video_source: Optional[VideoSource] = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )
            if video_source and video_source.platform:
                platform = video_source.platform
            elif run.source_type:
                platform = run.source_type.lower()
            else:
                platform = "youtube"
        else:
            platform = "youtube"

    try:
        log_with_context(
            logger,
            logging.INFO,
            f"Starting video download worker for run_id={run_id}",
            run_id=run_id,
            stage="download_video",
            platform=platform,
        )

        run_uuid = uuid.UUID(run_id)
        with session_scope() as db:
            run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).one_or_none()
            if run is None:
                raise ValueError(f"Run {run_id} not found")
            
            # Определяем платформу из run или из video_source
            video_source: Optional[VideoSource] = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )
            if video_source and video_source.platform:
                platform = video_source.platform
            elif run.source_type:
                platform = run.source_type.lower()
            else:
                platform = "youtube"
            
            source = _get_source_for_run(db, run_uuid)

        # Проверка идемпотентности: если video уже скачан, пропускаем
        # Получаем platform_video_id из video_source для точной проверки
        with session_scope() as db:
            video_source: Optional[VideoSource] = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )
            platform_video_id = (
                video_source.normalized_video_id if video_source and video_source.normalized_video_id else source
            )

        can_skip, reason = is_stage_idempotent(platform, platform_video_id, "download")
        if can_skip:
            log_with_context(
                logger,
                logging.INFO,
                f"Video download stage skipped (idempotent): {reason}",
                run_id=run_id,
                stage="download_video",
                platform=platform,
            )
            return

        adapter = get_adapter(platform)
        adapter.download_video(source, run_id=run_id)

        elapsed = time.time() - start_time
        fetcher_download_latency_seconds.labels(platform=platform).observe(elapsed)
        fetcher_videos_downloaded_total.labels(platform=platform).inc()

        log_with_context(
            logger,
            logging.INFO,
            f"Video download worker completed for run_id={run_id} in {elapsed:.2f}s",
            run_id=run_id,
            stage="download_video",
            platform=platform,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        fetcher_videos_failed_total.labels(platform=platform, reason=type(e).__name__).inc()
        log_with_context(
            logger,
            logging.ERROR,
            f"Video download worker failed for run_id={run_id}: {e}",
            run_id=run_id,
            stage="download_video",
            platform=platform,
            exception=str(e),
        )
        raise


def run_video_worker(run_id: str) -> None:
    """Обёртка для совместимости с существующими тестами.

    Делегирует выполнение основному worker'у скачивания видео.
    """
    run_video_download_worker(run_id)


__all__ = ["run_video_download_worker", "run_video_worker"]


