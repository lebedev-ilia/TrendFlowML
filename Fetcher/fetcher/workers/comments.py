from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from ..db import session_scope
from ..idempotency import is_stage_idempotent
from ..logging import get_logger, log_with_context
from ..metrics import fetcher_comments_latency_seconds, fetcher_videos_failed_total
from ..models import Run, VideoSource
from ..platforms.registry import get_adapter

logger = get_logger(__name__)


def _get_source_for_run(db: Session, run_id: uuid.UUID) -> str:
    """Получить source (URL или normalized_video_id) для указанного run_id.

    Использует ту же логику, что и metadata/video workers.
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


def run_comments_worker(run_id: str, limit: int = 100) -> None:
    """Синхронный comments worker для Fetcher (MVP).

    - проверяет наличие run'а;
    - получает source по `video_sources`;
    - вызывает `YouTubeAdapter.fetch_comments(...)` с указанным лимитом.

    Метрики:
    - `fetcher_comments_latency_seconds` (histogram) — время выполнения worker'а.
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
            f"Starting comments worker for run_id={run_id} (limit={limit})",
            run_id=run_id,
            stage="fetch_comments",
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

        # Проверка идемпотентности: если comments уже загружены, пропускаем
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

        can_skip, reason = is_stage_idempotent(platform, platform_video_id, "comments")
        if can_skip:
            log_with_context(
                logger,
                logging.INFO,
                f"Comments stage skipped (idempotent): {reason}",
                run_id=run_id,
                stage="fetch_comments",
                platform=platform,
            )
            return

        adapter = get_adapter(platform)
        adapter.fetch_comments(source, run_id=run_id, limit=limit)

        elapsed = time.time() - start_time
        fetcher_comments_latency_seconds.labels(platform=platform).observe(elapsed)

        log_with_context(
            logger,
            logging.INFO,
            f"Comments worker completed for run_id={run_id} in {elapsed:.2f}s (collected {limit} comments)",
            run_id=run_id,
            stage="fetch_comments",
            platform=platform,
        )
    except Exception as e:
        elapsed = time.time() - start_time
        fetcher_videos_failed_total.labels(platform=platform, reason=type(e).__name__).inc()
        log_with_context(
            logger,
            logging.ERROR,
            f"Comments worker failed for run_id={run_id}: {e}",
            run_id=run_id,
            stage="fetch_comments",
            platform=platform,
            exception=str(e),
        )
        raise


__all__ = ["run_comments_worker"]


