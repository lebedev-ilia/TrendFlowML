"""Celery задачи для Fetcher.

Задачи для очередей:
- fetch.metadata (high priority)
- fetch.video (low priority)
- fetch.comments (medium priority)
- fetch.finalize (high priority)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from .celery_app import celery_app
from .db import session_scope
from .logging import get_logger, log_with_context
from .models import Run, VideoSource
from .state_machine import (
    RUN_STATUS_CHECKING_CACHE,
    RUN_STATUS_DOWNLOADING_VIDEO,
    RUN_STATUS_FETCHING_COMMENTS,
    RUN_STATUS_FETCHING_METADATA,
    RUN_STATUS_FINALIZING,
    validate_transition,
)
from .utils import (
    all_artifacts_ready,
    check_cancel_requested,
    cancel_run_if_requested,
    get_missing_artifact_types,
)
from .backpressure import BackpressureError, check_backpressure
from .config import settings as fetcher_settings
from .errors import NonRetryableError, get_error_category, is_retryable_error
from .events import publish_job_failed, publish_job_finished, publish_job_started
from .lifecycle import run_lifecycle_cleanup
from .snapshots import create_snapshots_for_videos, get_videos_needing_snapshot
from .stats_aggregator import aggregate_stats
from .workers import (
    run_artifact_builder,
    run_comments_worker,
    run_metadata_worker,
    run_video_download_worker,
)
from .webhooks import send_webhook

logger = get_logger(__name__)

# Статусы, из которых при готовности артефактов можно перейти в FINALIZING (cache miss path)
_CACHE_MISS_STATUSES = (
    RUN_STATUS_FETCHING_METADATA,
    RUN_STATUS_DOWNLOADING_VIDEO,
    RUN_STATUS_FETCHING_COMMENTS,
)


def _maybe_enqueue_finalize_after_cache_miss(run_id: str) -> None:
    """Если все артефакты готовы и run ещё в статусе fetch_*, перевести в FINALIZING и поставить finalize.

    Вызывается из fetch_metadata_task, download_video_task, fetch_comments_task после успеха.
    Только один из воркеров «победит» (последний завершивший); остальные получат ValueError при переходе.
    """
    if not all_artifacts_ready(run_id):
        # Подробный лог, почему finalize пока не может стартовать
        missing, reason = get_missing_artifact_types(run_id)
        if reason or missing:
            logger.debug(
                "Finalize not enqueued yet for run_id=%s: reason=%s missing=%s",
                run_id,
                reason,
                sorted(missing) if missing else None,
            )
        return
    with session_scope() as db:
        run_uuid = uuid.UUID(run_id)
        run = db.query(Run).filter(Run.id == run_uuid).first()
        if not run or run.status not in _CACHE_MISS_STATUSES:
            return
        try:
            validate_transition(run.status, RUN_STATUS_FINALIZING, run_id=run_id)
            run.status = RUN_STATUS_FINALIZING
            db.commit()
        except ValueError:
            return  # уже FINALIZING или другой переход
    celery_app.send_task("fetcher.finalize", args=[run_id], queue="fetch.finalize")


@celery_app.task(
    name="fetcher.fetch_video",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    # Queue определяется динамически при вызове (fetcher.high, fetcher.normal, fetcher.low)
)
def fetch_video_task(self, run_id: str) -> None:
    """Celery задача-обёртка над orchestrator.fetch_video.

    Точка входа пайплайна: нормализация source, проверка кеша,
    постановка metadata/video/comments и, при cache hit, сразу finalize.
    """
    try:
        # Импортируем здесь, чтобы избежать циклического импорта при старте приложения
        from .orchestrator import fetch_video

        log_with_context(
            logger,
            logging.INFO,
            f"Starting fetch_video orchestrator for run_id={run_id}",
            run_id=run_id,
            stage="orchestrator",
        )
        fetch_video(run_id)
    except Exception as e:
        error_category = get_error_category(e)
        log_with_context(
            logger,
            logging.ERROR,
            f"fetch_video orchestrator failed for run_id={run_id}: {e} (category: {error_category})",
            run_id=run_id,
            stage="orchestrator",
            exception=str(e),
            error_category=error_category,
        )
        # Retry только для retryable ошибок
        if is_retryable_error(e) and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        raise


@celery_app.task(
    name="fetcher.fetch_metadata",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    # Queue определяется динамически при вызове (fetcher.high, fetcher.normal, fetcher.low)
    # или может быть переопределён через apply_async(queue=...)
)
def fetch_metadata_task(self, run_id: str) -> None:
    """Celery задача для metadata ingestion.

    Args:
        run_id: UUID run'а

    Retry policy:
    - RateLimitError: retry с exponential backoff
    - NetworkError: retry с exponential backoff
    - NonRetryableError: не ретраим (video removed, private, etc.)
    """
    # Получаем platform и platform_video_id для событий; обновляем run.status для синка с Backend
    platform = None
    platform_video_id = None
    with session_scope() as db:
        run_uuid = uuid.UUID(run_id)
        run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).first()
        if run:
            video_source: Optional[VideoSource] = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )
            if video_source:
                platform = video_source.platform or "youtube"
                platform_video_id = video_source.normalized_video_id
            try:
                validate_transition(run.status, RUN_STATUS_FETCHING_METADATA, run_id=run_id)
                run.status = RUN_STATUS_FETCHING_METADATA
                db.commit()
            except ValueError:
                pass  # переход уже сделан или не разрешён — продолжаем выполнение

    try:
        log_with_context(
            logger,
            logging.INFO,
            f"Starting metadata task for run_id={run_id}",
            run_id=run_id,
            stage="fetch_metadata",
            platform=platform or "youtube",
        )

        # Публикуем событие начала job'а
        publish_job_started(
            run_id=run_id,
            job_type="fetch_metadata",
            job_id=run_id,  # Используем run_id как job_id для простоты
            platform=platform,
            platform_video_id=platform_video_id,
            stage="fetch_metadata",
        )

        start_time = time.time()
        run_metadata_worker(run_id)
        duration_ms = int((time.time() - start_time) * 1000)

        log_with_context(
            logger,
            logging.INFO,
            f"Metadata task completed for run_id={run_id}",
            run_id=run_id,
            stage="fetch_metadata",
            platform=platform or "youtube",
        )

        # Публикуем событие завершения job'а
        publish_job_finished(
            run_id=run_id,
            job_type="fetch_metadata",
            job_id=run_id,
            duration_ms=duration_ms,
            platform=platform,
            platform_video_id=platform_video_id,
            stage="fetch_metadata",
        )
        _maybe_enqueue_finalize_after_cache_miss(run_id)
    except Exception as e:
        error_category = get_error_category(e)
        error_code = getattr(e, "error_code", error_category.upper())

        log_with_context(
            logger,
            logging.ERROR,
            f"Metadata task failed for run_id={run_id}: {e} (category: {error_category})",
            run_id=run_id,
            stage="fetch_metadata",
            platform=platform or "youtube",
            exception=str(e),
            error_category=error_category,
        )

        # Публикуем событие ошибки job'а
        publish_job_failed(
            run_id=run_id,
            job_type="fetch_metadata",
            job_id=run_id,
            error_code=error_code,
            error_message=str(e),
            platform=platform,
            platform_video_id=platform_video_id,
            stage="fetch_metadata",
        )

        # Retry только для retryable ошибок
        if is_retryable_error(e) and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        else:
            # Non-retryable или превышен лимит retry - fail fast
            raise


@celery_app.task(
    name="fetcher.download_video",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    queue="fetch.video",
    priority=1,
)
def download_video_task(self, run_id: str) -> None:
    """Celery задача для video download.

    Args:
        run_id: UUID run'а

    Особенности:
    - Использует distributed lock для предотвращения дублирующихся скачиваний
    - Проверяет кеш перед скачиванием
    - Retry только для сетевых ошибок
    """
    try:
        # Проверка cooperative cancellation
        if check_cancel_requested(run_id):
            log_with_context(
                logger,
                logging.INFO,
                f"Run {run_id} cancellation requested, skipping video download task",
                run_id=run_id,
                stage="download_video",
                platform="youtube",
            )
            cancel_run_if_requested(run_id)
            return

        # Обновляем run.status для синка с Backend
        with session_scope() as db:
            run_uuid = uuid.UUID(run_id)
            run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).first()
            if run:
                try:
                    validate_transition(run.status, RUN_STATUS_DOWNLOADING_VIDEO, run_id=run_id)
                    run.status = RUN_STATUS_DOWNLOADING_VIDEO
                    db.commit()
                except ValueError:
                    pass

        log_with_context(
            logger,
            logging.INFO,
            f"Starting video download task for run_id={run_id}",
            run_id=run_id,
            stage="download_video",
            platform="youtube",
        )

        run_video_download_worker(run_id)

        log_with_context(
            logger,
            logging.INFO,
            f"Video download task completed for run_id={run_id}",
            run_id=run_id,
            stage="download_video",
            platform="youtube",
        )
        _maybe_enqueue_finalize_after_cache_miss(run_id)
    except Exception as e:
        error_category = get_error_category(e)
        log_with_context(
            logger,
            logging.ERROR,
            f"Video download task failed for run_id={run_id}: {e} (category: {error_category})",
            run_id=run_id,
            stage="download_video",
            platform="youtube",
            exception=str(e),
            error_category=error_category,
        )

        # Retry только для retryable ошибок
        if is_retryable_error(e) and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=120 * (2 ** self.request.retries))
        else:
            # Non-retryable или превышен лимит retry - fail fast
            raise


@celery_app.task(
    name="fetcher.fetch_comments",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="fetch.comments",
    priority=5,
)
def fetch_comments_task(self, run_id: str, limit: int = 100) -> None:
    """Celery задача для comments ingestion.

    Args:
        run_id: UUID run'а
        limit: Максимальное количество комментариев (default: 100)
    """
    # Получаем platform и platform_video_id для событий
    platform = None
    platform_video_id = None
    with session_scope() as db:
        run_uuid = uuid.UUID(run_id)
        run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).first()
        if run:
            video_source: Optional[VideoSource] = (
                db.query(VideoSource)
                .filter(VideoSource.run_id == run_uuid)
                .order_by(VideoSource.created_at)
                .first()
            )
            if video_source:
                platform = video_source.platform or "youtube"
                platform_video_id = video_source.normalized_video_id

    try:
        # Проверка cooperative cancellation
        if check_cancel_requested(run_id):
            log_with_context(
                logger,
                logging.INFO,
                f"Run {run_id} cancellation requested, skipping comments task",
                run_id=run_id,
                stage="fetch_comments",
                platform=platform or "youtube",
            )
            cancel_run_if_requested(run_id)
            return

        # Обновляем run.status для синка с Backend
        with session_scope() as db:
            run_uuid = uuid.UUID(run_id)
            run: Optional[Run] = db.query(Run).filter(Run.id == run_uuid).first()
            if run:
                try:
                    validate_transition(run.status, RUN_STATUS_FETCHING_COMMENTS, run_id=run_id)
                    run.status = RUN_STATUS_FETCHING_COMMENTS
                    db.commit()
                except ValueError:
                    pass

        log_with_context(
            logger,
            logging.INFO,
            f"Starting comments task for run_id={run_id} (limit={limit})",
            run_id=run_id,
            stage="fetch_comments",
            platform=platform or "youtube",
        )

        # Публикуем событие начала job'а
        publish_job_started(
            run_id=run_id,
            job_type="fetch_comments",
            job_id=run_id,
            platform=platform,
            platform_video_id=platform_video_id,
            stage="fetch_comments",
        )

        start_time = time.time()
        run_comments_worker(run_id, limit=limit)
        duration_ms = int((time.time() - start_time) * 1000)

        log_with_context(
            logger,
            logging.INFO,
            f"Comments task completed for run_id={run_id}",
            run_id=run_id,
            stage="fetch_comments",
            platform=platform or "youtube",
        )

        # Публикуем событие завершения job'а
        publish_job_finished(
            run_id=run_id,
            job_type="fetch_comments",
            job_id=run_id,
            duration_ms=duration_ms,
            platform=platform,
            platform_video_id=platform_video_id,
            stage="fetch_comments",
        )
        _maybe_enqueue_finalize_after_cache_miss(run_id)
    except Exception as e:
        error_category = get_error_category(e)
        error_code = getattr(e, "error_code", error_category.upper())

        log_with_context(
            logger,
            logging.ERROR,
            f"Comments task failed for run_id={run_id}: {e} (category: {error_category})",
            run_id=run_id,
            stage="fetch_comments",
            platform=platform or "youtube",
            exception=str(e),
            error_category=error_category,
        )

        # Публикуем событие ошибки job'а
        publish_job_failed(
            run_id=run_id,
            job_type="fetch_comments",
            job_id=run_id,
            error_code=error_code,
            error_message=str(e),
            platform=platform,
            platform_video_id=platform_video_id,
            stage="fetch_comments",
        )

        # Retry только для retryable ошибок
        if is_retryable_error(e) and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        else:
            # Non-retryable или превышен лимит retry - fail fast
            raise


@celery_app.task(
    name="fetcher.finalize",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="fetch.finalize",
    priority=9,
)
def finalize_task(self, run_id: str) -> None:
    """Celery задача для artifact builder и завершения pipeline.

    Логика:
    1. Ждёт завершения всех обязательных задач (metadata, video, comments)
    2. Строит manifest.json
    3. Загружает manifest в storage
    4. Обновляет статус run на COMPLETED
    5. Enqueue process_run(run_id) в DataProcessor (если нужно)

    Args:
        run_id: UUID run'а
    """
    try:
        log_with_context(
            logger,
            logging.INFO,
            f"Starting finalize task for run_id={run_id}",
            run_id=run_id,
            stage="finalize",
            platform="youtube",
        )

        # Идемпотентность: если run уже завершён, ничего не делаем (защита от повторной постановки)
        run_uuid = uuid.UUID(run_id)
        with session_scope() as db:
            run = db.query(Run).filter(Run.id == run_uuid).first()
            if run and run.status == "COMPLETED":
                log_with_context(
                    logger,
                    logging.INFO,
                    f"Run {run_id} already completed, skipping finalize",
                    run_id=run_id,
                    stage="finalize",
                    platform="youtube",
                )
                return

        # Проверка backpressure перед обработкой
        if check_backpressure():
            log_with_context(
                logger,
                logging.WARNING,
                f"Backpressure detected, retrying finalize task later for run_id={run_id}",
                run_id=run_id,
                stage="finalize",
                platform="youtube",
            )
            # Retry через 5 минут (300 секунд)
            raise BackpressureError(
                "DataProcessor queue is overloaded, retrying later",
                retry_after=300,
            )

        # Ждём готовности всех артефактов (polling)
        max_wait_time = 1800  # 30 минут
        poll_interval = 5  # 5 секунд
        log_wait_interval = 30  # логировать каждые 30 с
        start_time = time.time()
        last_log_time = start_time

        while not all_artifacts_ready(run_id):
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                missing_set, reason = get_missing_artifact_types(run_id)
                detail = reason or (f"missing: {sorted(missing_set)}" if missing_set else "unknown")
                raise TimeoutError(f"Artifacts not ready after {max_wait_time}s ({detail})")
            time.sleep(poll_interval)
            # Логируем раз в log_wait_interval, чтобы не засорять лог
            if time.time() - last_log_time >= log_wait_interval:
                last_log_time = time.time()
                missing_set, reason = get_missing_artifact_types(run_id)
                if reason:
                    msg = f"Waiting for artifacts for run_id={run_id}: {reason} (elapsed {int(elapsed)}s)"
                else:
                    msg = f"Waiting for artifacts for run_id={run_id}: missing {sorted(missing_set or set())} (elapsed {int(elapsed)}s)"
                log_with_context(
                    logger,
                    logging.WARNING,
                    msg,
                    run_id=run_id,
                    stage="finalize",
                    platform="youtube",
                )

        # Строим manifest
        run_artifact_builder(run_id)

        # Обновляем статус run на COMPLETED с валидацией перехода
        run_uuid = uuid.UUID(run_id)
        webhook_url: Optional[str] = None
        platform: Optional[str] = None
        platform_video_id: Optional[str] = None

        with session_scope() as db:
            run: Run | None = db.query(Run).filter(Run.id == run_uuid).first()
            if run:
                from .state_machine import RUN_STATUS_COMPLETED, RUN_STATUS_FAILED, RUN_STATUS_FINALIZING, validate_transition

                old_status = run.status
                if old_status == RUN_STATUS_FAILED:
                    # Восстановление: run был помечен FAILED при предыдущей ошибке finalize (например до фикса retry).
                    log_with_context(
                        logger,
                        logging.INFO,
                        f"Recovering run {run_id} from FAILED to COMPLETED after successful finalize",
                        run_id=run_id,
                        stage="finalize",
                        platform="youtube",
                    )
                else:
                    validate_transition(old_status, RUN_STATUS_COMPLETED, run_id=run_id)
                run.status = RUN_STATUS_COMPLETED
                run.finished_at = datetime.utcnow()
                if run.error:
                    run.error = None  # очистить ошибку от предыдущей попытки

                # Получаем webhook_url и platform для отправки webhook
                webhook_url = getattr(run, "webhook_url", None)
                if webhook_url:
                    # Получаем platform из video_source
                    from .models import VideoSource

                    video_source = (
                        db.query(VideoSource)
                        .filter(VideoSource.run_id == run_uuid)
                        .order_by(VideoSource.created_at)
                        .first()
                    )
                    if video_source:
                        platform = video_source.platform
                        platform_video_id = video_source.normalized_video_id

                db.commit()

        # Отправляем webhook если настроен
        if webhook_url:
            try:
                # Используем asyncio для async функции send_webhook
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Если loop уже запущен, создаём task
                    asyncio.create_task(
                        send_webhook(
                            webhook_url=webhook_url,
                            run_id=run_id,
                            status="COMPLETED",
                            platform=platform,
                            platform_video_id=platform_video_id,
                        )
                    )
                else:
                    # Если loop не запущен, запускаем
                    loop.run_until_complete(
                        send_webhook(
                            webhook_url=webhook_url,
                            run_id=run_id,
                            status="COMPLETED",
                            platform=platform,
                            platform_video_id=platform_video_id,
                        )
                    )
            except Exception as e:
                log_with_context(
                    logger,
                    logging.ERROR,
                    f"Failed to send webhook for run_id={run_id}: {e}",
                    run_id=run_id,
                    stage="finalize",
                    platform=platform or "unknown",
                )

        # Phase 2: вызов Backend trigger-processing для запуска DataProcessor
        backend_base = getattr(fetcher_settings, "backend_base_url", None)
        if backend_base:
            import httpx

            url = f"{backend_base.rstrip('/')}/api/runs/{run_id}/trigger-processing"
            headers = {"Content-Type": "application/json"}
            api_key = getattr(fetcher_settings, "backend_trigger_api_key", None)
            if api_key:
                headers["X-API-Key"] = api_key
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, headers=headers)
                    if resp.status_code == 202:
                        log_with_context(
                            logger,
                            logging.INFO,
                            f"Backend trigger-processing accepted for run_id={run_id}",
                            run_id=run_id,
                            stage="finalize",
                            platform="youtube",
                        )
                    else:
                        log_with_context(
                            logger,
                            logging.WARNING,
                            f"Backend trigger returned {resp.status_code} for run_id={run_id}: {resp.text[:200]}",
                            run_id=run_id,
                            stage="finalize",
                            platform="youtube",
                        )
            except Exception as e:
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"Failed to call Backend trigger for run_id={run_id}: {e}",
                    run_id=run_id,
                    stage="finalize",
                    platform="youtube",
                )

        log_with_context(
            logger,
            logging.INFO,
            f"Finalize task completed for run_id={run_id}",
            run_id=run_id,
            stage="finalize",
            platform="youtube",
        )
    except BackpressureError as e:
        # Backpressure - retry с указанным retry_after
        log_with_context(
            logger,
            logging.WARNING,
            f"Backpressure detected for run_id={run_id}, retrying after {e.retry_after}s",
            run_id=run_id,
            stage="finalize",
            platform="youtube",
        )
        raise self.retry(exc=e, countdown=e.retry_after)
    except Exception as e:
        # Переводим в FAILED только когда не будем ретраить — иначе retry не сможет перевести в COMPLETED
        will_retry = is_retryable_error(e) and self.request.retries < self.max_retries
        if not will_retry:
            run_uuid = uuid.UUID(run_id)
            try:
                with session_scope() as db:
                    run: Run | None = db.query(Run).filter(Run.id == run_uuid).first()
                    if run:
                        from .state_machine import RUN_STATUS_FAILED, validate_transition

                        old_status = run.status
                        try:
                            validate_transition(old_status, RUN_STATUS_FAILED, run_id=run_id)
                        except ValueError:
                            log_with_context(
                                logger,
                                logging.WARNING,
                                f"Invalid transition to FAILED from {old_status}, "
                                f"but allowing it due to error: {e}",
                                run_id=run_id,
                                stage="finalize",
                            )
                        run.status = RUN_STATUS_FAILED
                        run.error = str(e)
                        run.finished_at = datetime.utcnow()

                        # Получаем webhook_url и platform для отправки webhook
                        webhook_url = getattr(run, "webhook_url", None)
                        platform = None
                        platform_video_id = None
                        if webhook_url:
                            from .models import VideoSource

                            video_source = (
                                db.query(VideoSource)
                                .filter(VideoSource.run_id == run_uuid)
                                .order_by(VideoSource.created_at)
                                .first()
                            )
                            if video_source:
                                platform = video_source.platform
                                platform_video_id = video_source.normalized_video_id

                        db.commit()

                        if webhook_url:
                            try:
                                import asyncio

                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    asyncio.create_task(
                                        send_webhook(
                                            webhook_url=webhook_url,
                                            run_id=run_id,
                                            status="FAILED",
                                            platform=platform,
                                            platform_video_id=platform_video_id,
                                            error=str(e),
                                        )
                                    )
                                else:
                                    loop.run_until_complete(
                                        send_webhook(
                                            webhook_url=webhook_url,
                                            run_id=run_id,
                                            status="FAILED",
                                            platform=platform,
                                            platform_video_id=platform_video_id,
                                            error=str(e),
                                        )
                                    )
                            except Exception as webhook_error:
                                log_with_context(
                                    logger,
                                    logging.ERROR,
                                    f"Failed to send webhook for failed run_id={run_id}: {webhook_error}",
                                    run_id=run_id,
                                    stage="finalize",
                                    platform=platform or "unknown",
                                )
            except Exception as db_error:
                log_with_context(
                    logger,
                    logging.WARNING,
                    f"Failed to update run status to FAILED: {db_error}",
                    run_id=run_id,
                    stage="finalize",
                )

        error_category = get_error_category(e)
        log_with_context(
            logger,
            logging.ERROR,
            f"Finalize task failed for run_id={run_id}: {e} (category: {error_category})",
            run_id=run_id,
            stage="finalize",
            platform="youtube",
            exception=str(e),
            error_category=error_category,
        )

        # Retry только для retryable ошибок
        if is_retryable_error(e) and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=30 * (2 ** self.request.retries))
        else:
            # Non-retryable или превышен лимит retry - fail fast
            raise


@celery_app.task(
    name="fetcher.lifecycle_cleanup",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="fetch.maintenance",
    priority=1,
)
def lifecycle_cleanup_task(self) -> dict:
    """Celery периодическая задача для lifecycle cleanup.

    Запускается автоматически через Celery Beat (ежедневно).
    Можно также запускать вручную.

    Returns:
        Dict с результатами очистки по каждому типу
    """
    from .config import settings

    try:
        log_with_context(
            logger,
            logging.INFO,
            "Starting lifecycle cleanup task",
            stage="lifecycle_cleanup",
        )

        results = run_lifecycle_cleanup(
            raw_video_retention_days=settings.raw_video_retention_days,
            raw_comments_retention_days=settings.raw_comments_retention_days,
            raw_comments_hard_cap_days=settings.raw_comments_hard_cap_days,
            temp_files_retention_days=settings.temp_files_retention_days,
            failed_runs_retention_days=settings.failed_runs_retention_days,
        )

        log_with_context(
            logger,
            logging.INFO,
            f"Lifecycle cleanup task completed: {results}",
            stage="lifecycle_cleanup",
        )

        return results
    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"Lifecycle cleanup task failed: {e}",
            stage="lifecycle_cleanup",
            exception=str(e),
        )
        # Retry для retryable ошибок
        raise self.retry(exc=e, countdown=300 * (2 ** self.request.retries))


@celery_app.task(
    name="fetcher.periodic_snapshots",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="fetch.maintenance",
    priority=1,
)
def periodic_snapshots_task(self) -> dict:
    """Celery периодическая задача для создания периодических snapshots.

    Запускается автоматически через Celery Beat (ежедневно).
    Обрабатывает видео, которым нужен новый snapshot согласно schedule.

    Returns:
        Dict с результатами: {"processed": N, "created": M, "failed": K, "skipped": L}
    """
    from .config import settings

    try:
        log_with_context(
            logger,
            logging.INFO,
            "Starting periodic snapshots task",
            stage="periodic_snapshots",
        )

        if not settings.enable_snapshots:
            log_with_context(
                logger,
                logging.INFO,
                "Snapshots disabled, skipping periodic snapshots task",
                stage="periodic_snapshots",
            )
            return {"processed": 0, "created": 0, "failed": 0, "skipped": 0}

        # Получаем список видео, которым нужен snapshot
        videos = get_videos_needing_snapshot(
            schedule_days=settings.snapshot_schedule_days,
            batch_size=100,  # Обрабатываем по 100 видео за раз
        )

        if not videos:
            log_with_context(
                logger,
                logging.INFO,
                "No videos need snapshots at this time",
                stage="periodic_snapshots",
            )
            return {"processed": 0, "created": 0, "failed": 0, "skipped": 0}

        # Создаём snapshots
        results = create_snapshots_for_videos(videos)

        log_with_context(
            logger,
            logging.INFO,
            f"Periodic snapshots task completed: processed={len(videos)}, {results}",
            stage="periodic_snapshots",
            processed=len(videos),
            **results,
        )

        return {
            "processed": len(videos),
            "created": results["created"],
            "failed": results["failed"],
            "skipped": results["skipped"],
        }

    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"Periodic snapshots task failed: {e}",
            stage="periodic_snapshots",
            exception=str(e),
        )
        # Retry для retryable ошибок
        if is_retryable_error(e) and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=300 * (2 ** self.request.retries))
        else:
            raise


@celery_app.task(
    name="fetcher.requeue_stuck_finalize",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="fetch.maintenance",
    priority=2,
)
def requeue_stuck_finalize_task(self) -> dict:
    """Периодически перепоставить finalize для run'ов, застрявших в FINALIZING (сетка безопасности).

    Находит run'ы со статусом FINALIZING без finished_at, созданные более 2 минут назад,
    и вызывает finalize_task.delay(run_id). finalize_task идемпотентен (пропускает уже COMPLETED).
    """
    from datetime import datetime, timedelta
    from .state_machine import RUN_STATUS_FINALIZING

    try:
        stuck_since = datetime.utcnow() - timedelta(seconds=120)
        with session_scope() as db:
            stuck = (
                db.query(Run)
                .filter(Run.status == RUN_STATUS_FINALIZING)
                .filter(Run.finished_at.is_(None))
                .filter(Run.created_at < stuck_since)
                .limit(20)
                .all()
            )
            run_ids = [str(r.id) for r in stuck]
        if not run_ids:
            return {"requeued": 0}
        for run_id in run_ids:
            celery_app.send_task("fetcher.finalize", args=[run_id], queue="fetch.finalize")
        log_with_context(
            logger,
            logging.INFO,
            f"Requeued finalize for {len(run_ids)} stuck run(s): {run_ids[:5]}{'...' if len(run_ids) > 5 else ''}",
            stage="requeue_stuck_finalize",
        )
        return {"requeued": len(run_ids)}
    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"requeue_stuck_finalize failed: {e}",
            stage="requeue_stuck_finalize",
            exception=str(e),
        )
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task(
    name="fetcher.tasks.aggregate_stats_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue="fetch.maintenance",
    priority=1,
)
def aggregate_stats_task(self) -> None:
    """Celery задача для агрегации статистики. Запускается по расписанию (beat) каждую минуту."""
    try:
        log_with_context(
            logger,
            logging.INFO,
            "Starting aggregate_stats task",
            stage="aggregate_stats",
        )
        aggregate_stats()
        log_with_context(
            logger,
            logging.INFO,
            "aggregate_stats task completed",
            stage="aggregate_stats",
        )
    except Exception as e:
        log_with_context(
            logger,
            logging.ERROR,
            f"aggregate_stats task failed: {e}",
            stage="aggregate_stats",
            exception=str(e),
        )
        if is_retryable_error(e) and self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        raise


__all__ = [
    "fetch_metadata_task",
    "download_video_task",
    "fetch_comments_task",
    "finalize_task",
    "lifecycle_cleanup_task",
    "periodic_snapshots_task",
    "requeue_stuck_finalize_task",
    "aggregate_stats_task",
]

