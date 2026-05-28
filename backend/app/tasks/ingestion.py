"""Celery: ingestion runs (Fetcher) и синхронизация статусов."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from uuid import UUID

import httpx

from ..config import Settings
from ..db import session_scope
from ..dbv2 import models as v2_models
from ..services.dataprocessor import run_dataprocessor_async, wait_for_run_completion_hybrid
from ..services.dataprocessor_adapter import build_ingestion_payload_from_fetcher
from ..services.fetcher_client import get_run as fetcher_get_run
from ..worker import celery_app
from .events import _emit_stage, _emit_status, _utcnow

_logger_ingestion = logging.getLogger(__name__)


@celery_app.task(name="process_ingestion_run")
def process_ingestion_run(run_id: str) -> None:
    """
    Обработка run'а ингестии: забирает manifest и артефакты из Fetcher (Phase 3:
    build_ingestion_payload_from_fetcher), передаёт video_url в DataProcessor;
    DataProcessor скачивает видео в свой кэш и обрабатывает.

    Вызывается после POST /api/runs/{run_id}/trigger-processing (из Fetcher finalize).
    См. docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md, docs/PHASE3_ARTIFACTS_CONTRACT.md.
    """
    run_uuid = UUID(run_id)
    settings = Settings()
    paths = settings.resolve_paths()

    try:
        with session_scope() as db:
            ingestion = (
                db.query(v2_models.IngestionRun)
                .filter(v2_models.IngestionRun.run_id == run_uuid)
                .first()
            )
            if not ingestion:
                _logger_ingestion.warning("IngestionRun not found for run_id=%s", run_id)
                return
            ingestion.ingestion_status = "processing"
            db.commit()

        payload = build_ingestion_payload_from_fetcher(run_id, settings=settings)
        if not payload.video_url:
            with session_scope() as db:
                ing = (
                    db.query(v2_models.IngestionRun)
                    .filter(v2_models.IngestionRun.run_id == run_uuid)
                    .first()
                )
                if ing:
                    ing.ingestion_status = "failed"
                    db.commit()
            _logger_ingestion.error("No video_file URL in payload for run_id=%s", run_id)
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                run_dataprocessor_async(
                    video_url=payload.video_url,
                    platform_id=payload.platform_id,
                    video_id=payload.video_id,
                    run_id=payload.run_id,
                    profile_config=payload.profile_config,
                    result_store_base=paths.result_store_base,
                    frames_dir_base=paths.frames_dir_base,
                    visual_cfg_default=paths.visual_cfg_default,
                )
            )
            final_status = loop.run_until_complete(
                wait_for_run_completion_hybrid(
                    run_id=payload.run_id,
                    webhook_timeout=30,
                    timeout_seconds=settings.dataprocessor_timeout_seconds,
                    poll_interval=settings.dataprocessor_poll_interval,
                )
            )
            with session_scope() as db:
                ing = (
                    db.query(v2_models.IngestionRun)
                    .filter(v2_models.IngestionRun.run_id == run_uuid)
                    .first()
                )
                if ing:
                    status = (final_status.get("status") or "").lower()
                    if status == "success":
                        ing.ingestion_status = "completed"
                    else:
                        ing.ingestion_status = "failed"
                        error_message = (
                            final_status.get("error")
                            or final_status.get("message")
                            or f"Unexpected DataProcessor status: {status or 'unknown'}"
                        )
                        ing.fetcher_error_code = final_status.get("error_code") or "DATAPROCESSOR_FAILED"
                        ing.fetcher_error_message = error_message
                    db.commit()
        finally:
            if loop != asyncio.get_event_loop():
                try:
                    loop.close()
                except Exception:
                    pass
    except ValueError as e:
        _logger_ingestion.warning("process_ingestion_run validation failed for run_id=%s: %s", run_id, e)
        with session_scope() as db:
            ing = (
                db.query(v2_models.IngestionRun)
                .filter(v2_models.IngestionRun.run_id == run_uuid)
                .first()
            )
            if ing:
                ing.ingestion_status = "failed"
                db.commit()
    except Exception as e:
        _logger_ingestion.exception("process_ingestion_run failed for run_id=%s: %s", run_id, e)
        with session_scope() as db:
            ing = (
                db.query(v2_models.IngestionRun)
                .filter(v2_models.IngestionRun.run_id == run_uuid)
                .first()
            )
            if ing:
                ing.ingestion_status = "failed"
                db.commit()
        raise


_logger_sync = logging.getLogger(__name__)

# Если beat вызывает sync до того, как Fetcher успел записать run (POST ещё в полёте),
# GET /runs/{id} даёт 404 — не помечаем ingestion failed в первые N секунд после создания.
_FETCHER_404_GRACE_AFTER_CREATE_SEC = 120

_FETCHER_STATUS_TO_INGESTION = {
    "PENDING": "pending",
    "RUNNING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "failed",
    # Промежуточные статусы Fetcher (state_machine) → running
    "NORMALIZING_SOURCE": "running",
    "CHECKING_CACHE": "running",
    "FETCHING_METADATA": "running",
    "FETCHING_CHANNEL": "running",
    "FETCHING_COMMENTS": "running",
    "DOWNLOADING_VIDEO": "running",
    "UPLOADING_ARTIFACTS": "running",
    "FINALIZING": "running",
}


@celery_app.task(name="sync_ingestion_run_status")
def sync_ingestion_run_status(batch_size: int = 50) -> None:
    """
    Опрашивает Fetcher для run'ов в статусе pending/running и обновляет БД + публикует события.

    Phase 4: транспорт Fetcher → Backend — polling GET /api/v1/runs/{run_id}.
    Запускать по расписанию (Celery beat), например каждые 10–30 секунд.

    Args:
        batch_size: Максимум run'ов за один вызов (по умолчанию 50).
    """
    settings = Settings()
    sync_cutoff = _utcnow() - timedelta(hours=max(1, settings.ingestion_sync_lookback_hours))
    with session_scope() as db:
        rows = (
            db.query(v2_models.IngestionRun)
            .filter(
                v2_models.IngestionRun.ingestion_status.in_(("pending", "running")),
                v2_models.IngestionRun.updated_at >= sync_cutoff,
            )
            .order_by(v2_models.IngestionRun.updated_at.desc())
            .limit(batch_size)
            .all()
        )
        if not rows:
            return
        run_ids = [r.run_id for r in rows]

    for i, run_uuid in enumerate(run_ids):
        if i > 0:
            time.sleep(0.15)  # снизить burst и риск 429 со стороны Fetcher
        run_id_str = str(run_uuid)
        try:
            data = fetcher_get_run(run_uuid, settings=settings)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Fetcher rate limit: пауза и один повтор через 5 с
                _logger_sync.warning(
                    "Fetcher rate limit (429) for run_id=%s, retrying once in 5s",
                    run_id_str,
                )
                time.sleep(5)
                try:
                    data = fetcher_get_run(run_uuid, settings=settings)
                except httpx.HTTPStatusError as e2:
                    if e2.response.status_code == 429:
                        _logger_sync.warning(
                            "Fetcher still 429 for run_id=%s, skipping this cycle",
                            run_id_str,
                        )
                        break
                    raise
                # retry успешен — data задан, обрабатываем ниже
            else:
                if e.response.status_code == 404:
                    within_grace = False
                    with session_scope() as db:
                        ing = (
                            db.query(v2_models.IngestionRun)
                            .filter(v2_models.IngestionRun.run_id == run_uuid)
                            .first()
                        )
                        if ing:
                            if ing.ingestion_status == "pending" and ing.created_at is not None:
                                c_at = ing.created_at
                                if getattr(c_at, "tzinfo", None) is not None:
                                    c_at = c_at.replace(tzinfo=None)
                                within_grace = (_utcnow() - c_at) < timedelta(
                                    seconds=_FETCHER_404_GRACE_AFTER_CREATE_SEC
                                )
                            if within_grace:
                                _logger_sync.info(
                                    "Fetcher 404 for run_id=%s within %ss of create — "
                                    "assuming race with POST /runs; not failing",
                                    run_id_str,
                                    _FETCHER_404_GRACE_AFTER_CREATE_SEC,
                                )
                            else:
                                ing.ingestion_status = "failed"
                                ing.fetcher_error_code = "RUN_NOT_FOUND"
                                ing.fetcher_error_message = "Run not found in Fetcher"
                                db.commit()
                                _emit_status(
                                    run_id_str,
                                    status="failed",
                                    error_code="RUN_NOT_FOUND",
                                    error_message="Run not found in Fetcher",
                                )
                _logger_sync.warning(
                    "Fetcher get_run HTTP error for run_id=%s: %s",
                    run_id_str,
                    e,
                    exc_info=False,
                )
                if e.response.status_code == 429:
                    break
                continue
        except Exception as e:
            _logger_sync.warning(
                "Fetcher get_run failed for run_id=%s: %s",
                run_id_str,
                e,
                exc_info=False,
            )
            continue

        status = (data.get("status") or "").upper()
        ingestion_status = _FETCHER_STATUS_TO_INGESTION.get(
            status, "running" if status else "pending"
        )
        error = data.get("error")
        error_code = data.get("error_code")
        progress = data.get("progress") or {}
        completed_stages = progress.get("completed_stages") or []
        stage = progress.get("stage") or (completed_stages[-1] if completed_stages else None)

        with session_scope() as db:
            ing = (
                db.query(v2_models.IngestionRun)
                .filter(v2_models.IngestionRun.run_id == run_uuid)
                .first()
            )
            if not ing:
                continue
            if ing.ingestion_status not in ("pending", "running"):
                # Состояние изменилось после выборки batch'а (например run уже в processing).
                continue
            prev_status = ing.ingestion_status
            prev_stage = ing.fetcher_stage
            ing.ingestion_status = ingestion_status
            ing.fetcher_stage = stage
            ing.fetcher_error_code = error_code
            ing.fetcher_error_message = error
            db.commit()

        if prev_status != ingestion_status:
            _emit_status(
                run_id_str,
                status=ingestion_status,
                error_code=error_code,
                error_message=error,
            )
        if stage and prev_stage != stage:
            _emit_stage(run_id_str, stage)
