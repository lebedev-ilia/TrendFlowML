from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..dbv2.enums import AnalysisStatus
from ..dbv2.models import (
    AnalysisJob,
    AnalysisSnapshot,
    Prediction,
    ProcessingConfig,
    Video,
    Workspace,
    WorkspaceMember,
)
from ..dbv2.models import User as CoreUser
from ..deps import get_current_user, get_db
from ..schemas import (
    AnalysisJobCreate,
    AnalysisJobOut,
    PredictionCreate,
    PredictionOut,
    PublicReportOut,
    ShareLinkOut,
    SnapshotOut,
    SnapshotUpsert,
)
from ..services import billing
from ..services.dataprocessor import request_dataprocessor_cancel
from ..tasks import process_analysis_job

router = APIRouter(prefix="/api", tags=["analysis"])
logger = logging.getLogger(__name__)


def _require_workspace_member(db: Session, workspace_id: uuid.UUID, user_id: uuid.UUID) -> None:
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")
    m = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.workspace_id == workspace_id, WorkspaceMember.user_id == user_id)
        .first()
    )
    if not m:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workspace access denied")


@router.post(
    "/workspaces/{workspace_id}/videos/{video_id}/analysis",
    response_model=AnalysisJobOut,
    status_code=status.HTTP_201_CREATED,
)
def create_analysis_job(
    workspace_id: uuid.UUID,
    video_id: uuid.UUID,
    payload: AnalysisJobCreate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_workspace_member(db, workspace_id, user.id)
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    job = AnalysisJob(
        workspace_id=workspace_id,
        video_id=video_id,
        triggered_by_user_id=user.id,
        processing_config_id=payload.processing_config_id,
        model_version_id=payload.model_version_id,
        status=AnalysisStatus.queued,
        retry_count=0,
    )
    db.add(job)
    db.flush()

    # Списание за запуск. Стоимость берётся из конфигурации; если она не
    # заполнена, анализ проводится без списания — выдумывать цену нельзя.
    config = (
        db.query(ProcessingConfig)
        .filter(ProcessingConfig.id == payload.processing_config_id)
        .first()
    )
    cost_units = (config.estimated_cost_units or 0) if config else 0

    if cost_units > 0:
        try:
            billing.charge_for_analysis(
                db,
                workspace_id=workspace_id,
                user_id=user.id,
                analysis_job_id=job.id,
                cost_units=cost_units,
                description=f"Анализ «{v.title}»",
            )
        except billing.InsufficientFunds as e:
            # Задача не создаётся: транзакция откатывается целиком.
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Недостаточно единиц: на балансе {e.balance}, "
                    f"требуется {e.required}"
                ),
            ) from e

    db.commit()
    db.refresh(job)

    # Ставим задачу в очередь Celery для обработки
    process_analysis_job.delay(str(job.id))

    return job


@router.post("/analysis/{analysis_job_id}/cancel")
def cancel_analysis_job(
    analysis_job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Отмена AnalysisJob.

    **Семантика:** ``queued`` → сразу ``canceled`` в БД; ``processing`` → запрос к DataProcessor
    ``POST /api/v1/runs/{run_id}/cancel`` (``run_id`` = ``analysis_job.id``); терминальные статусы → noop.
    Подробнее: ``docs/OPERATIONS.md``, ``GAPS_AND_ALIGNMENT.md`` §5.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)

    if job.status in (
        AnalysisStatus.completed,
        AnalysisStatus.failed,
        AnalysisStatus.canceled,
    ):
        return {
            "status": "noop",
            "analysis_job_id": str(job.id),
            "job_status": job.status.value,
        }

    if job.status == AnalysisStatus.queued:
        job.status = AnalysisStatus.canceled
        job.completed_at = datetime.utcnow()
        db.commit()
        return {"status": "canceled", "analysis_job_id": str(job.id)}

    if job.status == AnalysisStatus.processing:
        dp_ok = request_dataprocessor_cancel(str(job.id))
        if not dp_ok:
            logger.warning(
                "cancel_analysis_job: DataProcessor cancel did not succeed "
                "(run may not exist yet or API error); job_id=%s",
                job.id,
            )
        db.commit()
        return {
            "status": "cancel_requested",
            "analysis_job_id": str(job.id),
            "dataprocessor_notified": dp_ok,
        }

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unexpected analysis job status",
    )


@router.get("/analysis/{analysis_job_id}", response_model=AnalysisJobOut)
def get_analysis_job(
    analysis_job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Одна задача анализа по идентификатору (docs/API.md §6)."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)
    return job


@router.get("/workspaces/{workspace_id}/analysis", response_model=List[AnalysisJobOut])
def list_analysis_jobs(
    workspace_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    _require_workspace_member(db, workspace_id, user.id)
    return (
        db.query(AnalysisJob)
        .filter(AnalysisJob.workspace_id == workspace_id)
        .order_by(AnalysisJob.created_at.desc())
        .all()
    )


@router.post("/analysis/{analysis_job_id}/predictions", response_model=PredictionOut)
def create_prediction(
    analysis_job_id: uuid.UUID,
    payload: PredictionCreate,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)

    p = Prediction(
        analysis_job_id=analysis_job_id,
        horizon_days=payload.horizon_days,
        predicted_views=payload.predicted_views,
        predicted_likes=payload.predicted_likes,
        percentile_score=payload.percentile_score,
        confidence_lower=payload.confidence_lower,
        confidence_upper=payload.confidence_upper,
        model_version_id=payload.model_version_id,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.get("/analysis/{analysis_job_id}/predictions", response_model=List[PredictionOut])
def list_predictions(
    analysis_job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)
    return (
        db.query(Prediction)
        .filter(Prediction.analysis_job_id == analysis_job_id)
        .order_by(Prediction.horizon_days.asc())
        .all()
    )


@router.post("/analysis/{analysis_job_id}/share", response_model=ShareLinkOut)
def share_analysis(
    analysis_job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Создаёт (или возвращает существующую) публичную ссылку на отчёт.

    Делиться можно только завершённым анализом — у незавершённого нет прогноза.
    Токен генерируется один раз и переиспользуется при повторном запросе.
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)

    if job.status != AnalysisStatus.completed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поделиться можно только завершённым анализом",
        )

    if not job.share_token:
        job.share_token = secrets.token_urlsafe(16)
        db.commit()
        db.refresh(job)
    return ShareLinkOut(analysis_job_id=job.id, share_token=job.share_token)


@router.delete("/analysis/{analysis_job_id}/share")
def revoke_share(
    analysis_job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Отзывает публичную ссылку — старый URL перестаёт работать."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)

    job.share_token = None
    db.commit()
    return {"status": "ok"}


@router.get("/analysis/{analysis_job_id}/snapshot", response_model=SnapshotOut)
def get_snapshot(
    analysis_job_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """snapshot_0 анализа — состояние видео и канала на момент сбора."""
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)

    snap = (
        db.query(AnalysisSnapshot)
        .filter(AnalysisSnapshot.analysis_job_id == analysis_job_id)
        .first()
    )
    if not snap:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Snapshot not found")
    return snap


@router.put("/analysis/{analysis_job_id}/snapshot", response_model=SnapshotOut)
def upsert_snapshot(
    analysis_job_id: uuid.UUID,
    payload: SnapshotUpsert,
    db: Session = Depends(get_db),
    user: CoreUser = Depends(get_current_user),
):
    """Запись/обновление snapshot_0.

    Контракт для Fetcher/DataProcessor: состояние собирается на момент анализа
    и передаётся сюда. Одна запись на анализ (upsert по analysis_job_id).
    """
    job = db.query(AnalysisJob).filter(AnalysisJob.id == analysis_job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis job not found")
    _require_workspace_member(db, job.workspace_id, user.id)

    snap = (
        db.query(AnalysisSnapshot)
        .filter(AnalysisSnapshot.analysis_job_id == analysis_job_id)
        .first()
    )
    if snap is None:
        snap = AnalysisSnapshot(analysis_job_id=analysis_job_id)
        db.add(snap)
    for field, value in payload.model_dump().items():
        setattr(snap, field, value)
    db.commit()
    db.refresh(snap)
    return snap


# Публичный отчёт — БЕЗ авторизации, доступен по токену. Отдаём только top-line
# данные (прогноз): состояние канала и внутренние идентификаторы не раскрываем.
public_router = APIRouter(prefix="/api/public", tags=["public"])


@public_router.get("/reports/{share_token}", response_model=PublicReportOut)
def public_report(share_token: str, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.share_token == share_token).first()
    if not job or job.status != AnalysisStatus.completed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    video = db.query(Video).filter(Video.id == job.video_id).first()
    predictions = (
        db.query(Prediction)
        .filter(Prediction.analysis_job_id == job.id)
        .order_by(Prediction.horizon_days.asc())
        .all()
    )
    return PublicReportOut(
        video_title=video.title if video else "Видео",
        source="youtube" if (video and video.source_url) else "upload",
        predictions=predictions,
    )
