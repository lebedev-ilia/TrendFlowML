from __future__ import annotations

import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from ..auth import decode_token
from ..config import Settings
from ..deps import get_current_user, get_db
from fastapi import Query, Response

from ..models import Artifact, AnalysisProfile, Run, RunComponent, RunLog, User, UserVideoLink, Video
from ..schemas import RunCreate, RunLogOut, RunOut, RunResultOut
from ..services.events import subscribe_run_events
from ..tasks import process_run


router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", response_model=RunOut)
def create_run(payload: RunCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    link = (
        db.query(UserVideoLink)
        .filter(UserVideoLink.user_id == user.id, UserVideoLink.video_id == payload.video_id)
        .first()
    )
    if not link:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    if payload.profile_id:
        profile = db.query(AnalysisProfile).filter(AnalysisProfile.id == payload.profile_id).first()
        if not profile or (profile.user_id is not None and profile.user_id != user.id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    run = Run(user_id=user.id, video_id=payload.video_id, profile_id=payload.profile_id, status="queued")
    db.add(run)
    db.flush()

    # Minimal run components for progress (segmenter + visual)
    for comp in ["segmenter", "visual"]:
        rc = RunComponent(run_id=run.id, component_name=comp, status="queued")
        db.add(rc)
    db.flush()

    process_run.delay(run.id)
    return RunOut(
        id=run.id,
        video_id=run.video_id,
        profile_id=run.profile_id,
        status=run.status,
        stage=run.stage,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_code=run.error_code,
        error_message=run.error_message,
    )


@router.get("", response_model=List[RunOut])
def list_runs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    runs = db.query(Run).filter(Run.user_id == user.id).order_by(Run.created_at.desc()).all()
    return [
        RunOut(
            id=r.id,
            video_id=r.video_id,
            profile_id=r.profile_id,
            status=r.status,
            stage=r.stage,
            created_at=r.created_at,
            started_at=r.started_at,
            finished_at=r.finished_at,
            error_code=r.error_code,
            error_message=r.error_message,
        )
        for r in runs
    ]


@router.get("/{run_id}", response_model=RunOut)
def get_run(run_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == user.id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RunOut(
        id=run.id,
        video_id=run.video_id,
        profile_id=run.profile_id,
        status=run.status,
        stage=run.stage,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error_code=run.error_code,
        error_message=run.error_message,
    )


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == user.id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    run.cancel_requested_at = run.cancel_requested_at or datetime.utcnow()
    db.flush()
    return {"status": "ok"}


@router.get("/{run_id}/logs", response_model=List[RunLogOut])
def run_logs(run_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == user.id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    logs = db.query(RunLog).filter(RunLog.run_id == run_id).order_by(RunLog.ts.asc()).all()
    return [RunLogOut(ts=l.ts, level=l.level, message=l.message) for l in logs]


@router.get("/{run_id}/manifest")
def run_manifest(run_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == user.id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    settings = Settings()
    paths = settings.resolve_paths()
    video = db.query(Video).filter(Video.id == run.video_id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    manifest = paths.result_store_base / video.platform_id / video.video_id / run.id / "manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manifest not found")
    return json.loads(manifest.read_text(encoding="utf-8"))


@router.get("/{run_id}/result", response_model=RunResultOut)
def run_result(run_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == user.id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    settings = Settings()
    paths = settings.resolve_paths()
    video = db.query(Video).filter(Video.id == run.video_id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    manifest_path = paths.result_store_base / video.platform_id / video.video_id / run.id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    artifacts = (
        db.query(Artifact)
        .filter(Artifact.run_id == run_id)
        .order_by(Artifact.component_name.asc())
        .all()
    )
    artifacts_payload = [
        {
            "component": a.component_name,
            "kind": a.kind,
            "object_key": a.object_key,
            "size_bytes": a.size_bytes,
            "sha256": a.sha256_hex,
        }
        for a in artifacts
    ]

    return RunResultOut(run_id=run_id, manifest=manifest, artifacts=artifacts_payload)


@router.get("/{run_id}/artifact")
def get_artifact(
    run_id: str,
    object_key: str = Query(..., description="Artifact path from /result"),
    token: str | None = Query(default=None, description="Optional bearer token for iframe access"),
    db: Session = Depends(get_db),
):
    user_id = None
    if token:
        user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    run = db.query(Run).filter(Run.id == run_id, Run.user_id == user_id).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    settings = Settings()
    paths = settings.resolve_paths()
    path = Path(object_key).resolve()
    if not str(path).startswith(str(paths.result_store_base.resolve())):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden artifact path")
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    data = path.read_bytes()
    media = "text/html" if path.suffix.lower() == ".html" else "application/octet-stream"
    return Response(content=data, media_type=media)


@router.websocket("/{run_id}/events")
async def run_events(run_id: str, ws: WebSocket):
    await ws.accept()
    try:
        async for event in subscribe_run_events(run_id):
            await ws.send_json(event)
    except WebSocketDisconnect:
        return

