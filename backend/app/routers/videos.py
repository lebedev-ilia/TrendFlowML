from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_current_user, get_db
from ..models import Upload, User, UserVideoLink, Video, VideoFile, VideoSource
from ..schemas import UploadInitOut, VideoOut
from ..services.storage import move_upload_to_storage, probe_video, sha256_file


router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post("/upload/init", response_model=UploadInitOut)
def upload_init(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video_id = str(uuid.uuid4())
    video = Video(platform_id="upload", video_id=video_id, source_type="upload")
    db.add(video)
    db.flush()

    upload = Upload(user_id=user.id, video_id=video.id, status="init")
    db.add(upload)
    db.flush()

    return UploadInitOut(upload_id=upload.id, video_id=video_id)


@router.put("/upload/{upload_id}")
def upload_file(
    upload_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    upload = db.query(Upload).filter(Upload.id == upload_id, Upload.user_id == user.id).first()
    if not upload:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")
    if upload.status not in {"init", "uploaded"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Upload in invalid state")

    settings = Settings()
    paths = settings.resolve_paths()
    paths.raw_uploads_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = paths.raw_uploads_dir / "tmp" / upload_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / (file.filename or "video.mp4")

    with open(temp_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    upload.status = "uploaded"
    upload.temp_path = str(temp_path)
    upload.filename = file.filename or "video.mp4"
    db.flush()
    return {"status": "ok"}


@router.post("/upload/complete", response_model=VideoOut)
def upload_complete(
    upload_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    upload = db.query(Upload).filter(Upload.id == upload_id, Upload.user_id == user.id).first()
    if not upload or not upload.temp_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload not found")
    if upload.status != "uploaded":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Upload not ready")

    video = db.query(Video).filter(Video.id == upload.video_id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    final_path, _ = move_upload_to_storage(upload.temp_path, video.video_id, upload.filename)
    meta = probe_video(final_path)
    file_sha = sha256_file(final_path)
    file_row = db.query(VideoFile).filter(VideoFile.sha256_hex == file_sha).first()
    if not file_row:
        file_row = VideoFile(
            sha256_hex=file_sha,
            size_bytes=Path(final_path).stat().st_size,
            mime_type=None,
            object_key=final_path,
        )
        db.add(file_row)
        db.flush()

    source = VideoSource(
        video_id=video.id,
        uploaded_file_id=file_row.id,
        duration_sec=meta.duration_sec,
        width=meta.width,
        height=meta.height,
    )
    db.add(source)

    db.add(UserVideoLink(user_id=user.id, video_id=video.id))
    upload.status = "completed"
    db.flush()

    return VideoOut(
        id=video.id,
        platform_id=video.platform_id,
        video_id=video.video_id,
        source_type=video.source_type,
        title=video.title,
        created_at=video.created_at,
    )


@router.get("", response_model=List[VideoOut])
def list_videos(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = (
        db.query(Video)
        .join(UserVideoLink, UserVideoLink.video_id == Video.id)
        .filter(UserVideoLink.user_id == user.id)
        .order_by(Video.created_at.desc())
    )
    return [
        VideoOut(
            id=v.id,
            platform_id=v.platform_id,
            video_id=v.video_id,
            source_type=v.source_type,
            title=v.title,
            created_at=v.created_at,
        )
        for v in q
    ]

