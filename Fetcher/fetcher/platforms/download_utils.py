from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yt_dlp
from yt_dlp.utils import DownloadError

from fetcher.checksums import compute_sha256
from fetcher.circuit_breaker import get_circuit_breaker
from fetcher.config import settings
from fetcher.cookies import apply_cookiefile
from fetcher.db import session_scope
from fetcher.models import Artifact, Video
from fetcher.proxies import get_next_proxy, record_proxy_result
from fetcher.rate_limiter import acquire_video_lock, release_video_lock
from fetcher.storage import storage_client


def download_video_ytdlp(
    *,
    platform: str,
    source: str,
    platform_video_id: str,
    rate_key: str,
    rate_limit: int,
    rate_window_sec: int,
) -> None:
    breaker = get_circuit_breaker("download")
    if breaker.is_open():
        raise RuntimeError("Circuit breaker is OPEN for download operations")

    from fetcher.rate_limiter import acquire_token

    if not acquire_token(key=rate_key, limit=rate_limit, window_sec=rate_window_sec):
        breaker.record_failure("rate_limit_exceeded")
        raise RuntimeError(f"{platform} download rate limit exceeded")

    with session_scope() as db:
        video: Optional[Video] = (
            db.query(Video)
            .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
            .first()
        )
        if video is None:
            video = Video(platform=platform, platform_video_id=platform_video_id)
            db.add(video)
            db.flush()
        existing: Optional[Artifact] = (
            db.query(Artifact)
            .filter(
                Artifact.video_id == video.id,
                Artifact.artifact_type == "video_file",
                Artifact.status == "COMPLETED",
            )
            .first()
        )
        if existing is not None:
            return

    if not acquire_video_lock(platform, platform_video_id):
        return

    tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_video"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_template = str(tmp_dir / f"{platform_video_id}.%(ext)s")
    proxy = get_next_proxy() if settings.enable_proxies else None
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": tmp_template,
        "quiet": True,
        "merge_output_format": "mp4",
        "proxy": proxy,
    }
    apply_cookiefile(ydl_opts)

    start = time.time()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=True)
            downloaded_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if proxy:
            record_proxy_result(proxy, success=True, operation="download", latency_ms=int((time.time() - start) * 1000))
    except DownloadError as e:
        if proxy:
            record_proxy_result(
                proxy,
                success=False,
                operation="download",
                latency_ms=int((time.time() - start) * 1000),
                error_message=str(e)[:500],
            )
        breaker.record_failure("download_error")
        release_video_lock(platform, platform_video_id)
        raise

    today = datetime.now(timezone.utc)
    storage_key = f"raw/{platform}/{today.strftime('%Y/%m/%d')}/{platform_video_id}/video.mp4"
    try:
        size_bytes = downloaded_path.stat().st_size
        checksum = f"sha256:{compute_sha256(downloaded_path)}"
        storage_client.upload_file(downloaded_path, bucket=settings.bucket_raw, key=storage_key)
        breaker.record_success()
    finally:
        try:
            downloaded_path.unlink()
        except FileNotFoundError:
            pass
        release_video_lock(platform, platform_video_id)

    with session_scope() as db:
        video = (
            db.query(Video)
            .filter(Video.platform == platform, Video.platform_video_id == platform_video_id)
            .one()
        )
        artifact: Optional[Artifact] = (
            db.query(Artifact)
            .filter(Artifact.video_id == video.id, Artifact.artifact_type == "video_file")
            .order_by(Artifact.created_at.desc())
            .first()
        )
        if artifact is None:
            db.add(
                Artifact(
                    video_id=video.id,
                    artifact_type="video_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
            )
        else:
            artifact.storage_path = storage_key
            artifact.status = "COMPLETED"
            artifact.size_bytes = size_bytes
            artifact.checksum = checksum
        db.flush()


__all__ = ["download_video_ytdlp"]
