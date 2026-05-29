from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yt_dlp
from yt_dlp.utils import DownloadError
from sqlalchemy.exc import IntegrityError

from fetcher.checksums import compute_sha256
from fetcher.circuit_breaker import get_circuit_breaker
from fetcher.config import settings
from fetcher.cookies import apply_cookiefile
from fetcher.db import session_scope
from fetcher.models import Artifact, ChannelMetadata, Video, VideoMetadata
from fetcher.platforms.base import PlatformAdapter
from fetcher.proxies import get_next_proxy, record_proxy_result
from fetcher.rate_limiter import acquire_token, acquire_video_lock, release_video_lock
from fetcher.storage import storage_client


class TikTokAdapter(PlatformAdapter):
    """TikTokAdapter for Fetcher.

    MVP goals:
    - same DB tables as YouTube (Video, VideoMetadata, ChannelMetadata, Artifact)
    - same required artifacts: meta.json, video.mp4, comments.json (even if empty)
    - storage layout: raw/tiktok/YYYY/MM/DD/<video_id>/{meta.json,video.mp4,comments.json}
    """

    platform: str = "tiktok"

    def fetch_metadata(self, source: str, *, run_id: str) -> None:
        breaker = get_circuit_breaker("metadata")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for metadata operations")

        if not acquire_token(
            key="rate:tiktok:metadata:default",
            limit=getattr(settings, "tiktok_metadata_limit_per_window", 400),
            window_sec=getattr(settings, "tiktok_metadata_window_sec", 3600),
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("TikTok metadata rate limit exceeded")

        proxy = get_next_proxy() if settings.enable_proxies else None
        ydl_opts = {"skip_download": True, "quiet": True, "proxy": proxy}
        apply_cookiefile(ydl_opts)

        start_time = time.time()
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
            latency_ms = int((time.time() - start_time) * 1000)
            if proxy:
                record_proxy_result(proxy, success=True, operation="metadata", latency_ms=latency_ms)
        except DownloadError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            msg = str(e)
            if proxy:
                record_proxy_result(
                    proxy,
                    success=False,
                    operation="metadata",
                    latency_ms=latency_ms,
                    error_message=msg[:500],
                )
            breaker.record_failure("download_error")
            raise

        platform_video_id = info.get("id")
        if not platform_video_id:
            breaker.record_failure("no_video_id")
            raise ValueError("yt-dlp did not return video id")

        breaker.record_success()

        duration = info.get("duration")

        # Upsert DB rows
        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .first()
            )
            if video is None:
                video = Video(platform=self.platform, platform_video_id=platform_video_id)
                db.add(video)
                try:
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    video = (
                        db.query(Video)
                        .filter(
                            Video.platform == self.platform,
                            Video.platform_video_id == platform_video_id,
                        )
                        .one()
                    )

            if isinstance(duration, int):
                video.duration_seconds = duration

            vm: Optional[VideoMetadata] = (
                db.query(VideoMetadata).filter(VideoMetadata.video_id == video.id).first()
            )
            if vm is None:
                vm = VideoMetadata(video_id=video.id)
                db.add(vm)

            vm.title = info.get("title")
            vm.description = info.get("description")
            vm.language = info.get("language")
            if isinstance(duration, int):
                vm.duration_seconds = duration
            # TikTok timestamps vary; keep raw_json if enabled.
            vm.raw_json = info if settings.retain_raw_meta else None

            cm: Optional[ChannelMetadata] = (
                db.query(ChannelMetadata).filter(ChannelMetadata.video_id == video.id).first()
            )
            if cm is None:
                cm = ChannelMetadata(video_id=video.id)
                db.add(cm)

            # Best-effort mapping: uploader_id/name if present.
            cm.channel_id = info.get("uploader_id") or info.get("channel_id") or cm.channel_id
            cm.channel_title = info.get("uploader") or info.get("channel") or cm.channel_title
            cm.raw_json = info.get("channel") if settings.retain_raw_meta else None
            db.flush()

        # Save meta.json to storage and register artifact
        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/{self.platform}/{date_prefix}/{platform_video_id}/meta.json"

        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_meta"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{platform_video_id}_meta.json"

        meta_to_save = info.copy() if settings.retain_raw_meta else {
            "id": info.get("id"),
            "title": info.get("title"),
            "duration": info.get("duration"),
            "uploader": info.get("uploader"),
            "uploader_id": info.get("uploader_id"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
        }
        tmp_path.write_text(json.dumps(meta_to_save, ensure_ascii=False), encoding="utf-8")
        size_bytes = tmp_path.stat().st_size
        checksum_hex = compute_sha256(tmp_path)
        checksum = f"sha256:{checksum_hex}"

        try:
            storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .one()
            )
            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(Artifact.video_id == video.id, Artifact.artifact_type == "metadata_file")
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="metadata_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()

    def download_video(self, source: str, *, run_id: str) -> None:
        if getattr(settings, "tiktok_mock_video_download", False):
            self._download_video_mock(source)
            return

        breaker = get_circuit_breaker("download")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for download operations")

        if not acquire_token(
            key="rate:tiktok:download:default",
            limit=getattr(settings, "tiktok_download_limit_per_window", 80),
            window_sec=getattr(settings, "tiktok_download_window_sec", 3600),
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("TikTok download rate limit exceeded")

        proxy = get_next_proxy() if settings.enable_proxies else None
        ydl_opts = {"skip_download": True, "quiet": True, "proxy": proxy}
        apply_cookiefile(ydl_opts)

        start_time = time.time()
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
            latency_ms = int((time.time() - start_time) * 1000)
            if proxy:
                record_proxy_result(proxy, success=True, operation="download", latency_ms=latency_ms)
        except DownloadError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            msg = str(e)
            if proxy:
                record_proxy_result(
                    proxy,
                    success=False,
                    operation="download",
                    latency_ms=latency_ms,
                    error_message=msg[:500],
                )
            breaker.record_failure("download_error")
            raise

        platform_video_id = info.get("id")
        if not platform_video_id:
            breaker.record_failure("no_video_id")
            raise ValueError("yt-dlp did not return video id")

        # Skip if artifact already exists
        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .first()
            )
            if video is None:
                video = Video(platform=self.platform, platform_video_id=platform_video_id)
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

        if not acquire_video_lock(self.platform, platform_video_id):
            return

        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_video"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_template = str(tmp_dir / f"{platform_video_id}.%(ext)s")

        proxy = get_next_proxy() if settings.enable_proxies else None
        ydl_download_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": tmp_template,
            "quiet": True,
            "merge_output_format": "mp4",
            "proxy": proxy,
        }
        apply_cookiefile(ydl_download_opts)

        download_start_time = time.time()
        try:
            with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
                info = ydl.extract_info(source, download=True)
                downloaded_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
            download_latency_ms = int((time.time() - download_start_time) * 1000)
            if proxy:
                record_proxy_result(proxy, success=True, operation="download", latency_ms=download_latency_ms)
        except DownloadError as e:
            download_latency_ms = int((time.time() - download_start_time) * 1000)
            msg = str(e)
            if proxy:
                record_proxy_result(
                    proxy,
                    success=False,
                    operation="download",
                    latency_ms=download_latency_ms,
                    error_message=msg[:500],
                )
            breaker.record_failure("download_error")
            release_video_lock(self.platform, platform_video_id)
            raise

        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/{self.platform}/{date_prefix}/{platform_video_id}/video.mp4"

        try:
            size_bytes = downloaded_path.stat().st_size
            checksum_hex = compute_sha256(downloaded_path)
            checksum = f"sha256:{checksum_hex}"
        except FileNotFoundError:
            size_bytes = None
            checksum = None

        try:
            storage_client.upload_file(downloaded_path, bucket=settings.bucket_raw, key=storage_key)
            breaker.record_success()
        finally:
            try:
                downloaded_path.unlink()
            except FileNotFoundError:
                pass
            release_video_lock(self.platform, platform_video_id)

        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .one()
            )
            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(Artifact.video_id == video.id, Artifact.artifact_type == "video_file")
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="video_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()

    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None:
        """Fetch comments for TikTok.

        MVP: write an (optionally empty) comments.json artifact to satisfy pipeline invariants.
        """
        breaker = get_circuit_breaker("comments")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for comments operations")

        # Rate limit slot even if we write empty comments (keeps behavior consistent).
        if not acquire_token(
            key="rate:tiktok:comments:default",
            limit=getattr(settings, "tiktok_comments_limit_per_window", 400),
            window_sec=getattr(settings, "tiktok_comments_window_sec", 3600),
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("TikTok comments rate limit exceeded")

        # Best-effort: try to resolve video id via yt-dlp metadata call (fast), fallback to source.
        platform_video_id = source
        try:
            proxy = get_next_proxy() if settings.enable_proxies else None
            ydl_opts = {"skip_download": True, "quiet": True, "proxy": proxy}
            apply_cookiefile(ydl_opts)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
            if isinstance(info, dict) and info.get("id"):
                platform_video_id = str(info["id"])
        except Exception:
            pass

        breaker.record_success()

        # Ensure Video exists (for artifact FK)
        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .one_or_none()
            )
            if video is None:
                video = Video(platform=self.platform, platform_video_id=platform_video_id)
                db.add(video)
                try:
                    db.flush()
                except IntegrityError:
                    db.rollback()
                    video = (
                        db.query(Video)
                        .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                        .one()
                    )

        persisted_comments: list[dict] = []

        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/{self.platform}/{date_prefix}/{platform_video_id}/comments.json"

        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_comments"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{platform_video_id}_comments.json"

        tmp_path.write_text(json.dumps(persisted_comments, ensure_ascii=False), encoding="utf-8")
        size_bytes = tmp_path.stat().st_size
        checksum_hex = compute_sha256(tmp_path)
        checksum = f"sha256:{checksum_hex}"

        try:
            storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .one()
            )
            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(Artifact.video_id == video.id, Artifact.artifact_type == "comments_file")
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="comments_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()

    def _download_video_mock(self, source: str) -> None:
        platform_video_id = source

        sample_dir = Path(
            getattr(settings, "tiktok_mock_sample_video_dir", None) or "/tmp/fetcher_sample_videos"
        )
        count = int(getattr(settings, "tiktok_mock_sample_video_count", 8) or 8)
        index = int(hashlib.sha256(platform_video_id.encode("utf-8")).hexdigest(), 16) % max(count, 1)
        sample_path = sample_dir / f"sample_{index}.mp4"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample video not found for mock download: {sample_path}")

        today = datetime.now(timezone.utc)
        date_prefix = today.strftime("%Y/%m/%d")
        storage_key = f"raw/{self.platform}/{date_prefix}/{platform_video_id}/video.mp4"

        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_video_mock"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{platform_video_id}.mp4"
        tmp_path.write_bytes(sample_path.read_bytes())

        try:
            size_bytes = tmp_path.stat().st_size
            checksum_hex = compute_sha256(tmp_path)
            checksum = f"sha256:{checksum_hex}"
        except FileNotFoundError:
            size_bytes = None
            checksum = None

        try:
            storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        with session_scope() as db:
            video: Optional[Video] = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .one_or_none()
            )
            if video is None:
                video = Video(platform=self.platform, platform_video_id=platform_video_id)
                db.add(video)
                db.flush()

            artifact: Optional[Artifact] = (
                db.query(Artifact)
                .filter(Artifact.video_id == video.id, Artifact.artifact_type == "video_file")
                .order_by(Artifact.created_at.desc())
                .first()
            )
            if artifact is None:
                artifact = Artifact(
                    video_id=video.id,
                    artifact_type="video_file",
                    storage_path=storage_key,
                    status="COMPLETED",
                    size_bytes=size_bytes,
                    checksum=checksum,
                )
                db.add(artifact)
            else:
                artifact.storage_path = storage_key
                artifact.status = "COMPLETED"
                artifact.size_bytes = size_bytes
                artifact.checksum = checksum
            db.flush()


__all__ = ["TikTokAdapter"]

