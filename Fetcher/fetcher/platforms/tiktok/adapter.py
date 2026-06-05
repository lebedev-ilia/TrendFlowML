from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fetcher.circuit_breaker import get_circuit_breaker
from fetcher.config import settings
from fetcher.platforms.adapter_utils import persist_empty_comments, persist_metadata
from fetcher.platforms.base import PlatformAdapter
from fetcher.platforms.download_utils import download_video_ytdlp
from fetcher.platforms.dual_provider import fetch_with_fallback
from fetcher.platforms.platform_clients import (
    provider_mode_for,
    tiktok_api_client,
    tiktok_sdk_client,
)
from fetcher.proxies import get_next_proxy
from fetcher.rate_limiter import acquire_token
from fetcher.storage import storage_client
from fetcher.db import session_scope
from fetcher.models import Artifact, Video
from fetcher.checksums import compute_sha256
from datetime import datetime, timezone


class TikTokAdapter(PlatformAdapter):
    platform: str = "tiktok"

    def fetch_metadata(self, source: str, *, run_id: str) -> None:
        breaker = get_circuit_breaker("metadata")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for metadata operations")

        if not acquire_token(
            key="rate:tiktok:metadata:default",
            limit=settings.tiktok_metadata_limit_per_window,
            window_sec=settings.tiktok_metadata_window_sec,
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("TikTok metadata rate limit exceeded")

        api = tiktok_api_client()
        proxy = get_next_proxy() if settings.enable_proxies else None
        sdk = tiktok_sdk_client(proxy=proxy)
        video_id = source if source.isdigit() else source

        def _api():
            client = api
            assert client is not None
            vid = video_id
            if not vid.isdigit() and sdk is not None:
                vid = sdk.get_video_metadata(source).video_id
            return client.get_video_metadata(vid)

        def _sdk():
            client = sdk
            assert client is not None
            return client.get_video_metadata(source)

        try:
            dto = fetch_with_fallback(
                platform="tiktok",
                mode=provider_mode_for("tiktok"),
                api_fn=_api,
                sdk_fn=_sdk,
                api_available=api is not None,
                sdk_available=sdk is not None,
            )
            persist_metadata(platform=self.platform, dto=dto, run_id=run_id)
            breaker.record_success()
        except Exception:
            breaker.record_failure("metadata_fetch_error")
            raise

    def download_video(self, source: str, *, run_id: str) -> None:
        if settings.tiktok_mock_video_download:
            self._download_video_mock(source)
            return
        video_id = source if source.isdigit() else source
        try:
            sdk = tiktok_sdk_client(proxy=get_next_proxy() if settings.enable_proxies else None)
            if sdk:
                video_id = sdk.get_video_metadata(source).video_id
        except Exception:
            pass
        url = source if source.startswith("http") else f"https://www.tiktok.com/@unknown/video/{video_id}"
        download_video_ytdlp(
            platform=self.platform,
            source=url,
            platform_video_id=video_id,
            rate_key="rate:tiktok:download:default",
            rate_limit=settings.tiktok_download_limit_per_window,
            rate_window_sec=settings.tiktok_download_window_sec,
        )

    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None:
        breaker = get_circuit_breaker("comments")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for comments operations")
        if not acquire_token(
            key="rate:tiktok:comments:default",
            limit=settings.tiktok_comments_limit_per_window,
            window_sec=settings.tiktok_comments_window_sec,
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("TikTok comments rate limit exceeded")
        platform_video_id = source if source.isdigit() else source
        try:
            sdk = tiktok_sdk_client()
            if sdk:
                platform_video_id = sdk.get_video_metadata(source).video_id
        except Exception:
            pass
        persist_empty_comments(platform=self.platform, platform_video_id=platform_video_id)
        breaker.record_success()

    def _download_video_mock(self, source: str) -> None:
        platform_video_id = source
        sample_dir = Path(settings.tiktok_mock_sample_video_dir or "/tmp/fetcher_sample_videos")
        count = int(settings.tiktok_mock_sample_video_count or 8)
        index = int(hashlib.sha256(platform_video_id.encode()).hexdigest(), 16) % max(count, 1)
        sample_path = sample_dir / f"sample_{index}.mp4"
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample video not found: {sample_path}")

        today = datetime.now(timezone.utc)
        storage_key = f"raw/{self.platform}/{today.strftime('%Y/%m/%d')}/{platform_video_id}/video.mp4"
        tmp_dir = Path(os.getenv("FETCHER_TMP_DIR", "/tmp/fetcher_video_mock"))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{platform_video_id}.mp4"
        tmp_path.write_bytes(sample_path.read_bytes())
        size_bytes = tmp_path.stat().st_size
        checksum = f"sha256:{compute_sha256(tmp_path)}"
        try:
            storage_client.upload_file(tmp_path, bucket=settings.bucket_raw, key=storage_key)
        finally:
            tmp_path.unlink(missing_ok=True)

        with session_scope() as db:
            video = (
                db.query(Video)
                .filter(Video.platform == self.platform, Video.platform_video_id == platform_video_id)
                .one_or_none()
            )
            if video is None:
                video = Video(platform=self.platform, platform_video_id=platform_video_id)
                db.add(video)
                db.flush()
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
            db.flush()


__all__ = ["TikTokAdapter"]
