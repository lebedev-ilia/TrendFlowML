from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow
from fetcher.platforms.dual_provider import fetch_with_fallback
from fetcher.platforms.platform_clients import (
    provider_mode_for,
    tiktok_api_client,
    tiktok_sdk_client,
)
from fetcher.proxies import get_next_proxy
from fetcher.config import settings


class TikTokDiscoveryAdapter:
    platform = "tiktok"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=True,
        comments=False,
        downloads=True,
    )

    def discover(
        self,
        *,
        category: str,
        query: str,
        limit: int,
        published_after: Optional[datetime] = None,
        published_before: Optional[datetime] = None,
        time_interval: Optional[str] = None,
    ) -> Iterable[CollectedVideo]:
        api = tiktok_api_client()
        proxy = get_next_proxy() if settings.enable_proxies else None
        sdk = tiktok_sdk_client(proxy=proxy)
        mode = provider_mode_for("tiktok")

        items = []
        if api is not None:
            try:
                items = api.list_user_videos(count=limit)
            except Exception:
                items = []

        if not items and sdk is not None:
            try:
                items = sdk.discover_trending(count=limit)
            except Exception:
                items = []

        for dto in items[:limit]:
            now = utcnow()
            snapshot = Snapshot(
                snapshot_index=0,
                time_get=format_time_get(now),
                collected_at=now,
                viewCount=str(dto.view_count),
                likeCount=str(dto.like_count),
                commentCount=str(dto.comment_count),
                raw=dto.raw_json,
            )
            yield CollectedVideo(
                platform=self.platform,
                video_id=dto.video_id,
                url=dto.webpage_url or f"https://www.tiktok.com/@unknown/video/{dto.video_id}",
                category=category,
                query=query,
                metadata={
                    "title": dto.title,
                    "description": dto.description,
                    "duration_seconds": dto.duration_seconds,
                    "channel_id": dto.channel_id,
                    "source_provider": dto.source_provider,
                    "raw": dto.raw_json,
                },
                snapshot_0=snapshot,
                time_interval=time_interval,
                discovered_at=now,
                platform_capabilities=self.capabilities.__dict__,
            )

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        api = tiktok_api_client()
        sdk = tiktok_sdk_client(proxy=get_next_proxy() if settings.enable_proxies else None)

        def _api():
            assert api is not None
            return api.get_video_metadata(video_id)

        def _sdk():
            assert sdk is not None
            return sdk.get_video_metadata(video_id)

        dto = fetch_with_fallback(
            platform="tiktok",
            mode=provider_mode_for("tiktok"),
            api_fn=_api,
            sdk_fn=_sdk,
            api_available=api is not None,
            sdk_available=sdk is not None,
        )
        now = utcnow()
        return Snapshot(
            snapshot_index=snapshot_index,
            time_get=format_time_get(now),
            collected_at=now,
            viewCount=str(dto.view_count),
            likeCount=str(dto.like_count),
            commentCount=str(dto.comment_count),
            raw=dto.raw_json,
        )
