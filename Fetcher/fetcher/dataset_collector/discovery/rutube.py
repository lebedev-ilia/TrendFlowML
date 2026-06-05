from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow
from fetcher.platforms.platform_clients import rutube_sdk_client
from fetcher.config import settings
from fetcher.proxies import get_next_proxy


class RutubeDiscoveryAdapter:
    platform = "rutube"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=True,
        comments=False,
        downloads=False,
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
        client = rutube_sdk_client(proxy=get_next_proxy() if settings.enable_proxies else None)
        for dto in client.discover_by_query(query, limit=limit):
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
                url=dto.webpage_url or f"https://rutube.ru/video/{dto.video_id}/",
                category=category,
                query=query,
                metadata={
                    "title": dto.title,
                    "description": dto.description,
                    "duration_seconds": dto.duration_seconds,
                    "channel_id": dto.channel_id,
                    "channelTitle": dto.channel_title,
                    "source_provider": dto.source_provider,
                    "raw": dto.raw_json,
                },
                snapshot_0=snapshot,
                time_interval=time_interval,
                discovered_at=now,
                platform_capabilities=self.capabilities.__dict__,
            )

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        client = rutube_sdk_client(proxy=get_next_proxy() if settings.enable_proxies else None)
        url = video_id if video_id.startswith("http") else f"https://rutube.ru/video/{video_id}/"
        dto = client.get_video_metadata(url)
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
