from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import httpx

from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow


class RutubeDiscoveryAdapter:
    platform = "rutube"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=True,
        comments=False,
        downloads=False,
    )

    def __init__(self, *, timeout: float = 10.0) -> None:
        self.client = httpx.Client(timeout=timeout)

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
        response = self.client.get(
            "https://rutube.ru/api/search/video/",
            params={"query": query, "page_size": min(limit, 100)},
        )
        response.raise_for_status()
        for item in response.json().get("results") or []:
            yield self._record_from_item(
                item,
                category=category,
                query=query,
                snapshot_index=0,
                time_interval=time_interval,
            )

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        response = self.client.get(f"https://rutube.ru/api/video/{video_id}/")
        response.raise_for_status()
        item = response.json()
        now = utcnow()
        return Snapshot(
            snapshot_index=snapshot_index,
            time_get=format_time_get(now),
            collected_at=now,
            viewCount=str(item.get("hits") or item.get("view_count") or 0),
            likeCount=str(item.get("likes") or 0),
            commentCount=str(item.get("comments_count") or 0),
            raw=item,
        )

    def _record_from_item(
        self,
        item: dict,
        *,
        category: str,
        query: str,
        snapshot_index: int,
        time_interval: Optional[str] = None,
    ) -> CollectedVideo:
        now = utcnow()
        video_id = str(item.get("id") or "")
        snapshot = Snapshot(
            snapshot_index=snapshot_index,
            time_get=format_time_get(now),
            collected_at=now,
            viewCount=str(item.get("hits") or item.get("view_count") or 0),
            likeCount=str(item.get("likes") or 0),
            commentCount=str(item.get("comments_count") or 0),
            raw=item,
        )
        return CollectedVideo(
            platform=self.platform,
            video_id=video_id,
            url=item.get("video_url") or item.get("html") or f"https://rutube.ru/video/{video_id}/",
            category=category,
            query=query,
            metadata={
                "title": item.get("title"),
                "description": item.get("description"),
                "duration_seconds": item.get("duration"),
                "publishedAt": item.get("created_ts"),
                "channel_id": (item.get("author") or {}).get("id"),
                "channelTitle": (item.get("author") or {}).get("name"),
                "raw": item,
            },
            snapshot_0=snapshot,
            time_interval=time_interval,
            discovered_at=now,
            platform_capabilities=self.capabilities.__dict__,
        )
