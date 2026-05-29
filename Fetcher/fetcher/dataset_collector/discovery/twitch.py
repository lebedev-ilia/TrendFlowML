from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import httpx

from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow


class TwitchDiscoveryAdapter:
    platform = "twitch"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=False,
        comments=False,
        downloads=False,
    )

    def __init__(self, *, client_id: str, access_token: str, timeout: float = 10.0) -> None:
        self.client_id = client_id
        self.access_token = access_token
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
        game_id = self._find_game_id(query)
        if not game_id:
            return
        response = self.client.get(
            "https://api.twitch.tv/helix/videos",
            headers=self._headers(),
            params={"game_id": game_id, "first": min(limit, 100), "sort": "views", "type": "archive"},
        )
        response.raise_for_status()
        videos = []
        for item in response.json().get("data") or []:
            now = utcnow()
            snapshot = Snapshot(
                snapshot_index=0,
                time_get=format_time_get(now),
                collected_at=now,
                viewCount=str(item.get("view_count") or 0),
                raw=item,
            )
            videos.append(
                CollectedVideo(
                    platform=self.platform,
                    video_id=item.get("id") or "",
                    url=item.get("url") or "",
                    category=category,
                    query=query,
                    metadata={
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "duration": item.get("duration"),
                        "publishedAt": item.get("published_at"),
                        "channel_id": item.get("user_id"),
                        "channelTitle": item.get("user_name"),
                        "raw": item,
                    },
                    snapshot_0=snapshot,
                    time_interval=time_interval,
                    discovered_at=now,
                    platform_capabilities=self.capabilities.__dict__,
                )
            )
        return videos

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        raise NotImplementedError("Twitch snapshots are not supported by this adapter")

    def _find_game_id(self, query: str) -> Optional[str]:
        response = self.client.get(
            "https://api.twitch.tv/helix/search/categories",
            headers=self._headers(),
            params={"query": query, "first": 1},
        )
        response.raise_for_status()
        data = response.json().get("data") or []
        return data[0].get("id") if data else None

    def _headers(self) -> dict[str, str]:
        return {
            "Client-Id": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
        }
