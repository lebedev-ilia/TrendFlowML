from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import yt_dlp

from fetcher.dataset_collector.cookies import CookieRotator, apply_cookiefile
from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.proxy import ProxyRotator
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow


class InstagramDiscoveryAdapter:
    """Minimal discover via yt-dlp search (site:instagram.com). Snapshots not supported."""

    platform = "instagram"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=False,
        comments=False,
        downloads=True,
    )

    def __init__(
        self,
        *,
        proxy_rotator: ProxyRotator | None = None,
        cookie_rotator: CookieRotator | None = None,
    ) -> None:
        self.proxy_rotator = proxy_rotator or ProxyRotator()
        self.cookie_rotator = cookie_rotator

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
        search_url = f"ytsearch{limit}:site:instagram.com {query}"
        ydl_opts = self._ydl_opts()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
        for entry in (info or {}).get("entries") or []:
            video_id = str(entry.get("id") or entry.get("display_id") or "")
            if not video_id:
                continue
            now = utcnow()
            snapshot = Snapshot(
                snapshot_index=0,
                time_get=format_time_get(now),
                collected_at=now,
                viewCount=str(entry.get("view_count") or 0),
                likeCount=str(entry.get("like_count") or 0),
                commentCount=str(entry.get("comment_count") or 0),
                raw=entry,
            )
            yield CollectedVideo(
                platform=self.platform,
                video_id=video_id,
                url=entry.get("webpage_url") or entry.get("url") or "",
                category=category,
                query=query,
                metadata={
                    "title": entry.get("title"),
                    "description": entry.get("description"),
                    "duration_seconds": entry.get("duration"),
                    "channel_id": entry.get("uploader_id") or entry.get("channel_id"),
                    "raw": entry,
                },
                snapshot_0=snapshot,
                time_interval=time_interval,
                discovered_at=now,
                platform_capabilities=self.capabilities.__dict__,
            )

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        raise NotImplementedError("Instagram snapshots are not supported by this adapter")

    def _ydl_opts(self) -> dict:
        opts = {"quiet": True, "skip_download": True}
        proxy = self.proxy_rotator.next()
        if proxy:
            opts["proxy"] = proxy
        return apply_cookiefile(opts, self.cookie_rotator)
