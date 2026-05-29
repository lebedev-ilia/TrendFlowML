from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

import yt_dlp

from fetcher.dataset_collector.cookies import CookieRotator, apply_cookiefile
from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.proxy import ProxyRotator
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow


class TikTokDiscoveryAdapter:
    platform = "tiktok"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=True,
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
        search_url = f"ytsearch{limit}:site:tiktok.com {query}"
        ydl_opts = self._ydl_opts()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
        for entry in (info or {}).get("entries") or []:
            video_id = str(entry.get("id") or "")
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
                    "channel_id": entry.get("uploader_id"),
                    "raw": entry,
                },
                snapshot_0=snapshot,
                time_interval=time_interval,
                discovered_at=now,
                platform_capabilities=self.capabilities.__dict__,
            )

    def collect_snapshot(self, video_id: str, *, snapshot_index: int, comments_limit: int) -> Snapshot:
        url = video_id if video_id.startswith("http") else f"https://www.tiktok.com/@unknown/video/{video_id}"
        ydl_opts = self._ydl_opts()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        now = utcnow()
        return Snapshot(
            snapshot_index=snapshot_index,
            time_get=format_time_get(now),
            collected_at=now,
            viewCount=str((info or {}).get("view_count") or 0),
            likeCount=str((info or {}).get("like_count") or 0),
            commentCount=str((info or {}).get("comment_count") or 0),
            raw=info or {},
        )

    def _ydl_opts(self) -> dict:
        opts = {"quiet": True, "skip_download": True}
        proxy = self.proxy_rotator.next()
        if proxy:
            opts["proxy"] = proxy
        return apply_cookiefile(opts, self.cookie_rotator)
