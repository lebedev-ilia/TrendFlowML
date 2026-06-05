from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from fetcher.dataset_collector.discovery.base import DiscoveryCapabilities
from fetcher.dataset_collector.schemas import CollectedVideo, Snapshot
from fetcher.dataset_collector.state import format_time_get, utcnow
from fetcher.platforms.dual_provider import fetch_with_fallback
from fetcher.platforms.platform_clients import (
    provider_mode_for,
    twitch_api_client,
    twitch_sdk_client,
)


class TwitchDiscoveryAdapter:
    platform = "twitch"
    capabilities = DiscoveryCapabilities(
        search=True,
        metadata=True,
        snapshots=True,
        comments=False,
        downloads=False,
    )

    def __init__(self, *, client_id: str | None = None, access_token: str | None = None) -> None:
        self._legacy_client_id = client_id
        self._legacy_access_token = access_token

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
        api = twitch_api_client()
        sdk = twitch_sdk_client()
        items = []

        if api is not None:
            try:
                items = api.discover_by_game(query, limit=limit)
            except Exception:
                items = []
        if not items and sdk is not None:
            try:
                items = sdk.discover_by_game(query, limit=limit)
            except Exception:
                items = []

        for dto in items[:limit]:
            now = utcnow()
            snapshot = Snapshot(
                snapshot_index=0,
                time_get=format_time_get(now),
                collected_at=now,
                viewCount=str(dto.view_count),
                raw=dto.raw_json,
            )
            yield CollectedVideo(
                platform=self.platform,
                video_id=dto.video_id,
                url=dto.webpage_url or "",
                category=category,
                query=query,
                metadata={
                    "title": dto.title,
                    "description": dto.description,
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
        api = twitch_api_client()
        sdk = twitch_sdk_client()

        def _api():
            assert api is not None
            return api.get_video_metadata(video_id)

        def _sdk():
            assert sdk is not None
            return sdk.get_video_metadata(video_id)

        dto = fetch_with_fallback(
            platform="twitch",
            mode=provider_mode_for("twitch"),
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
            raw=dto.raw_json,
        )
