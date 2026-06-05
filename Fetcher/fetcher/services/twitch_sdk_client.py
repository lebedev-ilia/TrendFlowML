from __future__ import annotations

import asyncio
from typing import Optional

from fetcher.schemas.platform_video import PlatformVideoDto, from_twitch_helix


class TwitchSdkClient:
    """Обёртка twitchAPI с sync-интерфейсом."""

    def __init__(self, *, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret

    def get_video_metadata(self, video_id: str) -> PlatformVideoDto:
        item = self._run(self._fetch_video(video_id))
        return from_twitch_helix(item)

    def discover_by_game(self, query: str, *, limit: int = 20) -> list[PlatformVideoDto]:
        items = self._run(self._fetch_by_game(query, limit))
        return [from_twitch_helix(i) for i in items]

    async def _fetch_video(self, video_id: str) -> dict:
        from twitchAPI.twitch import Twitch

        twitch = await Twitch(self.client_id, self.client_secret)
        async for video in twitch.get_videos(ids=[video_id]):
            return {
                "id": video.id,
                "user_id": video.user_id,
                "user_name": video.user_name,
                "title": video.title,
                "description": video.description,
                "created_at": video.created_at.isoformat() if video.created_at else None,
                "published_at": video.published_at.isoformat() if video.published_at else None,
                "url": video.url,
                "view_count": video.view_count,
                "duration": video.duration,
                "thumbnail_url": video.thumbnail_url,
            }
        raise RuntimeError(f"Twitch video not found: {video_id}")

    async def _fetch_by_game(self, query: str, limit: int) -> list[dict]:
        from twitchAPI.twitch import Twitch

        twitch = await Twitch(self.client_id, self.client_secret)
        game_id = None
        async for cat in twitch.search_categories(query=query, first=1):
            game_id = cat.id
            break
        if not game_id:
            return []
        results = []
        async for video in twitch.get_videos(game_id=game_id, first=min(limit, 100), sort="views", video_type="archive"):
            results.append(
                {
                    "id": video.id,
                    "user_id": video.user_id,
                    "user_name": video.user_name,
                    "title": video.title,
                    "description": video.description,
                    "published_at": video.published_at.isoformat() if video.published_at else None,
                    "url": video.url,
                    "view_count": video.view_count,
                    "duration": video.duration,
                    "thumbnail_url": video.thumbnail_url,
                }
            )
        return results

    def _run(self, coro):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)


__all__ = ["TwitchSdkClient"]
