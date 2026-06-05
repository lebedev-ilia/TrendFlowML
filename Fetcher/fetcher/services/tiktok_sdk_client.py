from __future__ import annotations

import asyncio
from typing import Iterable, Optional

from fetcher.schemas.platform_video import PlatformVideoDto, from_tiktok_sdk


class TikTokSdkClient:
    """Обёртка TikTokApi (Playwright) с sync-интерфейсом."""

    def __init__(self, *, ms_token: str, proxy: str | None = None) -> None:
        self.ms_token = ms_token
        self.proxy = proxy

    def get_video_metadata(self, video_id_or_url: str) -> PlatformVideoDto:
        url = video_id_or_url
        if video_id_or_url.isdigit():
            url = f"https://www.tiktok.com/@unknown/video/{video_id_or_url}"
        video = self._run(self._fetch_video(url))
        return from_tiktok_sdk(video)

    def resolve_video_id(self, url: str) -> str:
        video = self._run(self._fetch_video(url))
        return str(getattr(video, "id", "") or "")

    def discover_trending(self, *, count: int = 20) -> list[PlatformVideoDto]:
        videos = self._run(self._iter_trending(count))
        return [from_tiktok_sdk(v) for v in videos]

    async def _fetch_video(self, url: str):
        from TikTokApi import TikTokApi

        async with TikTokApi() as api:
            await api.create_sessions(
                ms_tokens=[self.ms_token],
                num_sessions=1,
                sleep_after=2,
                proxies=[self.proxy] if self.proxy else None,
            )
            return api.video(url=url)

    async def _iter_trending(self, count: int) -> list:
        from TikTokApi import TikTokApi

        items = []
        async with TikTokApi() as api:
            await api.create_sessions(
                ms_tokens=[self.ms_token],
                num_sessions=1,
                sleep_after=2,
                proxies=[self.proxy] if self.proxy else None,
            )
            async for video in api.trending.videos(count=count):
                items.append(video)
        return items

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


__all__ = ["TikTokSdkClient"]
