from __future__ import annotations

from typing import Any, Optional

import yt_dlp

from fetcher.cookies import apply_cookiefile
from fetcher.proxies import get_next_proxy
from fetcher.config import settings
from fetcher.schemas.platform_video import PlatformVideoDto, from_ytdlp


class RutubeYtdlpClient:
    def __init__(self, *, proxy: str | None = None) -> None:
        self.proxy = proxy

    def get_video_metadata(self, source: str) -> PlatformVideoDto:
        info = self._extract_info(source)
        return from_ytdlp(info)

    def discover_by_query(self, query: str, *, limit: int = 20) -> list[PlatformVideoDto]:
        search_url = f"ytsearch{limit}:site:rutube.ru {query}"
        info = self._extract_info(search_url)
        entries = (info or {}).get("entries") or []
        return [from_ytdlp(entry) for entry in entries if entry.get("id")]

    def _extract_info(self, source: str) -> dict[str, Any]:
        proxy = self.proxy
        if proxy is None and settings.enable_proxies:
            proxy = get_next_proxy()
        opts: dict[str, Any] = {"quiet": True, "skip_download": True, "proxy": proxy}
        apply_cookiefile(opts)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
        if not isinstance(info, dict):
            raise ValueError("yt-dlp returned invalid info")
        return info


__all__ = ["RutubeYtdlpClient"]
