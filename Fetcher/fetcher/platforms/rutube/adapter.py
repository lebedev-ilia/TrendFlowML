from __future__ import annotations

from fetcher.circuit_breaker import get_circuit_breaker
from fetcher.config import settings
from fetcher.platforms.adapter_utils import persist_empty_comments, persist_metadata
from fetcher.platforms.base import PlatformAdapter
from fetcher.platforms.download_utils import download_video_ytdlp
from fetcher.platforms.platform_clients import provider_mode_for, rutube_sdk_client
from fetcher.proxies import get_next_proxy
from fetcher.rate_limiter import acquire_token


class RutubeAdapter(PlatformAdapter):
    platform: str = "rutube"

    def fetch_metadata(self, source: str, *, run_id: str) -> None:
        breaker = get_circuit_breaker("metadata")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for metadata operations")
        if not acquire_token(
            key="rate:rutube:metadata:default",
            limit=settings.rutube_metadata_limit_per_window,
            window_sec=settings.rutube_metadata_window_sec,
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("RuTube metadata rate limit exceeded")

        url = self._normalize_url(source)
        client = rutube_sdk_client(proxy=get_next_proxy() if settings.enable_proxies else None)
        try:
            dto = client.get_video_metadata(url)
            persist_metadata(platform=self.platform, dto=dto, run_id=run_id)
            breaker.record_success()
        except Exception:
            breaker.record_failure("metadata_fetch_error")
            raise

    def download_video(self, source: str, *, run_id: str) -> None:
        url = self._normalize_url(source)
        client = rutube_sdk_client(proxy=get_next_proxy() if settings.enable_proxies else None)
        dto = client.get_video_metadata(url)
        download_video_ytdlp(
            platform=self.platform,
            source=url,
            platform_video_id=dto.video_id,
            rate_key="rate:rutube:download:default",
            rate_limit=settings.rutube_download_limit_per_window,
            rate_window_sec=settings.rutube_download_window_sec,
        )

    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None:
        breaker = get_circuit_breaker("comments")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for comments operations")
        url = self._normalize_url(source)
        client = rutube_sdk_client()
        dto = client.get_video_metadata(url)
        persist_empty_comments(platform=self.platform, platform_video_id=dto.video_id)
        breaker.record_success()

    @staticmethod
    def _normalize_url(source: str) -> str:
        if source.startswith("http"):
            return source
        return f"https://rutube.ru/video/{source}/"


__all__ = ["RutubeAdapter"]
