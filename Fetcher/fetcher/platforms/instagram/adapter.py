from __future__ import annotations

import re

from fetcher.circuit_breaker import get_circuit_breaker
from fetcher.config import settings
from fetcher.platforms.adapter_utils import persist_empty_comments, persist_metadata
from fetcher.platforms.base import PlatformAdapter
from fetcher.platforms.download_utils import download_video_ytdlp
from fetcher.platforms.dual_provider import fetch_with_fallback
from fetcher.platforms.platform_clients import (
    instagram_api_client,
    instagram_sdk_client,
    provider_mode_for,
)
from fetcher.rate_limiter import acquire_token


class InstagramAdapter(PlatformAdapter):
    platform: str = "instagram"

    def fetch_metadata(self, source: str, *, run_id: str) -> None:
        breaker = get_circuit_breaker("metadata")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for metadata operations")
        if not acquire_token(
            key="rate:instagram:metadata:default",
            limit=settings.instagram_metadata_limit_per_window,
            window_sec=settings.instagram_metadata_window_sec,
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("Instagram metadata rate limit exceeded")

        api = instagram_api_client()
        sdk = instagram_sdk_client()
        shortcode = self._shortcode(source)

        def _api():
            client = api
            assert client is not None
            return client.get_media_metadata(shortcode)

        def _sdk():
            client = sdk
            assert client is not None
            return client.get_post_metadata(shortcode)

        try:
            dto = fetch_with_fallback(
                platform="instagram",
                mode=provider_mode_for("instagram"),
                api_fn=_api,
                sdk_fn=_sdk,
                api_available=api is not None,
                sdk_available=sdk is not None,
            )
            persist_metadata(platform=self.platform, dto=dto, run_id=run_id)
            breaker.record_success()
        except Exception:
            breaker.record_failure("metadata_fetch_error")
            raise

    def download_video(self, source: str, *, run_id: str) -> None:
        api = instagram_api_client()
        sdk = instagram_sdk_client()
        shortcode = self._shortcode(source)

        def _api():
            assert api is not None
            return api.get_media_metadata(shortcode)

        def _sdk():
            assert sdk is not None
            return sdk.get_post_metadata(shortcode)

        dto = fetch_with_fallback(
            platform="instagram",
            mode=provider_mode_for("instagram"),
            api_fn=_api,
            sdk_fn=_sdk,
            api_available=api is not None,
            sdk_available=sdk is not None,
        )
        url = dto.media_url or dto.webpage_url or source
        download_video_ytdlp(
            platform=self.platform,
            source=url,
            platform_video_id=dto.video_id,
            rate_key="rate:instagram:download:default",
            rate_limit=settings.instagram_metadata_limit_per_window,
            rate_window_sec=settings.instagram_metadata_window_sec,
        )

    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None:
        breaker = get_circuit_breaker("comments")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for comments operations")
        shortcode = self._shortcode(source)
        persist_empty_comments(platform=self.platform, platform_video_id=shortcode)
        breaker.record_success()

    @staticmethod
    def _shortcode(source: str) -> str:
        if "/p/" in source or "/reel/" in source or "/reels/" in source:
            m = re.search(r"/(?:p|reel|reels)/([^/?#]+)", source)
            if m:
                return m.group(1)
        return source.strip("/")


__all__ = ["InstagramAdapter"]
