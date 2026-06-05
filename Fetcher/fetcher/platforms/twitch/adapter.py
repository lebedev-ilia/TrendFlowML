from __future__ import annotations

from fetcher.circuit_breaker import get_circuit_breaker
from fetcher.config import settings
from fetcher.platforms.adapter_utils import persist_empty_comments, persist_metadata
from fetcher.platforms.base import PlatformAdapter
from fetcher.platforms.download_utils import download_video_ytdlp
from fetcher.platforms.dual_provider import fetch_with_fallback
from fetcher.platforms.platform_clients import (
    provider_mode_for,
    twitch_api_client,
    twitch_sdk_client,
)
from fetcher.rate_limiter import acquire_token


class TwitchAdapter(PlatformAdapter):
    platform: str = "twitch"

    def fetch_metadata(self, source: str, *, run_id: str) -> None:
        breaker = get_circuit_breaker("metadata")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for metadata operations")
        if not acquire_token(
            key="rate:twitch:metadata:default",
            limit=settings.twitch_metadata_limit_per_window,
            window_sec=settings.twitch_metadata_window_sec,
        ):
            breaker.record_failure("rate_limit_exceeded")
            raise RuntimeError("Twitch metadata rate limit exceeded")

        video_id = self._video_id(source)
        api = twitch_api_client()
        sdk = twitch_sdk_client()

        def _api():
            client = api
            assert client is not None
            return client.get_video_metadata(video_id)

        def _sdk():
            client = sdk
            assert client is not None
            return client.get_video_metadata(video_id)

        try:
            dto = fetch_with_fallback(
                platform="twitch",
                mode=provider_mode_for("twitch"),
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
        video_id = self._video_id(source)
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
        url = dto.webpage_url or f"https://www.twitch.tv/videos/{video_id}"
        download_video_ytdlp(
            platform=self.platform,
            source=url,
            platform_video_id=video_id,
            rate_key="rate:twitch:download:default",
            rate_limit=settings.twitch_metadata_limit_per_window,
            rate_window_sec=settings.twitch_metadata_window_sec,
        )

    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None:
        breaker = get_circuit_breaker("comments")
        if breaker.is_open():
            raise RuntimeError("Circuit breaker is OPEN for comments operations")
        video_id = self._video_id(source)
        persist_empty_comments(platform=self.platform, platform_video_id=video_id)
        breaker.record_success()

    @staticmethod
    def _video_id(source: str) -> str:
        if source.isdigit():
            return source
        if "/videos/" in source:
            return source.rstrip("/").split("/videos/")[-1].split("?")[0]
        return source


__all__ = ["TwitchAdapter"]
