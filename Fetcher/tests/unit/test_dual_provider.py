from __future__ import annotations

import pytest

from fetcher.platforms.dual_provider import fetch_with_fallback, is_fallback_eligible
from fetcher.platforms.provider_mode import ProviderMode
from fetcher.schemas.platform_video import PlatformVideoDto
from fetcher.services.youtube_data_client import QuotaExceededError


def _dto(vid: str, provider: str) -> PlatformVideoDto:
    return PlatformVideoDto(video_id=vid, title=provider, source_provider=provider)


def test_api_first_falls_back_on_quota():
    calls = {"api": 0, "sdk": 0}

    def api():
        calls["api"] += 1
        raise QuotaExceededError("quota")

    def sdk():
        calls["sdk"] += 1
        return _dto("1", "sdk")

    result = fetch_with_fallback(
        platform="youtube",
        mode=ProviderMode.API_FIRST,
        api_fn=api,
        sdk_fn=sdk,
    )
    assert result.video_id == "1"
    assert calls == {"api": 1, "sdk": 1}


def test_api_only_raises_without_fallback():
    with pytest.raises(QuotaExceededError):
        fetch_with_fallback(
            platform="youtube",
            mode=ProviderMode.API_ONLY,
            api_fn=lambda: (_ for _ in ()).throw(QuotaExceededError("quota")),
            sdk_fn=lambda: _dto("1", "sdk"),
        )


def test_parallel_merges_fields():
    result = fetch_with_fallback(
        platform="instagram",
        mode=ProviderMode.PARALLEL,
        api_fn=lambda: PlatformVideoDto(video_id="1", title="api", source_provider="api"),
        sdk_fn=lambda: PlatformVideoDto(
            video_id="1",
            title="",
            thumbnail_url="http://thumb",
            source_provider="sdk",
        ),
    )
    assert result.title == "api"
    assert result.thumbnail_url == "http://thumb"
    assert result.source_provider == "merged"


def test_is_fallback_eligible_429():
    assert is_fallback_eligible(RuntimeError("HTTP 429 Too Many Requests"))
