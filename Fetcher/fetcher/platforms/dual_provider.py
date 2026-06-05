from __future__ import annotations

import logging
from typing import Callable, Optional, TypeVar

from fetcher.metrics import fetcher_provider_fallback_total
from fetcher.platforms.provider_mode import ProviderMode
from fetcher.schemas.platform_video import PlatformVideoDto

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that trigger SDK fallback in api_first mode.
FALLBACK_EXCEPTION_NAMES = frozenset(
    {
        "QuotaExceededError",
        "TikTokQuotaExceededError",
        "InstagramQuotaExceededError",
        "TwitchQuotaExceededError",
        "YouTubeAPIError",
        "TikTokAPIError",
        "InstagramAPIError",
        "TwitchAPIError",
    }
)


def is_fallback_eligible(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in FALLBACK_EXCEPTION_NAMES:
        return True
    text = str(exc).lower()
    if any(marker in text for marker in ("429", "quota", "rate limit", "too many requests")):
        return True
    if any(marker in text for marker in ("500", "502", "503", "504", "timeout", "temporarily unavailable")):
        return True
    return False


def fetch_with_fallback(
    *,
    platform: str,
    mode: ProviderMode,
    api_fn: Callable[[], PlatformVideoDto],
    sdk_fn: Callable[[], PlatformVideoDto],
    api_available: bool = True,
    sdk_available: bool = True,
) -> PlatformVideoDto:
    """Получить метаданные с учётом режима провайдера."""

    if mode == ProviderMode.SDK_ONLY:
        if not sdk_available:
            raise RuntimeError(f"{platform}: SDK provider not configured")
        return sdk_fn()

    if mode == ProviderMode.API_ONLY:
        if not api_available:
            raise RuntimeError(f"{platform}: API provider not configured")
        return api_fn()

    if mode == ProviderMode.PARALLEL:
        api_result: Optional[PlatformVideoDto] = None
        sdk_result: Optional[PlatformVideoDto] = None
        if api_available:
            try:
                api_result = api_fn()
            except Exception as exc:
                logger.warning("%s API failed in parallel mode: %s", platform, exc)
        if sdk_available:
            try:
                sdk_result = sdk_fn()
            except Exception as exc:
                logger.warning("%s SDK failed in parallel mode: %s", platform, exc)
        if api_result and sdk_result:
            return api_result.merge_from(sdk_result)
        if api_result:
            return api_result
        if sdk_result:
            return sdk_result
        raise RuntimeError(f"{platform}: both API and SDK providers failed")

    # API_FIRST (default)
    if api_available:
        try:
            return api_fn()
        except Exception as exc:
            if sdk_available and is_fallback_eligible(exc):
                logger.info(
                    "%s API failed (%s), falling back to SDK",
                    platform,
                    type(exc).__name__,
                )
                fetcher_provider_fallback_total.labels(
                    platform=platform,
                    from_provider="api",
                    to_provider="sdk",
                ).inc()
                return sdk_fn()
            raise

    if sdk_available:
        return sdk_fn()

    raise RuntimeError(f"{platform}: no provider available")


def fetch_comments_with_fallback(
    *,
    platform: str,
    mode: ProviderMode,
    api_fn: Callable[[], list],
    sdk_fn: Callable[[], list],
    api_available: bool = True,
    sdk_available: bool = True,
) -> list:
    """Аналог fetch_with_fallback для списков комментариев."""

    if mode == ProviderMode.SDK_ONLY:
        return sdk_fn() if sdk_available else []
    if mode == ProviderMode.API_ONLY:
        return api_fn() if api_available else []

    if mode == ProviderMode.PARALLEL:
        api_items: list = []
        sdk_items: list = []
        if api_available:
            try:
                api_items = api_fn()
            except Exception:
                pass
        if sdk_available:
            try:
                sdk_items = sdk_fn()
            except Exception:
                pass
        if api_items:
            return api_items
        return sdk_items

    if api_available:
        try:
            return api_fn()
        except Exception as exc:
            if sdk_available and is_fallback_eligible(exc):
                fetcher_provider_fallback_total.labels(
                    platform=platform,
                    from_provider="api",
                    to_provider="sdk",
                ).inc()
                return sdk_fn()
            raise
    if sdk_available:
        return sdk_fn()
    return []


__all__ = ["fetch_comments_with_fallback", "fetch_with_fallback", "is_fallback_eligible"]
