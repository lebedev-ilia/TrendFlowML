from __future__ import annotations

from typing import Dict, Type

from fetcher.config import settings
from fetcher.platforms.base import PlatformAdapter


class PlatformDisabledError(RuntimeError):
    pass


class UnknownPlatformError(ValueError):
    pass


_ADAPTERS: Dict[str, Type[PlatformAdapter]] = {}


def _ensure_registered() -> None:
    """Lazy-register built-in adapters to avoid import side effects at startup/tests."""
    if _ADAPTERS:
        return

    # Import locally to keep module import lightweight.
    from fetcher.platforms.youtube import YouTubeAdapter

    _ADAPTERS["youtube"] = YouTubeAdapter

    try:
        from fetcher.platforms.tiktok import TikTokAdapter

        _ADAPTERS["tiktok"] = TikTokAdapter
    except Exception:
        # TikTok adapter may be absent in older versions; keep registry usable.
        pass


def get_adapter(platform: str) -> PlatformAdapter:
    """Return adapter instance for given platform.

    Enforces feature flags via `settings.enabled_platforms` and per-platform
    `*_enabled` flags when present.
    """
    if not platform:
        raise UnknownPlatformError("Platform is empty")

    p = platform.lower().strip()
    _ensure_registered()

    cls = _ADAPTERS.get(p)
    if cls is None:
        raise UnknownPlatformError(f"Unknown platform: {p}")

    if settings.enabled_platforms and p not in {x.lower() for x in settings.enabled_platforms}:
        raise PlatformDisabledError(f"Platform disabled: {p}")

    enabled_flag_name = f"{p}_enabled"
    if hasattr(settings, enabled_flag_name):
        if not bool(getattr(settings, enabled_flag_name)):
            raise PlatformDisabledError(f"Platform disabled by flag: {enabled_flag_name}=false")

    return cls()


__all__ = ["get_adapter", "PlatformDisabledError", "UnknownPlatformError"]

