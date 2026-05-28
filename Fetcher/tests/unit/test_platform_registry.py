import pytest

from fetcher.platforms.registry import (
    PlatformDisabledError,
    UnknownPlatformError,
    get_adapter,
)
from fetcher.platforms.tiktok import TikTokAdapter


@pytest.mark.unit
class TestPlatformRegistry:
    def test_get_adapter_unknown_platform(self):
        with pytest.raises(UnknownPlatformError):
            get_adapter("unknown_platform_xyz")

    def test_get_adapter_tiktok_disabled_by_enabled_platforms(self, monkeypatch):
        import fetcher.platforms.registry as registry

        monkeypatch.setattr(registry.settings, "enabled_platforms", ["youtube"])
        monkeypatch.setattr(registry.settings, "tiktok_enabled", True)
        with pytest.raises(PlatformDisabledError, match="Platform disabled"):
            get_adapter("tiktok")

    def test_get_adapter_tiktok_disabled_by_flag(self, monkeypatch):
        import fetcher.platforms.registry as registry

        monkeypatch.setattr(registry.settings, "enabled_platforms", ["youtube", "tiktok"])
        monkeypatch.setattr(registry.settings, "tiktok_enabled", False)
        with pytest.raises(PlatformDisabledError, match="disabled"):
            get_adapter("tiktok")

    def test_get_adapter_tiktok_enabled(self, monkeypatch):
        import fetcher.platforms.registry as registry

        monkeypatch.setattr(registry.settings, "enabled_platforms", ["youtube", "tiktok"])
        monkeypatch.setattr(registry.settings, "tiktok_enabled", True)
        adapter = get_adapter("tiktok")
        assert isinstance(adapter, TikTokAdapter)

