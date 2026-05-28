import pytest

from fetcher.orchestrator import normalize_source


@pytest.mark.unit
class TestTikTokNormalizeSource:
    def test_tiktok_user_video_url_parsed(self):
        platform, vid = normalize_source("https://www.tiktok.com/@someuser/video/7351234567890123456")
        assert platform == "tiktok"
        assert vid == "7351234567890123456"

    def test_tiktok_short_url_without_yt_dlp_fails(self):
        with pytest.raises(ValueError, match="Failed to parse TikTok video id"):
            normalize_source("https://www.tiktok.com/t/ZTRfakeCode/")

