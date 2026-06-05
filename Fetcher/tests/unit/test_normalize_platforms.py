from __future__ import annotations

import re


def test_instagram_shortcode_regex():
    url = "https://www.instagram.com/reel/ABC123xyz/"
    m = re.search(r"/(?:p|reel|reels)/([^/?#]+)", url)
    assert m and m.group(1) == "ABC123xyz"


def test_rutube_video_id_regex():
    url = "https://rutube.ru/video/abcdef0123456789/"
    m = re.search(r"/video/([a-f0-9]+)", url.lower())
    assert m and m.group(1) == "abcdef0123456789"


def test_twitch_video_id_regex():
    url = "https://www.twitch.tv/videos/1234567890"
    m = re.search(r"/videos/(\d+)", url.lower())
    assert m and m.group(1) == "1234567890"
