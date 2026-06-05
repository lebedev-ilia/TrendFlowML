from __future__ import annotations

import re
from typing import Optional

from fetcher.schemas.platform_video import PlatformVideoDto, from_instaloader_post


class InstagramSdkClient:
    def __init__(self, *, session_file: str | None = None, username: str | None = None) -> None:
        self.session_file = session_file
        self.username = username

    def _loader(self):
        import instaloader

        loader = instaloader.Instaloader(quiet=True, download_pictures=False, download_videos=False)
        if self.session_file:
            loader.load_session_from_file(self.username or "", self.session_file)
        return loader

    def get_post_metadata(self, shortcode_or_url: str) -> PlatformVideoDto:
        shortcode = self._extract_shortcode(shortcode_or_url)
        import instaloader

        loader = self._loader()
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        return from_instaloader_post(post)

    def discover_by_hashtag(self, hashtag: str, *, limit: int = 25) -> list[PlatformVideoDto]:
        tag = hashtag.lstrip("#").strip()
        import instaloader

        loader = self._loader()
        hashtag_obj = instaloader.Hashtag.from_name(loader.context, tag)
        results: list[PlatformVideoDto] = []
        for post in hashtag_obj.get_posts():
            if not post.is_video:
                continue
            results.append(from_instaloader_post(post))
            if len(results) >= limit:
                break
        return results

    def _extract_shortcode(self, value: str) -> str:
        if "/p/" in value or "/reel/" in value or "/reels/" in value:
            m = re.search(r"/(?:p|reel|reels)/([^/?#]+)", value)
            if m:
                return m.group(1)
        return value.strip("/")


__all__ = ["InstagramSdkClient"]
