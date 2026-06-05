from __future__ import annotations

import time
from typing import Any

import httpx

from fetcher.schemas.platform_video import PlatformVideoDto, from_instagram_graph

GRAPH_BASE = "https://graph.facebook.com/v17.0"
MEDIA_FIELDS = "id,caption,media_type,media_url,thumbnail_url,timestamp,like_count,comments_count,permalink"


class InstagramAPIError(Exception):
    pass


class InstagramQuotaExceededError(InstagramAPIError):
    pass


class InstagramGraphClient:
    def __init__(self, *, access_token: str, ig_user_id: str, timeout: float = 15.0) -> None:
        self.access_token = access_token
        self.ig_user_id = ig_user_id
        self.client = httpx.Client(timeout=timeout)

    def get_media_metadata(self, media_id: str) -> PlatformVideoDto:
        data = self._get(f"/{media_id}", params={"fields": MEDIA_FIELDS})
        if data.get("media_type") not in (None, "VIDEO", "REELS"):
            pass
        return from_instagram_graph(data)

    def list_user_media(self, *, limit: int = 25) -> list[PlatformVideoDto]:
        data = self._get(
            f"/{self.ig_user_id}/media",
            params={"fields": MEDIA_FIELDS, "limit": min(limit, 50)},
        )
        return [from_instagram_graph(item) for item in data.get("data") or []]

    def discover_by_hashtag(self, hashtag: str, *, limit: int = 25) -> list[PlatformVideoDto]:
        tag = hashtag.lstrip("#").strip()
        if not tag:
            return []
        search = self._get("/ig_hashtag_search", params={"user_id": self.ig_user_id, "q": tag})
        items = search.get("data") or []
        if not items:
            return []
        hashtag_id = items[0].get("id")
        recent = self._get(
            f"/{hashtag_id}/recent_media",
            params={"user_id": self.ig_user_id, "fields": MEDIA_FIELDS, "limit": min(limit, 50)},
        )
        return [from_instagram_graph(item) for item in recent.get("data") or []]

    def _get(self, path: str, *, params: dict[str, Any] | None = None, max_retries: int = 3) -> dict:
        query = dict(params or {})
        query["access_token"] = self.access_token
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.client.get(f"{GRAPH_BASE}{path}", params=query)
                if resp.status_code == 429:
                    raise InstagramQuotaExceededError(resp.text[:300])
                if resp.status_code >= 500:
                    raise InstagramAPIError(f"Instagram server error {resp.status_code}")
                data = resp.json()
                if "error" in data:
                    err = data["error"]
                    code = err.get("code")
                    if code in (4, 17, 32, 613):
                        raise InstagramQuotaExceededError(str(err))
                    raise InstagramAPIError(str(err))
                return data
            except (InstagramQuotaExceededError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
        if last_exc:
            raise last_exc
        raise InstagramAPIError("Instagram request failed")


__all__ = [
    "InstagramAPIError",
    "InstagramGraphClient",
    "InstagramQuotaExceededError",
]
