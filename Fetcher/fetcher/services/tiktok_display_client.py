from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from fetcher.schemas.platform_video import PlatformVideoDto, from_tiktok_api

BASE_URL = "https://open.tiktokapis.com"


class TikTokAPIError(Exception):
    pass


class TikTokQuotaExceededError(TikTokAPIError):
    pass


class TikTokDisplayClient:
    def __init__(
        self,
        *,
        access_token: str,
        open_id: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.access_token = access_token
        self.open_id = open_id
        self.client = httpx.Client(timeout=timeout)

    def get_video_metadata(self, video_id: str) -> PlatformVideoDto:
        response = self._request(
            "POST",
            "/v2/video/query/",
            json_body={
                "filters": {"video_ids": [video_id]},
                "fields": [
                    "id",
                    "create_time",
                    "video_description",
                    "duration",
                    "like_count",
                    "comment_count",
                    "share_count",
                    "view_count",
                    "cover_image_url",
                    "share_url",
                ],
            },
        )
        videos = (response.get("data") or {}).get("videos") or []
        if not videos:
            raise TikTokAPIError(f"TikTok video not found: {video_id}")
        return from_tiktok_api(videos[0])

    def list_user_videos(self, *, count: int = 20, cursor: int = 0) -> list[PlatformVideoDto]:
        if not self.open_id:
            raise TikTokAPIError("open_id required for list_user_videos")
        response = self._request(
            "POST",
            "/v2/video/list/",
            json_body={"max_count": min(count, 20), "cursor": cursor},
            params={"open_id": self.open_id},
        )
        videos = (response.get("data") or {}).get("videos") or []
        return [from_tiktok_api(v) for v in videos]

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.client.request(
                    method,
                    f"{BASE_URL}{path}",
                    headers=headers,
                    json=json_body,
                    params=params,
                )
                if resp.status_code == 429:
                    raise TikTokQuotaExceededError(resp.text[:300])
                if resp.status_code >= 500:
                    raise TikTokAPIError(f"TikTok server error {resp.status_code}")
                if resp.status_code >= 400:
                    raise TikTokAPIError(resp.text[:500])
                data = resp.json()
                error = data.get("error") or {}
                if error.get("code") and str(error.get("code")) not in ("ok", "0"):
                    code = str(error.get("code"))
                    if "rate" in code.lower() or "limit" in code.lower():
                        raise TikTokQuotaExceededError(str(error))
                    raise TikTokAPIError(str(error))
                return data
            except (TikTokQuotaExceededError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
        if last_exc:
            raise last_exc
        raise TikTokAPIError("TikTok request failed")


__all__ = [
    "TikTokAPIError",
    "TikTokDisplayClient",
    "TikTokQuotaExceededError",
]
