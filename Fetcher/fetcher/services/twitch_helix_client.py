from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from fetcher.schemas.platform_video import PlatformVideoDto, from_twitch_helix

HELIX_BASE = "https://api.twitch.tv/helix"


class TwitchAPIError(Exception):
    pass


class TwitchQuotaExceededError(TwitchAPIError):
    pass


class TwitchHelixClient:
    def __init__(
        self,
        *,
        client_id: str,
        access_token: str,
        timeout: float = 15.0,
    ) -> None:
        self.client_id = client_id
        self.access_token = access_token
        self.client = httpx.Client(timeout=timeout)

    def get_video_metadata(self, video_id: str) -> PlatformVideoDto:
        data = self._get("/videos", params={"id": video_id})
        items = data.get("data") or []
        if not items:
            raise TwitchAPIError(f"Twitch video not found: {video_id}")
        return from_twitch_helix(items[0])

    def discover_by_game(self, query: str, *, limit: int = 20) -> list[PlatformVideoDto]:
        game_id = self._find_game_id(query)
        if not game_id:
            return []
        data = self._get(
            "/videos",
            params={
                "game_id": game_id,
                "first": min(limit, 100),
                "sort": "views",
                "type": "archive",
            },
        )
        return [from_twitch_helix(item) for item in data.get("data") or []]

    def _find_game_id(self, query: str) -> Optional[str]:
        data = self._get("/search/categories", params={"query": query, "first": 1})
        items = data.get("data") or []
        return items[0].get("id") if items else None

    def _headers(self) -> dict[str, str]:
        return {
            "Client-Id": self.client_id,
            "Authorization": f"Bearer {self.access_token}",
        }

    def _get(self, path: str, *, params: dict[str, Any] | None = None, max_retries: int = 3) -> dict:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = self.client.get(f"{HELIX_BASE}{path}", headers=self._headers(), params=params or {})
                if resp.status_code == 429:
                    raise TwitchQuotaExceededError(resp.text[:300])
                if resp.status_code >= 500:
                    raise TwitchAPIError(f"Twitch server error {resp.status_code}")
                if resp.status_code >= 400:
                    raise TwitchAPIError(resp.text[:500])
                return resp.json()
            except (TwitchQuotaExceededError, httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                raise
        if last_exc:
            raise last_exc
        raise TwitchAPIError("Twitch request failed")


__all__ = ["TwitchAPIError", "TwitchHelixClient", "TwitchQuotaExceededError"]
