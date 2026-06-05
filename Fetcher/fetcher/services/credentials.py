from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _mask_secret(value: str | None, *, visible: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}...{value[-visible:]}"


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _env_or_file(env_name: str, file_data: dict[str, Any], file_key: str) -> str | None:
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip() or None
    file_value = file_data.get(file_key)
    if file_value is None:
        return None
    text = str(file_value).strip()
    return text or None


@dataclass
class YouTubeCredentials:
    api_keys: list[str] = field(default_factory=list)

    def masked(self) -> dict[str, Any]:
        return {"api_keys": [_mask_secret(k) for k in self.api_keys]}


@dataclass
class TikTokCredentials:
    client_key: str | None = None
    client_secret: str | None = None
    access_token: str | None = None
    open_id: str | None = None
    ms_token: str | None = None

    @property
    def api_configured(self) -> bool:
        return bool(self.client_key and self.client_secret and self.access_token)

    @property
    def sdk_configured(self) -> bool:
        return bool(self.ms_token)

    def masked(self) -> dict[str, Any]:
        return {
            "client_key": _mask_secret(self.client_key),
            "client_secret": _mask_secret(self.client_secret),
            "access_token": _mask_secret(self.access_token),
            "open_id": self.open_id,
            "ms_token": _mask_secret(self.ms_token),
        }


@dataclass
class InstagramCredentials:
    access_token: str | None = None
    ig_user_id: str | None = None
    instaloader_session: str | None = None

    @property
    def api_configured(self) -> bool:
        return bool(self.access_token and self.ig_user_id)

    @property
    def sdk_configured(self) -> bool:
        return bool(self.instaloader_session or self.access_token)

    def masked(self) -> dict[str, Any]:
        return {
            "access_token": _mask_secret(self.access_token),
            "ig_user_id": self.ig_user_id,
            "instaloader_session": _mask_secret(self.instaloader_session),
        }


@dataclass
class TwitchCredentials:
    client_id: str | None = None
    client_secret: str | None = None
    access_token: str | None = None

    @property
    def api_configured(self) -> bool:
        return bool(self.client_id and self.access_token)

    @property
    def sdk_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def masked(self) -> dict[str, Any]:
        return {
            "client_id": _mask_secret(self.client_id),
            "client_secret": _mask_secret(self.client_secret),
            "access_token": _mask_secret(self.access_token),
        }


@dataclass
class RutubeCredentials:
    cookie_file: str | None = None

    def masked(self) -> dict[str, Any]:
        return {"cookie_file": self.cookie_file}


class CredentialsStore:
    """Загрузка credentials из env и JSON-файлов."""

    def __init__(
        self,
        *,
        credentials_dir: str | Path | None = None,
        youtube_keys_file: str | Path | None = None,
        tiktok_credentials_file: str | Path | None = None,
        instagram_credentials_file: str | Path | None = None,
        twitch_credentials_file: str | Path | None = None,
        rutube_credentials_file: str | Path | None = None,
    ) -> None:
        base = Path(credentials_dir or os.getenv("FETCHER_CREDENTIALS_DIR") or "fetcher/credentials")
        self.credentials_dir = base
        self._youtube_keys_file = Path(youtube_keys_file) if youtube_keys_file else base / "youtube_keys.txt"
        self._tiktok_file = Path(tiktok_credentials_file) if tiktok_credentials_file else base / "tiktok.json"
        self._instagram_file = (
            Path(instagram_credentials_file) if instagram_credentials_file else base / "instagram.json"
        )
        self._twitch_file = Path(twitch_credentials_file) if twitch_credentials_file else base / "twitch.json"
        self._rutube_file = Path(rutube_credentials_file) if rutube_credentials_file else base / "rutube.json"

    def youtube(self) -> YouTubeCredentials:
        keys: list[str] = []
        env_keys = os.getenv("FETCHER_YOUTUBE_DATA_API_KEYS") or os.getenv("FETCHER_YOUTUBE_DATA_API_KEY")
        if env_keys:
            keys.extend([k.strip() for k in env_keys.split(",") if k.strip()])
        single = os.getenv("FETCHER_YOUTUBE_DATA_API_KEY")
        if single and single.strip() not in keys:
            keys.append(single.strip())
        if self._youtube_keys_file.is_file():
            for line in self._youtube_keys_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.append(line)
        return YouTubeCredentials(api_keys=list(dict.fromkeys(keys)))

    def tiktok(self) -> TikTokCredentials:
        data = _read_json_file(self._tiktok_file)
        return TikTokCredentials(
            client_key=_env_or_file("FETCHER_TIKTOK_CLIENT_KEY", data, "client_key"),
            client_secret=_env_or_file("FETCHER_TIKTOK_CLIENT_SECRET", data, "client_secret"),
            access_token=_env_or_file("FETCHER_TIKTOK_ACCESS_TOKEN", data, "access_token"),
            open_id=_env_or_file("FETCHER_TIKTOK_OPEN_ID", data, "open_id"),
            ms_token=_env_or_file("FETCHER_TIKTOK_MS_TOKEN", data, "ms_token"),
        )

    def instagram(self) -> InstagramCredentials:
        data = _read_json_file(self._instagram_file)
        return InstagramCredentials(
            access_token=_env_or_file("FETCHER_INSTAGRAM_ACCESS_TOKEN", data, "access_token"),
            ig_user_id=_env_or_file("FETCHER_INSTAGRAM_IG_USER_ID", data, "ig_user_id"),
            instaloader_session=_env_or_file("FETCHER_INSTAGRAM_INSTALOADER_SESSION", data, "instaloader_session"),
        )

    def twitch(self) -> TwitchCredentials:
        data = _read_json_file(self._twitch_file)
        return TwitchCredentials(
            client_id=_env_or_file("FETCHER_TWITCH_CLIENT_ID", data, "client_id"),
            client_secret=_env_or_file("FETCHER_TWITCH_CLIENT_SECRET", data, "client_secret"),
            access_token=_env_or_file("FETCHER_TWITCH_ACCESS_TOKEN", data, "access_token"),
        )

    def rutube(self) -> RutubeCredentials:
        data = _read_json_file(self._rutube_file)
        cookie = _env_or_file("FETCHER_RUTUBE_COOKIE_FILE", data, "cookie_file")
        return RutubeCredentials(cookie_file=cookie)

    def masked_summary(self) -> dict[str, Any]:
        return {
            "credentials_dir": str(self.credentials_dir),
            "youtube": self.youtube().masked(),
            "tiktok": self.tiktok().masked(),
            "instagram": self.instagram().masked(),
            "twitch": self.twitch().masked(),
            "rutube": self.rutube().masked(),
        }


_default_store: Optional[CredentialsStore] = None


def get_credentials_store(**kwargs: Any) -> CredentialsStore:
    global _default_store
    if kwargs:
        return CredentialsStore(**kwargs)
    if _default_store is None:
        _default_store = CredentialsStore()
    return _default_store


__all__ = [
    "CredentialsStore",
    "InstagramCredentials",
    "RutubeCredentials",
    "TikTokCredentials",
    "TwitchCredentials",
    "YouTubeCredentials",
    "get_credentials_store",
]
