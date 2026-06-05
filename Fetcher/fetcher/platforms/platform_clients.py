from __future__ import annotations

from typing import Optional

from fetcher.config import settings
from fetcher.platforms.provider_mode import ProviderMode
from fetcher.services.credentials import CredentialsStore, get_credentials_store
from fetcher.services.instagram_graph_client import InstagramGraphClient
from fetcher.services.instagram_sdk_client import InstagramSdkClient
from fetcher.services.rutube_ytdlp_client import RutubeYtdlpClient
from fetcher.services.tiktok_display_client import TikTokDisplayClient
from fetcher.services.tiktok_sdk_client import TikTokSdkClient
from fetcher.services.twitch_helix_client import TwitchHelixClient
from fetcher.services.twitch_sdk_client import TwitchSdkClient


def provider_mode_for(platform: str) -> ProviderMode:
    attr = f"{platform}_provider_mode"
    return ProviderMode.from_value(getattr(settings, attr, None))


def _store() -> CredentialsStore:
    return get_credentials_store(credentials_dir=settings.credentials_dir)


def tiktok_api_client() -> Optional[TikTokDisplayClient]:
    creds = _store().tiktok()
    token = settings.tiktok_access_token or creds.access_token
    open_id = settings.tiktok_open_id or creds.open_id
    if not settings.tiktok_data_enabled or not token:
        return None
    return TikTokDisplayClient(access_token=token, open_id=open_id)


def tiktok_sdk_client(*, proxy: str | None = None) -> Optional[TikTokSdkClient]:
    creds = _store().tiktok()
    ms_token = settings.tiktok_ms_token or creds.ms_token
    if not ms_token:
        return None
    return TikTokSdkClient(ms_token=ms_token, proxy=proxy)


def instagram_api_client() -> Optional[InstagramGraphClient]:
    creds = _store().instagram()
    token = settings.instagram_access_token or creds.access_token
    ig_user_id = settings.instagram_ig_user_id or creds.ig_user_id
    if not settings.instagram_data_enabled or not token or not ig_user_id:
        return None
    return InstagramGraphClient(access_token=token, ig_user_id=ig_user_id)


def instagram_sdk_client() -> Optional[InstagramSdkClient]:
    creds = _store().instagram()
    session = settings.instagram_instaloader_session or creds.instaloader_session
    if not session and not (settings.instagram_access_token or creds.access_token):
        return None
    return InstagramSdkClient(session_file=session)


def twitch_api_client() -> Optional[TwitchHelixClient]:
    creds = _store().twitch()
    client_id = settings.twitch_client_id or creds.client_id
    token = settings.twitch_access_token or creds.access_token
    if not settings.twitch_data_enabled or not client_id or not token:
        return None
    return TwitchHelixClient(client_id=client_id, access_token=token)


def twitch_sdk_client() -> Optional[TwitchSdkClient]:
    creds = _store().twitch()
    client_id = settings.twitch_client_id or creds.client_id
    secret = settings.twitch_client_secret or creds.client_secret
    if not client_id or not secret:
        return None
    return TwitchSdkClient(client_id=client_id, client_secret=secret)


def rutube_sdk_client(*, proxy: str | None = None) -> RutubeYtdlpClient:
    return RutubeYtdlpClient(proxy=proxy)


__all__ = [
    "instagram_api_client",
    "instagram_sdk_client",
    "provider_mode_for",
    "rutube_sdk_client",
    "tiktok_api_client",
    "tiktok_sdk_client",
    "twitch_api_client",
    "twitch_sdk_client",
]
