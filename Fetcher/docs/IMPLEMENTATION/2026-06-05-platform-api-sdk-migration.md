# Platform API + SDK Migration (2026-06-05)

## Summary

Миграция парсинга TikTok, Instagram, RuTube и Twitch на схему «официальный API → SDK fallback» по образцу YouTube.

## Changes

### Infrastructure
- `ProviderMode` enum + `dual_provider.fetch_with_fallback()`
- `PlatformVideoDto` — каноническая модель полей
- `CredentialsStore` + `fetcher/credentials/` directory
- `fetcher_provider_fallback_total` metric

### Platforms
- **YouTube:** `youtube_provider_mode=api_first` с auto-fallback API→yt-dlp
- **TikTok:** Display API + TikTokApi SDK; убран ytsearch discovery
- **Instagram:** Graph API + Instaloader; новый `InstagramAdapter`
- **Twitch:** Helix API + twitchAPI SDK; новый `TwitchAdapter`
- **RuTube:** yt-dlp only; убран `rutube.ru/api`; новый `RutubeAdapter`

### Registry / Orchestrator
- Все 5 адаптеров в `platforms/registry.py`
- `normalize_source()` для instagram/rutube/twitch
- TikTok short links → TikTokApi resolve

### Dependencies
- `TikTokApi`, `playwright`, `instaloader`, `twitchAPI` в `requirements.txt`
- Optional: `requirements-platforms.txt`

## Docs
- `docs/DUAL_MODE_PROVIDERS.md`
- `docs/PLATFORM_CREDENTIALS.md`
- `fetcher/credentials/README.md`

## Tests
- `tests/unit/test_dual_provider.py`
- `tests/unit/test_credentials.py`
- `tests/unit/test_platform_video_dto.py`
