# Dual-Mode Providers (Official API + SDK)

Fetcher использует единый паттерн **API-first с автоматическим fallback на SDK** для всех платформ.

## Режимы (`ProviderMode`)

| Режим | Поведение |
|-------|-----------|
| `api_first` (default) | Сначала официальный API; при 429/quota/5xx → SDK |
| `api_only` | Только API, без fallback |
| `sdk_only` | Только SDK (yt-dlp, TikTokApi, Instaloader, …) |
| `parallel` | Оба источника; SDK дополняет пустые поля API |

Настраивается per-platform: `FETCHER_<PLATFORM>_PROVIDER_MODE`.

## Архитектура

```
PlatformAdapter.fetch_metadata()
  → dual_provider.fetch_with_fallback(api_fn, sdk_fn, mode)
  → PlatformVideoDto (канонические поля)
  → adapter_utils.persist_metadata()
```

Ключевые модули:

- [`fetcher/platforms/provider_mode.py`](../fetcher/platforms/provider_mode.py)
- [`fetcher/platforms/dual_provider.py`](../fetcher/platforms/dual_provider.py)
- [`fetcher/schemas/platform_video.py`](../fetcher/schemas/platform_video.py)
- [`fetcher/platforms/platform_clients.py`](../fetcher/platforms/platform_clients.py)

## Платформы

| Платформа | API (primary) | SDK (fallback) |
|-----------|---------------|----------------|
| YouTube | YouTube Data API v3 | yt-dlp |
| TikTok | TikTok Display API | TikTokApi |
| Instagram | Instagram Graph API | Instaloader |
| Twitch | Helix REST API | twitchAPI |
| RuTube | — (нет official API) | yt-dlp |

## Метрики

`fetcher_provider_fallback_total{platform, from_provider, to_provider}` — счётчик fallback API→SDK.

## См. также

- [PLATFORM_CREDENTIALS.md](PLATFORM_CREDENTIALS.md)
- [PLATFORM_ADAPTERS.md](PLATFORM_ADAPTERS.md)
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
