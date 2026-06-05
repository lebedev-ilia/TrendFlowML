# Platform Credentials

Храните API keys и tokens здесь или в переменных окружения `FETCHER_*`.

## Quick start

```bash
cp youtube_keys.txt.example youtube_keys.txt
cp tiktok.json.example tiktok.json
cp instagram.json.example instagram.json
cp twitch.json.example twitch.json
# Заполните значения, затем:
python scripts/check_platform_credentials.py
```

## Приоритет загрузки

1. Переменные окружения `FETCHER_<PLATFORM>_*`
2. JSON/txt файлы в этом каталоге
3. Пути из `dataset_campaign.json` (`*_credentials_file`)

## Файлы

| Файл | Платформа |
|------|-----------|
| `youtube_keys.txt` | YouTube Data API keys (по одному на строку) |
| `tiktok.json` | TikTok Display API + msToken для SDK |
| `instagram.json` | Graph API token + Instaloader session |
| `twitch.json` | Helix Client ID/Secret/Token |
| `rutube.json` | Опциональный cookie file для yt-dlp |

См. [docs/PLATFORM_CREDENTIALS.md](../../docs/PLATFORM_CREDENTIALS.md).
