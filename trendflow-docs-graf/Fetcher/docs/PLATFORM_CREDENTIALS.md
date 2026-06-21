# Настройка credentials для платформ

Все ключи и токены настраиваются **без правок кода** — через env или файлы в `fetcher/credentials/`.

## Быстрый старт

```bash
cd Fetcher
cp fetcher/credentials/youtube_keys.txt.example fetcher/credentials/youtube_keys.txt
cp fetcher/credentials/tiktok.json.example fetcher/credentials/tiktok.json
# заполните значения
python scripts/check_platform_credentials.py
```

## YouTube

| Переменная | Описание |
|------------|----------|
| `FETCHER_YOUTUBE_DATA_API_KEY` | Один API key |
| `FETCHER_YOUTUBE_DATA_API_KEYS` | Несколько ключей через запятую |
| `FETCHER_YOUTUBE_DATA_ENABLED` | `true` для включения API |
| `FETCHER_YOUTUBE_PROVIDER_MODE` | `api_first` (default) |

Файл: `fetcher/credentials/youtube_keys.txt` (один ключ на строку).

## TikTok

| Переменная | Описание |
|------------|----------|
| `FETCHER_TIKTOK_CLIENT_KEY` | OAuth Client Key |
| `FETCHER_TIKTOK_CLIENT_SECRET` | OAuth Client Secret |
| `FETCHER_TIKTOK_ACCESS_TOKEN` | User/Client Access Token |
| `FETCHER_TIKTOK_OPEN_ID` | open_id пользователя |
| `FETCHER_TIKTOK_MS_TOKEN` | msToken cookie для TikTokApi SDK |
| `FETCHER_TIKTOK_DATA_ENABLED` | `true` для Display API |
| `FETCHER_TIKTOK_PROVIDER_MODE` | `api_first` (default) |

Файл: `fetcher/credentials/tiktok.json`

## Instagram

| Переменная | Описание |
|------------|----------|
| `FETCHER_INSTAGRAM_ACCESS_TOKEN` | Long-lived Graph API token |
| `FETCHER_INSTAGRAM_IG_USER_ID` | Business/Creator IG user id |
| `FETCHER_INSTAGRAM_INSTALOADER_SESSION` | Путь к session-файлу Instaloader |
| `FETCHER_INSTAGRAM_DATA_ENABLED` | `true` для Graph API |
| `FETCHER_INSTAGRAM_PROVIDER_MODE` | `api_first` (default) |

Файл: `fetcher/credentials/instagram.json`

**Ограничение:** Graph API требует Business/Creator аккаунт. Discovery — hashtag-based.

## Twitch

| Переменная | Описание |
|------------|----------|
| `FETCHER_TWITCH_CLIENT_ID` | Application Client-ID |
| `FETCHER_TWITCH_CLIENT_SECRET` | Client Secret |
| `FETCHER_TWITCH_ACCESS_TOKEN` | App/User OAuth token |
| `FETCHER_TWITCH_DATA_ENABLED` | `true` для Helix API |
| `FETCHER_TWITCH_PROVIDER_MODE` | `api_first` (default) |

Файл: `fetcher/credentials/twitch.json`

## RuTube

Официального API нет. Используется только yt-dlp SDK.

| Переменная | Описание |
|------------|----------|
| `FETCHER_RUTUBE_PROVIDER_MODE` | `sdk_only` (default) |
| `FETCHER_RUTUBE_COOKIE_FILE` | Опциональный Netscape cookie file |

## Dataset Collector

В `dataset_campaign.json` можно указать пути:

```json
{
  "credentials_dir": "fetcher/credentials",
  "youtube_keys_file": "fetcher/credentials/youtube_keys.txt",
  "tiktok_credentials_file": "fetcher/credentials/tiktok.json",
  "instagram_credentials_file": "fetcher/credentials/instagram.json",
  "twitch_credentials_file": "fetcher/credentials/twitch.json"
}
```

## Валидация

```bash
python scripts/check_platform_credentials.py --credentials-dir fetcher/credentials
python scripts/check_platform_credentials.py --json
```
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
