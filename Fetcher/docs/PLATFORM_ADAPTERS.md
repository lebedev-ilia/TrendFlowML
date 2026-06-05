## Platform adapters для Fetcher

Этот документ описывает целевой дизайн **адаптеров платформ** (YouTube, TikTok, Instagram, …) для Fetcher.

Цели:

- инкапсулировать особенности конкретных платформ;
- предоставить Orchestrator’у единый интерфейс для операций `fetch_metadata`, `download_video`, `fetch_comments`;
- обеспечить идемпотентность и корректную обработку ошибок.

Основан на `Fetcher/docs/plan.md`, раздел 2.3.

---

## 1. Базовый интерфейс `PlatformAdapter`

Логический интерфейс (Python‑уровень, без привязки к конкретной реализации):

```python
class PlatformAdapter:
    def fetch_metadata(self, source: str, *, run_id: str) -> None: ...
    def download_video(self, source: str, *, run_id: str) -> None: ...
    def fetch_comments(self, source: str, *, run_id: str, limit: int = 100) -> None: ...
```

Где:

- `source`:
  - на ранних шагах — исходный URL;
  - после `NORMALIZE_SOURCE` — нормализованный `platform_video_id` (рекомендуемый вариант для production).
- `run_id`:
  - обязателен для логирования, событий и связи с БД/артефактами;
  - должен использоваться во всех логах и событиях как ключевой идентификатор.
- `limit`:
  - верхняя граница числа комментариев (по умолчанию ≤100, см. чеклист Phase 1).

Реализации адаптеров не должны напрямую знать о Backend или DataProcessor — только о Fetcher (БД, storage, события).

---

## 2. Общие инварианты адаптеров

Все реализации `PlatformAdapter` обязаны соблюдать:

### 2.1. Идемпотентность

- Повторный вызов методов для того же `(platform, platform_video_id)` **не должен** создавать дубликаты:
  - `fetch_metadata`:
    - при наличии актуальной записи в `video_metadata` / `channel_metadata` может ничего не делать;
  - `download_video`:
    - при наличии `artifacts` с `artifact_type='video_file'` и `status='COMPLETED'` не скачивает файл повторно;
  - `fetch_comments`:
    - при наличии достаточного числа комментариев в `comments` (или соответствующего `comments.json`) может пропустить работу.

### 2.2. Ошибки и retry

- Ошибки делятся на:
  - **retryable** (сетевые, timeouts, rate‑limit 429, временные 5xx);
  - **non‑retryable** (`VIDEO_NOT_FOUND`, `PRIVATE_VIDEO`, `AGE_RESTRICTED` без необходимого конфига).
- Каждая реализация адаптера должна маппить ошибки в нормализованные коды:
  - например: `YOUTUBE_429`, `YOUTUBE_VIDEO_NOT_FOUND`, `YOUTUBE_AGE_RESTRICTED`, `DOWNLOAD_TIMEOUT`.
- Эти коды используются:
  - в событиях (`job.failed`, `run.status_changed` — поле `error_code`);
  - в логах и, при необходимости, в агрегированных полях БД.

### 2.3. Лимиты и ресурсы

- Адаптеры обязаны уважать:
  - конфигурируемые лимиты времени (`timeout_seconds` на операцию);
  - лимиты попыток (`max_retries` для retryable‑ошибок);
  - Redis‑rate‑лимитер и proxy‑pool (будут реализованы в Phase 0/1, см. чеклист `Rate limiting & locking`).

---

## 3. Dual-Mode Providers

Все адаптеры поддерживают режимы `api_first`, `api_only`, `sdk_only`, `parallel`.
См. [DUAL_MODE_PROVIDERS.md](DUAL_MODE_PROVIDERS.md) и [PLATFORM_CREDENTIALS.md](PLATFORM_CREDENTIALS.md).

| Платформа | API | SDK fallback |
|-----------|-----|--------------|
| YouTube | Data API v3 | yt-dlp |
| TikTok | Display API | TikTokApi |
| Instagram | Graph API | Instaloader |
| Twitch | Helix | twitchAPI |
| RuTube | — | yt-dlp |

---

## 4. YouTubeAdapter (первая реализация)

Первая целевая реализация — `YouTubeAdapter(PlatformAdapter)`.

### 3.1. Инструменты

- `yt-dlp` для:
  - получения метаданных (`--dump-json`, `--skip-download`);
  - скачивания видео (`-f "bestvideo[height<=720]+bestaudio/best"`, с ограничениями качества).
- (опционально) YouTube Data API для дополнительных полей и/или обхода ограничений.
- Proxy‑пул и cookies для age‑restricted / geo‑restricted контента.

### 3.2. Метаданные (fetch_metadata)

Задачи:

- вызвать `yt-dlp` (и/или API) и собрать:
  - `title`, `description`, `tags`, `language`, `duration_seconds`, `published_at`;
  - данные о канале: `channel_id`, `channel_title`, `subscriber_count`, `video_count`, `view_count_channel`;
  - `thumbnails`, `formats`, `captions`, `chapters`, `age_limit`, `country`, и др.;
- записать:
  - `video_metadata` / `channel_metadata` в БД Fetcher;
  - опционально `meta.json` в `video-analytics-raw` (см. `STORAGE_LAYOUT.md`) + запись в `artifacts`.

### 3.3. Видео (download_video)

Задачи:

- скачать видео с ограничением качества (например, ≤720p) и нормализовать контейнер/кодеки через `ffmpeg`;
- pipeline:
  - download → `/tmp` → upload в `video-analytics-raw` → удаление `/tmp` файла;
- записать:
  - `artifacts` (`artifact_type='video_file'`, `storage_path`, `size_bytes`, `checksum`, `status`);
  - события (`job.started`, `job.finished` / `job.failed`).

Особые требования:

- использовать **distributed lock** перед скачиванием (см. чеклист `Rate limiting & locking`);
- уважать rate‑лимит и proxy‑пул, чтобы не ловить массовые 429/403.

### 3.4. Комментарии (fetch_comments)

Задачи:

- собрать top‑N комментариев (по умолчанию ≤100):
  - текст, лайки, число ответов, автор, дата, `raw_json`;
- хранение:
  - либо в таблице `comments` (`raw_json` + агрегированные поля);
  - либо в `comments.json` в `video-analytics-raw` + ссылка в `artifacts`.

Требования:

- поддержка ограничений:
  - лимит по числу комментариев;
  - лимит по времени (timeout);
  - опционально — лимит по страницам.
- базовая PII‑фильтрация (email/phone/url) закладывается как следующий этап (Phase 1/3).

---

## 4. Включение/выключение платформ (feature‑флаги)

### 4.1. Конфигурация

Активация платформ контролируется конфигом:

- глобальный список:

```text
FETCHER_ENABLED_PLATFORMS = ["youtube"]
```

- индивидуальные флаги:

```text
FETCHER_YOUTUBE_ENABLED = true
FETCHER_TIKTOK_ENABLED = false
FETCHER_INSTAGRAM_ENABLED = false
```

Значения и источник (файлы конфигурации, переменные окружения) будут зафиксированы в отдельном config‑документе.

### 4.2. Поведение Orchestrator’а

- при выборе адаптера по `platform`:
  - если платформа не включена (`platform not in ENABLED_PLATFORMS`) → Orchestrator:
    - не запускает ingestion;
    - помечает run как `FAILED` с кодом `PLATFORM_DISABLED`;
    - публикует соответствующее событие.

Таким образом, новыми платформами можно управлять конфигурационно, без изменения кода Orchestrator’а.

---

## 5. Связанные документы

- `Fetcher/docs/plan.md` — общая архитектура Fetcher и раздел о platform adapters.
- `Fetcher/docs/BACKEND_CONTRACTS.md` — события и manifest.json (адаптеры должны соблюдать эти контракты).
- `Fetcher/docs/DATABASE.md` — таблицы `video_metadata`, `channel_metadata`, `comments`, `artifacts`.
- `Fetcher/docs/STORAGE_LAYOUT.md` — layout `video-analytics-raw` для `video.mp4`, `meta.json`, `comments.json`.


