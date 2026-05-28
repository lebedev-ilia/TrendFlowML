# План внедрения YouTube Data API v3 в Fetcher

Документ описывает архитектуру и этапы миграции Fetcher на использование **YouTube Data API v3** для получения метадаты и комментариев, с возможностью мокать скачивание видео для офлайн/E2E сценариев.

---

## 1. Цели

- **Стабильный E2E без жёсткой зависимости от yt-dlp и сети к YouTube**.
- **Единый источник правды по метадате/комментариям** — YouTube Data API v3 (официальный API, понятные квоты).
- **Гибкий режим работы**:
  - production: yt-dlp +/или Data API (выбор по фичефлагам);
  - dev/e2e: Data API + моки скачивания видео (без реального трафика к YouTube).
- **Минимальные изменения контрактов** между Backend ↔ Fetcher ↔ DataProcessor (артефакты, manifest, schema).

---

## 2. Архитектура

### 2.1 Новый модуль клиента YouTube Data API

- **Файл**: `Fetcher/fetcher/services/youtube_data_client.py` (или `fetcher/youtube/client.py`).
- **Ответственность**:
  - Обёртка над HTTP‑клиентом к `https://youtube.googleapis.com/youtube/v3`.
  - Методы:
    - `get_video_metadata(video_id: str) -> VideoMetadataDto`
    - `iter_comments(video_id: str, *, max_count: int) -> Iterator[CommentDto]`
  - Обработка квот и rate limiting (минимум — экспоненциальный backoff по 429/5xx).
- **Конфигурация** (через `FetcherSettings`):
  - `youtube_data_api_key: str | None` — ключ API.
  - `youtube_data_enabled: bool = False` — флаг включения клиента.
  - `youtube_data_max_comments: int = 1000` — лимит количества комментариев на видео для выборки.

#### 2.1.1 Quota tracking, rate limiting и retry

- **QuotaTracker**:
  - знает суточный лимит (по умолчанию 10 000 units, настраивается через env);
  - учитывает «стоимость» каждого вызова (`videos.list` = 1, `commentThreads.list` = 1 и т.п.);
  - перед обращением к API вызывается `quota.consume(units)`, при превышении лимита — `QuotaExceededError` и понятный `error_code` на уровне job.

- **Rate limiter**:
  - отдельный лимитер запросов (token‑bucket/линейный), например 5 RPS с небольшим burst;
  - параметры лимитера выносятся в конфиг и могут отличаться для prod/dev.

- **Retry‑политика**:
  - до 3 попыток с backoff 1s → 2s → 4s;
  - retriable: 429, 5xx + сетевые таймауты;
  - остальные ошибки считаем фатальными и пробрасываем наверх.

### 2.2 Интеграция в pipeline Fetcher

#### 2.2.1 fetch_metadata

- **Текущее поведение**: yt-dlp качает/парсит данные, сохраняем в `video_metadata` (`raw_json` и агрегаты).
- **Новая логика (при включенном Data API)**:
  - На стадии `fetch_metadata`:
    1. Если `youtube_data_enabled` и платформа `youtube` → вызываем `youtube_data_client.get_video_metadata(video_id)`.
    2. Маппим ответ API в нашу модель `VideoMetadata`:
       - title, description, channel, duration, statistics (view_count, like_count, comment_count), published_at.
       - Полный JSON ответа сохраняем в `raw_json` для отладки.
    3. Для других платформ или если нет ключа API → fallback к текущему поведению (yt-dlp) **или** сразу помечаем задачу как `FAILED` с понятным кодом ошибки.

#### 2.2.2 fetch_comments

- **Текущее поведение**: yt-dlp получает комментарии пачками, пишем в `comments` и артефакт `comments_file`.

- **Новая логика (при включенном Data API)**:
  - Используем `commentThreads.list` / `comments.list` (YouTube Data API v3) со **streaming‑итератором**, без накопления всех комментариев в памяти:

    ```python
    def iter_comments(video_id: str, *, max_count: int) -> Iterator[CommentDto]:
        page_token: str | None = None
        count = 0

        while True:
            resp = self._request_comments_page(video_id, page_token)
            items = resp.get("items") or []
            if not items:
                return

            for item in items:
                yield CommentDto.from_api(item)
                count += 1
                if count >= max_count:
                    return

            page_token = resp.get("nextPageToken")
            if not page_token:
                return
    ```

  - Worker по мере прихода `CommentDto` сразу пишет их в БД и/или в файл (JSONL), чтобы не держать 1000+ комментариев в памяти.
  - Формируем артефакт `comments_file` (JSONL/JSON) на основе данных из БД (как и сейчас).

#### 2.2.3 download_video (мок для e2e)

- **Цель**: для e2e не обязательно качать реальный `.mp4` из YouTube.

- **Подход**:
  - Ввести флаг `youtube_mock_video_download: bool` в `FetcherSettings`.
  - Если он **true** и платформа `youtube`:
    - вместо реального скачивания:
      - использовать набор sample‑видео, например директорию `sample_videos/` с `sample_0.mp4`, `sample_1.mp4`, …;
      - выбирать sample **детерминированно** по `video_id`, чтобы разные видео стабильно маппились на разные файлы (полезно для ML‑pipeline):

        ```python
        index = int(hashlib.sha256(video_id.encode()).hexdigest(), 16) % settings.youtube_mock_sample_video_count
        sample_path = Path(settings.youtube_mock_sample_video_dir) / f"sample_{index}.mp4"
        ```

      - копируем `sample_path` во временный файл и загружаем его в MinIO/S3 как `video_file` артефакт.
    - обновляем `artifacts` и статус job как `COMPLETED`.
  - В проде этот флаг выключен — используется существующая логика yt-dlp.

---

## 3. Конфигурация и env‑переменные

### 3.1 Новые переменные

Добавить в `FetcherSettings` (`fetcher/config.py`):

- `youtube_data_api_key: str | None` — читается из `FETCHER_YOUTUBE_DATA_API_KEY`.
- `youtube_data_enabled: bool = Field(False, env="FETCHER_YOUTUBE_DATA_ENABLED")`.
- `youtube_data_max_comments: int = Field(1000, env="FETCHER_YOUTUBE_DATA_MAX_COMMENTS")`.
- `youtube_mock_video_download: bool = Field(False, env="FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD")`.
- `youtube_daily_quota_limit: int = Field(10_000, env="FETCHER_YOUTUBE_DAILY_QUOTA_LIMIT")` — суточный лимит квоты YouTube API.
- `youtube_rate_limit_rps: int = Field(5, env="FETCHER_YOUTUBE_RATE_LIMIT_RPS")` — базовый лимит запросов в секунду.
- `youtube_mock_sample_video_dir: str | None = Field(None, env="FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_DIR")` — директория с sample‑видео для мока.
- `youtube_mock_sample_video_count: int = Field(8, env="FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_COUNT")` — количество sample‑файлов в директории.

### 3.2 Режимы работы

- **Production**:
  - `FETCHER_YOUTUBE_DATA_ENABLED=true`
  - `FETCHER_YOUTUBE_DATA_API_KEY=<prod-key>`
  - `FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD=false`
  - `FETCHER_YOUTUBE_USE_YT_DLP=true` (по умолчанию) — можно оставить yt-dlp как fallback, либо постепенно выключать.

- **Dev / E2E локальный** (без настоящего трафика к YouTube):
  - `FETCHER_YOUTUBE_DATA_ENABLED=true`
  - `FETCHER_YOUTUBE_DATA_API_KEY=<dev-key>` (ключ, который даёшь ты)
  - `FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD=true`
  - `FETCHER_YOUTUBE_USE_YT_DLP=false`

---

## 4. Изменения в коде

### 4.1 Новый модуль клиента

**Файл**: `fetcher/services/youtube_data_client.py`

- Использует `httpx` или `requests` с `BASE_URL = "https://youtube.googleapis.com/youtube/v3"`.
- Реализует минимум:

```python
class YouTubeDataClient:
    def __init__(self, api_key: str, timeout: float = 10.0): ...

    def get_video_metadata(self, video_id: str) -> VideoMetadataDto: ...

    def iter_comments(self, video_id: str, *, max_count: int) -> Iterator[CommentDto]: ...
```

- DTO можно хранить в этом же модуле или в `fetcher/schemas/youtube.py`.

### 4.2 Интеграция в tasks/workers

- В `fetcher/tasks.py` и/или `fetcher/workers/metadata.py` / `fetcher/workers/comments.py`:
  - Внедрить зависимость на `YouTubeDataClient` через функцию‑фабрику (чтобы в тестах можно было подменять моком).
  - Логику условного ветвления по `settings.youtube_data_enabled`:
    - если включен — берём данные из Data API;
    - иначе — используем текущую реализацию (yt-dlp) или падаем с явным кодом ошибки.

### 4.3 Provider‑абстракция

Для production‑уровня удобно ввести интерфейс **провайдера видео**:

```python
class VideoProvider(Protocol):
    def get_metadata(self, video_id: str) -> VideoMetadataDto: ...
    def iter_comments(self, video_id: str, *, max_count: int) -> Iterator[CommentDto]: ...
```

Реализации:

- `YouTubeDataAPIProvider` — поверх YouTube Data API v3.
- `YtDlpProvider` — поверх yt-dlp.
- `MockProvider` — для тестов/e2e (фиксированная или детерминированная метадата и комменты).

Выбор провайдера — через настройки (feature flag), что упрощает тестирование и даёт гибкость в проде (fallback на yt-dlp, если Data API недоступен).

### 4.4 DTO и валидация ответов

- DTO для метадаты:

```python
class VideoMetadataDto(BaseModel):
    video_id: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    duration_seconds: int
    view_count: int
    like_count: int
    comment_count: int
    published_at: datetime
```

- При вызове `videos.list` обязательно валидировать, что `items` не пустой:

```python
items = resp.get("items") or []
if not items:
    raise VideoNotFoundError(video_id)
```

Это защищает от кейсов, когда YouTube API возвращает пустой список для существующего видео.

### 4.5 Кэширование метадаты

- Добавить простой кэш для метадаты (in‑memory или Redis):
  - ключ: `video_id`;
  - значение: сериализованный `VideoMetadataDto` + `fetched_at`;
  - TTL: по умолчанию 24 часа.
- Перед обращением к Data API:
  - если в кэше есть свежая запись (меньше TTL) → используем её и **не тратим квоту**;
  - иначе — идём в Data API и обновляем кэш.
- Кэш логичнее размещать **над** HTTP‑клиентом (на уровне `VideoProvider` или отдельного слоя `MetadataCache`), чтобы клиент оставался «тонкой» обёрткой над YouTube API.
- Для поля `raw_json` в БД в проде стоит предусмотреть стратегию объёма:
  - хранить сжатым (gzip/jsonb‑compression);
  - либо выносить большие `raw_json` в объектное хранилище (S3/MinIO), а в БД держать только ссылку.

### 4.3 Моковое скачивание видео

- В `fetcher/workers/video.py` / `fetcher/workers/artifacts.py`:

```python
if settings.youtube_mock_video_download and platform == "youtube":
    sample_path = Path(settings.youtube_mock_sample_video or "tests/data/sample.mp4")
    # копируем в tmp и загружаем в S3 как обычно
```

- Добавить в настройки путь к sample‑видео (`FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO`, опционально).

---

## 5. Тестирование

### 5.1 Unit‑тесты клиента

- Новый файл: `Fetcher/tests/unit/test_youtube_data_client.py`.
- Мокировать HTTP‑вызовы (через `responses`/`httpx_mock`) и проверять:
  - корректный URL и параметры запросов;
  - разбор успешных ответов в DTO;
  - обработку ошибок и ретраев.

### 5.2 Интеграционные тесты Fetcher

- Расширить `Fetcher/tests/integration/test_full_pipeline.py`:
  - режим с `FETCHER_YOUTUBE_DATA_ENABLED=true` и замоком клиента (без реального вызова API):
    - проверка, что `video_metadata` и `comments` заполняются данными из DTO;
    - проверка, что `artifacts` для `metadata_file` и `comments_file` создаются.

### 5.3 E2E сценарий

- В `backend/tests/e2e/test_ingestion_e2e.py` (или новый тест):
  - использовать `FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD=true` и замоканный клиент Data API, чтобы тест был полностью детерминированным.
  - Проверять полный путь: Backend → Fetcher → DataProcessor без реальных запросов к YouTube.

---

## 6. Миграция и rollout

1. **Этап 1 – Клиент + фичефлаг**
   - Реализовать `YouTubeDataClient` и новые настройки, НЕ меняя существующий код pipeline.
   - Добавить unit‑тесты клиента.

2. **Этап 2 – Интеграция в fetch_metadata/fetch_comments (opt‑in)**
   - Обернуть существующую логику условием `if settings.youtube_data_enabled`.
   - В dev включить Data API и проверить, что всё работает на одном/двух видео.

3. **Этап 3 – Моковое скачивание видео для E2E**
   - Добавить `youtube_mock_video_download` и sample‑видео.
   - Переключить e2e‑скрипт и чеклист на использование этого режима.

4. **Этап 4 – Постепенный отказ от yt-dlp (опционально)**
   - После стабилизации и проверки квот YouTube Data API можно:
     - оставить yt-dlp только как fallback на редкие ошибки Data API;
     - или полностью удалить зависимости от yt-dlp для production.

---

## 7. Связанные документы

- `Fetcher/docs/DATABASE.md` — схема таблиц `video_metadata`, `comments`, `artifacts`.
- `Fetcher/docs/DEVELOPMENT.md` — переменные окружения и локальный запуск.
- `backend/docs/E2E_FULL_CHECKLIST.md` — чеклист полного e2e Backend → Fetcher → DataProcessor (нужно будет обновить после реализации Data API режима).

---

## 8. Статус реализации (март 2026)

- **Этап 1 – Клиент + фичефлаг** — ✅ выполнен:
  - добавлены поля `youtube_data_api_key`, `youtube_data_enabled`, `youtube_data_max_comments`, `youtube_daily_quota_limit`, `youtube_rate_limit_rps`, `youtube_mock_video_download`, `youtube_mock_sample_video_dir`, `youtube_mock_sample_video_count` в `FetcherSettings`;
  - реализован модуль `fetcher/services/youtube_data_client.py` с DTO (`VideoMetadataDto`, `CommentDto`), `QuotaTracker`, retry‑политикой и учётом квоты/RPS;
  - добавлены unit‑тесты `tests/unit/test_youtube_data_client.py`.
- **Этап 2 – Интеграция в fetch_metadata/fetch_comments (opt‑in)** — ✅ выполнен:
  - `YouTubeAdapter.fetch_metadata` и `YouTubeAdapter.fetch_comments` используют Data API при `settings.youtube_data_enabled=True` и сохраняют метадату/комментарии в БД + артефакты (`meta.json`, `comments.json`);
  - при выключенном фичефлаге сохраняется текущее поведение через `yt-dlp`.
- **Этап 3 – Моковое скачивание видео для E2E** — ✅ выполнен:
  - `YouTubeAdapter.download_video` поддерживает режим `settings.youtube_mock_video_download=True` с детерминированным выбором sample‑видео и загрузкой его как `video_file` артефакта.
- **Этап 4 – Постепенный отказ от yt-dlp** — ⏳ не начинался:
  - yt-dlp по‑прежнему используется как основной путь для скачивания видео и как fallback для метадаты/комментариев при выключенном Data API.

