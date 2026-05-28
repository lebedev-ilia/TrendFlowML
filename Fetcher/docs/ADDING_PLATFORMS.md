# Добавление новых платформ (TikTok, Instagram и др.)

Краткий чеклист для реализации нового платформенного адаптера в Fetcher.

---

## 1. Интерфейс

- Базовый класс: `fetcher/platforms/base.py` — `PlatformAdapter`.
- Обязательные методы:
  - `fetch_metadata(source, *, run_id)` — метаданные видео и канала в БД/storage.
  - `download_video(source, *, run_id)` — скачать видео и загрузить в S3/MinIO.
  - `fetch_comments(source, *, run_id, limit=100)` — комментарии в БД и/или `comments.json`.

Контракт и инварианты см. в `BACKEND_CONTRACTS.md` и `plan.md` (раздел Platform adapters).

---

## 2. Шаги реализации

### 2.1. Модуль адаптера

- Создать каталог `fetcher/platforms/<platform>/` (например, `tiktok`, `instagram`).
- Реализовать класс `<Platform>Adapter(PlatformAdapter)` в `adapter.py`.
- Использовать те же таблицы и артефакты, что и YouTube: `Video`, `VideoMetadata`, `ChannelMetadata`, `Comment`, `Artifact`, `VideoSnapshot` (при необходимости).
- В `fetcher/platforms/__init__.py` или в оркестраторе зарегистрировать фабрику/выбор адаптера по `platform`.

### 2.2. Нормализация URL

- В `orchestrator.normalize_source(url)` добавить ветку для новой платформы:
  - определение платформы по домену;
  - извлечение `platform_video_id` (через API, yt-dlp, или парсинг URL).
- Возвращать `(platform, platform_video_id)`.

### 2.3. Конфиг и feature flag

- В `config.py` при необходимости добавить:
  - `<platform>_enabled: bool`;
  - лимиты и окна rate limit для платформы.
- Добавить платформу в `enabled_platforms` (или проверять `<platform>_enabled` при выборе адаптера).

### 2.4. Оркестратор и воркеры

- В `orchestrator.py` и воркерах (`workers/metadata.py`, `workers/video.py`, `workers/comments.py`) выбор адаптера по `platform` (сейчас захардкожен YouTube).
- Общий паттерн: `get_adapter(platform) -> PlatformAdapter` и вызов его методов.

### 2.5. Метрики (опционально)

- При необходимости завести метрики по платформе: например `fetcher_*_total{platform="tiktok"}`, rate limit errors по платформе.

### 2.6. Тесты

- Unit-тесты адаптера с моками (API/yt-dlp, storage, DB).
- Интеграционный тест: один run с тестовым URL (если есть тестовое окружение).

---

## 3. Пример: TikTok

- **Нормализация**: парсинг URL вида `https://www.tiktok.com/@user/video/1234567890` → `platform_video_id`.
- **Метаданные**: TikTok API или неофициальный клиент / yt-dlp (если поддерживает).
- **Скачивание**: yt-dlp или отдельная библиотека по политике платформы.
- **Комментарии**: по API/скрапингу с учётом лимитов и PII.

---

## 4. Пример: Instagram

- **Нормализация**: URL Reels/видео → shortcode или media ID.
- **Метаданные/комментарии**: через Instagram Graph API (требуются токены и права).
- Учесть ограничения API и rate limits.

---

## 5. Связанные файлы

- `fetcher/platforms/base.py` — интерфейс.
- `fetcher/platforms/youtube/` — эталонная реализация.
- `fetcher/orchestrator.py` — `normalize_source`, выбор адаптера.
- `fetcher/config.py` — `enabled_platforms`, лимиты.
- `fetcher/workers/*.py` — вызов адаптера по `platform`.

После реализации добавить платформу в описание в `plan.md` и в чеклист в `checklist.md`.
