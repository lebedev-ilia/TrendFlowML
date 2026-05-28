## Core Ingestion Logic (Phase 1)

Этот документ описывает дизайн **основных ingestion‑воркеров Fetcher**:

- Metadata worker
- Video download worker
- Comments worker
- Artifact Builder и управление артефактами

Основан на `Fetcher/docs/plan.md` (разделы про workers и artifact builder) и чеклисте Phase 1.

---

## 1. Metadata worker

### 1.1. Задачи

- Получить полные метаданные видео (и канала) с платформы (на старте — YouTube).
- Сохранить:
  - нормализованные поля в БД (`video_metadata`, `channel_metadata`);
  - `raw_json` (исходные данные) для будущего анализа/диагностики;
  - при необходимости — `meta.json` в object storage + запись в `artifacts`.

### 1.2. Входы и выходы

- **Вход**:
  - `run_id` (UUID);
  - `platform`, `platform_video_id` (после `NORMALIZE_SOURCE`);
  - (опционально) исходный URL.
- **Выход**:
  - строки в `video_metadata`, `channel_metadata`;
  - (опционально) объект `raw/.../meta.json` в `video-analytics-raw`;
  - запись `artifacts` (`artifact_type='metadata_file'`).

### 1.3. Поля, которые worker обязан поддерживать

Согласно чеклисту:

- `title`
- `description`
- `tags`
- `language`
- `duration_seconds`
- `published_at`
- channel stats:
  - `subscriber_count`
  - `video_count`
  - `view_count_channel`
- дополнительные:
  - `thumbnails`
  - `captions`

Они должны либо быть записаны в нормализованные поля БД, либо присутствовать в `raw_json`.

### 1.4. Инварианты

- Worker **идемпотентен**:
  - при повторном запуске для того же `video_id`:
    - если в `video_metadata`/`channel_metadata` уже есть актуальные записи — повторная запись либо не происходит, либо выполняется безопасный upsert.
- Перед обращением к платформе:
  - worker использует rate‑лимитер (`RATE_LIMITING_AND_LOCKS.md`);
  - корректно обрабатывает retryable/non‑retryable ошибки.

---

## 2. Video download worker

### 2.1. Задачи

- Скачать и нормализовать видео (ограничение по качеству, контейнеру/кодекам).
- Загрузить файл в `video-analytics-raw` (S3/MinIO).
- Зарегистрировать артефакт в таблице `artifacts`.

### 2.2. Входы и выходы

- **Вход**:
  - `run_id`, `platform`, `platform_video_id`;
  - информация о `video_id` из БД.
- **Выход**:
  - файл `video.mp4` в `video-analytics-raw` (`raw/{platform}/YYYY/MM/DD/VIDEO_ID/video.mp4`);
  - запись в `artifacts` (`video_file`, `storage_path`, `size_bytes`, `checksum`, `status`).

### 2.3. Алгоритм (high‑level)

1. Проверить глобальный кеш:
   - если в `artifacts` уже есть `video_file` со статусом `COMPLETED` для данного `video_id` → reuse, скачивание не требуется.
2. Получить distributed lock:
   - `lock:video:{platform}:{platform_video_id}`.
3. При необходимости скачать видео (`yt-dlp`) и нормализовать (`ffmpeg`).
4. Загрузить файл в `video-analytics-raw` через `StorageClient`.
5. Посчитать `checksum` (SHA256), `size_bytes` и обновить `artifacts`.
6. Удалить временный файл из `/tmp`.

### 2.4. Инварианты

- Никогда не оставлять большие файлы в `/tmp` после завершения шага.
- Не скачивать одно и то же видео несколько раз параллельно (distributed lock).
- При неуспешном upload’е:
  - артефакт помечается как `FAILED`;
  - при resume шаг может быть повторён.

---

## 3. Comments worker

### 3.1. Задачи

- Собрать ограниченное число релевантных комментариев с платформы:
  - текст, лайки, число ответов, автор, дата, `raw_json`.
- Сохранить:
  - в таблицу `comments` (по одному ряду на комментарий);
  - либо в `comments.json` в object storage + запись в `artifacts`.

### 3.2. Ограничения и stopping conditions

Чеклист Phase 1 требует:

- лимит по числу комментариев (по умолчанию ≤100);
- pagination streaming (постраничная загрузка);
- контроль timeout’а (≈30s);
- stopping conditions:
  - достигнут лимит по времени;
  - достигнут лимит по страницам;
  - достигнут лимит по количеству комментариев.

### 3.3. PII filtering (дальнейшие фазы)

- PII‑фильтрация (email/phone/url) будет добавлена в более поздней фазе (Phase 1/3);
- Comments worker должен быть спроектирован так, чтобы:
  - можно было добавить фильтрацию как отдельный шаг/pipe без слома контракта.

---

## 4. Artifact Builder и lifecycle артефактов

### 4.1. Artifact Builder

Задачи:

- дождаться завершения всех обязательных ingestion‑шагов:
  - metadata (обновлена БД / создан `meta.json`);
  - video (создан `video_file` в `artifacts`);
  - comments (созданы записи в `comments` или `comments.json`).
- собрать `manifest.json` (`FetcherManifest`) с:
  - `manifest_version`, `platform`, `video_id`, `duration_seconds`;
  - путями к основным артефактам (`video_file`, `meta_file`, `comments_file`);
  - `checksum` для ключевых артефактов.
- записать `manifest.json` в `video-analytics-raw` или `video-analytics-processed` (в зависимости от финального дизайна);
- обновить `artifacts` для manifest’а;
- инициировать запуск DataProcessor (через Backend/DP API).

### 4.2. Artifact lifecycle

Статусы артефактов:

- `PENDING`
- `UPLOADING`
- `COMPLETED`
- `FAILED`

Основные правила:

- при старте upload’а:
  - статус `PENDING` → `UPLOADING`;
- при успешном завершении:
  - `UPLOADING` → `COMPLETED`;
- при сбое upload’а:
  - `UPLOADING` → `FAILED`, с `error`/`retry_count` (в `fetch_jobs` или отдельном поле).

### 4.3. Частично скачанные/загруженные артефакты

Сценарии:

- локальный `/tmp` файл есть, но `artifacts` записи нет;
- есть запись в `artifacts` со статусом `PENDING`/`FAILED`, но файла в storage нет;
- есть файл в storage, но checksum/metadata в БД отсутствуют.

Требования:

- при обнаружении таких ситуаций во время resume:
  - либо повторить upload с обновлением `artifacts`;
  - либо пометить артефакт как `FAILED` и удалить временные/битые данные;
  - логировать инцидент в `fetch_logs` для последующего анализа.

---

## 5. Связанные документы

- `Fetcher/docs/plan.md` — общая архитектура Fetcher, описание workers и pipeline.
- `Fetcher/docs/DATABASE.md` — таблицы `video_metadata`, `channel_metadata`, `comments`, `artifacts`, `fetch_jobs`, `fetch_logs`.
- `Fetcher/docs/STORAGE_LAYOUT.md` — layout `video-analytics-raw` для `video.mp4`, `meta.json`, `comments.json`.
- `Fetcher/docs/BACKEND_CONTRACTS.md` и `Fetcher/schemas/manifest.py` — контракт `manifest.json`.


