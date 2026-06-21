## Object Storage layout для Fetcher

Этот документ описывает **целевую структуру бакетов и ключей в S3/MinIO** для Fetcher.

Цель:

- отделить краткоживущие raw‑данные от долгоживущих обработанных артефактов;
- обеспечить человекочитаемый и предсказуемый layout (удобно дебажить руками);
- подготовить надёжный контракт для `manifest.json` и DataProcessor.

Основан на `Fetcher/docs/plan.md`, раздел 6.

---

## 1. Бакеты и их назначение

Рекомендуется использовать как минимум три бакета:

- `video-analytics-raw`
  - raw‑видео и сырые JSON (метаданные, комментарии);
  - основное место, откуда Fetcher строит `manifest.json`.
- `video-analytics-processed`
  - артефакты DataProcessor (NPZ, агрегированные фичи, результаты моделей);
  - не управляется Fetcher напрямую, но Fetcher должен быть совместим с layout’ом.
- `video-analytics-temp`
  - временные файлы (если нужны промежуточные артефакты/логи);
  - агрессивная очистка по TTL.

---

## 2. Layout внутри `video-analytics-raw`

Fetcher не хранит видео локально долговременно:  
`download → /tmp → upload → rm /tmp`.

Рекомендуемая структура ключей:

```text
video-analytics-raw/
  raw/
    youtube/YYYY/MM/DD/VIDEO_ID/video.mp4
    youtube/YYYY/MM/DD/VIDEO_ID/meta.json
    youtube/YYYY/MM/DD/VIDEO_ID/comments.json
```

Где:

- `YYYY/MM/DD` — дата загрузки/ингеста (по UTC);
- `VIDEO_ID` — нормализованный `platform_video_id` (совпадает с `video_id` в manifest);
- `meta.json` — результат `yt-dlp --dump-json` и/или YouTube API (может быть агрегирован);
- `comments.json` — top‑N комментариев (по умолчанию ≤100) с полями, описанными в plan.md.

**Плюсы такого layout’а**:

- равномерное распределение объектов по директориям;
- удобная очистка по дате (S3 lifecycle policies);
- человекочитаемые пути (упрощает дебаг).

---

## 3. Связь с manifest.json

`Fetcher/docs/BACKEND_CONTRACTS.md` описывает контракт `manifest.json`.  
Схема из `Fetcher/schemas/manifest.py` (`FetcherManifest`) ожидает, что:

- `artifacts.video_file.path` указывает на `raw/youtube/YYYY/MM/DD/VIDEO_ID/video.mp4`;
- `artifacts.meta_file.path` указывает на `raw/youtube/YYYY/MM/DD/VIDEO_ID/meta.json`;
- `artifacts.comments_file.path` указывает на `raw/youtube/YYYY/MM/DD/VIDEO_ID/comments.json`.

Пример:

```json
{
  "manifest_version": "1.0",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "abc123",
  "platform": "youtube",
  "duration_seconds": 540,
  "storage_layout_version": "1.0",
  "artifacts": {
    "video_file": {
      "path": "raw/youtube/2026/03/05/abc123/video.mp4",
      "checksum": "sha256:...",
      "size_bytes": 123456789
    },
    "meta_file": {
      "path": "raw/youtube/2026/03/05/abc123/meta.json",
      "checksum": "sha256:...",
      "size_bytes": 12345
    },
    "comments_file": {
      "path": "raw/youtube/2026/03/05/abc123/comments.json",
      "checksum": "sha256:...",
      "size_bytes": 67890,
      "comment_count": 100
    }
  }
}
```

---

## 4. Клиент для S3/MinIO

Fetcher должен использовать абстракцию storage‑клиента поверх конкретного SDK (например, `boto3`).

Пример минимального интерфейса:

```python
class StorageClient:
    def upload_file(self, local_path: str, bucket: str, key: str) -> None: ...
    def download_file(self, bucket: str, key: str, local_path: str) -> None: ...
    def object_exists(self, bucket: str, key: str) -> bool: ...
```

Python‑пример для MinIO/S3 (см. также `plan.md`, раздел 6.4):

```python
import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id="minio",
    aws_secret_access_key="minio123",
)

def upload_video(local_path: str, storage_key: str) -> None:
    s3.upload_file(
        local_path,
        "video-analytics-raw",
        storage_key,
    )
```

---

## 5. Lifecycle‑политики хранения

При реальной нагрузке (десятки тысяч видео в день) raw‑storage быстро растёт до терабайт.  
Рекомендуемые политики:

- `video-analytics-raw`:
  - удалить или перевести в cold‑storage через N дней (например, удалить через 30 дней);
- `video-analytics-processed`:
  - хранить дольше (артефакты и фичи — основной источник для ML/аналитики);
- `video-analytics-temp`:
  - агрессивная очистка (TTL ~7 дней или меньше).

Backend и ML‑pipeline должны опираться на processed‑артефакты (NPZ/агрегаты), а не на гарантированную доступность исходных `video.mp4`.

---

## 6. Следующие шаги

- Реализовать `StorageClient` и обвязку вокруг S3/MinIO в коде Fetcher.
- Добавить конфигурацию endpoint’а, ключей доступа и имён бакетов в отдельный config‑модуль.
- Задокументировать переменные окружения для Fetcher storage (по аналогии с `DataProcessor/api/docs/ENVIRONMENT_VARIABLES.md`).
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
