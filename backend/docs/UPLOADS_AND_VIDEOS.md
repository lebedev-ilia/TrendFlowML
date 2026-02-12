# Uploads and videos

Backend реализует upload‑flow для пользовательских видео и хранит
минимальные метаданные (duration/width/height через ffprobe).

## 1) Upload flow (as implemented)

1. `POST /api/videos/upload/init`
   - создаёт `Video` (`platform_id="upload"`, `source_type="upload"`)
   - создаёт `Upload` со статусом `init`
2. `PUT /api/videos/upload/{upload_id}`
   - сохраняет файл во временную папку `storage/raw/tmp/<upload_id>`
   - помечает `Upload.status="uploaded"`
3. `POST /api/videos/upload/complete`
   - переносит файл в `storage/raw/<video_id>/video.<ext>`
   - копирует файл в `example/example_videos`
   - вычисляет `sha256_hex` и создаёт/использует `video_files`
   - создаёт `video_sources` и `user_video_links`
   - помечает `Upload.status="completed"`

## 2) Dedup

Dedup реализован через `video_files.sha256_hex`.  
Если файл уже есть:

- новый `Video` всё равно создаётся
- `video_sources.uploaded_file_id` ссылается на существующий `VideoFile`
- доступ даётся только через `user_video_links`

## 3) Метаданные

`ffprobe` вызывается в `services/storage.py::probe_video` и записывает:

- `duration_sec`
- `width`
- `height`

Если `ffprobe` не установлен — upload complete завершится ошибкой.

## 4) Видео‑идентификаторы

Для upload‑видео каноничный `video_id` генерируется backend (UUID).
Пользователь не задаёт `video_id`.

