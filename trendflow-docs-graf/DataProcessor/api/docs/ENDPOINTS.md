# API Endpoints

Полная документация всех endpoints DataProcessor API.

## Базовый URL

```
http://localhost:8000/api/v1
```

## Аутентификация

Все endpoints (кроме `/health`) требуют аутентификации через API Key:

```
X-API-Key: your-api-key
```

## Endpoints

### 1. POST /api/v1/process

Запуск обработки видео.

**Описание**: Принимает запрос на обработку видео и ставит его в очередь. Возвращает `run_id` и статус `queued`.

**Request Body**:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "video-123",
  "platform_id": "youtube",
  "video_path": "/path/to/video.mp4",
  "config_hash": "abc123...",
  "profile_config": {
    "visual": {"enabled": true},
    "audio": {"enabled": true},
    "text": {"enabled": true}
  },
  "profile_version": "v1",
  "feature_schema_version": "v1",
  "pipeline_version": "prod"
}
```

**Response** (202 Accepted):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Processing queued successfully"
}
```

**Ошибки**:
- `400 Bad Request` - Невалидный payload
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ
- `409 Conflict` - Run с таким `run_id` уже существует
- `503 Service Unavailable` - Backpressure (очередь переполнена)

**Rate Limiting**: 100 запросов в час на backend instance

---

### 2. GET /api/v1/runs/{run_id}

Получение метаданных run'а.

**Описание**: Возвращает метаданные run'а (run_id, video_id, platform_id, created_at, etc.).

**Path Parameters**:
- `run_id` (string, required) - UUID run'а

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "video-123",
  "platform_id": "youtube",
  "created_at": "2024-01-01T12:00:00Z",
  "status": "running"
}
```

**Ошибки**:
- `404 Not Found` - Run не найден
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ

---

### 3. GET /api/v1/runs/{run_id}/status

Получение детального статуса обработки.

**Описание**: Возвращает текущий статус обработки, прогресс, стадии и компоненты.

**Path Parameters**:
- `run_id` (string, required) - UUID run'а

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.65,
  "current_stage": "visual",
  "current_component": "core_clip",
  "started_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:05:00Z",
  "stages": {
    "visual": {
      "status": "running",
      "progress": 0.65,
      "current_component": "core_clip",
      "components": {
        "core_clip": {"status": "running", "progress": 0.5},
        "places365": {"status": "pending"}
      }
    },
    "audio": {"status": "pending"},
    "text": {"status": "pending"}
  }
}
```

**Ошибки**:
- `404 Not Found` - Run не найден
- `410 Gone` - Run завершён и удалён (retention policy)
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ

---

### 4. GET /api/v1/runs/{run_id}/events

Server-Sent Events (SSE) стрим событий.

**Описание**: Возвращает поток событий в реальном времени с обновлениями статуса и прогресса.

**Path Parameters**:
- `run_id` (string, required) - UUID run'а

**Query Parameters**:
- `api_key` (string, optional) - API ключ (альтернатива заголовку `X-API-Key`)

**Response** (200 OK, Content-Type: text/event-stream):
```
event: status
data: {"run_id": "...", "status": "running", "progress": 0.1}

event: stage
data: {"run_id": "...", "stage": "visual", "status": "started"}

event: component
data: {"run_id": "...", "component": "core_clip", "status": "started"}

event: log
data: {"run_id": "...", "level": "info", "message": "Processing..."}

event: status
data: {"run_id": "...", "status": "success", "progress": 1.0}
```

**Типы событий**:
- `status` - Обновление статуса run'а
- `stage` - Обновление стадии (visual, audio, text)
- `component` - Обновление компонента
- `log` - Лог сообщение

**Ошибки**:
- `404 Not Found` - Run не найден
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ
- `429 Too Many Requests` - Превышен лимит SSE соединений на run_id

---

### 5. GET /api/v1/runs/{run_id}/manifest

Получение manifest.json.

**Описание**: Возвращает manifest.json с метаданными результатов обработки.

**Path Parameters**:
- `run_id` (string, required) - UUID run'а

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "video-123",
  "platform_id": "youtube",
  "status": "success",
  "finished_at": "2024-01-01T12:30:00Z",
  "artifacts": {
    "visual": {
      "path": "visual.npz",
      "size": 1024000,
      "checksum": "sha256:abc123..."
    },
    "audio": {
      "path": "audio.npz",
      "size": 512000,
      "checksum": "sha256:def456..."
    },
    "text": {
      "path": "text.npz",
      "size": 256000,
      "checksum": "sha256:ghi789..."
    }
  },
  "metadata": {
    "duration": 300.5,
    "fps": 30,
    "resolution": "1920x1080"
  }
}
```

**Ошибки**:
- `404 Not Found` - Run не найден или manifest не существует
- `410 Gone` - Run завершён и удалён (retention policy)
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ

---

### 6. GET /api/v1/runs/{run_id}/artifacts/{artifact_path}

Получение артефактов обработки.

**Описание**: Возвращает файл артефакта (NPZ файлы, JSON файлы, etc.).

**Path Parameters**:
- `run_id` (string, required) - UUID run'а
- `artifact_path` (string, required) - Путь к артефакту (например, `visual.npz`, `audio.npz`)

**Response** (200 OK, Content-Type: application/octet-stream):
Бинарные данные файла

**Ошибки**:
- `404 Not Found` - Run не найден или артефакт не существует
- `410 Gone` - Run завершён и удалён (retention policy)
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ

**Примеры**:
- `/api/v1/runs/{run_id}/artifacts/visual.npz`
- `/api/v1/runs/{run_id}/artifacts/audio.npz`
- `/api/v1/runs/{run_id}/artifacts/text.npz`
- `/api/v1/runs/{run_id}/artifacts/manifest.json`

---

### 7. POST /api/v1/runs/{run_id}/cancel

Отмена активной обработки.

**Описание**: Отменяет активную обработку run'а. Run переходит в статус `cancelled`.

**Path Parameters**:
- `run_id` (string, required) - UUID run'а

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "cancelled",
  "message": "Run cancelled successfully"
}
```

**Ошибки**:
- `404 Not Found` - Run не найден
- `400 Bad Request` - Run уже завершён или отменён
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ

---

### 8. GET /api/v1/health

Health check API сервера.

**Описание**: Проверяет состояние API сервера и зависимостей (Redis, Storage, Triton).

**Response** (200 OK или 503 Service Unavailable):
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "dependencies": {
    "redis": {
      "status": "healthy",
      "latency_ms": 2
    },
    "storage": {
      "status": "healthy"
    },
    "triton": {
      "status": "healthy",
      "latency_ms": 10
    }
  },
  "queue": {
    "length": 5,
    "active_runs": 2
  },
  "total_runs_today": 100
}
```

**Статусы**:
- `200 OK` - API здоров
- `503 Service Unavailable` - API unhealthy (зависимости недоступны)

**Примечание**: Не требует аутентификации

---

### 9. GET /api/v1/metrics

Prometheus метрики.

**Описание**: Возвращает метрики в формате Prometheus text format.

**Response** (200 OK, Content-Type: text/plain; version=0.0.4; charset=utf-8):
```
# HELP dataprocessor_queue_length Current queue length
# TYPE dataprocessor_queue_length gauge
dataprocessor_queue_length{priority="high"} 5.0
dataprocessor_queue_length{priority="normal"} 10.0

# HELP dataprocessor_processing_seconds Processing time per run
# TYPE dataprocessor_processing_seconds histogram
dataprocessor_processing_seconds_bucket{processor="visual",component="core_clip",le="60.0"} 10.0
...
```

**Примечание**: Не требует аутентификации

---

### 10. POST /api/v1/admin/retention/cleanup

Ручной запуск retention cleanup (требует аутентификации).

**Описание**: Запускает очистку старых данных (Redis state и Storage).

**Response** (200 OK):
```json
{
  "redis": {
    "deleted_keys": 100,
    "deleted_bytes": 1024000
  },
  "storage": {
    "deleted_runs": 10,
    "deleted_bytes": 104857600
  }
}
```

**Ошибки**:
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ

---

## Коды статуса HTTP

- `200 OK` - Успешный запрос
- `202 Accepted` - Запрос принят (асинхронная обработка)
- `400 Bad Request` - Невалидный запрос
- `401 Unauthorized` - Требуется аутентификация
- `403 Forbidden` - Невалидный API ключ
- `404 Not Found` - Ресурс не найден
- `409 Conflict` - Конфликт (например, дубликат run_id)
- `410 Gone` - Ресурс больше не доступен (завершённый run)
- `429 Too Many Requests` - Превышен rate limit
- `500 Internal Server Error` - Внутренняя ошибка сервера
- `503 Service Unavailable` - Сервис недоступен (backpressure, unhealthy)

## Версионирование

API версионируется через префикс пути: `/api/v1/`

Текущая версия: `v1`

## Rate Limiting

Endpoint `POST /api/v1/process` ограничен до 100 запросов в час на backend instance.

Используйте заголовок `X-Backend-ID` для идентификации backend instance.

## Swagger UI

Интерактивная документация доступна по адресу:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json
---

## Навигация

[README](README.md) · [Module README](../README.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
