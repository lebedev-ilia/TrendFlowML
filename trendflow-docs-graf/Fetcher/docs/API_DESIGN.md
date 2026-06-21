# Fetcher API Design

**Дата**: 2026-03-05  
**Версия**: 2.0 (обновлено с учётом архитектурных улучшений)  
**Статус**: План

## Обзор

Fetcher должен предоставлять REST API для взаимодействия с Backend и другими сервисами. API должен поддерживать создание runs, запуск ingestion, получение статусов и управление pipeline.

**Важные архитектурные решения**:
- API не запускает ingestion синхронно — использует event-driven подход (Kafka/Celery)
- Cursor-based pagination вместо offset для масштабируемости
- Логи не возвращаются напрямую — предоставляется URL для доступа
- Artifacts возвращают signed URLs для безопасного доступа
- Cooperative cancellation для остановки длительных операций
- Idempotency keys для безопасных retry
- Bulk ingestion для эффективной обработки множественных запросов
- Webhooks для асинхронных уведомлений

## Принципы дизайна

1. **RESTful API**: Использование стандартных HTTP методов и кодов ответов
2. **Версионирование**: Все endpoints под `/api/v1/`
3. **Аутентификация**: Поддержка API keys или JWT (для production)
4. **Документация**: OpenAPI/Swagger спецификация
5. **Обработка ошибок**: Стандартизированные error responses
6. **Пагинация**: Cursor-based для масштабируемости (не offset)
7. **Event-driven**: API публикует события, не запускает ingestion напрямую
8. **Idempotency**: Поддержка Idempotency-Key header

## Endpoints

### 1. Runs Management

#### 1.1. POST /api/v1/runs
**Описание**: Создать новый run и запустить ingestion (event-driven, не синхронно).

**Headers**:
- `Idempotency-Key` (optional): Ключ для идемпотентности. Если передан и run уже существует, возвращается существующий run.

**Request Body**:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",  // UUID от Backend
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "platform": "youtube",  // опционально, может быть определен автоматически
  "priority": "normal",  // опционально: "low", "normal", "high" (определяет очередь: fetcher.low, fetcher.normal, fetcher.high)
  "webhook_url": "https://backend.example.com/runs/callback",  // опционально, для уведомлений
  "max_run_duration_seconds": 7200  // опционально, default: 2 часа, watchdog отменит run если превышен
}
```

**Deduplication**:
- API проверяет canonical video ID для source_url
- Если run с таким canonical ID уже существует, возвращается существующий run (409 Conflict или 200 OK с existing_run_id)
- Это предотвращает дублирование одинаковых видео с разными URL форматами (youtube.com/watch?v=abc vs youtu.be/abc)

**Response** (201 Created):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "platform": "youtube",
  "created_at": "2026-03-05T12:00:00Z",
  "message": "Run created and ingestion queued"
}
```

**Ошибки**:
- `400 Bad Request`: Невалидный URL или run_id
- `409 Conflict`: Run с таким run_id уже существует (или idempotency key уже использован)
- `429 Too Many Requests`: Превышен rate limit
- `500 Internal Server Error`: Ошибка при создании run'а

**Логика (Event-driven)**:
1. Проверка Idempotency-Key (если передан, проверяем существующий run)
2. Валидация run_id (UUID формат)
3. Валидация source_url
4. Создание записи в таблице `runs` со статусом `PENDING`
5. Создание записи в таблице `video_sources`
6. **Публикация события в очередь** (Celery task или Kafka):
   - `fetch_metadata_task.delay(run_id)` через Celery
   - Или публикация в Kafka topic `fetcher.tasks.fetch_metadata`
7. Публикация события `run.created` в Kafka (если включен)
8. Возврат информации о созданном run'е

**Важно**: API не запускает ingestion синхронно. Вместо этого публикует событие в очередь для обработки workers.

#### 1.2. GET /api/v1/runs/{run_id}
**Описание**: Получить информацию о run'е.

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "FETCHING_METADATA",
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "platform": "youtube",
  "platform_video_id": "dQw4w9WgXcQ",
  "created_at": "2026-03-05T12:00:00Z",
  "started_at": "2026-03-05T12:00:01Z",
  "finished_at": null,
  "error": null,
  "error_code": null,
  "video_id": "123e4567-e89b-12d3-a456-426614174000",  // UUID видео в Fetcher БД
  "artifacts": {
    "video_file": "s3://bucket/video.mp4",
    "meta_file": "s3://bucket/meta.json",
    "comments_file": "s3://bucket/comments.json",
    "manifest_file": "s3://bucket/manifest.json"
  },
  "progress": {
    "stage": "fetch_metadata",
    "completed_stages": ["normalize_source", "check_cache"],
    "total_stages": 7
  }
}
```

**Ошибки**:
- `404 Not Found`: Run не найден

#### 1.3. GET /api/v1/runs
**Описание**: Получить список runs с фильтрацией и cursor-based пагинацией.

**Query Parameters**:
- `status` (optional): Фильтр по статусу (PENDING, FETCHING_METADATA, COMPLETED, FAILED, etc.)
- `platform` (optional): Фильтр по платформе (youtube, tiktok, etc.)
- `limit` (optional, default: 50, max: 100): Количество результатов
- `cursor` (optional): Cursor для пагинации (base64 encoded JSON с timestamp и run_id)
- `created_after` (optional): Фильтр по дате создания (ISO 8601) - для начальной точки
- `created_before` (optional): Фильтр по дате создания (ISO 8601)

**Response** (200 OK):
```json
{
  "runs": [
    {
      "run_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "COMPLETED",
      "platform": "youtube",
      "created_at": "2026-03-05T12:00:00Z",
      "finished_at": "2026-03-05T12:05:00Z"
    }
  ],
  "pagination": {
    "limit": 50,
    "has_more": true,
    "next_cursor": "eyJjcmVhdGVkX2F0IjoiMjAyNi0wMy0wNVQxMjowNTowMFoiLCJydW5faWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAifQ=="
  }
}
```

**Cursor format** (base64 encoded JSON):
```json
{
  "created_at": "2026-03-05T12:05:00Z",
  "run_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Важно**: Используется cursor-based pagination вместо offset для масштабируемости при больших объёмах данных.

#### 1.4. POST /api/v1/runs/{run_id}/retry
**Описание**: Перезапустить ingestion для существующего run'а (event-driven).

**Использование**: Для retry failed runs или manual restart.

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "message": "Ingestion queued for retry"
}
```

**Ошибки**:
- `404 Not Found`: Run не найден
- `400 Bad Request`: Run в статусе, который нельзя перезапустить (например, COMPLETED)
- `429 Too Many Requests`: Превышен rate limit

**Логика**: Публикует событие в очередь для перезапуска, не запускает синхронно.

#### 1.5. PATCH /api/v1/runs/{run_id}
**Описание**: Обновить run (например, запросить отмену).

**Request Body**:
```json
{
  "cancel_requested": true
}
```

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "CANCELLING",
  "cancel_requested": true,
  "message": "Cancellation requested. Run will be cancelled at next checkpoint."
}
```

**Ошибки**:
- `404 Not Found`: Run не найден
- `400 Bad Request`: Run уже завершён или отменён
- `422 Unprocessable Entity`: Невалидное поле для обновления

**Альтернативный вариант** (если нужен отдельный endpoint):
#### 1.5a. POST /api/v1/runs/{run_id}/actions/cancel
**Описание**: Запросить отмену выполнения run'а (cooperative cancellation).

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "CANCELLING",
  "message": "Cancellation requested. Run will be cancelled at next checkpoint."
}
```

**Логика (Cooperative Cancellation)**:
1. Установка флага `cancel_requested = true` в БД
2. Workers проверяют этот флаг между стадиями
3. При обнаружении флага worker останавливается и устанавливает статус `CANCELLED`
4. Для длительных операций (например, download 500MB video) отмена происходит на следующем checkpoint

**Важно**: Отмена не мгновенная. Workers должны проверять `run.cancel_requested` между стадиями.

### 2. Artifacts & Manifest

#### 2.1. GET /api/v1/runs/{run_id}/artifacts
**Описание**: Получить список артефактов для run'а с signed URLs для скачивания.

**Query Parameters**:
- `expires_in` (optional, default: 3600): Время жизни signed URL в секундах (max: 86400)

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "artifacts": [
    {
      "artifact_type": "video_file",
      "storage_key": "s3://bucket/video.mp4",
      "download_url": "https://s3.amazonaws.com/bucket/video.mp4?X-Amz-Algorithm=...&X-Amz-Signature=...",
      "download_url_expires_at": "2026-03-05T13:00:00Z",
      "size_bytes": 10485760,
      "checksum": "sha256:abc123...",
      "status": "COMPLETED",
      "created_at": "2026-03-05T12:02:00Z"
    },
    {
      "artifact_type": "meta_file",
      "storage_key": "s3://bucket/meta.json",
      "download_url": "https://s3.amazonaws.com/bucket/meta.json?X-Amz-Algorithm=...&X-Amz-Signature=...",
      "download_url_expires_at": "2026-03-05T13:00:00Z",
      "size_bytes": 1024,
      "checksum": "sha256:def456...",
      "status": "COMPLETED",
      "created_at": "2026-03-05T12:01:00Z"
    }
  ]
}
```

**Важно**: Возвращаются signed URLs для безопасного доступа к артефактам. Внутренние storage_key не должны быть доступны клиентам.

#### 2.2. GET /api/v1/runs/{run_id}/manifest
**Описание**: Получить manifest.json для run'а.

**Response** (200 OK):
```json
{
  "manifest_version": "1.0",
  "platform": "youtube",
  "video_id": "dQw4w9WgXcQ",
  "duration_seconds": 212,
  "artifacts": {
    "video_file": {
      "path": "s3://bucket/video.mp4",
      "size_bytes": 10485760,
      "checksum": "sha256:abc123..."
    },
    "meta_file": {
      "path": "s3://bucket/meta.json",
      "size_bytes": 1024,
      "checksum": "sha256:def456..."
    }
  },
  "created_at": "2026-03-05T12:05:00Z"
}
```

**Ошибки**:
- `404 Not Found`: Run не найден или manifest не создан
- `503 Service Unavailable`: Manifest ещё не готов (run в процессе)

### 3. Logs

#### 3.1. GET /api/v1/runs/{run_id}/logs_url
**Описание**: Получить URL для доступа к логам run'а (логи хранятся в централизованном хранилище, не в БД).

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "logs_url": "https://grafana.example.com/explore?orgId=1&left=%5B%22now-1h%22,%22now%22,%22Loki%22%5D&query=%7Brun_id%3D%22550e8400...%22%7D",
  "logs_backend": "loki",  // или "elasticsearch", "cloudwatch"
  "message": "Logs are available in centralized logging system"
}
```

**Ошибки**:
- `404 Not Found`: Run не найден
- `503 Service Unavailable`: Централизованное логирование не настроено

**Важно**: Логи не хранятся в PostgreSQL. Они отправляются в централизованное хранилище (Loki, Elasticsearch, CloudWatch). API возвращает URL для доступа к логам через Grafana или другой интерфейс.

**Альтернативный endpoint** (если нужен прямой доступ):
#### 3.2. GET /api/v1/runs/{run_id}/logs (опционально)
**Описание**: Получить последние N логов из БД (только для отладки, не для production).

**Query Parameters**:
- `limit` (optional, default: 50, max: 200): Количество последних логов

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "logs": [
    {
      "timestamp": "2026-03-05T12:00:00Z",
      "level": "info",
      "stage": "fetch_metadata",
      "message": "Starting metadata worker"
    }
  ],
  "warning": "Only recent logs are returned. For full logs, use logs_url endpoint.",
  "logs_url": "https://grafana.example.com/..."
}
```

**Важно**: Этот endpoint возвращает только последние логи из БД (таблица `fetch_logs`). Для полных логов используется `logs_url`.

### 4. Video Cache

#### 4.1. GET /api/v1/videos/{platform}/{video_id}
**Описание**: Получить информацию о видео из кеша Fetcher.

**Response** (200 OK):
```json
{
  "video_id": "123e4567-e89b-12d3-a456-426614174000",
  "platform": "youtube",
  "platform_video_id": "dQw4w9WgXcQ",
  "channel_id": "UCuAXFkgsw1L7xaCfnd5JJOw",
  "duration_seconds": 212,
  "cached_at": "2026-03-05T10:00:00Z",
  "artifacts_available": ["video_file", "meta_file", "comments_file"],
  "snapshots_count": 3,
  "comments_count": 150
}
```

**Ошибки**:
- `404 Not Found`: Видео не найдено в кеше

### 5. Statistics

#### 5.1. GET /api/v1/stats
**Описание**: Получить статистику по ingestion.

**Query Parameters**:
- `period` (optional, default: "24h"): Период (1h, 24h, 7d, 30d)

**Response** (200 OK):
```json
{
  "period": "24h",
  "runs": {
    "total": 1000,
    "completed": 950,
    "failed": 30,
    "running": 20
  },
  "throughput": {
    "videos_per_hour": 41.67,
    "videos_per_day": 1000
  },
  "cache": {
    "hit_rate": 0.75,
    "hits": 750,
    "misses": 250
  },
  "platforms": {
    "youtube": 900,
    "tiktok": 100
  },
  "errors": {
    "YOUTUBE_429": 10,
    "NETWORK_ERROR": 5,
    "VIDEO_NOT_FOUND": 15
  }
}
```

### 6. Admin Endpoints (уже реализованы)

- `GET /metrics` - Prometheus metrics
- `GET /health` - Health check
- `GET /admin/validation` - Валидация компонентов
- `POST /admin/lifecycle/cleanup` - Lifecycle cleanup

## Error Responses

Стандартизированный формат ошибок:

```json
{
  "error": {
    "code": "RUN_NOT_FOUND",
    "message": "Run with id 550e8400-... not found",
    "details": {
      "run_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

**Коды ошибок**:
- `INVALID_REQUEST`: Невалидный запрос
- `RUN_NOT_FOUND`: Run не найден
- `RUN_ALREADY_EXISTS`: Run с таким ID уже существует
- `INVALID_STATUS_TRANSITION`: Недопустимый переход статуса
- `VIDEO_NOT_FOUND`: Видео не найдено
- `ARTIFACT_NOT_READY`: Артефакт ещё не готов
- `RATE_LIMIT_EXCEEDED`: Превышен rate limit
- `INTERNAL_ERROR`: Внутренняя ошибка сервера

## Аутентификация

Для production рекомендуется:

1. **API Key Authentication**:
   - Header: `X-API-Key: <key>`
   - Или: Query parameter `?api_key=<key>`

2. **JWT Authentication**:
   - Header: `Authorization: Bearer <token>`

3. **Rate Limiting**:
   - По API key или IP
   - Конфигурируемые лимиты

## Версионирование

- Все endpoints под `/api/v1/`
- При breaking changes: `/api/v2/`
- Поддержка нескольких версий одновременно

## OpenAPI/Swagger

- Автогенерация из FastAPI
- Доступно по `/docs` (Swagger UI) и `/redoc` (ReDoc)
- Включить примеры запросов и ответов

## Пагинация

Для списков использовать:
- `limit` и `offset` (простая пагинация)
- Или cursor-based (для больших объёмов)

## Приоритеты реализации

**Phase 1 (MVP)**:
1. POST /api/v1/runs - создание и запуск ingestion
2. GET /api/v1/runs/{run_id} - получение статуса
3. GET /api/v1/runs/{run_id}/manifest - получение manifest

**Phase 2**:
4. GET /api/v1/runs - список runs
5. GET /api/v1/runs/{run_id}/artifacts - список артефактов
6. GET /api/v1/runs/{run_id}/logs - логи

**Phase 3**:
7. POST /api/v1/runs/{run_id}/fetch - перезапуск
8. POST /api/v1/runs/{run_id}/cancel - отмена
9. GET /api/v1/videos/{platform}/{video_id} - информация о видео
10. GET /api/v1/stats - статистика

**Phase 4**:
11. Аутентификация
12. Rate limiting для API
13. OpenAPI документация
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
