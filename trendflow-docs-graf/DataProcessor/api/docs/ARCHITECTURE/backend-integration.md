# Backend Integration Architecture

## Обзор

Backend интегрируется с DataProcessor через HTTP API вместо прямого вызова через subprocess. Это позволяет:
- Разделить backend и DataProcessor на отдельные сервисы
- Масштабировать DataProcessor независимо
- Использовать Redis Queue и Worker процессы
- Улучшить мониторинг и обработку ошибок

## Архитектура

```
┌─────────────┐         HTTP         ┌──────────────────┐
│   Backend   │ ────────────────────> │ DataProcessor API│
│             │  POST /api/v1/process │                  │
│  Celery     │                       │  ┌─────────────┐ │
│  Task       │                       │  │   Queue    │ │
└─────────────┘                       │  │  (Redis)   │ │
       │                               │  └─────────────┘ │
       │                               │        │         │
       │                               │        ▼         │
       │                               │  ┌─────────────┐ │
       │  Polling                      │  │   Worker   │ │
       │  GET /api/v1/runs/{id}/status │  │  Process    │ │
       └───────────────────────────────> │  └─────────────┘ │
                                         └──────────────────┘
```

## Компоненты

### Backend

**Файл**: `backend/app/services/dataprocessor.py`

**Функция**: `run_dataprocessor_async()`
- Отправляет HTTP POST запрос к DataProcessor API
- Использует `httpx.AsyncClient` для async HTTP запросов
- Поддерживает API Key аутентификацию
- Обрабатывает ошибки HTTP

**Конфигурация**: `backend/app/config.py`
- `dataprocessor_api_url`: URL DataProcessor API
- `dataprocessor_api_key`: API Key для аутентификации
- `dataprocessor_poll_interval`: Интервал polling статуса
- `dataprocessor_timeout_seconds`: Timeout для обработки

### DataProcessor API

**Endpoint**: `POST /api/v1/process`
- Принимает запрос на обработку
- Валидирует payload
- Добавляет задачу в Redis Queue
- Возвращает статус и run_id

**Обработка**:
1. Валидация запроса
2. Проверка idempotency lock
3. Проверка backpressure
4. Добавление в Redis Queue
5. Возврат ответа

## Поток данных

### 1. Запрос на обработку

```
Backend → POST /api/v1/process → DataProcessor API
```

**Payload**:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "dQw4w9WgXcQ",
  "platform_id": "youtube",
  "video_path": "/data/videos/dQw4w9WgXcQ.mp4",
  "config_hash": "abc123def456",
  "profile_config": {...},
  "rs_base": "/data/result_store",
  "output": "/data/frames_dir",
  ...
}
```

**Response**:
```json
{
  "status": "queued",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Processing request accepted"
}
```

### 2. Polling статуса (Этап 6.2)

```
Backend → GET /api/v1/runs/{run_id}/status → DataProcessor API
```

**Response**:
```json
{
  "status": "running",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": {
    "overall": 0.5,
    "components": {
      "segmenter": {"status": "success", "progress": 1.0},
      "audio": {"status": "running", "progress": 0.3}
    }
  }
}
```

### 3. Завершение обработки

```
Backend → GET /api/v1/runs/{run_id}/status → DataProcessor API
```

**Response** (success):
```json
{
  "status": "success",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "progress": {
    "overall": 1.0,
    "components": {
      "segmenter": {"status": "success", "progress": 1.0},
      "audio": {"status": "success", "progress": 1.0},
      "visual": {"status": "success", "progress": 1.0}
    }
  }
}
```

## Обработка ошибок

### HTTP ошибки

1. **400 Bad Request**: Невалидный payload
   - Backend должен проверить payload перед отправкой
   - Исправить ошибки валидации

2. **401 Unauthorized**: Отсутствует API Key
   - Настроить `TF_BACKEND_DATAPROCESSOR_API_KEY`

3. **403 Forbidden**: Невалидный API Key
   - Проверить правильность API Key

4. **409 Conflict**: Run с таким ID уже существует
   - Использовать другой run_id или проверить существующий run

5. **503 Service Unavailable**: Backpressure
   - Повторить запрос позже (Retry-After header)
   - Уменьшить нагрузку

### Ошибки соединения

- **ConnectionError**: DataProcessor API недоступен
  - Проверить доступность API
  - Использовать retry логику

- **TimeoutError**: Timeout при запросе
  - Увеличить timeout или проверить производительность API

## Конфигурация

### Backend (.env)

```bash
# DataProcessor API
TF_BACKEND_DATAPROCESSOR_API_URL=http://dataprocessor:8000
TF_BACKEND_DATAPROCESSOR_API_KEY=your-api-key
TF_BACKEND_DATAPROCESSOR_POLL_INTERVAL=5
TF_BACKEND_DATAPROCESSOR_TIMEOUT_SECONDS=3600
```

### DataProcessor API (.env)

```bash
# API настройки
API_HOST=0.0.0.0
API_PORT=8000
DATAPROCESSOR_API_KEY=your-api-key

# Redis
REDIS_URL=redis://redis:6379/0

# Storage
STORAGE_TYPE=fs
STORAGE_ROOT=/data
```

## Мониторинг

### Метрики Backend

- Количество запросов к DataProcessor API
- Время ответа API
- Количество ошибок
- Статус обработки (queued, running, success, error)

### Метрики DataProcessor API

- Длина очереди
- Количество активных run'ов
- Время обработки
- Количество ошибок

## Безопасность

1. **API Key аутентификация**: Все запросы должны включать валидный API Key
2. **HTTPS в production**: Использовать HTTPS для защиты данных
3. **Валидация путей**: Проверка разрешенных директорий для video_path
4. **Rate limiting**: Ограничение количества запросов

## Масштабирование

### Горизонтальное масштабирование

- **Backend**: Можно запустить несколько экземпляров
- **DataProcessor API**: Можно запустить несколько экземпляров (stateless)
- **DataProcessor Worker**: Можно масштабировать независимо через `docker-compose up --scale dataprocessor-worker=5`

### Вертикальное масштабирование

- Увеличить ресурсы контейнеров (memory, CPU)
- Настроить лимиты параллелизма
- Оптимизировать обработку

## Ссылки

- [Этап 6.1: Замена subprocess на HTTP](../IMPLEMENTATION/2024-01-XX-stage-6.1-subprocess-to-http.md)
- [API Changes: Backend Integration](../API_CHANGES/2024-01-XX-backend-integration-http.md)
- [DataProcessor API Architecture](../../../docs/DATAPROCESSOR_API_ARCHITECTURE.md)
---

## Навигация

[README](README.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
