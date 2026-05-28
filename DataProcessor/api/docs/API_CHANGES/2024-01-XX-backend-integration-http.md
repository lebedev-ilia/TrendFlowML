# Backend Integration: Замена subprocess на HTTP

**Дата**: 2024-01-XX  
**Версия API**: v1  
**Тип изменения**: Интеграция

## Описание изменения

Backend теперь использует HTTP запросы к DataProcessor API вместо прямого вызова через subprocess. Это позволяет разделить backend и DataProcessor на отдельные сервисы и улучшить масштабируемость.

## Что изменилось

### Backend (`backend/app/services/dataprocessor.py`)

**Было** (subprocess):
```python
def run_dataprocessor(...):
    cmd = [sys.executable, str(dp_main), *args]
    proc = subprocess.run(cmd, check=True, ...)
```

**Стало** (HTTP):
```python
async def run_dataprocessor_async(...):
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{api_url}/api/v1/process",
            json=payload,
            headers={"X-API-Key": api_key}
        )
        response.raise_for_status()
        return response.json()
```

### Конфигурация (`backend/app/config.py`)

**Добавлены настройки**:
- `dataprocessor_api_url`: URL DataProcessor API (по умолчанию: `http://localhost:8001`)
- `dataprocessor_api_key`: API Key для аутентификации (опционально)
- `dataprocessor_poll_interval`: Интервал polling статуса (по умолчанию: 5 секунд)
- `dataprocessor_timeout_seconds`: Timeout для обработки (по умолчанию: 3600 секунд)

## Детали

### Endpoint

**URL**: `POST /api/v1/process`

**Headers**:
- `Content-Type: application/json`
- `X-API-Key: <api_key>` (опционально, если настроена аутентификация)

**Request Body**:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "dQw4w9WgXcQ",
  "platform_id": "youtube",
  "video_path": "/data/videos/dQw4w9WgXcQ.mp4",
  "config_hash": "abc123def456",
  "profile_config": {
    "processors": {
      "segmenter": {"enabled": true, "required": true},
      "audio": {"enabled": true, "required": false},
      "visual": {"enabled": true, "required": true}
    }
  },
  "rs_base": "/data/result_store",
  "output": "/data/frames_dir",
  "visual_cfg_path": "/path/to/visual_cfg.yaml",
  "dag_path": "/path/to/component_graph.yaml",
  "dag_stage": "baseline",
  "sampling_policy_version": "v1",
  "dataprocessor_version": "dev",
  "chunk_size": 64
}
```

**Response** (200 OK):
```json
{
  "status": "queued",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Processing request accepted"
}
```

**Error Responses**:
- `400 Bad Request`: Невалидный payload
- `401 Unauthorized`: Отсутствует API Key
- `403 Forbidden`: Невалидный API Key
- `409 Conflict`: Run с таким ID уже существует
- `503 Service Unavailable`: Backpressure (очередь переполнена)

## Миграция

### Шаг 1: Настроить переменные окружения

```bash
export TF_BACKEND_DATAPROCESSOR_API_URL=http://dataprocessor:8000
export TF_BACKEND_DATAPROCESSOR_API_KEY=your-api-key
export TF_BACKEND_DATAPROCESSOR_POLL_INTERVAL=5
export TF_BACKEND_DATAPROCESSOR_TIMEOUT_SECONDS=3600
```

### Шаг 2: Обновить код

Заменить вызовы `run_dataprocessor()` на `run_dataprocessor_async()`:

```python
# Было
run_paths = run_dataprocessor(
    video_path=video_path,
    platform_id=platform_id,
    video_id=video_id,
    run_id=run_id,
    profile_config=profile_config,
    result_store_base=result_store_base,
    frames_dir_base=frames_dir_base,
    visual_cfg_default=visual_cfg_default,
)

# Стало
run_paths = await run_dataprocessor_async(
    video_path=video_path,
    platform_id=platform_id,
    video_id=video_id,
    run_id=run_id,
    profile_config=profile_config,
    result_store_base=result_store_base,
    frames_dir_base=frames_dir_base,
    visual_cfg_default=visual_cfg_default,
)
```

### Шаг 3: Обновить Celery задачи

Celery задачи должны быть async или использовать `asyncio.run()`:

```python
@celery_app.task
def process_analysis_job(analysis_job_id: str):
    # ...
    run_paths = asyncio.run(run_dataprocessor_async(...))
    # ...
```

## Breaking Changes

1. **Функция теперь async**: `run_dataprocessor_async()` требует `await`
2. **Требуется httpx**: Добавьте `httpx>=0.25.0` в `backend/requirements.txt`
3. **Требуется DataProcessor API**: DataProcessor API должен быть запущен и доступен

## Обратная совместимость

Старая функция `run_dataprocessor()` сохранена для обратной совместимости, но рекомендуется мигрировать на `run_dataprocessor_async()`.

## Связанные изменения

- [Этап 6.1: Замена subprocess на HTTP](../IMPLEMENTATION/2024-01-XX-stage-6.1-subprocess-to-http.md)
- [DataProcessor API Endpoint](../../endpoints/process.py)
- [Архитектура интеграции](../../ARCHITECTURE/backend-integration.md)

