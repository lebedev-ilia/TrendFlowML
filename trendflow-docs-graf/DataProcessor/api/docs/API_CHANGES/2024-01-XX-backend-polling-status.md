# Backend Integration: Polling для статуса

**Дата**: 2024-01-XX  
**Версия API**: v1  
**Тип изменения**: Интеграция

## Описание изменения

Backend теперь использует polling для отслеживания статуса обработки run'а до завершения. Это заменяет ожидание завершения subprocess на периодические запросы к DataProcessor API.

## Что изменилось

### Backend (`backend/app/tasks.py`)

**Было** (subprocess):
```python
proc = subprocess.Popen(cmd, ...)
exit_code = proc.wait()
if exit_code != 0:
    # Обработка ошибки
```

**Стало** (HTTP + polling):
```python
# Отправить запрос на обработку
run_paths = asyncio.run(run_dataprocessor_async(...))

# Polling статуса до завершения
final_status = asyncio.run(poll_run_status(...))

# Обработать финальный статус
if status == "success":
    # Успешное завершение
elif status == "error":
    # Ошибка обработки
```

### Новая функция (`backend/app/services/dataprocessor.py`)

**Добавлена функция**:
```python
async def poll_run_status(
    run_id: str,
    timeout_seconds: Optional[int] = None,
    poll_interval: Optional[int] = None
) -> Dict[str, Any]:
```

## Детали

### Endpoint

**URL**: `GET /api/v1/runs/{run_id}/status`

**Headers**:
- `X-API-Key: <api_key>` (опционально, если настроена аутентификация)

**Response** (200 OK):
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_id": "dQw4w9WgXcQ",
  "platform_id": "youtube",
  "status": "running",
  "progress": {
    "overall": 0.5,
    "current_processor": "visual",
    "components": {
      "segmenter": {"status": "success", "progress": 1.0}
    }
  },
  "started_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:05:00Z"
}
```

**Финальные статусы**:
- `success` - обработка завершена успешно
- `error` - обработка завершена с ошибкой
- `empty` - обработка завершена, но результат пустой
- `skipped` - обработка пропущена
- `cancelled` - обработка отменена

### Параметры polling

- **poll_interval**: Интервал между запросами (по умолчанию: 5 секунд)
- **timeout_seconds**: Максимальное время ожидания (по умолчанию: 3600 секунд = 1 час)

### Обработка ошибок

1. **TimeoutError**: Если обработка не завершилась за timeout_seconds
   - Backend обновляет статус на `failed` с error_code="timeout"

2. **ValueError**: Если run не найден (404)
   - Backend обновляет статус на `failed` с error_code="validation_error"

3. **HTTP ошибки**: Другие ошибки HTTP
   - Backend обновляет статус на `failed` с error_code="api_error"

4. **Transient ошибки**: Ошибки соединения
   - Polling продолжается, ошибки логируются

## Миграция

### Шаг 1: Обновить код

Код уже обновлен в `backend/app/tasks.py`. Старый subprocess код удален.

### Шаг 2: Настроить переменные окружения

```bash
export TF_BACKEND_DATAPROCESSOR_API_URL=http://dataprocessor:8000
export TF_BACKEND_DATAPROCESSOR_API_KEY=your-api-key
export TF_BACKEND_DATAPROCESSOR_POLL_INTERVAL=5
export TF_BACKEND_DATAPROCESSOR_TIMEOUT_SECONDS=3600
```

### Шаг 3: Проверить работу

1. Создать задачу обработки через API backend
2. Проверить логи на наличие polling запросов
3. Убедиться что статус обновляется корректно

## Breaking Changes

1. **Удален subprocess код**: Больше не используется прямой вызов DataProcessor
2. **Удален state events tailer**: Больше не tail'ится state_events.jsonl
3. **Требуется DataProcessor API**: DataProcessor API должен быть запущен и доступен

## Обратная совместимость

Старая функция `run_dataprocessor()` сохранена для обратной совместимости, но не используется в Celery задаче.

## Преимущества

1. **Разделение сервисов**: Backend и DataProcessor разделены
2. **Масштабируемость**: DataProcessor может масштабироваться независимо
3. **Мониторинг**: Легче отслеживать статус через API
4. **Обработка ошибок**: Улучшенная обработка ошибок через HTTP

## Недостатки

1. **Polling overhead**: Периодические запросы создают нагрузку
2. **Задержка обновлений**: Обновления не в реальном времени (5 секунд задержка)
3. **Нет логов subprocess**: Логи stdout/stderr больше не доступны

## Связанные изменения

- [Этап 6.1: Замена subprocess на HTTP](../IMPLEMENTATION/2024-01-XX-stage-6.1-subprocess-to-http.md)
- [Этап 6.2: Polling для статуса](../IMPLEMENTATION/2024-01-XX-stage-6.2-polling-status.md)
- [DataProcessor API Status Endpoint](../../endpoints/runs.py)
---

## Навигация

[README](README.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
