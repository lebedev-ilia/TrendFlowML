# Backend Integration: Hybrid подход (Webhook + Polling Fallback)

**Дата**: 2024-01-XX  
**Версия API**: v1  
**Тип изменения**: Интеграция

## Описание изменения

Backend теперь использует production-grade hybrid подход для отслеживания статуса обработки:
- **Webhook endpoint** для получения real-time уведомлений от DataProcessor API
- **SSE listener** для streaming обновлений
- **Polling fallback** если webhook/SSE не работают

## Что изменилось

### Backend (`backend/app/routers/webhooks.py`)

**Новый endpoint**:
```python
POST /api/webhooks/dataprocessor
```

**Функциональность**:
- Валидация webhook signature через заголовок `X-Webhook-Signature`
- Обновление статуса AnalysisJob в БД
- Отправка WebSocket событий через Redis pubsub

**Payload**:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "success",
  "progress": {...},
  "error": null,
  "error_code": null,
  "stage": "visual",
  "component": "feature_extraction",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Backend (`backend/app/services/dataprocessor.py`)

**Новые функции**:

1. **`stream_run_events_sse()`** - SSE listener:
   - Подключение к SSE endpoint DataProcessor API
   - Парсинг SSE формата (event:, data:)
   - Обработка финальных событий (complete, error)

2. **`wait_for_run_completion_hybrid()`** - Hybrid подход:
   - Попытка 1: SSE listener для real-time обновлений
   - Fallback: Polling если SSE не работает
   - Graceful degradation при ошибках

### Backend (`backend/app/tasks.py`)

**Изменено**:
- Заменен `poll_run_status()` на `wait_for_run_completion_hybrid()`
- Использование SSE для real-time обновлений с fallback на polling

**Было**:
```python
final_status = asyncio.run(
    poll_run_status(
        run_id=payload.run_id,
        timeout_seconds=settings.dataprocessor_timeout_seconds,
        poll_interval=settings.dataprocessor_poll_interval
    )
)
```

**Стало**:
```python
final_status = asyncio.run(
    wait_for_run_completion_hybrid(
        run_id=payload.run_id,
        webhook_timeout=30,
        timeout_seconds=settings.dataprocessor_timeout_seconds,
        poll_interval=settings.dataprocessor_poll_interval
    )
)
```

## Детали

### Webhook signature

Для MVP используется простая проверка через API Key:
```python
expected_signature = settings.dataprocessor_api_key
return hmac.compare_digest(signature, expected_signature)
```

В production рекомендуется использовать HMAC-SHA256:
```python
expected_signature = hmac.new(
    settings.dataprocessor_api_key.encode(),
    payload_body,
    hashlib.sha256
).hexdigest()
```

### SSE формат

DataProcessor API отправляет события в SSE формате:
```
event: progress
data: {"run_id": "...", "progress": 0.5, ...}

event: complete
data: {"run_id": "...", "status": "success", ...}
```

### Hybrid стратегия

1. **SSE listener** (приоритет):
   - Real-time обновления
   - Меньше нагрузка на API
   - Лучший UX

2. **Polling fallback**:
   - Надежность
   - Работает даже если SSE недоступен
   - Простота реализации

## Миграция

### Шаг 1: Обновить код

Код уже обновлен в `backend/app/tasks.py`. Hybrid подход используется автоматически.

### Шаг 2: Настроить webhook URL (TODO)

DataProcessor API должен знать URL webhook'а для отправки уведомлений:
- Добавить поле `webhook_url` в ProcessRequest
- DataProcessor API должен отправлять webhook'и по этому URL

### Шаг 3: Проверить работу

1. Создать задачу обработки через API backend
2. Проверить что SSE stream работает (если доступен)
3. Проверить что polling fallback работает (если SSE недоступен)
4. Проверить webhook endpoint (если DataProcessor API отправляет webhook'и)

## Breaking Changes

Нет breaking changes. Hybrid подход обратно совместим с polling.

## Обратная совместимость

- Polling fallback обеспечивает обратную совместимость
- Если SSE недоступен, используется polling (как раньше)
- Webhook опционален (не критичен для работы)

## Преимущества

1. **Real-time обновления**: SSE обеспечивает обновления в реальном времени
2. **Надежность**: Polling fallback гарантирует работу даже при ошибках SSE
3. **Меньше нагрузка**: SSE уменьшает количество HTTP запросов
4. **Лучший UX**: Real-time обновления улучшают пользовательский опыт

## Недостатки

1. **Сложность**: Hybrid подход сложнее чем простой polling
2. **Webhook URL**: Требуется настройка webhook URL в DataProcessor API (TODO)
3. **SSE timeout**: SSE stream может зависнуть при длительных обработках

## Связанные изменения

- [Этап 6.1: Замена subprocess на HTTP](../IMPLEMENTATION/2024-01-XX-stage-6.1-subprocess-to-http.md)
- [Этап 6.2: Polling для статуса](../IMPLEMENTATION/2024-01-XX-stage-6.2-polling-status.md)
- [Этап 6.3: Hybrid подход](../IMPLEMENTATION/2024-01-XX-stage-6.3-hybrid-webhook-polling.md)
- [DataProcessor API SSE Endpoint](../../endpoints/runs.py)
---

## Навигация

[README](README.md) · [DataProcessor](../../../docs/MAIN_INDEX.md) · [Vault](../../../../docs/MAIN_INDEX.md)
