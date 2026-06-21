# Тесты для DataProcessor API

## Структура

```
api/tests/
├── __init__.py
├── conftest.py          # Общие фикстуры и конфигурация
├── unit/                # Unit тесты
│   ├── test_state_machine.py
│   ├── test_task_manager.py
│   └── test_error_handling.py
└── integration/         # Integration тесты
    ├── test_process_endpoint.py
    └── test_health_endpoint.py
```

## Установка зависимостей

```bash
pip install -r requirements-test.txt
```

## Запуск тестов

### Все тесты
```bash
pytest
```

### Только unit тесты
```bash
pytest api/tests/unit/
```

### Только integration тесты
```bash
pytest api/tests/integration/
```

### С coverage отчетом
```bash
pytest --cov=api --cov-report=html
```

Отчет будет доступен в `htmlcov/index.html`.

### С маркерами
```bash
# Только unit тесты
pytest -m unit

# Только integration тесты
pytest -m integration

# Тесты, требующие Redis
pytest -m redis

# Медленные тесты
pytest -m slow
```

## Покрытие кода

Цель покрытия: **>80%**

Текущее покрытие можно проверить:
```bash
pytest --cov=api --cov-report=term-missing
```

## Написание новых тестов

### Unit тесты

Unit тесты должны быть быстрыми и не требовать внешних зависимостей (Redis, Storage).

Пример:
```python
import pytest
from api.services.state_machine import can_transition
from api.schemas.state import RunStatus

def test_pending_to_queued():
    assert can_transition(RunStatus.PENDING, RunStatus.QUEUED) is True
```

### Integration тесты

Integration тесты могут использовать моки для внешних зависимостей.

Пример:
```python
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

def test_process_endpoint(client):
    with patch("api.endpoints.process.enqueue_run", new_callable=AsyncMock):
        response = client.post("/api/v1/process", json=payload)
        assert response.status_code == 202
```

## Фикстуры

Доступные фикстуры в `conftest.py`:

- `api_config` - Конфигурация API
- `mock_storage` - Мок Storage
- `mock_key_layout` - Мок KeyLayout
- `task_manager` - TaskManager instance
- `client` - FastAPI TestClient
- `mock_redis_client` - Мок Redis клиента
- `sample_process_request` - Пример ProcessRequest

## Известные ограничения

1. **Redis тесты**: Требуют fakeredis или реальный Redis instance
2. **Storage тесты**: Требуют моки или временную файловую систему
3. **Worker тесты**: Требуют изоляции процессов (будут добавлены позже)
---

## Навигация

[DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
