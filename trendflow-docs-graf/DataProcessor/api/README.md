# DataProcessor API

HTTP API для управления обработкой видео через DataProcessor. Предоставляет REST API для запуска обработки, отслеживания статуса, получения результатов и управления задачами.

## Быстрый старт

### Локальный запуск

```bash
# 1. Установка зависимостей
pip install -r requirements-api.txt

# 2. Настройка переменных окружения (опционально)
cp env.example .env
# Отредактируйте .env при необходимости

# 3. Запуск API сервера
python -m api.main

# Или через uvicorn напрямую
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose

```bash
# Запуск всех сервисов (API, Worker, Redis)
docker-compose up -d

# Просмотр логов
docker-compose logs -f dataprocessor-api

# Остановка
docker-compose down
```

### Доступ к API

После запуска API доступен по адресу:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Структура проекта

```
api/
├── __init__.py          # Модуль API
├── main.py              # FastAPI приложение и точка входа
├── config.py            # Настройки API сервера
├── dependencies.py      # FastAPI dependencies
├── security.py          # Аутентификация и авторизация
│
├── endpoints/           # API endpoints
│   ├── __init__.py
│   ├── process.py       # POST /api/v1/process
│   ├── runs.py          # GET /api/v1/runs/{run_id}/*
│   ├── health.py        # GET /api/v1/health
│   ├── artifacts.py     # GET /api/v1/runs/{run_id}/artifacts/*
│   ├── cancel.py        # POST /api/v1/runs/{run_id}/cancel
│   ├── metrics.py       # GET /api/v1/metrics
│   └── retention.py     # POST /api/v1/admin/retention/cleanup
│
├── schemas/             # Pydantic models
│   ├── __init__.py
│   ├── requests.py      # Request models
│   ├── responses.py     # Response models
│   └── state.py         # State models
│
├── services/            # Бизнес-логика
│   ├── __init__.py
│   ├── processor.py     # Интеграция с main.py
│   ├── state_reader.py  # Чтение state из storage
│   ├── task_manager.py  # Управление задачами
│   ├── queue.py         # Redis Streams queue
│   ├── metrics.py       # Prometheus метрики
│   └── tracing.py       # OpenTelemetry tracing
│
├── middleware/          # Middleware
│   └── request_id.py   # Request ID middleware
│
└── utils/               # Утилиты
    ├── __init__.py
    ├── errors.py        # Кастомные исключения
    └── validators.py    # Валидация payload
```

## Основные возможности

### 1. Обработка видео

**POST** `/api/v1/process` - Запуск обработки видео

```bash
curl -X POST "http://localhost:8000/api/v1/process" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "run_id": "unique-run-id",
    "video_id": "video-123",
    "platform_id": "youtube",
    "video_path": "/path/to/video.mp4",
    "profile_config": {
      "visual": {"enabled": true},
      "audio": {"enabled": true},
      "text": {"enabled": true}
    }
  }'
```

**Ответ**: `202 Accepted` с `run_id` и статусом `queued`

### 2. Отслеживание статуса

**GET** `/api/v1/runs/{run_id}/status` - Получение статуса обработки

```bash
curl "http://localhost:8000/api/v1/runs/{run_id}/status" \
  -H "X-API-Key: your-api-key"
```

**Ответ**: JSON с текущим статусом, прогрессом и метаданными

### 3. События в реальном времени

**GET** `/api/v1/runs/{run_id}/events` - Server-Sent Events (SSE) стрим

```bash
curl "http://localhost:8000/api/v1/runs/{run_id}/events" \
  -H "X-API-Key: your-api-key" \
  -N
```

**Ответ**: Поток событий в формате SSE с обновлениями статуса и прогресса

### 4. Получение результатов

**GET** `/api/v1/runs/{run_id}/manifest` - Получение manifest.json

```bash
curl "http://localhost:8000/api/v1/runs/{run_id}/manifest" \
  -H "X-API-Key: your-api-key"
```

**GET** `/api/v1/runs/{run_id}/artifacts/{artifact_path}` - Получение артефактов

```bash
curl "http://localhost:8000/api/v1/runs/{run_id}/artifacts/visual.npz" \
  -H "X-API-Key: your-api-key" \
  -o visual.npz
```

### 5. Отмена обработки

**POST** `/api/v1/runs/{run_id}/cancel` - Отмена активной обработки

```bash
curl -X POST "http://localhost:8000/api/v1/runs/{run_id}/cancel" \
  -H "X-API-Key: your-api-key"
```

### 6. Health Check

**GET** `/api/v1/health` - Проверка состояния API

```bash
curl "http://localhost:8000/api/v1/health"
```

**Ответ**: JSON с статусом API и зависимостей

### 7. Метрики Prometheus

**GET** `/api/v1/metrics` - Prometheus метрики

```bash
curl "http://localhost:8000/api/v1/metrics"
```

## Переменные окружения

См. [Документация переменных окружения](docs/ENVIRONMENT_VARIABLES.md) для полного списка.

### Основные переменные

**API настройки**:
- `API_HOST` - хост API сервера (по умолчанию: `0.0.0.0`)
- `API_PORT` - порт API сервера (по умолчанию: `8000`)
- `API_VERSION` - версия API (по умолчанию: `0.1.0`)
- `DEBUG` - режим отладки (по умолчанию: `false`)

**Параллелизм**:
- `MAX_CONCURRENT_RUNS` - максимальное количество одновременных run'ов (по умолчанию: `4`)
- `MAX_QUEUE_LENGTH` - лимит длины очереди для backpressure (по умолчанию: `100`)

**Storage**:
- `STORAGE_TYPE` - тип storage (`fs` или `s3`, по умолчанию: `fs`)
- `STORAGE_ROOT` - корневая директория для файловой системы storage

**Redis**:
- `REDIS_URL` - URL подключения к Redis (например: `redis://localhost:6379/0`)
- `REDIS_HOST` - хост Redis (по умолчанию: `localhost`)
- `REDIS_PORT` - порт Redis (по умолчанию: `6379`)

**Аутентификация**:
- `DATAPROCESSOR_API_KEY` - API ключ для аутентификации
- `AUTH_TYPE` - тип аутентификации (`api_key` или `mtls`, по умолчанию: `api_key`)

**Логирование**:
- `LOG_LEVEL` - уровень логирования (`DEBUG`, `INFO`, `WARNING`, `ERROR`, по умолчанию: `INFO`)
- `LOG_FORMAT` - формат логов (`json` или `text`, по умолчанию: `json`)

**CORS**:
- `CORS_ORIGINS` - разрешённые origins для CORS (разделённые запятой, по умолчанию: `*`)

**OpenTelemetry Tracing** (опционально):
- `ENABLE_TRACING` - включить distributed tracing (по умолчанию: `false`)
- `TRACING_EXPORTER` - экспортер трейсов (`jaeger` или `otlp`, по умолчанию: `jaeger`)
- `JAEGER_AGENT_HOST` - хост Jaeger agent (по умолчанию: `localhost`)
- `JAEGER_AGENT_PORT` - порт Jaeger agent (по умолчанию: `6831`)

## Примеры использования

См. [Примеры использования](docs/EXAMPLES.md) для подробных примеров.

### Python

```python
import httpx

async with httpx.AsyncClient() as client:
    # Запуск обработки
    response = await client.post(
        "http://localhost:8000/api/v1/process",
        headers={"X-API-Key": "your-api-key"},
        json={
            "run_id": "run-123",
            "video_id": "video-456",
            "platform_id": "youtube",
            "video_path": "/path/to/video.mp4",
            "profile_config": {
                "visual": {"enabled": True},
                "audio": {"enabled": True},
                "text": {"enabled": True}
            }
        }
    )
    run_id = response.json()["run_id"]
    
    # Отслеживание статуса
    while True:
        status_response = await client.get(
            f"http://localhost:8000/api/v1/runs/{run_id}/status",
            headers={"X-API-Key": "your-api-key"}
        )
        status = status_response.json()
        
        if status["status"] in ["success", "error", "cancelled"]:
            break
        
        await asyncio.sleep(5)
    
    # Получение результатов
    manifest_response = await client.get(
        f"http://localhost:8000/api/v1/runs/{run_id}/manifest",
        headers={"X-API-Key": "your-api-key"}
    )
    manifest = manifest_response.json()
```

### JavaScript/TypeScript

```typescript
const API_URL = 'http://localhost:8000';
const API_KEY = 'your-api-key';

// Запуск обработки
const processResponse = await fetch(`${API_URL}/api/v1/process`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
  },
  body: JSON.stringify({
    run_id: 'run-123',
    video_id: 'video-456',
    platform_id: 'youtube',
    video_path: '/path/to/video.mp4',
    profile_config: {
      visual: { enabled: true },
      audio: { enabled: true },
      text: { enabled: true }
    }
  })
});

const { run_id } = await processResponse.json();

// Отслеживание статуса через SSE
const eventSource = new EventSource(
  `${API_URL}/api/v1/runs/${run_id}/events?api_key=${API_KEY}`
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Status update:', data);
  
  if (data.status === 'success' || data.status === 'error') {
    eventSource.close();
  }
};
```

## API Документация

### Swagger UI

Интерактивная документация доступна по адресу:
- **Swagger UI**: http://localhost:8000/docs

### ReDoc

Альтернативная документация:
- **ReDoc**: http://localhost:8000/redoc

### OpenAPI Schema

JSON схема OpenAPI:
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Мониторинг

### Prometheus метрики

Метрики доступны по адресу:
- **Metrics**: http://localhost:8000/api/v1/metrics

### Grafana дашборды

См. [Мониторинг](../monitoring/README.md) для инструкций по настройке Grafana.

### Distributed Tracing

См. [Мониторинг](../monitoring/README.md) для инструкций по настройке OpenTelemetry/Jaeger.

## Troubleshooting

См. [Troubleshooting Guide](docs/TROUBLESHOOTING.md) для решения распространённых проблем.

## Документация

- **Архитектура**: `DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md`
- **Чеклист разработки**: `DataProcessor/docs/API_DEVELOPMENT_CHECKLIST.md`
- **Документация изменений**: `api/docs/`
  - [Индекс документации](docs/INDEX.md)
  - [Журнал изменений](docs/CHANGELOG.md)
  - [Отчеты о реализации](docs/IMPLEMENTATION/)
  - [Изменения в API](docs/API_CHANGES/)
  - [Архитектурные изменения](docs/ARCHITECTURE/)

## Лицензия

Proprietary
---

## Навигация

[DataProcessor](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
