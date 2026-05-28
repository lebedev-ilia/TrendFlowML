# Переменные окружения DataProcessor API

Этот документ описывает все переменные окружения, используемые DataProcessor API.

## Основные настройки API

### `API_HOST`
- **Описание**: Хост, на котором будет слушать API сервер
- **Тип**: `string`
- **По умолчанию**: `0.0.0.0`
- **Пример**: `API_HOST=0.0.0.0`

### `API_PORT`
- **Описание**: Порт, на котором будет слушать API сервер
- **Тип**: `integer`
- **По умолчанию**: `8000`
- **Пример**: `API_PORT=8000`

### `API_VERSION`
- **Описание**: Версия API
- **Тип**: `string`
- **По умолчанию**: `0.1.0`
- **Пример**: `API_VERSION=0.1.0`

### `DEBUG`
- **Описание**: Включить режим отладки (автоперезагрузка, детальные ошибки)
- **Тип**: `boolean`
- **По умолчанию**: `false`
- **Пример**: `DEBUG=true`

## Параллелизм и производительность

### `MAX_CONCURRENT_RUNS`
- **Описание**: Максимальное количество одновременных run'ов
- **Тип**: `integer`
- **По умолчанию**: `4`
- **Пример**: `MAX_CONCURRENT_RUNS=10`

### `MAX_QUEUE_LENGTH`
- **Описание**: Лимит длины очереди для backpressure. Если очередь превышает этот лимит, API возвращает 503 Service Unavailable
- **Тип**: `integer`
- **По умолчанию**: `100`
- **Пример**: `MAX_QUEUE_LENGTH=200`

### `SUBPROCESS_MEMORY_LIMIT_MB`
- **Описание**: Лимит памяти для subprocess в MB. Если установлен, subprocess будет убит при превышении лимита
- **Тип**: `integer` (опционально)
- **По умолчанию**: `None` (без лимита)
- **Пример**: `SUBPROCESS_MEMORY_LIMIT_MB=8000`

## Storage настройки

### `STORAGE_TYPE`
- **Описание**: Тип storage для хранения результатов
- **Тип**: `string` (`fs` или `s3`)
- **По умолчанию**: `fs`
- **Пример**: `STORAGE_TYPE=s3`

### `STORAGE_ROOT`
- **Описание**: Корневая директория для файловой системы storage (используется только при `STORAGE_TYPE=fs`)
- **Тип**: `string` (опционально)
- **По умолчанию**: Автоматически определяется относительно репозитория
- **Пример**: `STORAGE_ROOT=/data/storage`

### S3 настройки (для `STORAGE_TYPE=s3`)

#### `S3_ENDPOINT`
- **Описание**: Endpoint S3-совместимого хранилища
- **Тип**: `string`
- **Пример**: `S3_ENDPOINT=http://minio:9000`

#### `S3_BUCKET`
- **Описание**: Имя bucket для хранения результатов
- **Тип**: `string`
- **Пример**: `S3_BUCKET=trendflow`

#### `S3_PREFIX`
- **Описание**: Префикс для ключей в bucket
- **Тип**: `string`
- **Пример**: `S3_PREFIX=trendflowml`

#### `AWS_ACCESS_KEY_ID`
- **Описание**: Access key ID для S3
- **Тип**: `string`
- **Пример**: `AWS_ACCESS_KEY_ID=trendflow`

#### `AWS_SECRET_ACCESS_KEY`
- **Описание**: Secret access key для S3
- **Тип**: `string`
- **Пример**: `AWS_SECRET_ACCESS_KEY=trendflow123`

#### `AWS_DEFAULT_REGION`
- **Описание**: Регион по умолчанию для S3
- **Тип**: `string`
- **По умолчанию**: `us-east-1`
- **Пример**: `AWS_DEFAULT_REGION=us-east-1`

## Redis настройки

### `REDIS_URL`
- **Описание**: Полный URL подключения к Redis (приоритет над отдельными параметрами)
- **Тип**: `string` (опционально)
- **Пример**: `REDIS_URL=redis://localhost:6379/0`

### `REDIS_HOST`
- **Описание**: Хост Redis сервера
- **Тип**: `string` (опционально)
- **По умолчанию**: `localhost`
- **Пример**: `REDIS_HOST=redis`

### `REDIS_PORT`
- **Описание**: Порт Redis сервера
- **Тип**: `integer`
- **По умолчанию**: `6379`
- **Пример**: `REDIS_PORT=6379`

### `REDIS_DB`
- **Описание**: Номер базы данных Redis
- **Тип**: `integer`
- **По умолчанию**: `0`
- **Пример**: `REDIS_DB=0`

### `REDIS_PASSWORD`
- **Описание**: Пароль для подключения к Redis
- **Тип**: `string` (опционально)
- **Пример**: `REDIS_PASSWORD=secret`

## SSE (Server-Sent Events) настройки

### `MAX_SSE_CONNECTIONS_PER_RUN`
- **Описание**: Максимальное количество одновременных SSE соединений на run_id
- **Тип**: `integer`
- **По умолчанию**: `10`
- **Пример**: `MAX_SSE_CONNECTIONS_PER_RUN=20`

### `SSE_STREAM_READ_TIMEOUT`
- **Описание**: Таймаут для чтения новых событий из Redis Streams (мс)
- **Тип**: `integer`
- **По умолчанию**: `5000`
- **Пример**: `SSE_STREAM_READ_TIMEOUT=10000`

## Логирование

### `LOG_LEVEL`
- **Описание**: Уровень логирования
- **Тип**: `string` (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- **По умолчанию**: `INFO`
- **Пример**: `LOG_LEVEL=DEBUG`

### `LOG_FORMAT`
- **Описание**: Формат логов
- **Тип**: `string` (`json` или `text`)
- **По умолчанию**: `json`
- **Пример**: `LOG_FORMAT=text`

## CORS настройки

### `CORS_ORIGINS`
- **Описание**: Разрешённые origins для CORS (разделённые запятой)
- **Тип**: `string`
- **По умолчанию**: `*` (все origins)
- **Пример**: `CORS_ORIGINS=http://localhost:3000,https://example.com`

## Аутентификация

### `DATAPROCESSOR_API_KEY`
- **Описание**: API ключ для аутентификации запросов
- **Тип**: `string` (опционально)
- **Пример**: `DATAPROCESSOR_API_KEY=your-secret-api-key`

### `AUTH_TYPE`
- **Описание**: Тип аутентификации
- **Тип**: `string` (`api_key` или `mtls`)
- **По умолчанию**: `api_key`
- **Пример**: `AUTH_TYPE=api_key`

## Security настройки

### `ALLOWED_VIDEO_PATHS`
- **Описание**: Разрешённые директории для video_path (разделённые запятой)
- **Тип**: `string`
- **По умолчанию**: `/data/videos,/data/uploads`
- **Пример**: `ALLOWED_VIDEO_PATHS=/data/videos,/data/uploads,/mnt/storage`

### `AUDIT_LOG_ENABLED`
- **Описание**: Включить audit log для отслеживания действий
- **Тип**: `boolean`
- **По умолчанию**: `true`
- **Пример**: `AUDIT_LOG_ENABLED=true`

### `AUDIT_LOG_TTL`
- **Описание**: TTL для audit log записей в Redis (в секундах)
- **Тип**: `integer`
- **По умолчанию**: `2592000` (30 дней)
- **Пример**: `AUDIT_LOG_TTL=604800` (7 дней)

## Triton настройки (опционально)

### `TRITON_ENDPOINT`
- **Описание**: Endpoint Triton Inference Server
- **Тип**: `string` (опционально)
- **Пример**: `TRITON_ENDPOINT=http://triton:8000`

## OpenTelemetry Tracing (опционально)

### `ENABLE_TRACING`
- **Описание**: Включить distributed tracing
- **Тип**: `boolean`
- **По умолчанию**: `false`
- **Пример**: `ENABLE_TRACING=true`

### `TRACING_EXPORTER`
- **Описание**: Экспортер трейсов
- **Тип**: `string` (`jaeger` или `otlp`)
- **По умолчанию**: `jaeger`
- **Пример**: `TRACING_EXPORTER=jaeger`

### `JAEGER_AGENT_HOST`
- **Описание**: Хост Jaeger agent (используется при `TRACING_EXPORTER=jaeger`)
- **Тип**: `string`
- **По умолчанию**: `localhost`
- **Пример**: `JAEGER_AGENT_HOST=jaeger`

### `JAEGER_AGENT_PORT`
- **Описание**: Порт Jaeger agent (используется при `TRACING_EXPORTER=jaeger`)
- **Тип**: `integer`
- **По умолчанию**: `6831`
- **Пример**: `JAEGER_AGENT_PORT=6831`

### `OTLP_ENDPOINT`
- **Описание**: OTLP endpoint (используется при `TRACING_EXPORTER=otlp`)
- **Тип**: `string`
- **По умолчанию**: `http://localhost:4317`
- **Пример**: `OTLP_ENDPOINT=http://localhost:4317`

### `SERVICE_NAME`
- **Описание**: Имя сервиса для трейсов
- **Тип**: `string`
- **По умолчанию**: `dataprocessor-api`
- **Пример**: `SERVICE_NAME=dataprocessor-api`

### `SERVICE_VERSION`
- **Описание**: Версия сервиса для трейсов
- **Тип**: `string`
- **По умолчанию**: `0.1.0`
- **Пример**: `SERVICE_VERSION=0.1.0`

## Модели (опционально)

### `DP_MODELS_ROOT`
- **Описание**: Корневая директория для моделей ML
- **Тип**: `string`
- **Пример**: `DP_MODELS_ROOT=/app/models`

## Пример конфигурации

### Development

```bash
# .env для development
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
LOG_LEVEL=DEBUG
LOG_FORMAT=text
MAX_CONCURRENT_RUNS=2
STORAGE_TYPE=fs
CORS_ORIGINS=*
```

### Production

```bash
# .env для production
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json
MAX_CONCURRENT_RUNS=10
STORAGE_TYPE=s3
S3_ENDPOINT=http://minio:9000
S3_BUCKET=trendflow
S3_PREFIX=trendflowml
AWS_ACCESS_KEY_ID=trendflow
AWS_SECRET_ACCESS_KEY=trendflow123
REDIS_URL=redis://redis:6379/0
DATAPROCESSOR_API_KEY=your-secret-api-key
CORS_ORIGINS=https://app.example.com
ENABLE_TRACING=true
TRACING_EXPORTER=jaeger
JAEGER_AGENT_HOST=jaeger
```

## Примечания

- Все переменные окружения опциональны и имеют значения по умолчанию
- Переменные можно задавать через `.env` файл или через переменные окружения системы
- Приоритет: переменные окружения системы > `.env` файл > значения по умолчанию
- Для production рекомендуется использовать секреты из секретного хранилища (например, Kubernetes Secrets)

