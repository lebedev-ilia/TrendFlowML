# Local Development Environment для Fetcher

Руководство по настройке и использованию local development окружения для Fetcher.

## Обзор

Local dev окружение включает:
- **PostgreSQL** - база данных Fetcher
- **Redis** - для rate limiting, locks и очередей Celery
- **MinIO** - S3-совместимое object storage
- **Fetcher API** - HTTP API сервис
- **Celery Worker** - воркеры для обработки задач
- **Celery Beat** - планировщик периодических задач

## Требования

- Docker и Docker Compose
- Python 3.11+ (для локальной разработки без Docker)

## Быстрый старт

### 1. Запуск всех сервисов

```bash
docker-compose up -d
```

Это запустит:
- PostgreSQL на порту 5432
- Redis на порту 6379
- MinIO на портах 9000 (API) и 9001 (Console)
- Fetcher API на порту 8000
- Celery Worker
- Celery Beat

### 2. Проверка статуса

```bash
docker-compose ps
```

### 3. Просмотр логов

```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f fetcher-api
docker-compose logs -f fetcher-worker
```

### 4. Остановка сервисов

```bash
docker-compose down
```

### 5. Очистка данных

```bash
# Остановить и удалить volumes
docker-compose down -v
```

## Настройка БД

### Создание миграций Alembic

```bash
# Создать новую миграцию
alembic revision --autogenerate -m "Initial migration"

# Применить миграции
alembic upgrade head

# Откатить последнюю миграцию
alembic downgrade -1

# Просмотр текущей версии
alembic current

# Просмотр истории миграций
alembic history
```

### Первоначальная настройка

```bash
# 1. Применить миграции
alembic upgrade head

# 2. Создать bucket в MinIO (через консоль или API)
# MinIO Console: http://localhost:9001
# Login: minioadmin / minioadmin123
# Создать bucket: video-analytics-raw
```

## Переменные окружения

Переменные окружения настраиваются в `docker-compose.yml`. Для локальной разработки можно создать `.env` файл:

```bash
# .env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=fetcher
POSTGRES_PASSWORD=fetcher_password
POSTGRES_DB=fetcher_db
POSTGRES_SSL_MODE=disable

REDIS_URL=redis://localhost:6379/0
REDIS_SSL=false

S3_ENDPOINT_URL=http://localhost:9000
S3_ACCESS_KEY_ID=minioadmin
S3_SECRET_ACCESS_KEY=minioadmin123
S3_BUCKET_RAW=video-analytics-raw
S3_USE_SSL=false
S3_VERIFY_SSL=false

CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

## Локальная разработка (без Docker)

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск сервисов вручную

```bash
# PostgreSQL (если не используете Docker)
# Убедитесь, что PostgreSQL запущен и доступен

# Redis (если не используете Docker)
# Убедитесь, что Redis запущен и доступен

# MinIO (если не используете Docker)
# Запустите MinIO локально или используйте Docker только для MinIO
```

### 3. Запуск Fetcher API

```bash
uvicorn fetcher.api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Запуск Celery Worker

```bash
celery -A fetcher.celery_app worker --loglevel=info --concurrency=4
```

### 5. Запуск Celery Beat

```bash
celery -A fetcher.celery_app beat --loglevel=info
```

## Доступ к сервисам

### PostgreSQL

```bash
# Подключение через psql
psql -h localhost -U fetcher -d fetcher_db

# Пароль: fetcher_password
```

### Redis

```bash
# Подключение через redis-cli
redis-cli -h localhost -p 6379
```

### MinIO

- **API**: http://localhost:9000
- **Console**: http://localhost:9001
- **Login**: minioadmin / minioadmin123

### Fetcher API

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **API Docs**: http://localhost:8000/docs

## Тестирование

### Запуск load test

```bash
# Используя docker-compose окружение
python scripts/load_test.py --target 100 --duration 3600
```

### Проверка валидации

```bash
curl http://localhost:8000/admin/validation
```

## Отладка

### Просмотр логов

```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f fetcher-api
docker-compose logs -f fetcher-worker
docker-compose logs -f postgres
docker-compose logs -f redis
docker-compose logs -f minio
```

### Подключение к контейнерам

```bash
# Fetcher API
docker-compose exec fetcher-api bash

# PostgreSQL
docker-compose exec postgres psql -U fetcher -d fetcher_db

# Redis
docker-compose exec redis redis-cli
```

## Известные проблемы

### Проблемы с подключением к БД

Если возникают проблемы с подключением к PostgreSQL:

1. Проверьте, что PostgreSQL запущен: `docker-compose ps`
2. Проверьте логи: `docker-compose logs postgres`
3. Убедитесь, что переменные окружения корректны

### Проблемы с MinIO

Если возникают проблемы с MinIO:

1. Проверьте, что MinIO запущен: `docker-compose ps`
2. Проверьте логи: `docker-compose logs minio`
3. Убедитесь, что bucket создан: http://localhost:9001

### Проблемы с Celery

Если задачи не выполняются:

1. Проверьте, что Celery Worker запущен: `docker-compose ps`
2. Проверьте логи: `docker-compose logs fetcher-worker`
3. Проверьте подключение к Redis: `docker-compose logs redis`

## Следующие шаги

- Настройка IDE для разработки
- Настройка pre-commit hooks
- Настройка CI/CD для автоматического тестирования

