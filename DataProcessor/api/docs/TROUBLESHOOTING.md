# Troubleshooting Guide

Руководство по решению распространённых проблем при работе с DataProcessor API.

## Содержание

1. [Проблемы с запуском](#проблемы-с-запуском)
2. [Проблемы с подключением](#проблемы-с-подключением)
3. [Проблемы с обработкой](#проблемы-с-обработкой)
4. [Проблемы с производительностью](#проблемы-с-производительностью)
5. [Проблемы с зависимостями](#проблемы-с-зависимостями)
6. [Логи и диагностика](#логи-и-диагностика)

## Проблемы с запуском

### API сервер не запускается

**Симптомы**:
- Ошибка при запуске `python -m api.main`
- Порт уже занят

**Решения**:

1. **Проверить, что порт свободен**:
   ```bash
   # Linux/Mac
   lsof -i :8000
   # Windows
   netstat -ano | findstr :8000
   ```

2. **Изменить порт**:
   ```bash
   export API_PORT=8001
   python -m api.main
   ```

3. **Проверить зависимости**:
   ```bash
   pip install -r requirements-api.txt
   ```

4. **Проверить переменные окружения**:
   ```bash
   # Проверить .env файл
   cat .env
   
   # Или установить переменные вручную
   export API_HOST=0.0.0.0
   export API_PORT=8000
   ```

### Ошибка импорта модулей

**Симптомы**:
- `ModuleNotFoundError: No module named 'api'`
- `ImportError: cannot import name 'X'`

**Решения**:

1. **Убедиться, что вы находитесь в правильной директории**:
   ```bash
   # Должны быть в корне DataProcessor
   cd /path/to/DataProcessor
   python -m api.main
   ```

2. **Проверить PYTHONPATH**:
   ```bash
   export PYTHONPATH=/path/to/DataProcessor:$PYTHONPATH
   python -m api.main
   ```

3. **Установить зависимости**:
   ```bash
   pip install -r requirements-api.txt
   ```

## Проблемы с подключением

### 401 Unauthorized

**Симптомы**:
- Запросы возвращают `401 Unauthorized`
- Сообщение "Authentication required"

**Решения**:

1. **Проверить API Key**:
   ```bash
   # Проверить переменную окружения
   echo $DATAPROCESSOR_API_KEY
   
   # Или в .env файле
   grep DATAPROCESSOR_API_KEY .env
   ```

2. **Передать API Key в запросе**:
   ```bash
   curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/health
   ```

3. **Проверить, что API Key установлен в API сервере**:
   ```bash
   # В логах API должно быть:
   # "API Key authentication enabled"
   ```

### 403 Forbidden

**Симптомы**:
- Запросы возвращают `403 Forbidden`
- Сообщение "Invalid API key"

**Решения**:

1. **Проверить правильность API Key**:
   - API Key должен совпадать с `DATAPROCESSOR_API_KEY` в API сервере

2. **Проверить формат заголовка**:
   ```bash
   # Правильно
   curl -H "X-API-Key: your-api-key" ...
   
   # Неправильно
   curl -H "Authorization: Bearer your-api-key" ...
   ```

### 503 Service Unavailable

**Симптомы**:
- Запросы возвращают `503 Service Unavailable`
- Сообщение "Service overloaded"

**Решения**:

1. **Проверить длину очереди**:
   ```bash
   curl http://localhost:8000/api/v1/health | jq '.queue.length'
   ```

2. **Подождать и повторить запрос**:
   ```bash
   # Проверить заголовок Retry-After
   curl -I http://localhost:8000/api/v1/process
   # Retry-After: 60
   ```

3. **Увеличить лимит очереди** (если возможно):
   ```bash
   export MAX_QUEUE_LENGTH=200
   ```

## Проблемы с обработкой

### Run остаётся в статусе "queued"

**Симптомы**:
- Run не переходит в статус "running"
- Очередь не обрабатывается

**Решения**:

1. **Проверить, что Worker запущен**:
   ```bash
   # Docker Compose
   docker-compose ps dataprocessor-worker
   
   # Локально
   ps aux | grep worker.py
   ```

2. **Проверить Redis подключение**:
   ```bash
   curl http://localhost:8000/api/v1/health | jq '.dependencies.redis'
   ```

3. **Проверить логи Worker**:
   ```bash
   docker-compose logs -f dataprocessor-worker
   ```

4. **Проверить лимит параллелизма**:
   ```bash
   # Проверить активные run'ы
   curl http://localhost:8000/api/v1/health | jq '.queue.active_runs'
   
   # Если достигнут лимит, подождать или увеличить MAX_CONCURRENT_RUNS
   ```

### Run падает с ошибкой

**Симптомы**:
- Run переходит в статус "error"
- В статусе есть `error_message`

**Решения**:

1. **Проверить error_message в статусе**:
   ```bash
   curl http://localhost:8000/api/v1/runs/{run_id}/status | jq '.error_message'
   ```

2. **Проверить логи Worker**:
   ```bash
   docker-compose logs dataprocessor-worker | grep {run_id}
   ```

3. **Проверить доступность видео файла**:
   ```bash
   # Проверить, что файл существует и доступен
   ls -la /path/to/video.mp4
   ```

4. **Проверить Storage**:
   ```bash
   curl http://localhost:8000/api/v1/health | jq '.dependencies.storage'
   ```

### Run не завершается (зависает)

**Симптомы**:
- Run остаётся в статусе "running" долгое время
- Нет обновлений прогресса

**Решения**:

1. **Проверить heartbeat**:
   ```bash
   # Проверить последнее обновление статуса
   curl http://localhost:8000/api/v1/runs/{run_id}/status | jq '.updated_at'
   ```

2. **Проверить логи Worker**:
   ```bash
   docker-compose logs -f dataprocessor-worker | grep {run_id}
   ```

3. **Отменить run и перезапустить**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/runs/{run_id}/cancel \
     -H "X-API-Key: your-api-key"
   ```

4. **Проверить ресурсы системы**:
   ```bash
   # CPU и память
   docker stats dataprocessor-worker
   ```

## Проблемы с производительностью

### Медленная обработка

**Симптомы**:
- Обработка занимает больше времени, чем ожидалось
- Низкая пропускная способность

**Решения**:

1. **Проверить метрики Prometheus**:
   ```bash
   curl http://localhost:8000/api/v1/metrics | grep processing_time
   ```

2. **Увеличить параллелизм** (если ресурсы позволяют):
   ```bash
   export MAX_CONCURRENT_RUNS=10
   ```

3. **Проверить использование ресурсов**:
   ```bash
   docker stats dataprocessor-worker
   ```

4. **Проверить Triton** (если используется):
   ```bash
   curl http://localhost:8000/api/v1/health | jq '.dependencies.triton'
   ```

### Высокое использование памяти

**Симптомы**:
- OOM (Out of Memory) ошибки
- Контейнер перезапускается

**Решения**:

1. **Установить лимит памяти для subprocess**:
   ```bash
   export SUBPROCESS_MEMORY_LIMIT_MB=8000
   ```

2. **Уменьшить параллелизм**:
   ```bash
   export MAX_CONCURRENT_RUNS=2
   ```

3. **Увеличить лимиты Docker**:
   ```yaml
   # docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 16G
   ```

## Проблемы с зависимостями

### Redis недоступен

**Симптомы**:
- Health check показывает `redis.status: "unhealthy"`
- Ошибки подключения к Redis

**Решения**:

1. **Проверить, что Redis запущен**:
   ```bash
   docker-compose ps redis
   # Или
   redis-cli ping
   ```

2. **Проверить URL подключения**:
   ```bash
   echo $REDIS_URL
   # Должно быть: redis://localhost:6379/0
   ```

3. **Проверить сеть Docker**:
   ```bash
   # Если в docker-compose, использовать имя сервиса
   REDIS_URL=redis://redis:6379/0
   ```

4. **Перезапустить Redis**:
   ```bash
   docker-compose restart redis
   ```

### Storage недоступен

**Симптомы**:
- Health check показывает `storage.status: "unhealthy"`
- Ошибки при чтении/записи файлов

**Решения**:

1. **Проверить тип Storage**:
   ```bash
   echo $STORAGE_TYPE
   # fs или s3
   ```

2. **Для файловой системы**:
   ```bash
   # Проверить права доступа
   ls -la $STORAGE_ROOT
   
   # Проверить, что директория существует
   mkdir -p $STORAGE_ROOT
   ```

3. **Для S3**:
   ```bash
   # Проверить переменные окружения
   echo $S3_ENDPOINT
   echo $S3_BUCKET
   echo $AWS_ACCESS_KEY_ID
   
   # Проверить подключение
   aws s3 ls s3://$S3_BUCKET --endpoint-url $S3_ENDPOINT
   ```

### Triton недоступен

**Симптомы**:
- Health check показывает `triton.status: "unhealthy"`
- Ошибки при обращении к моделям

**Решения**:

1. **Проверить, что Triton запущен**:
   ```bash
   docker-compose ps triton
   ```

2. **Проверить endpoint**:
   ```bash
   echo $TRITON_ENDPOINT
   # Должно быть: http://triton:8000
   ```

3. **Проверить доступность**:
   ```bash
   curl http://triton:8000/v2/health/ready
   ```

4. **Triton опционален** - API будет работать без него, но некоторые модели могут быть недоступны

## Логи и диагностика

### Просмотр логов

**Docker Compose**:
```bash
# Все сервисы
docker-compose logs -f

# Только API
docker-compose logs -f dataprocessor-api

# Только Worker
docker-compose logs -f dataprocessor-worker

# Последние 100 строк
docker-compose logs --tail=100 dataprocessor-api
```

**Локально**:
```bash
# Логи выводятся в stdout/stderr
# Для JSON формата:
export LOG_FORMAT=json

# Для текстового формата:
export LOG_FORMAT=text
```

### Уровни логирования

```bash
# DEBUG - детальные логи
export LOG_LEVEL=DEBUG

# INFO - стандартные логи
export LOG_LEVEL=INFO

# WARNING - только предупреждения и ошибки
export LOG_LEVEL=WARNING
```

### Метрики Prometheus

```bash
# Получить все метрики
curl http://localhost:8000/api/v1/metrics

# Конкретные метрики
curl http://localhost:8000/api/v1/metrics | grep queue_length
curl http://localhost:8000/api/v1/metrics | grep processing_time
```

### Health Check

```bash
# Полный health check
curl http://localhost:8000/api/v1/health | jq '.'

# Только статус
curl http://localhost:8000/api/v1/health | jq '.status'

# Только зависимости
curl http://localhost:8000/api/v1/health | jq '.dependencies'
```

## Часто задаваемые вопросы

### Как проверить, что API работает?

```bash
curl http://localhost:8000/api/v1/health
```

### Как найти run_id в логах?

```bash
docker-compose logs dataprocessor-api | grep "run_id"
```

### Как проверить статус конкретного run'а?

```bash
curl http://localhost:8000/api/v1/runs/{run_id}/status \
  -H "X-API-Key: your-api-key"
```

### Как отменить зависший run?

```bash
curl -X POST http://localhost:8000/api/v1/runs/{run_id}/cancel \
  -H "X-API-Key: your-api-key"
```

### Как увеличить производительность?

1. Увеличить `MAX_CONCURRENT_RUNS`
2. Увеличить ресурсы Docker контейнера
3. Использовать S3 storage вместо файловой системы
4. Настроить Triton для GPU ускорения

## Получение помощи

Если проблема не решена:

1. **Собрать информацию**:
   - Версия API: `curl http://localhost:8000/api/v1/health | jq '.version'`
   - Логи: `docker-compose logs > logs.txt`
   - Конфигурация: `env | grep -E 'API_|STORAGE_|REDIS_'`

2. **Проверить документацию**:
   - [Архитектура](../docs/DATAPROCESSOR_API_ARCHITECTURE.md)
   - [Чеклист разработки](../docs/API_DEVELOPMENT_CHECKLIST.md)

3. **Создать issue** с описанием проблемы и собранной информацией

