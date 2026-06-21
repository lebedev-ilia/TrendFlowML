# Runbooks & Playbooks для Fetcher

Операционные руководства для решения типичных проблем и инцидентов в Fetcher.

## Содержание

1. [Падение прокси](#падение-прокси)
2. [Рост 429 ошибок](#рост-429-ошибок)
3. [Экспоненциальный рост очереди](#экспоненциальный-рост-очереди)
4. [Проблемы с БД](#проблемы-с-бд)
5. [Проблемы с Storage](#проблемы-с-storage)
6. [Circuit Breaker открыт](#circuit-breaker-открыт)
7. [Backpressure от DataProcessor](#backpressure-от-dataprocessor)

---

## Падение прокси

### Симптомы

- Высокий `proxy_failure_rate` в метриках
- Много ошибок в логах: `Proxy connection failed`
- Низкий `fetcher_videos_downloaded_total`
- Worker'ы не могут скачать видео

### Диагностика

1. Проверить метрики:
   ```bash
   curl http://localhost:8000/metrics | grep proxy_failure_rate
   ```

2. Проверить логи:
   ```bash
   docker-compose logs fetcher-worker | grep -i proxy
   ```

3. Проверить валидацию прокси:
   ```bash
   curl http://localhost:8000/admin/validation | jq .proxy_rotation
   ```

### Решение

#### Вариант 1: Автоматическое исключение

Fetcher автоматически исключает прокси с высоким failure rate (>50%). Проверить, что это работает:

```bash
# Проверить health score прокси
curl http://localhost:8000/admin/validation
```

#### Вариант 2: Ручное отключение прокси

1. Подключиться к БД:
   ```bash
   docker-compose exec postgres psql -U fetcher -d fetcher_db
   ```

2. Отключить проблемный прокси:
   ```sql
   UPDATE proxies SET enabled = false WHERE url = 'http://problematic-proxy:8080';
   ```

3. Проверить, что прокси больше не используется:
   ```bash
   docker-compose logs fetcher-worker | grep -i proxy
   ```

#### Вариант 3: Добавление новых прокси

1. Добавить прокси в БД:
   ```sql
   INSERT INTO proxies (url, country, enabled) 
   VALUES ('http://new-proxy:8080', 'US', true);
   ```

2. Или обновить настройки:
   ```bash
   # В docker-compose.yml или .env
   FETCHER_PROXIES=http://new-proxy1:8080,http://new-proxy2:8080
   ```

3. Перезапустить worker'ы:
   ```bash
   docker-compose restart fetcher-worker
   ```

### Профилактика

- Регулярно мониторить `proxy_failure_rate`
- Настроить алерты на высокий failure rate (>30%)
- Регулярно проверять health прокси через `/admin/validation`

---

## Рост 429 ошибок

### Симптомы

- Высокий `fetcher_youtube_429_total` в метриках
- Много ошибок в логах: `YouTube rate limit exceeded`
- Circuit breaker открыт для metadata/download/comments
- Низкий throughput

### Диагностика

1. Проверить метрики:
   ```bash
   curl http://localhost:8000/metrics | grep fetcher_youtube_429
   ```

2. Проверить circuit breaker:
   ```bash
   curl http://localhost:8000/admin/validation | jq .circuit_breaker
   ```

3. Проверить rate limiter:
   ```bash
   curl http://localhost:8000/admin/validation | jq .rate_limiter
   ```

### Решение

#### Вариант 1: Увеличить лимиты rate limiter

1. Обновить настройки в `config.py` или через переменные окружения:
   ```bash
   # Увеличить лимит для metadata
   FETCHER_YOUTUBE_METADATA_RATE_LIMIT=200  # было 400
   FETCHER_YOUTUBE_METADATA_WINDOW_SEC=3600
   ```

2. Перезапустить worker'ы:
   ```bash
   docker-compose restart fetcher-worker
   ```

#### Вариант 2: Использовать больше прокси

1. Добавить прокси из разных стран:
   ```sql
   INSERT INTO proxies (url, country, enabled) VALUES
   ('http://proxy-us:8080', 'US', true),
   ('http://proxy-gb:8080', 'GB', true),
   ('http://proxy-de:8080', 'DE', true);
   ```

2. Перезапустить worker'ы

#### Вариант 3: Временно снизить нагрузку

1. Остановить новые задачи:
   ```bash
   # Остановить orchestrator или уменьшить количество worker'ов
   docker-compose scale fetcher-worker=1
   ```

2. Дождаться cooldown circuit breaker (по умолчанию 5 минут)

3. Постепенно увеличивать нагрузку

### Профилактика

- Мониторить `fetcher_youtube_429_total` и настроить алерты
- Использовать достаточное количество прокси
- Настроить circuit breaker для автоматической защиты
- Регулярно проверять rate limiter через `/admin/validation`

---

## Экспоненциальный рост очереди

### Симптомы

- Высокий размер очереди Celery в Redis
- Много задач в статусе `PENDING` или `QUEUED`
- Высокая задержка между постановкой и выполнением задач
- Worker'ы не справляются с нагрузкой

### Диагностика

1. Проверить размер очереди:
   ```bash
   docker-compose exec redis redis-cli LLEN celery
   ```

2. Проверить метрики:
   ```bash
   curl http://localhost:8000/metrics | grep celery
   ```

3. Проверить статусы run'ов:
   ```bash
   docker-compose exec postgres psql -U fetcher -d fetcher_db -c \
     "SELECT status, COUNT(*) FROM runs GROUP BY status;"
   ```

### Решение

#### Вариант 1: Увеличить количество worker'ов

1. Увеличить concurrency:
   ```bash
   # В docker-compose.yml
   command: celery -A fetcher.celery_app worker --loglevel=info --concurrency=8
   ```

2. Или масштабировать:
   ```bash
   docker-compose scale fetcher-worker=4
   ```

#### Вариант 2: Оптимизировать приоритеты задач

1. Проверить приоритеты в `celery_app.py`:
   ```python
   task_routes={
       "fetcher.tasks.fetch_metadata_task": {"queue": "fetch.metadata", "priority": 9},
       "fetcher.tasks.download_video_task": {"queue": "fetch.video", "priority": 1},
   }
   ```

2. Убедиться, что важные задачи имеют высокий приоритет

#### Вариант 3: Очистить старые задачи

1. Очистить failed задачи:
   ```bash
   docker-compose exec redis redis-cli DEL celery:failed
   ```

2. Очистить старые run'ы:
   ```bash
   curl -X POST http://localhost:8000/admin/lifecycle/cleanup
   ```

#### Вариант 4: Временно остановить новые задачи

1. Остановить orchestrator:
   ```bash
   docker-compose stop fetcher-api
   ```

2. Дождаться обработки текущей очереди

3. Постепенно возобновить работу

### Профилактика

- Мониторить размер очереди и настроить алерты
- Использовать достаточное количество worker'ов
- Настроить backpressure control для защиты от перегрузки
- Регулярно очищать старые задачи и run'ы

---

## Проблемы с БД

### Симптомы

- Ошибки подключения к БД в логах
- Health check возвращает `database.status != "healthy"`
- Медленные запросы
- Таймауты при работе с БД

### Диагностика

1. Проверить health check:
   ```bash
   curl http://localhost:8000/health | jq .dependencies.database
   ```

2. Проверить подключение:
   ```bash
   docker-compose exec postgres psql -U fetcher -d fetcher_db -c "SELECT 1;"
   ```

3. Проверить логи:
   ```bash
   docker-compose logs postgres | tail -50
   ```

### Решение

#### Вариант 1: Перезапуск PostgreSQL

```bash
docker-compose restart postgres
```

#### Вариант 2: Проверка ресурсов

1. Проверить использование диска:
   ```bash
   docker-compose exec postgres df -h
   ```

2. Проверить использование памяти:
   ```bash
   docker stats fetcher-postgres
   ```

#### Вариант 3: Оптимизация запросов

1. Проверить медленные запросы:
   ```sql
   SELECT * FROM pg_stat_statements 
   ORDER BY total_time DESC LIMIT 10;
   ```

2. Добавить индексы при необходимости

### Профилактика

- Регулярно мониторить health check
- Настроить алерты на проблемы с БД
- Регулярно делать резервные копии
- Мониторить использование ресурсов

---

## Проблемы с Storage

### Симптомы

- Ошибки при upload/download артефактов
- Health check возвращает `storage.status != "healthy"`
- Медленные операции с storage
- Ошибки checksum validation

### Диагностика

1. Проверить health check:
   ```bash
   curl http://localhost:8000/health | jq .dependencies.storage
   ```

2. Проверить MinIO:
   ```bash
   curl http://localhost:9000/minio/health/live
   ```

3. Проверить логи:
   ```bash
   docker-compose logs minio | tail -50
   ```

### Решение

#### Вариант 1: Перезапуск MinIO

```bash
docker-compose restart minio
```

#### Вариант 2: Проверка bucket'ов

1. Проверить существование bucket'ов:
   ```bash
   # Через MinIO Console: http://localhost:9001
   # Или через API
   ```

2. Создать bucket если отсутствует:
   ```bash
   # Через MinIO Console или API
   ```

#### Вариант 3: Проверка дискового пространства

1. Проверить использование диска:
   ```bash
   docker-compose exec minio df -h
   ```

2. Очистить старые артефакты:
   ```bash
   curl -X POST http://localhost:8000/admin/lifecycle/cleanup
   ```

### Профилактика

- Регулярно мониторить health check
- Настроить алерты на проблемы с storage
- Регулярно очищать старые артефакты
- Мониторить использование дискового пространства

---

## Circuit Breaker открыт

### Симптомы

- Circuit breaker в статусе `OPEN`
- Операции блокируются
- Низкий throughput
- Много ошибок в логах

### Диагностика

1. Проверить валидацию:
   ```bash
   curl http://localhost:8000/admin/validation | jq .circuit_breaker
   ```

2. Проверить метрики:
   ```bash
   curl http://localhost:8000/metrics | grep circuit_breaker_tripped_total
   ```

### Решение

#### Вариант 1: Дождаться cooldown

Circuit breaker автоматически переходит в `HALF_OPEN` после cooldown (по умолчанию 5 минут).

#### Вариант 2: Ручной сброс (если нужно)

1. Перезапустить worker'ы:
   ```bash
   docker-compose restart fetcher-worker
   ```

2. Это сбросит in-memory circuit breakers

#### Вариант 3: Исправить причину ошибок

1. Определить причину (429, 403, network errors)
2. Исправить проблему (добавить прокси, увеличить лимиты, etc.)
3. Дождаться cooldown

### Профилактика

- Мониторить `circuit_breaker_tripped_total`
- Настроить алерты на открытие circuit breaker
- Регулярно проверять валидацию через `/admin/validation`

---

## Backpressure от DataProcessor

### Симптомы

- Много задач в статусе `FINALIZING`
- Ошибки `BackpressureError` в логах
- Низкий throughput финализации
- DataProcessor queue перегружен

### Диагностика

1. Проверить логи:
   ```bash
   docker-compose logs fetcher-worker | grep -i backpressure
   ```

2. Проверить статусы run'ов:
   ```bash
   docker-compose exec postgres psql -U fetcher -d fetcher_db -c \
     "SELECT status, COUNT(*) FROM runs WHERE status = 'FINALIZING' GROUP BY status;"
   ```

### Решение

#### Вариант 1: Дождаться освобождения очереди DataProcessor

Backpressure control автоматически retry задачи после указанного времени.

#### Вариант 2: Увеличить backpressure threshold

1. Обновить настройки:
   ```bash
   FETCHER_BACKPRESSURE_THRESHOLD=1000  # было 500
   ```

2. Перезапустить worker'ы

#### Вариант 3: Временно остановить финализацию

1. Остановить finalize worker'ы:
   ```bash
   docker-compose stop fetcher-worker
   # Запустить только metadata/video/comments worker'ы
   ```

2. Дождаться освобождения очереди DataProcessor

3. Возобновить финализацию

### Профилактика

- Мониторить размер очереди DataProcessor
- Настроить алерты на backpressure
- Использовать backpressure control для автоматической защиты

---

## Общие рекомендации

### Мониторинг

- Регулярно проверять health check: `curl http://localhost:8000/health`
- Мониторить метрики Prometheus: `http://localhost:8000/metrics`
- Настроить Grafana dashboard для визуализации
- Настроить алерты на критические метрики

### Логирование

- Использовать structured logging для лучшей читаемости
- Регулярно проверять логи: `docker-compose logs -f`
- Настроить централизованное логирование (ELK, Loki, etc.)

### Резервные копии

- Регулярно делать резервные копии БД
- Настроить автоматические резервные копии
- Тестировать восстановление из резервных копий

### Обновления

- Регулярно обновлять зависимости
- Тестировать обновления в staging окружении
- Использовать версионирование для миграций БД
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
