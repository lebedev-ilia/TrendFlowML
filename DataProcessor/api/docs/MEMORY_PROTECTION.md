# Memory Protection

## Обзор

Memory Protection защищает систему от превышения памяти через:
- **Container limits**: Ограничение памяти на уровне Docker контейнера (16G для worker)
- **Subprocess monitoring**: Мониторинг использования памяти каждого subprocess и автоматическое завершение при превышении лимита

## Конфигурация

### Container Limits

В `docker-compose.yml` настроены следующие лимиты для worker:

```yaml
dataprocessor-worker:
  deploy:
    resources:
      limits:
        memory: 16G  # Лимит памяти для worker контейнера
        cpus: '4'    # Лимит CPU
      reservations:
        memory: 8G  # Резервирование памяти
        cpus: '2'    # Резервирование CPU
```

### Subprocess Memory Limit

Лимит памяти для каждого subprocess настраивается через переменную окружения:

```bash
export SUBPROCESS_MEMORY_LIMIT_MB=8000  # 8GB (по умолчанию)
```

В `docker-compose.yml`:
```yaml
environment:
  - SUBPROCESS_MEMORY_LIMIT_MB=8000  # Лимит памяти для subprocess (8GB)
```

## Как это работает

### 1. Container Limits

Docker автоматически ограничивает использование памяти контейнером. При превышении лимита:
- Контейнер может быть остановлен Docker'ом
- OOM Killer может убить процессы внутри контейнера

### 2. Subprocess Memory Monitoring

Для каждого запущенного subprocess:
1. **Мониторинг**: Каждые 10 секунд проверяется использование памяти через `psutil`
2. **Логирование**: 
   - `INFO` при нормальном использовании
   - `WARNING` при превышении 80% лимита
   - `WARNING` при превышении лимита с последующим kill
3. **Метрики**: Обновление Prometheus метрики `dataprocessor_memory_bytes`
4. **Kill**: При превышении лимита процесс убивается через `process.kill()`

### Пример логов

```
INFO: Run abc-123 memory usage: 2048MB / 8000MB
WARNING: Run abc-123 memory usage: 6500MB / 8000MB (80%+ threshold)
WARNING: Run abc-123 exceeded memory limit (8500MB > 8000MB), killing process
```

## Мониторинг

### Prometheus метрики

Метрика `dataprocessor_memory_bytes` обновляется каждые 10 секунд для каждого активного run'а:

```prometheus
# HELP dataprocessor_memory_bytes Memory usage per run
# TYPE dataprocessor_memory_bytes gauge
dataprocessor_memory_bytes{run_id="abc-123"} 2.147483648e+09
```

### Логи

Проверьте логи worker'а для мониторинга использования памяти:

```bash
# Docker Compose
docker-compose logs dataprocessor-worker | grep -i memory

# Kubernetes
kubectl logs -l app=dataprocessor-worker | grep -i memory
```

## Обработка OOM

При обнаружении превышения лимита памяти:

1. **Процесс убивается**: `process.kill()` отправляет SIGKILL
2. **Exit code**: `-9` (SIGKILL)
3. **Error type**: `killed_by_memory_limit`
4. **Recovery**: Worker может автоматически re-enqueue run с lower priority ("low")

## Тестирование

Запустите unit тесты:

```bash
pytest api/tests/unit/test_memory_protection.py -v
```

Тесты покрывают:
- Нормальное использование памяти
- Превышение лимита
- Обработку ошибок (NoSuchProcess, ImportError)
- Обновление метрик
- Проверку интервала мониторинга

## Требования

- `psutil>=5.9.0` (уже в `requirements-api.txt`)
- Docker с поддержкой resource limits
- Достаточная память на хосте для контейнеров

## Рекомендации

1. **Настройка лимитов**:
   - Worker container: 16G (достаточно для обработки видео)
   - Subprocess: 8GB (достаточно для большинства задач)
   - При необходимости увеличьте лимиты

2. **Мониторинг**:
   - Настройте алерты на Prometheus метрику `dataprocessor_memory_bytes`
   - Мониторьте логи на наличие предупреждений о памяти

3. **Оптимизация**:
   - Если часто происходят OOM, рассмотрите:
     - Увеличение лимитов
     - Оптимизацию обработки (batch size, streaming)
     - Использование более эффективных алгоритмов

## Ссылки

- [Архитектура Memory Protection](../docs/DATAPROCESSOR_API_ARCHITECTURE.md#L2378)
- [Чеклист разработки](../docs/API_DEVELOPMENT_CHECKLIST.md#L892)
- [psutil документация](https://psutil.readthedocs.io/)

