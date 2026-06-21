# Load Testing для Fetcher

**Дата**: 2026-03-05  
**Стадия**: Quality Assurance Checklist  
**Статус**: ✅ Реализовано

## Обзор

Реализован скрипт для load-тестирования Fetcher на целевой нагрузке (10k видео/день). Это критически важно для проверки производительности, масштабируемости и стабильности системы под нагрузкой.

## Требования

Из Quality Assurance Checklist:
- ✅ Load‑тесты на целевую нагрузку (например, 10k видео/день)

## Реализация

### 1. Скрипт `scripts/load_test.py`

**Класс `LoadTestRunner`**:

- **`__init__(target_requests, duration_seconds, video_urls=None)`**:
  - Инициализация load test runner
  - Генерация тестовых URL если не указаны

- **`create_run(url)`**:
  - Создание run'а в БД для тестирования
  - Возвращает UUID run'а

- **`run_load_test()`**:
  - Запуск load test
  - Создание run'ов с заданной частотой
  - Запуск orchestrator для каждого run'а
  - Сбор метрик и результатов

- **`_collect_results(run_ids, completed, failed)`**:
  - Сбор финальных результатов теста
  - Вычисление метрик (throughput, latency, statuses)

- **`print_results(results)`**:
  - Вывод результатов теста в читаемом формате

**Функции**:

- **`main()`**:
  - Главная функция для запуска load test
  - Парсинг аргументов командной строки
  - Обработка ошибок и прерываний

### 2. Параметры командной строки

- `--target`: Целевое количество запросов (по умолчанию 10000)
- `--duration`: Длительность теста в секундах (по умолчанию 86400 = 1 день)
- `--urls`: Список URL видео для тестирования (опционально)

### 3. Метрики

Скрипт собирает следующие метрики:

- **Throughput**: Количество запросов в секунду
- **Latency**: Среднее, минимальное и максимальное время выполнения
- **Success/Failure rate**: Процент успешных и неуспешных запросов
- **Run statuses**: Распределение статусов run'ов (COMPLETED, FAILED, etc.)

## Использование

### Базовый запуск

```bash
# 10k запросов за 24 часа
python scripts/load_test.py --target 10000 --duration 86400

# Быстрый тест (100 запросов за 1 час)
python scripts/load_test.py --target 100 --duration 3600

# Тест с конкретными URL
python scripts/load_test.py --target 50 --duration 300 --urls \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

### Пример вывода

```
================================================================================
LOAD TEST RESULTS
================================================================================

Test Configuration:
  Target requests: 10000
  Duration: 86400s
  Actual duration: 86400.00s

Requests:
  Total: 10000
  Completed: 9950
  Failed: 50
  Throughput: 0.116 req/s

Run Statuses:
  COMPLETED: 9950
  FAILED: 50

Latency:
  Avg success: 45.234s
  Avg failed: 12.345s
  Min success: 2.123s
  Max success: 180.456s

Timestamp: 2026-03-05T12:00:00
================================================================================
```

## Мониторинг

### Prometheus метрики

Во время load test можно мониторить:

- `fetcher_videos_downloaded_total` - количество скачанных видео
- `fetcher_videos_failed_total` - количество неудачных попыток
- `fetcher_cache_hits_total` - количество cache hits
- `fetcher_cache_miss_total` - количество cache misses
- `fetcher_metadata_latency_seconds` - latency для metadata worker
- `fetcher_download_latency_seconds` - latency для video download worker
- `fetcher_comments_latency_seconds` - latency для comments worker

### Health Check

```bash
curl http://localhost:8000/health
```

## Рекомендации

### Подготовка

1. Используйте staging окружение, максимально близкое к production
2. Убедитесь, что есть достаточно тестовых URL видео
3. Настройте мониторинг (Prometheus, Grafana) перед тестом
4. Сделайте резервные копии БД перед тестом

### Выполнение

1. Начните с малой нагрузки и постепенно увеличивайте
2. Следите за использованием CPU, памяти, диска, сети
3. Включите детальное логирование для анализа проблем
4. Тест должен длиться достаточно долго для выявления проблем стабильности

### Анализ результатов

1. **Throughput**: Должен соответствовать целевой нагрузке
2. **Latency**: Должен быть в разумных пределах
3. **Error rate**: Должен быть минимальным (< 1%)
4. **Resource usage**: CPU, память, диск не должны быть перегружены

## Известные ограничения

- Скрипт использует синхронный подход (может быть медленным для больших нагрузок)
- Не учитывает реальные ограничения YouTube API (rate limits)
- Не моделирует реальное распределение нагрузки (может быть равномерным)

## Улучшения

В будущем можно добавить:

1. **Асинхронный подход**: Использовать asyncio для параллельных запросов
2. **Распределение нагрузки**: Моделировать реальное распределение (peak hours, etc.)
3. **Интеграция с Locust/K6**: Использовать специализированные инструменты для load testing
4. **Автоматический анализ**: Автоматический анализ результатов и генерация отчётов
5. **Chaos testing**: Добавить искусственные сбои для проверки устойчивости

## Связанные файлы

- `scripts/load_test.py` - скрипт для load-тестирования
- `docs/LOAD_TESTING.md` - документация по load testing
- `docs/checklist.md` - Quality Assurance Checklist
- `docs/GRAFANA_DASHBOARD.md` - Grafana Dashboard

## Статус

✅ **Реализовано**:
- Скрипт `load_test.py` для load-тестирования
- Поддержка целевой нагрузки (10k видео/день)
- Сбор метрик (throughput, latency, success/failure rate)
- Документация по load testing

## Следующие шаги

- Добавить асинхронный подход для улучшения производительности
- Интегрировать с Locust/K6 для более продвинутого load testing
- Добавить автоматический анализ результатов и генерацию отчётов
---

## Навигация

[README](README.md) · [Fetcher](../INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
