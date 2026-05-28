# Мониторинг и Observability

**Дата**: 2024-01-XX  
**Раздел**: Дополнительные задачи из [API_DEVELOPMENT_CHECKLIST.md](../../../docs/API_DEVELOPMENT_CHECKLIST.md)  
**Статус**: ✅ Завершено

## Обзор

Настроена полная инфраструктура мониторинга и observability для DataProcessor API:
- **Grafana дашборды** для визуализации метрик
- **Prometheus алерты** для критических событий
- **OpenTelemetry distributed tracing** для отслеживания запросов

## Выполненные задачи

### ✅ Настроить Grafana дашборды для метрик

**Создан дашборд** `monitoring/grafana/dashboards/dataprocessor-overview.json`:

**Панели**:
1. **Queue Length** - график длины очереди по приоритетам (high, normal, low)
2. **Active Runs** - stat panel с цветовой индикацией (зеленый < 20, желтый 20-30, красный > 30)
3. **Crashed Runs (Total)** - общее количество упавших run'ов
4. **Processing Time (95th percentile)** - время обработки по процессорам и компонентам
5. **Failure Rate** - rate ошибок по процессорам/компонентам/типам ошибок
6. **Memory Usage** - использование памяти по run_id
7. **Queue Wait Time (95th percentile)** - время ожидания в очереди

**Provisioning**:
- `monitoring/grafana/provisioning/dashboards/dashboard.yml` - автоматическая загрузка дашбордов
- `monitoring/grafana/provisioning/datasources/prometheus.yml` - автоматическое подключение Prometheus

**Docker Compose**:
- Добавлен сервис `grafana` с портом 3000
- Volumes для данных и provisioning конфигурации
- Зависимость от prometheus

### ✅ Настроить алерты в Prometheus

**Создан файл** `monitoring/prometheus/alerts.yml` с правилами:

1. **DataProcessorQueueLengthHigh**
   - Условие: `sum(dataprocessor_queue_length) > 100`
   - Длительность: 5 минут
   - Severity: warning
   - Описание: Queue length превышает лимит (100)

2. **DataProcessorCrashedRunsHigh**
   - Условие: `rate(crashed_runs_total[5m]) / rate(processing_seconds_count[5m]) > 0.1`
   - Длительность: 5 минут
   - Severity: critical
   - Описание: Процент crashed runs превышает 10%

3. **DataProcessorProcessingTimeHigh**
   - Условие: `current_time(95th) > baseline(95th) * 2`
   - Длительность: 10 минут
   - Severity: warning
   - Описание: Время обработки превышает baseline × 2

4. **DataProcessorMemoryUsageHigh**
   - Условие: `sum(dataprocessor_memory_bytes) > 8GB`
   - Длительность: 5 минут
   - Severity: warning
   - Описание: Использование памяти превышает 8GB

5. **DataProcessorActiveRunsHigh**
   - Условие: `dataprocessor_active_runs > 30`
   - Длительность: 5 минут
   - Severity: warning
   - Описание: Количество активных run'ов превышает лимит (30)

6. **DataProcessorFailureRateHigh**
   - Условие: `rate(dataprocessor_failures_total[5m]) > 0.1`
   - Длительность: 5 минут
   - Severity: critical
   - Описание: Rate ошибок превышает 0.1 ошибок/сек

**Конфигурация Prometheus**:
- `monitoring/prometheus/prometheus.yml` - основная конфигурация
- Scrape интервал: 10 секунд для API, 15 секунд для других
- Targets: dataprocessor-api, dataprocessor-worker
- Загрузка правил алертов из `alerts.yml`

**Docker Compose**:
- Добавлен сервис `prometheus` с портом 9090
- Volumes для конфигурации и данных
- Зависимость от dataprocessor-api

### ✅ Настроить distributed tracing (OpenTelemetry)

**Зависимости**:
- Добавлены в `requirements-api.txt`:
  - `opentelemetry-api>=1.21.0`
  - `opentelemetry-sdk>=1.21.0`
  - `opentelemetry-instrumentation-fastapi>=0.42b0`
  - `opentelemetry-instrumentation-httpx>=0.42b0`
  - `opentelemetry-exporter-jaeger>=1.21.0`
  - `opentelemetry-exporter-otlp>=1.21.0`

**Инструментация FastAPI**:
- Добавлен импорт OpenTelemetry в `api/main.py`
- Функция `_setup_opentelemetry()` для инициализации tracing
- Инструментация FastAPI через `FastAPIInstrumentor.instrument_app()`
- Автоматическое создание spans для всех HTTP запросов

**Экспортеры**:
- **Jaeger** (по умолчанию):
  - UDP agent на порту 6831
  - Настройки: `JAEGER_AGENT_HOST`, `JAEGER_AGENT_PORT`
- **OTLP** (альтернатива):
  - gRPC endpoint на порту 4317
  - Настройки: `OTLP_ENDPOINT`

**Настройки в config.py**:
- `enable_tracing: bool` - включить/выключить tracing
- `tracing_exporter: str` - выбрать экспортер (jaeger/otlp)
- `jaeger_agent_host: str` - хост Jaeger agent
- `jaeger_agent_port: int` - порт Jaeger agent
- `otlp_endpoint: str` - OTLP endpoint
- `service_name: str` - имя сервиса
- `service_version: str` - версия сервиса

**Модуль tracing.py**:
- Создан `api/services/tracing.py` для работы с трейсами
- Функция `get_tracer()` для получения tracer
- Функция `create_span()` для создания spans с атрибутами

**Docker Compose**:
- Добавлен сервис `jaeger` (all-in-one)
- Порты: 16686 (UI), 6831 (UDP agent), 4317 (OTLP gRPC), 4318 (OTLP HTTP)
- Включен OTLP collector

## Измененные файлы

### `monitoring/prometheus/prometheus.yml` (новый)
- Конфигурация Prometheus для сбора метрик
- Scrape конфигурации для API и worker
- Загрузка правил алертов

### `monitoring/prometheus/alerts.yml` (новый)
- Правила алертов для критических событий
- 6 правил для различных метрик

### `monitoring/grafana/dashboards/dataprocessor-overview.json` (новый)
- Дашборд Grafana с панелями для всех метрик
- 7 панелей для визуализации

### `monitoring/grafana/provisioning/` (новый)
- Автоматическая загрузка дашбордов
- Автоматическое подключение Prometheus

### `monitoring/README.md` (новый)
- Документация по мониторингу
- Инструкции по запуску и использованию

### `DataProcessor/api/main.py`
- Добавлена инициализация OpenTelemetry tracing
- Инструментация FastAPI приложения

### `DataProcessor/api/config.py`
- Добавлены настройки для OpenTelemetry tracing

### `DataProcessor/api/services/tracing.py` (новый)
- Модуль для работы с трейсами
- Функции для создания spans

### `DataProcessor/requirements-api.txt`
- Добавлены зависимости OpenTelemetry

### `DataProcessor/docker-compose.yml`
- Добавлены сервисы: prometheus, grafana, jaeger
- Настроены volumes и networks

## Технические детали

### Prometheus Scraping

**Интервалы**:
- API: 10 секунд (частое обновление для real-time мониторинга)
- Worker: 10 секунд
- Prometheus self: 15 секунд

**Метрики**:
- Endpoint: `/api/v1/metrics`
- Формат: Prometheus text format
- Content-Type: `text/plain; version=0.0.4; charset=utf-8`

### Grafana Provisioning

**Автоматическая загрузка**:
- Дашборды из `dashboards/`
- Datasources из `datasources/`
- Обновление каждые 10 секунд

**Доступ**:
- URL: http://localhost:3000
- Логин: admin
- Пароль: admin (изменить в production!)

### OpenTelemetry Tracing

**Инструментация**:
- FastAPI автоматически создает spans для всех endpoints
- Атрибуты: `run_id`, `video_id`, `platform_id`, `http.method`, `http.route`, `http.status_code`

**Экспортеры**:
- **Jaeger**: UDP agent (быстрый, для development)
- **OTLP**: gRPC/HTTP (стандартный, для production)

**Spans**:
- Каждый HTTP запрос создает span
- Вложенные spans для вложенных операций
- Атрибуты добавляются автоматически

## Использование

### Запуск мониторинга

```bash
# Запустить все сервисы мониторинга
docker-compose up -d prometheus grafana jaeger

# Или запустить все сервисы
docker-compose up -d
```

### Доступ к сервисам

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger UI**: http://localhost:16686

### Включение tracing

```bash
export ENABLE_TRACING=true
export TRACING_EXPORTER=jaeger
export JAEGER_AGENT_HOST=jaeger
export JAEGER_AGENT_PORT=6831
```

### Просмотр метрик

1. **Prometheus**:
   - Открыть http://localhost:9090
   - Выполнить запросы: `dataprocessor_queue_length`, `dataprocessor_active_runs`, и т.д.

2. **Grafana**:
   - Открыть http://localhost:3000
   - Дашборд "DataProcessor API - Overview" загружается автоматически

3. **Jaeger**:
   - Открыть http://localhost:16686
   - Выбрать сервис: `dataprocessor-api`
   - Найти трейсы по run_id или другим атрибутам

## Тестирование

### Ручное тестирование

1. **Запустить сервисы**:
   ```bash
   docker-compose up -d prometheus grafana jaeger dataprocessor-api
   ```

2. **Проверить Prometheus**:
   - Открыть http://localhost:9090
   - Проверить что метрики собираются (Status → Targets)

3. **Проверить Grafana**:
   - Открыть http://localhost:3000
   - Проверить что дашборд загружен
   - Проверить что данные отображаются

4. **Проверить Jaeger**:
   - Включить tracing в API
   - Выполнить запрос к API
   - Проверить что трейсы появляются в Jaeger UI

### Unit тесты

TODO: Добавить unit тесты:
- Тесты для создания spans
- Тесты для экспортеров
- Тесты для алертов

## Известные проблемы

1. **Grafana пароль**: По умолчанию admin/admin
   - Решение: Изменить в production через переменные окружения

2. **OTLP insecure**: Используется insecure=True для development
   - Решение: Использовать TLS в production

3. **Jaeger all-in-one**: Не подходит для production
   - Решение: Использовать отдельные компоненты Jaeger в production

## Следующие шаги

1. **Настроить Alertmanager**:
   - Интеграция с Slack/Email/PagerDuty
   - Настройка routing правил

2. **Добавить больше дашбордов**:
   - Дашборд для ошибок
   - Дашборд для производительности
   - Дашборд для ресурсов

3. **Улучшить трейсинг**:
   - Инструментация worker процессов
   - Инструментация Redis операций
   - Инструментация Storage операций

## Ссылки

- [Архитектура мониторинга](../../../docs/DATAPROCESSOR_API_ARCHITECTURE.md#L2165)
- [Архитектура tracing](../../../docs/DATAPROCESSOR_API_ARCHITECTURE.md#L2546)
- [Чеклист разработки](../../../docs/API_DEVELOPMENT_CHECKLIST.md#L1025)
- [Документация мониторинга](../../monitoring/README.md)

