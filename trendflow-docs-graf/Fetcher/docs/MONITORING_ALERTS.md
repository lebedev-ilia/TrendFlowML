## Monitoring Alerts для Fetcher

Этот документ описывает рекомендуемые алерты для мониторинга Fetcher, которые должны быть настроены в Prometheus Alertmanager.

---

## Критичные алерты (Critical)

### 1. High Failure Rate

**Название**: `FetcherHighFailureRate`  
**Описание**: Слишком высокий процент неуспешных ингестий  
**Условие**: `sum(rate(fetcher_videos_failed_total[5m])) > 10`  
**Severity**: `critical`  
**Действие**: Немедленно проверить логи и состояние платформ (YouTube API, прокси)

**PromQL**:
```promql
sum(rate(fetcher_videos_failed_total[5m])) > 10
```

**Аннотации**:
- `summary`: "Fetcher failure rate превысил порог"
- `description`: "Failure rate: {{ $value }}/min. Проверьте логи и состояние платформ."

---

### 2. Circuit Breaker Tripped

**Название**: `FetcherCircuitBreakerTripped`  
**Описание**: Circuit breaker сработал слишком часто  
**Условие**: `sum(rate(circuit_breaker_tripped_total[5m])) > 5`  
**Severity**: `critical`  
**Действие**: Проверить состояние прокси и YouTube API, возможно требуется ротация прокси

**PromQL**:
```promql
sum(rate(circuit_breaker_tripped_total[5m])) > 5
```

**Аннотации**:
- `summary`: "Circuit breaker сработал слишком часто"
- `description`: "Срабатываний: {{ $value }}/min. Проверьте прокси и YouTube API."

---

### 3. Proxy Failure Rate High

**Название**: `FetcherProxyFailureRateHigh`  
**Описание**: Слишком высокий процент неуспешных запросов через прокси  
**Условие**: `proxy_failure_rate > 0.8`  
**Severity**: `critical`  
**Действие**: Проверить состояние прокси, возможно требуется замена или ротация

**PromQL**:
```promql
proxy_failure_rate > 0.8
```

**Аннотации**:
- `summary`: "Proxy failure rate критически высок"
- `description`: "Failure rate: {{ $value | humanizePercentage }}. Проверьте прокси {{ $labels.proxy_id }}."

---

### 4. Database Unavailable

**Название**: `FetcherDatabaseUnavailable`  
**Описание**: PostgreSQL база данных недоступна  
**Условие**: Health check endpoint возвращает `database.status != "healthy"`  
**Severity**: `critical`  
**Действие**: Немедленно проверить состояние PostgreSQL, возможно требуется перезапуск или восстановление

**PromQL** (через метрику health check):
```promql
fetcher_health_database_status != 1
```

**Аннотации**:
- `summary`: "PostgreSQL база данных недоступна"
- `description`: "Fetcher не может подключиться к базе данных. Проверьте состояние PostgreSQL."

---

### 5. Storage Unavailable

**Название**: `FetcherStorageUnavailable`  
**Описание**: S3/MinIO storage недоступен  
**Условие**: Health check endpoint возвращает `storage.status != "healthy"`  
**Severity**: `critical`  
**Действие**: Немедленно проверить состояние S3/MinIO, возможно проблемы с сетью или креденшелами

**PromQL** (через метрику health check):
```promql
fetcher_health_storage_status != 1
```

**Аннотации**:
- `summary`: "S3/MinIO storage недоступен"
- `description`: "Fetcher не может подключиться к storage. Проверьте состояние S3/MinIO."

---

## Предупреждающие алерты (Warning)

### 6. Low Cache Hit Ratio

**Название**: `FetcherLowCacheHitRatio`  
**Описание**: Низкий процент cache hits  
**Условие**: `cache_hit_ratio < 0.3`  
**Severity**: `warning`  
**Действие**: Проверить логику кеширования, возможно требуется оптимизация

**PromQL**:
```promql
sum(rate(fetcher_cache_hits_total[5m])) / 
(sum(rate(fetcher_cache_hits_total[5m])) + sum(rate(fetcher_cache_miss_total[5m]))) < 0.3
```

**Аннотации**:
- `summary`: "Cache hit ratio ниже порога"
- `description`: "Cache hit ratio: {{ $value | humanizePercentage }}. Проверьте логику кеширования."

---

### 7. High YouTube 429 Rate

**Название**: `FetcherHighYouTube429Rate`  
**Описание**: Слишком много 429 ошибок от YouTube (rate limit)  
**Условие**: `sum(rate(fetcher_youtube_429_total[5m])) > 10`  
**Severity**: `warning`  
**Действие**: Проверить rate limiting, возможно требуется увеличение интервалов между запросами или ротация прокси

**PromQL**:
```promql
sum(rate(fetcher_youtube_429_total[5m])) > 10
```

**Аннотации**:
- `summary`: "Слишком много 429 ошибок от YouTube"
- `description`: "429 ошибок: {{ $value }}/min. Проверьте rate limiting и прокси."

---

### 8. High Download Latency

**Название**: `FetcherHighDownloadLatency`  
**Описание**: Слишком высокое время скачивания видео  
**Условие**: `histogram_quantile(0.95, sum(rate(fetcher_download_latency_seconds_bucket[5m])) by (le, platform)) > 300`  
**Severity**: `warning`  
**Действие**: Проверить состояние сети, прокси и YouTube API

**PromQL**:
```promql
histogram_quantile(0.95, 
  sum(rate(fetcher_download_latency_seconds_bucket[5m])) by (le, platform)
) > 300
```

**Аннотации**:
- `summary`: "P95 download latency превысил порог"
- `description`: "P95 latency: {{ $value }}s. Проверьте сеть и прокси."

---

### 9. Queue Depth Exploding

**Название**: `FetcherQueueDepthExploding`  
**Описание**: Глубина очереди слишком большая  
**Условие**: `sum(fetcher_queue_length) > 1000`  
**Severity**: `warning`  
**Действие**: Проверить количество активных воркеров, возможно требуется масштабирование

**PromQL**:
```promql
sum(fetcher_queue_length) > 1000
```

**Аннотации**:
- `summary`: "Глубина очереди слишком большая"
- `description`: "Queue depth: {{ $value }}. Проверьте количество активных воркеров."

---

### 10. Redis Unavailable

**Название**: `FetcherRedisUnavailable`  
**Описание**: Redis недоступен (некритично, но влияет на rate limiting)  
**Условие**: Health check endpoint возвращает `redis.status == "unhealthy"`  
**Severity**: `warning`  
**Действие**: Проверить состояние Redis, возможно требуется перезапуск

**PromQL** (через метрику health check):
```promql
fetcher_health_redis_status == 0
```

**Аннотации**:
- `summary`: "Redis недоступен"
- `description`: "Fetcher не может подключиться к Redis. Rate limiting может не работать."

---

## Информационные алерты (Info)

### 11. High Ingestion Throughput

**Название**: `FetcherHighIngestionThroughput`  
**Описание**: Высокий throughput ингестии (информационный)  
**Условие**: `sum(rate(fetcher_videos_downloaded_total[5m])) > 50`  
**Severity**: `info`  
**Действие**: Мониторинг, возможно требуется масштабирование

**PromQL**:
```promql
sum(rate(fetcher_videos_downloaded_total[5m])) > 50
```

**Аннотации**:
- `summary`: "Высокий ingestion throughput"
- `description`: "Throughput: {{ $value }}/min. Рассмотрите возможность масштабирования."

---

## Настройка в Prometheus Alertmanager

Пример конфигурации `alertmanager.yml`:

```yaml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      continue: true
    - match:
        severity: warning
      receiver: 'warning-alerts'
      continue: true
    - match:
        severity: info
      receiver: 'info-alerts'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://alertmanager-webhook:5001/webhook'

  - name: 'critical-alerts'
    webhook_configs:
      - url: 'http://alertmanager-webhook:5001/webhook'
    # Можно добавить email, Slack, PagerDuty и т.д.

  - name: 'warning-alerts'
    webhook_configs:
      - url: 'http://alertmanager-webhook:5001/webhook'

  - name: 'info-alerts'
    webhook_configs:
      - url: 'http://alertmanager-webhook:5001/webhook'
```

---

## Правила Prometheus

Пример файла `fetcher_alerts.yml` для Prometheus:

```yaml
groups:
  - name: fetcher_critical
    interval: 30s
    rules:
      - alert: FetcherHighFailureRate
        expr: sum(rate(fetcher_videos_failed_total[5m])) > 10
        for: 2m
        labels:
          severity: critical
          service: fetcher
        annotations:
          summary: "Fetcher failure rate превысил порог"
          description: "Failure rate: {{ $value }}/min. Проверьте логи и состояние платформ."

      - alert: FetcherCircuitBreakerTripped
        expr: sum(rate(circuit_breaker_tripped_total[5m])) > 5
        for: 2m
        labels:
          severity: critical
          service: fetcher
        annotations:
          summary: "Circuit breaker сработал слишком часто"
          description: "Срабатываний: {{ $value }}/min. Проверьте прокси и YouTube API."

  - name: fetcher_warning
    interval: 30s
    rules:
      - alert: FetcherLowCacheHitRatio
        expr: |
          sum(rate(fetcher_cache_hits_total[5m])) / 
          (sum(rate(fetcher_cache_hits_total[5m])) + sum(rate(fetcher_cache_miss_total[5m]))) < 0.3
        for: 5m
        labels:
          severity: warning
          service: fetcher
        annotations:
          summary: "Cache hit ratio ниже порога"
          description: "Cache hit ratio: {{ $value | humanizePercentage }}. Проверьте логику кеширования."

      - alert: FetcherHighYouTube429Rate
        expr: sum(rate(fetcher_youtube_429_total[5m])) > 10
        for: 2m
        labels:
          severity: warning
          service: fetcher
        annotations:
          summary: "Слишком много 429 ошибок от YouTube"
          description: "429 ошибок: {{ $value }}/min. Проверьте rate limiting и прокси."
```

---

## Связанные документы

- [FETCHER_OBSERVABILITY.md](./FETCHER_OBSERVABILITY.md) — полное описание метрик и observability
- [GRAFANA_DASHBOARD.md](./GRAFANA_DASHBOARD.md) — описание структуры Grafana dashboard
- [checklist.md](./checklist.md) — Phase 2 — Observability

---

## Примечания

- Пороги алертов могут быть скорректированы на основе реальных данных в production
- Некоторые алерты (circuit breaker, proxy failure rate) будут актуальны после Phase 3 (Security & Privacy)
- Алерты для queue depth будут актуальны после Phase 4 (Queue & Orchestration)
- Health check метрики нужно будет добавить в Prometheus (через отдельный exporter или через метрики в `/metrics` endpoint)
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
