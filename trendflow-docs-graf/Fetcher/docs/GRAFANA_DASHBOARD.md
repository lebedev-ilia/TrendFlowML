## Grafana Dashboard для Fetcher

Этот документ описывает структуру и панели Grafana dashboard для мониторинга Fetcher.

---

## Общая структура

Dashboard состоит из следующих разделов (rows):

1. **Ingestion Overview** — общая статистика по ингестии
2. **Queue & Worker Health** — состояние очередей и воркеров
3. **Cache Efficiency** — эффективность кеширования
4. **Download Performance** — производительность скачивания видео
5. **Platform Health** — здоровье платформ (YouTube, TikTok, etc.)
6. **Error Analysis** — анализ ошибок

---

## 1. Ingestion Overview

### 1.1. Ingestion Throughput

**Тип**: Stat panel  
**Метрика**: `rate(fetcher_videos_downloaded_total[5m])`  
**Описание**: Количество успешно скачанных видео в минуту (за последние 5 минут)

**Дополнительные панели**:
- **Videos per hour**: `rate(fetcher_videos_downloaded_total[1h]) * 3600`
- **Videos per day**: `rate(fetcher_videos_downloaded_total[1d]) * 86400`

### 1.2. Success vs Failed Ratio

**Тип**: Pie chart или Bar gauge  
**Метрики**:
- Success: `sum(rate(fetcher_videos_downloaded_total[5m]))`
- Failed: `sum(rate(fetcher_videos_failed_total[5m]))`

**Описание**: Соотношение успешных и неуспешных ингестий

### 1.3. Total Videos Downloaded (Counter)

**Тип**: Stat panel  
**Метрика**: `sum(fetcher_videos_downloaded_total)`  
**Описание**: Общее количество скачанных видео (с момента запуска)

---

## 2. Queue & Worker Health

> **Примечание**: Эти панели будут актуальны после интеграции с очередью (Celery/Redis). Пока можно оставить placeholder'ы.

### 2.1. Queue Depth by Type

**Тип**: Bar gauge или Time series  
**Метрики** (будущие):
- Metadata queue: `fetcher_queue_length{type="metadata"}`
- Video download queue: `fetcher_queue_length{type="download"}`
- Comments queue: `fetcher_queue_length{type="comments"}`
- Finalize queue: `fetcher_queue_length{type="finalize"}`

**Описание**: Глубина очередей по типам задач

### 2.2. Active Workers

**Тип**: Stat panel  
**Метрика** (будущая): `fetcher_active_workers{type="..."}`  
**Описание**: Количество активных воркеров по типам

---

## 3. Cache Efficiency

### 3.1. Cache Hit Ratio

**Тип**: Gauge  
**Метрика**: 
```promql
sum(rate(fetcher_cache_hits_total[5m])) / 
(sum(rate(fetcher_cache_hits_total[5m])) + sum(rate(fetcher_cache_miss_total[5m])))
```

**Описание**: Доля cache hits от общего числа запросов (0.0 - 1.0)

**Alert**: Если cache hit ratio < 0.3 (30%), отправлять warning

### 3.2. Cache Hits/Misses Over Time

**Тип**: Time series (stacked area)  
**Метрики**:
- Cache hits: `sum(rate(fetcher_cache_hits_total[5m]))`
- Cache misses: `sum(rate(fetcher_cache_miss_total[5m]))`

**Описание**: Динамика cache hits и misses по времени

---

## 4. Download Performance

### 4.1. Download Latency Distribution

**Тип**: Histogram  
**Метрика**: `fetcher_download_latency_seconds`  
**Описание**: Распределение времени скачивания видео (гистограмма)

**Панели**:
- **P50 (median)**: `histogram_quantile(0.5, sum(rate(fetcher_download_latency_seconds_bucket[5m])) by (le, platform))`
- **P95**: `histogram_quantile(0.95, ...)`
- **P99**: `histogram_quantile(0.99, ...)`

### 4.2. Download Latency by Platform

**Тип**: Time series  
**Метрика**: `rate(fetcher_download_latency_seconds_sum[5m]) / rate(fetcher_download_latency_seconds_count[5m])`  
**Описание**: Среднее время скачивания по платформам

**Labels**: `platform` (youtube, tiktok, etc.)

### 4.3. Metadata & Comments Latency

**Тип**: Time series  
**Метрики**:
- Metadata latency: `rate(fetcher_metadata_latency_seconds_sum[5m]) / rate(fetcher_metadata_latency_seconds_count[5m])`
- Comments latency: `rate(fetcher_comments_latency_seconds_sum[5m]) / rate(fetcher_comments_latency_seconds_count[5m])`

**Описание**: Среднее время выполнения metadata и comments worker'ов

---

## 5. Platform Health

### 5.1. YouTube Rate Limit Errors (429)

**Тип**: Time series  
**Метрика**: `sum(rate(fetcher_youtube_429_total[5m])) by (operation)`  
**Описание**: Количество 429 ошибок от YouTube по операциям (metadata, download, comments)

**Alert**: Если rate > 10/min, отправлять warning

### 5.2. YouTube 403 Errors

**Тип**: Time series  
**Метрика**: `sum(rate(fetcher_youtube_403_total[5m])) by (operation, error_code)`  
**Описание**: Количество 403 ошибок от YouTube (geo-block, age-restriction, bot detection)

### 5.3. Proxy Failure Rate

**Тип**: Gauge  
**Метрика** (будущая): `proxy_failure_rate{proxy_id="...", country="..."}`  
**Описание**: Доля неуспешных запросов по каждому прокси

**Alert**: Если failure rate > 0.5 (50%), отправлять critical alert

### 5.4. Circuit Breaker Trips

**Тип**: Counter  
**Метрика** (будущая): `sum(rate(circuit_breaker_tripped_total[5m])) by (operation, reason)`  
**Описание**: Количество срабатываний circuit breaker'а

---

## 6. Error Analysis

### 6.1. Failed Ingestions by Reason

**Тип**: Pie chart или Bar gauge  
**Метрика**: `sum(rate(fetcher_videos_failed_total[5m])) by (platform, reason)`  
**Описание**: Распределение ошибок по причинам (rate_limit, download_failed, metadata_failed, etc.)

### 6.2. Error Rate Over Time

**Тип**: Time series  
**Метрика**: `sum(rate(fetcher_videos_failed_total[5m]))`  
**Описание**: Динамика ошибок по времени

**Alert**: Если error rate > 5/min, отправлять warning

---

## 7. Рекомендуемые алерты

### Critical

1. **High error rate**: `sum(rate(fetcher_videos_failed_total[5m])) > 10`
2. **Circuit breaker tripped**: `sum(rate(circuit_breaker_tripped_total[5m])) > 5`
3. **Proxy failure rate high**: `proxy_failure_rate > 0.8`

### Warning

1. **Low cache hit ratio**: `cache_hit_ratio < 0.3`
2. **High YouTube 429 rate**: `sum(rate(fetcher_youtube_429_total[5m])) > 10`
3. **High download latency**: `histogram_quantile(0.95, ...) > 300` (5 минут)

---

## 8. Переменные (Variables) для Dashboard

- `$platform`: Выбор платформы (youtube, tiktok, all)
- `$time_range`: Временной диапазон (5m, 1h, 6h, 24h)

---

## 9. Связанные документы

- `FETCHER_OBSERVABILITY.md` — полное описание метрик и observability
- `checklist.md` — Phase 2 — Observability
- Prometheus configuration для scraping метрик Fetcher

---

## 10. Пример конфигурации Prometheus

```yaml
scrape_configs:
  - job_name: 'fetcher'
    static_configs:
      - targets: ['fetcher:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

---

## 11. Примечания для реализации

- Dashboard можно создать вручную в Grafana UI или **импортировать готовый JSON**: см. `docs/grafana/fetcher-dashboard.json` и `docs/grafana/README.md`.
- Глубина очередей отображается метрикой `fetcher_celery_queue_pending{queue="..."}` (см. раздел 2).
- Готовые правила алертов для Prometheus: `docs/grafana/fetcher_alerts.yaml`.
- После реализации proxy rotation и circuit breaker метрики `proxy_failure_rate` и `circuit_breaker_tripped_total` уже объявлены и используются в дашборде и алертах.
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
