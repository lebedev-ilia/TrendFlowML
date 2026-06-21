# Grafana Dashboard для Fetcher

## Импорт дашборда

1. В Grafana: **Dashboards** → **New** → **Import**.
2. Загрузите файл `fetcher-dashboard.json` или вставьте его содержимое.
3. Выберите Prometheus datasource (если переменная `datasource` не подставилась — выберите вручную в настройках дашборда).
4. Сохраните дашборд.

## Панели

- **Ingestion Overview**: throughput, total downloaded, failed/min, cache hit ratio.
- **Queue & Worker Health**: глубина очередей Celery (`fetcher_celery_queue_pending`) по очередям и во времени.
- **Cache & Download Performance**: cache hits/misses, P50/P95 download latency.
- **Platform Health & Errors**: YouTube 429/403, failures по reason, circuit breaker, DataProcessor queue (backpressure).

## Требования

- Prometheus собирает метрики с Fetcher (`/metrics`, см. `GRAFANA_DASHBOARD.md` → раздел 10).
- В Prometheus настроен scrape на Fetcher (например, `fetcher:8000`).

## Алерты

Готовые правила алертов см. в `MONITORING_ALERTS.md`. Их нужно добавить в Prometheus Alertmanager или в раздел Alerting дашборда/панелей Grafana.
---

## Навигация

[Fetcher](../INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
