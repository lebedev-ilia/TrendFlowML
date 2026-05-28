# Prometheus + Grafana: локальный E2E-стек (задокументировано для Audit v4 / батч 60+)

**Статус:** на **2026-04-22** связка **Postgres/Redis/MinIO + Backend/Fetcher/DataProcessor + Prometheus + Grafana** поднимается штатными скриптами и **работает** (два scrape target’а, дашборд в Grafana, метрики воркера не теряются в отдельном процессе).

**Не заменяет** боевую среду батча: URL retention и доступы **прод/стенда** владелец фиксирует отдельно (чеклист [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) п. **4.2**, **4.7**).

---

## 1. Как поднять (канон)

Из корня репозитория (или `backend/`), с полной инфраструктурой:

```bash
./backend/scripts/start_e2e_stack.sh --with-infra
```

- **`--with-infra`** вызывает [setup_e2e_infra.sh](../../../backend/scripts/setup_e2e_infra.sh): Postgres, Redis, MinIO, миграции, MinIO buckets, **шаг [6/6]**: Prometheus + Grafana в Docker (compose из `DataProcessor/`, override `monitoring/docker-compose.prometheus-override-e2e.yml` — scrape **DataProcessor с хоста**).
- Затем скрипт стартует **на хосте**: Backend, Fetcher, **DataProcessor API** (`:8002`), **DataProcessor worker** и при наличии deps — **Embedding Service** (`:8005`).

Порты, не пересекающиеся с backend-api (**8001**): см. [e2e_env.sh](../../../backend/scripts/e2e_env.sh) — в т.ч. `DP_WORKER_METRICS_PORT` по умолчанию **8003** (метрики воркера **не** на 8001).

---

## 2. URL и порты (локально)

| Сервис | Как определить порт | Примечание |
|--------|----------------------|------------|
| **Grafana** | Обычно **3000**; если занят — авто-выбор в `setup_e2e_infra` | Учётка по умолчанию в compose: **admin** / **admin** |
| **Prometheus** | Обычно **9090**; при занятом 9090 — **9091+** (подбор свободного) | UI: **Status → Targets** |
| Фактические порты после `--with-infra` | `backend/.e2e/state/monitoring_ports.env` | `E2E_PROMETHEUS_HOST_PORT`, `E2E_GRAFANA_HOST_PORT` |

**DataProcessor:** API `http://127.0.0.1:8002`, метрики воркера `http://127.0.0.1:8003/metrics` (при `DP_WORKER_METRICS_PORT=8003`).

---

## 3. Scrape-конфигурация (суть)

- **API:** `GET http://<host>:8002/api/v1/metrics` — очередь, активные run, часть бизнес-метрик (процесс **uvicorn**).
- **Worker:** `GET http://<host>:8003/metrics` — гистограммы `dataprocessor_processing_seconds` / счётчики `dataprocessor_failures_total` для пайплайна (процесс **worker**); без отдельного HTTP на воркере эти серии **не** попадут в Prometheus, если скрейпить только API.
- **Prometheus в Docker** (E2E): `host.docker.internal` + порты 8002 / 8003 в [prometheus.e2e_host.yml](../../monitoring/prometheus/prometheus.e2e_host.yml).

Полный разбор путей: [monitoring/README.md](../../monitoring/README.md) (разделы «Батч 60+», «Пилот 15»).

---

## 4. Grafana: дашборд

- Провижининг: [monitoring/grafana/provisioning/](../../monitoring/grafana/provisioning/) — datasource Prometheus, дашборд **DataProcessor API - Overview** ([dataprocessor-overview.json](../../monitoring/grafana/dashboards/dataprocessor-overview.json)).
- **Операционные** метрики (очередь, p95, failures, `processor`/`component`); **не** содержимое NPZ/фич — см. [METRICS_REFERENCE.md](../../monitoring/METRICS_REFERENCE.md).

---

## 5. Проверка «всё работает»

1. `http://localhost:<prometheus_port>/targets` — job’ы `dataprocessor-api-e2e-host` и `dataprocessor-worker-e2e-host` (или имена из вашего `prometheus.yml`) в **UP**.
2. `curl -s "http://127.0.0.1:8002/api/v1/metrics" | head` и `curl -s "http://127.0.0.1:8003/metrics" | head` — текст Prometheus.
3. В Grafana **Explore** — запрос `{__name__=~"dataprocessor_.*"}[5m]`.

---

## 6. Останов

- Только приложения: `./backend/scripts/stop_e2e_stack.sh`
- Приложения + MinIO/Redis/Postgres + **Prometheus/Grafana:** `./backend/scripts/stop_e2e_stack.sh --with-infra`

---

## 7. Связь с чеклистом 60+

| П. чеклиста | Статус (локальный E2E) |
|-------------|-------------------------|
| **4.1** | Prometheus скрейпит API **и** (при `DP_WORKER_METRICS_PORT`) воркер — **выполнено** для описанного стека. |
| **4.2** | URL/пароль локальные — **да**; **прод** — отдельная запись. |
| **4.3** | Обзорный дашборд из репо — **да**; проверка на **боевом** datasource — владелец. |
| **4.5** | [METRICS_LABELS_INVENTORY_60PLUS.md](METRICS_LABELS_INVENTORY_60PLUS.md) — **по-прежнему** требуется **подтверждение на пилоте** (1–2 run тем же путём, что батч). |
| **4.6** | Для пути **API → очередь → worker → `main.py`** метрики согласованы; **чисто CLI** без API — вне этого документа, см. чеклист. |
| **4.7** | [monitoring/README.md](../../monitoring/README.md) + этот файл; retention **прод** — TBD. |

---

*Соседние артефакты: [METRICS_REFERENCE.md](../../monitoring/METRICS_REFERENCE.md), [BATCH_60PLUS_OWNER_TASKS.md](BATCH_60PLUS_OWNER_TASKS.md) §5.*
