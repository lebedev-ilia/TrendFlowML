# Dataset Collector — локальный мониторинг

Prometheus + Grafana для `discover --metrics-port 9095`.

## Быстрый старт

1. Запустите discover с метриками (на хосте):

```bash
cd Fetcher
source .fetcher_venv/bin/activate
python -m fetcher.dataset_collector.cli discover dataset_campaign.json \
  --category Sport \
  --metrics-port 9095
```

2. Поднимите стек:

```bash
cd Fetcher/monitoring
docker compose up -d
```

3. Откройте:

| Сервис | URL | Логин |
|--------|-----|-------|
| **Grafana** | http://127.0.0.1:3001 | admin / admin |
| **Prometheus** | http://127.0.0.1:9090 | — |

Дашборд **Dataset Collector** появится в папке *Dataset Collector* (home dashboard).

## Проверка scrape

```bash
# метрики на хосте
curl -s http://127.0.0.1:9095/metrics | head

# target в Prometheus
open http://127.0.0.1:9090/targets
```

Job `dataset-collector` → **UP**.

## Grafana пишет «Нет данных»

1. **URL** — именно http://127.0.0.1:3001 (не 3000), логин `admin` / `admin`.
2. **Discover** должен быть запущен с `--metrics-port 9095` (пока процесс жив, метрики на хосте).
3. **Категория** вверху дашборда — выберите **Sport**, не «All» (значение All ломает фильтр).
4. **Время** — «Last 30 minutes» (или шире, пока идёт сессия).
5. Обновите дашборд: `docker compose restart grafana` в `Fetcher/monitoring`, затем Ctrl+Shift+R в браузере.

Проверка в Prometheus: http://127.0.0.1:9090/graph → запрос `dataset_collector_session_accepted` — должно быть число ~700+.

## Остановка

```bash
docker compose down
```

## Полное распределение по диску (без Grafana)

```bash
python -m fetcher.dataset_collector.cli status dataset_campaign.json | python -m json.tool

Inventory gauges (queues, HF, local downloads): `dataset_collector_download_queue_pending`, `dataset_collector_videos_on_hf`, `dataset_collector_shards_on_hf`, etc. See runbook section «Инвентарь».
```

Секция `distributions`: min/p50/p90/p99 по view, like, comment, duration из всех shards.
