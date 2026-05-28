# Events and logging

Транспорт событий к WebSocket: **Redis pub/sub**, канал `run:{run_id}` (`app/services/events.py`). Обоснование: [adr/0001-celery-redis-pubsub-manifest-source-of-truth.md](adr/0001-celery-redis-pubsub-manifest-source-of-truth.md).

## 1) WebSocket events

Endpoint: `GET /api/runs/{run_id}/events?token=<JWT>` (WebSocket upgrade)

**Авторизация:** JWT в query (тот же токен, что для `Authorization: Bearer` на REST). Без валидного токена или для чужого run подключение отклоняется (закрытие **1008**). См. `SECURITY.md` §3.

В текущей реализации это **live‑only** поток.
Историю нужно получать из REST (`/logs`, `/runs/{id}`).

**Phase 4 (ingestion):** для run'ов ингестии (YouTube и др.) события публикуются при синхронизации статуса из Fetcher (задача `sync_ingestion_run_status`): при изменении статуса — `run.status_changed`, при смене стадии — `run.stage_changed`. Клиент подключается к `WS /api/runs/{run_id}/events` и получает те же форматы событий.

Формат payload зависит от источника, но базовые типы:

- `run.status_changed`
- `run.stage_changed`
- `component.started`
- `component.finished`
- `log.line`

## 2) Redis pubsub

Механика:

- `publish_run_event(run_id, payload)` публикует в Redis канал `run:{run_id}`
- `subscribe_run_events(run_id)` читает события и отдаёт их WS

Код: `backend/app/services/events.py`

## 3) Logs

Логи DataProcessor пишутся:

- stdout → `run_logs` c `level=info`
- stderr → `run_logs` c `level=error`

Клиент может:

- получать live `log.line` через WS
- получить всю историю через `GET /api/runs/{run_id}/logs`

