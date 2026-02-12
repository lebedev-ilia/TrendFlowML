# Events and logging

## 1) WebSocket events

Endpoint: `GET /api/runs/{run_id}/events` (WebSocket)

В текущей реализации это **live‑only** поток.
Историю нужно получать из REST (`/logs`, `/runs/{id}`).

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

