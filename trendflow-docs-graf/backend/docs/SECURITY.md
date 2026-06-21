# Security

## 1) Auth (JWT)

- регистрация: `/api/auth/register`
- логин: `/api/auth/login`
- токен JWT хранится у клиента и передаётся в `Authorization: Bearer <token>`

Проверка токена выполняется в `deps.get_current_user`.

## 2) Admin access

Доступ к `/api/admin/*` разрешён:

- пользователям с `role="admin"`
- или пользователям, чей email входит в `TF_BACKEND_ADMIN_EMAILS`

## 3) WebSocket и артефакты

**WebSocket** `GET /api/runs/{run_id}/events` (**upgrade**): обязателен query‑параметр **`token`**, значение — **JWT** того же формата, что выдаёт `POST /api/auth/login` (`sub` = id пользователя). Доступ только к run'ам, где `IngestionRun.user_id` совпадает с субъектом токена. Иначе соединение закрывается с кодом **1008** (политика/отказ в доступе) до принятия сокета. Реализация: `app/routers/runs.py::ws_run_events`.

Для HTML‑артефактов используется query‑param:

```
GET /api/runs/{run_id}/artifact?object_key=...&token=<bearer>
```

## 4) CORS

Список origin задаётся переменной **`TF_BACKEND_CORS_ORIGINS`**: значение `"*"` (по умолчанию) или перечень URL через запятую. В production задайте явные домены фронтенда; комбинация `allow_credentials=True` с `"*"` допустима только для локальной разработки.

## 5) Сервисные ключи (machine-to-machine)

- **`POST /api/runs/{run_id}/trigger-processing`** (вызов от Fetcher после finalize): если задан **`TF_BACKEND_RUN_TRIGGER_API_KEY`**, требуется заголовок **`X-API-Key`** с тем же значением.
- **Webhooks DataProcessor** (`/api/webhooks/...`): проверка подписи/секрета — см. реализацию в `app/routers/webhooks.py` и тесты `tests/integration/test_webhooks.py`.

Для демо рекрутеру полезно явно разделить **пользовательский JWT** и **ключи между сервисами**.

## 6) JWT secret в production

- Переменная **`TF_BACKEND_JWT_SECRET`** не должна оставаться **пустой** или на **известных демо-значениях** (`change-me`, `demo-change-me-in-production` и т.д., см. `app.config.is_weak_jwt_secret`) в среде **`TF_BACKEND_DEPLOYMENT_ENV=production`** или **`staging`**: при старте API вызывается **`validate_security_at_startup`** — процесс завершится с **`RuntimeError`** (fail-fast).
- В **`development`** (значение по умолчанию) слабый secret допускается для локальной работы, но в лог пишется **предупреждение** — смените секрет перед выкладкой.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
