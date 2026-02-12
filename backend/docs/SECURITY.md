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

WS endpoint `/api/runs/{run_id}/events` **не требует JWT** в текущем коде.
Это известный риск и должен быть закрыт (см. `GAPS_AND_ALIGNMENT.md`).

Для HTML‑артефактов используется query‑param:

```
GET /api/runs/{run_id}/artifact?object_key=...&token=<bearer>
```

## 4) CORS

В `main.py` включён `allow_origins=["*"]` (dev‑режим).
Для prod нужно сузить список доменов.

