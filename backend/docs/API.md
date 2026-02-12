# API (REST + WebSocket)

Все endpoints определены в `backend/app/routers/*`.

## 1) Auth

- `POST /api/auth/register`  
  body: `{email, password}` → `UserOut`
- `POST /api/auth/login`  
  body: `{email, password}` → `{access_token, token_type}`
- `GET /api/auth/me`  
  header: `Authorization: Bearer <token>` → `UserOut`

OAuth/2FA/reset — TODO (см. `GAPS_AND_ALIGNMENT.md`).

## 2) Videos / Upload

- `POST /api/videos/upload/init` → `{upload_id, video_id}`
- `PUT /api/videos/upload/{upload_id}`  
  multipart file (field `file`) → `{status:"ok"}`
- `POST /api/videos/upload/complete`  
  query: `upload_id=<id>` → `VideoOut`
- `GET /api/videos` → `[VideoOut]`

## 3) Profiles

- `GET /api/profiles` → `[ProfileOut]` (public)
- `GET /api/my/profiles` → `[ProfileOut]`
- `POST /api/my/profiles` → `ProfileOut`
- `PUT /api/my/profiles/{profile_id}` → `ProfileOut`
- `DELETE /api/my/profiles/{profile_id}` → `{status:"ok"}`

## 4) Runs

- `POST /api/runs`  
  body: `{video_id, profile_id?}` → `RunOut`
- `GET /api/runs` → `[RunOut]`
- `GET /api/runs/{run_id}` → `RunOut`
- `POST /api/runs/{run_id}/cancel` → `{status:"ok"}`
- `GET /api/runs/{run_id}/logs` → `[RunLogOut]`
- `GET /api/runs/{run_id}/manifest` → JSON manifest
- `GET /api/runs/{run_id}/result` → `{run_id, manifest, artifacts}`
- `GET /api/runs/{run_id}/artifact`  
  query: `object_key=<path>&token=<bearer>` → binary content
- `GET /api/runs/{run_id}/events` → WebSocket (live events)

## 5) Admin

Только админы:

- `GET /api/admin/users`
- `POST /api/admin/users`
- `PUT /api/admin/users/{id}`
- `DELETE /api/admin/users/{id}`

## 6) Авторизация

Все REST endpoints (кроме `register/login`) защищены JWT:

- заголовок: `Authorization: Bearer <token>`

WS endpoint `/api/runs/{run_id}/events` **в текущей реализации не проверяет JWT**.
Для HTML артефактов используется `token` в query‑param.

