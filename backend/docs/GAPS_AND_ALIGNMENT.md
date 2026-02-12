# Gaps and alignment with canonical docs

Этот документ фиксирует расхождения между **каноничными контрактами**
(`DataProcessor/docs/backend/*`, `DataProcessor/docs/contracts/*`) и
фактической реализацией backend в `backend/`.

## 1) Billing

Контракт требует:

- `billing_ledger`
- hold/charge/release
- блокировку run при недостатке средств

В коде backend:

- billing таблиц нет
- нет расчёта cost_units и hold
- `estimated_cost_units` и `actual_cost_units` не используются

## 2) YouTube Fetcher

Контракт требует Fetcher worker (`fetch_video(run_id)`), который:

- качает видео
- собирает метаданные/комменты
- кладёт в object storage
- запускает `process_run(run_id)`

В коде backend:

- поддержан только **upload** путь
- fetcher отсутствует
- `video_sources.youtube_url` не используется

## 3) Idempotency

Контракт требует `Idempotency-Key` для:

- `POST /api/runs`
- `POST /api/videos/upload/complete`

В коде backend идемпотентность не реализована.

## 4) WebSocket auth

Контракт предполагает bearer auth (или mTLS внутри k8s).  
В коде WS endpoint не проверяет токен.

## 5) Cancel semantics

Контракт: “cancel всегда доступен, прекращаем дальнейшие компоненты”.  
В коде:

- cancel только выставляет `cancel_requested_at`
- DataProcessor не останавливается

## 6) Delete endpoints и retention

Контракт: `DELETE /api/videos/{video_id}` удаляет артефакты/кэш/индексы.  
В коде endpoint отсутствует.  
Также отсутствует политика retention сырого видео.

## 7) Validation and limits

Контракты фиксируют:

- min/max длительность видео
- max resolution
- корректная обработка битых файлов

В коде:

- есть только `ffprobe`, без полноценной валидации

## 8) Admin/support roles

Контракт: роли `user/admin/support`.  
В коде: `role` есть, но используется только `admin`.

## 9) Health checks

Контракт: `/health` и `/health/live`.  
В коде endpoint нет.

---

Эти пункты лучше переносить в roadmap/backlog по мере стабилизации backend.

