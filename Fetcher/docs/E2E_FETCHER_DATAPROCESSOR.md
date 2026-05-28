# E2E: Fetcher → DataProcessor

Чеклист и сценарии для проверки сквозного потока от Fetcher до DataProcessor.

---

## Предусловия

- Fetcher запущен (API, Celery workers, Redis, PostgreSQL, MinIO/S3).
- DataProcessor запущен и принимает задачи (очередь или HTTP API).
- Доступ к YouTube (или мок) для тестового видео.

---

## Чеклист E2E

### 1. Создание run через Fetcher API

- [ ] **POST /api/v1/runs** с валидным YouTube URL.
  - Тело: `{"run_id": "<uuid>", "source_url": "https://www.youtube.com/watch?v=...", "platform": "youtube"}`.
  - Ожидание: `201 Created`, в ответе `run_id`, `status: "pending"` (или аналог).
- [ ] Проверка дедупликации: повторный POST с тем же видео (другой run_id) при включённой дедупликации возвращает существующий run или 409.
- [ ] **GET /api/v1/runs/{run_id}**: статус run обновляется (pending → fetching_metadata → … → completed или failed).

### 2. Прогресс и артефакты

- [ ] **GET /api/v1/runs/{run_id}**: поле `progress` и/или `artifacts` заполняются по мере выполнения.
- [ ] **GET /api/v1/runs/{run_id}/manifest**: после успешного завершения возвращается `manifest.json` (video_id, platform, storage_keys, checksums, duration_seconds).
- [ ] **GET /api/v1/runs/{run_id}/artifacts**: возвращаются артефакты с signed URLs (video, meta, comments).

### 3. Интеграция с DataProcessor (и Backend, Phase 2)

- [ ] После `COMPLETED` Fetcher вызывает Backend: `POST {FETCHER_BACKEND_BASE_URL}/api/runs/{run_id}/trigger-processing` (см. конфиг `backend_base_url`, `backend_trigger_api_key`). Backend ставит задачу `process_ingestion_run(run_id)`.
- [ ] Backend забирает у Fetcher manifest и артефакты (signed URL видео), скачивает видео и отправляет запрос в DataProcessor API. DataProcessor запускает пайплайн (segmenter, visual и т.д.).
- [ ] Результаты DataProcessor сохраняются в ожидаемое хранилище (NPZ, manifest и т.д.).

Документация интеграции: `backend/docs/FETCHER_INTEGRATION.md`, `docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md`.

### 4. Ошибки и отмена

- [ ] **PATCH /api/v1/runs/{run_id}** с `cancel_requested: true`: run переходит в отменённое состояние на следующем checkpoint.
- [ ] При недоступности DataProcessor или backpressure: Fetcher не падает, повторяет или откладывает постановку задачи (согласно backpressure logic).

### 5. Кеш и повторный запрос

- [ ] Повторный запрос на то же видео (тот же platform_video_id): Fetcher отдаёт cache hit, не скачивает заново, при необходимости переиспользует артефакты и снова ставит задачу в DataProcessor (если нужно переобработать).

---

## Пример ручной проверки (curl)

```bash
# 1. Создать run
RUN_ID=$(uuidgen)
curl -s -X POST "http://localhost:8000/api/v1/runs" \
  -H "Content-Type: application/json" \
  -d "{\"run_id\": \"$RUN_ID\", \"source_url\": \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\", \"platform\": \"youtube\"}"

# 2. Смотреть статус
curl -s "http://localhost:8000/api/v1/runs/$RUN_ID" | jq .

# 3. После completed — запросить manifest
curl -s "http://localhost:8000/api/v1/runs/$RUN_ID/manifest" | jq .

# 4. Артефакты (signed URLs)
curl -s "http://localhost:8000/api/v1/runs/$RUN_ID/artifacts" | jq .
```

---

## Автоматизация (опционально)

- Скрипт в `scripts/` или pytest в `tests/e2e/` может выполнять шаги 1–4: создать run, ждать completed/failed с таймаутом, проверить manifest и артефакты, при наличии DataProcessor — проверить появление результата в хранилище.
- Переменные окружения: `FETCHER_API_URL`, `DATAPROCESSOR_API_URL`, тестовый YouTube URL (или мок).

---

## Связанные документы

- `BACKEND_CONTRACTS.md` — контракты run_id, manifest, события.
- `checklist.md` — Phase 5 (ML Pipeline Compatibility), Phase 6 (REST API).
- DataProcessor: документация по приёму задач и формату manifest.
