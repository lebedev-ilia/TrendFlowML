# E2E: YouTube URL → Ingestion → DataProcessor → результат (Phase 5)

Чеклист для проверки полного цикла «создание run по YouTube URL → Fetcher → Backend → DataProcessor → результат». См. [BACKEND_FETCHER_INTEGRATION_ANALYSIS.md](./BACKEND_FETCHER_INTEGRATION_ANALYSIS.md), [backend/docs/FETCHER_INTEGRATION.md](../backend/docs/FETCHER_INTEGRATION.md).

**Подробная настройка и все исправления для ручного E2E** (Backend, Fetcher Docker, Celery очереди, нормализация URL без сети, импорты в tasks): [E2E_MANUAL_SETUP_AND_FIXES.md](./E2E_MANUAL_SETUP_AND_FIXES.md).

**Сборка Segmenter + Audio + Visual без Text** после готового Fetcher run: [backend/docs/E2E_PIPELINE_NO_TEXT.md](../backend/docs/E2E_PIPELINE_NO_TEXT.md).

---

## Предусловия

- [ ] Backend запущен (API + Celery worker + Celery beat для синхронизации статуса).
- [ ] Fetcher запущен и доступен по `TF_BACKEND_FETCHER_API_URL`.
- [ ] DataProcessor API запущен и доступен по `TF_BACKEND_DATAPROCESSOR_API_URL`.
- [ ] В Fetcher заданы `FETCHER_BACKEND_BASE_URL` и при необходимости `FETCHER_BACKEND_TRIGGER_API_KEY` для вызова trigger-processing после finalize.
- [ ] Redis доступен (Backend и Celery).
- [ ] PostgreSQL с применёнными миграциями (в т.ч. `0003_add_ingestion_runs`, `0004_ingestion_runs_fetcher_fields`).
- [ ] Получен JWT (например `POST /api/auth/login`) для вызовов Backend.

---

## Шаги

### 1. Создать run по YouTube URL

- [ ] `POST /api/runs` с телом `{"source_url": "https://www.youtube.com/watch?v=...", "workspace_id": "..." (опционально)}` и заголовком `Authorization: Bearer <token>`.
- [ ] Ответ 201, в теле `run_id`, `ingestion_status` (например `pending` или `running`), `source_url`.
- [ ] При повторном запросе с тем же `Idempotency-Key` — 201 с тем же `run_id` (идемпотентность).

### 2. Проверить статус в Fetcher (опционально)

- [ ] В Fetcher по `run_id`: run появился, статус переходит PENDING → RUNNING → COMPLETED (или FAILED при ошибке).

### 3. Синхронизация статуса в Backend (Phase 4)

- [ ] Celery beat запускает `sync_ingestion_run_status` по расписанию.
- [ ] `GET /api/runs/{run_id}` — поля `ingestion_status`, `fetcher_stage`, `fetcher_error_code`, `fetcher_error_message` обновляются (при успехе: `ingestion_status` → `completed`, при ошибке Fetcher — `failed` и заполнены error-поля).
- [ ] Подключение к WebSocket `GET /api/runs/{run_id}/events` — приходят события `run.status_changed`, `run.stage_changed`.

### 4. Trigger processing (после COMPLETED в Fetcher)

- [ ] Fetcher после finalize вызывает `POST {TF_BACKEND_FETCHER_BASE_URL}/api/runs/{run_id}/trigger-processing` (с `X-API-Key` при необходимости).
- [ ] Backend возвращает 202 Accepted и ставит задачу `process_ingestion_run(run_id)`.

### 5. Обработка DataProcessor

- [ ] Задача `process_ingestion_run` получает manifest и artifacts из Fetcher, передаёт в DataProcessor `video_url` (signed URL).
- [ ] DataProcessor скачивает видео в кэш, обрабатывает (segmenter, visual по умолчанию).
- [ ] В Backend `ingestion_status` переходит в `completed` при успехе или `failed` при ошибке (по логике задачи).

### 6. Результат

- [ ] Для run ингестии результат обработки (артефакты, manifest) остаётся в DataProcessor/result_store; при необходимости последующая интеграция с Backend (например сохранение ссылок в БД) может быть добавлена отдельно. Текущий критерий: run в Backend имеет `ingestion_status=completed` после успешного trigger и выполнения DataProcessor.

---

## Ошибки (Phase 5.2)

- [ ] **Fetcher недоступен при создании run:** `POST /api/runs` возвращает 502, в БД run с `ingestion_status=failed`, в ответе/`GET /api/runs/{run_id}` — понятное сообщение (например `Fetcher service error: ...`).
- [ ] **Таймаут Fetcher:** при таймауте при создании run — 502, run в БД с `failed` и сохранённым сообщением об ошибке.
- [ ] **Run failed в Fetcher:** после синхронизации `GET /api/runs/{run_id}` показывает `ingestion_status=failed`, `fetcher_error_code`, `fetcher_error_message`.

---

## Известные ограничения (на момент Phase 5)

- Результаты обработки DataProcessor для ingestion run (артефакты, прогнозы) не автоматически линкуются с сущностями Backend (Video, AnalysisJob); run остаётся записью в `ingestion_runs` с финальным статусом.
- WebSocket `/api/runs/{run_id}/events` не проверяет JWT (см. API.md).
- Отмена (cancel) ingestion run и распространение отмены до Fetcher/DataProcessor не реализованы (Phase 5.4 опционально).
