# Анализ интеграции Backend и Fetcher

Отчёт о текущем состоянии интеграции между **backend** и **Fetcher** в проекте TrendFlowML.

---

## 1. Целевая архитектура (по контрактам)

По `backend/docs/reference/backend_qna_contracts.md` и `Fetcher/docs/BACKEND_CONTRACTS.md` зафиксирован следующий поток:

1. **Backend** создаёт run (UUID `run_id`), пишет запись в свою БД, кладёт в очередь задачу `fetch_video(run_id)`.
2. **Fetcher** (worker) забирает задачу, выполняет ingestion (метаданные, комментарии, скачивание видео), кладёт сырьё в MinIO/S3, строит `manifest.json`, обновляет статус run.
3. **Fetcher** после успешного finalize ставит задачу `process_run(run_id)` для **DataProcessor**.
4. **DataProcessor** обрабатывает run по manifest и артефактам; Backend агрегирует статус и отдаёт результат.

Идентификаторы:

- `run_id` — UUID, **источник правды — Backend** (генерируется при создании run).
- Fetcher использует `run_id` как внешний ключ в своей БД; Backend в Fetcher БД не пишет, только через очередь/API.

---

## 2. Фактическая реализация

### 2.1 Backend

| Аспект | Контракт | Факт |
|--------|----------|------|
| Создание run для YouTube | `POST /api/runs` с YouTube URL, создание run_id, постановка `fetch_video(run_id)` в очередь Fetcher | **Нет**: endpoint `POST /api/runs` отсутствует. Есть только поток Workspace → Channel → Video → Analysis. |
| Конфигурация Fetcher | URL Fetcher API и/или общая очередь Celery с Fetcher | **Нет**: в `backend/app/config.py` есть только `dataprocessor_api_url`, `dataprocessor_api_key` и т.п. Переменных `fetcher_api_url` / `FETCHER_*` нет. |
| Celery task для Fetcher | Вызов или постановка `fetch_video(run_id)` | **Нет**: в `backend/app/tasks.py` только `process_analysis_job(analysis_job_id)`. Упоминаний Fetcher, `fetch_video`, HTTP-клиента к Fetcher нет. |
| Путь видео для анализа | Для YouTube — после Fetcher: артефакты в MinIO, путь из manifest | **Только upload**: в `dataprocessor_adapter.py` путь к видео берётся из `Video.storage_path` или legacy `VideoFile.object_key` (локальный/уже загруженный файл). `video_sources.youtube_url` не используется для запуска Fetcher. |

Источник: `backend/docs/GAPS_AND_ALIGNMENT.md` явно фиксирует:

- «поддержан только **upload** путь»;
- «**fetcher отсутствует**»;
- «`video_sources.youtube_url` не используется».

Текущий поток Backend → DataProcessor:

1. `POST /api/workspaces/{id}/videos/{video_id}/analysis` → создаётся `AnalysisJob`.
2. Celery: `process_analysis_job(analysis_job_id)` → адаптер готовит `DataProcessorPayload` с **локальным `video_path`**.
3. DataProcessor вызывается через HTTP API (`run_dataprocessor_async`) с путями к файловой системе backend.

То есть цепочка **Backend → Fetcher** в коде не реализована.

### 2.2 Fetcher

| Аспект | Ожидание контракта | Факт |
|--------|---------------------|------|
| Кто создаёт run | Backend создаёт run_id и передаёт в Fetcher (очередь или API) | Fetcher API `POST /api/v1/runs` **принимает** `run_id` в теле запроса (`CreateRunRequest.run_id`). То есть контракт «run_id от Backend» заложен, но вызывать этот endpoint Backend не умеет. |
| Собственная БД и очереди | Отдельная БД, свои Celery/Redis | Реализовано: своя БД (например `runs`, `video_sources`, `artifacts`), свои задачи `fetch_metadata`, `download_video`, `fetch_comments`, `finalize`. |
| После finalize | Постановка `process_run(run_id)` в DataProcessor (или вызов Backend) | **Не реализовано**: в `Fetcher/fetcher/tasks.py` в `finalize_task` после перевода run в `COMPLETED` и отправки webhook **нет** вызова Backend/DataProcessor для постановки `process_run(run_id)`. В комментарии указано «5. Enqueue process_run(run_id) в DataProcessor (если нужно)», но кода для этого нет. |
| Интеграция с DataProcessor | Backpressure по очереди DataProcessor | Реализовано: `dataprocessor_api_url`, backpressure через `/api/v1/health` и `/api/v1/metrics` (размер очереди). Используется только для ограничения постановки своих задач, не для вызова process_run. |

Итого: Fetcher рассчитан на то, что **кто-то извне** (по контракту — Backend) передаёт `run_id` и URL (например `POST /api/v1/runs`). Обратная связь «Fetcher завершил → запустить DataProcessor» в коде отсутствует.

### 2.3 Общая очередь Celery

- **Backend** и **Fetcher** используют Celery, но в коде не зафиксировано, что они разделяют один broker (например Redis) и что Backend ставит задачи в очереди Fetcher (`fetch.metadata`, `fetch.video`, `fetch.comments`, `fetch.finalize`).
- Даже при общем broker Backend сейчас **нигде не ставит** задачи Fetcher.

---

## 3. Сводка разрывов

1. **Backend не вызывает Fetcher**  
   Нет ни HTTP-клиента к Fetcher API, ни постановки `fetch_video(run_id)` в очередь. Нет конфигурации Fetcher в backend.

2. **Нет канонического «создания run по YouTube URL» в Backend**  
   Нет `POST /api/runs`; создание сущностей идёт через Workspace/Channel/Video и анализ по уже существующему видео с путём к файлу.

3. **Fetcher не запускает DataProcessor после finalize**  
   Нет шага «enqueue process_run(run_id)» ни в Backend, ни в DataProcessor API. Цепочка Fetcher → DataProcessor оборвана.

4. **Две разные модели «run»**  
   - Backend: legacy `runs` (core) и v2 `AnalysisJob` (core.analysis_jobs); анализ привязан к Video и job.  
   - Fetcher: своя таблица `runs` с `run_id` (UUID), ожидаемым от Backend.  
   Связь между ними (один и тот же run_id, обновление статуса в Backend по событиям Fetcher) не реализована.

5. **Хранение видео**  
   Backend в адаптере DataProcessor опирается на локальные пути (`storage_path`, `VideoFile.object_key`). Fetcher кладёт артефакты в S3/MinIO и формирует manifest. Интеграция «Backend берёт путь/manifest из Fetcher для запуска DataProcessor» отсутствует.

---

## 4. Что уже согласовано и готово к интеграции

- **Контракты** (BACKEND_CONTRACTS, backend_qna): run_id lifecycle, события (run.status_changed, run.stage_changed, log.line), формат manifest.json, backpressure.
- **Fetcher API**: `POST /api/v1/runs` с `run_id`, `source_url`, опционально `platform`, `webhook_url` и т.д.; GET статуса, manifest, artifacts.
- **Fetcher pipeline**: нормализация URL, кеш по (platform, platform_video_id), metadata/video/comments/finalize, manifest, webhook.
- **Backend**: есть модель Run (legacy), события (Redis pubsub), WebSocket; DataProcessor вызывается по job с путями. Не хватает только «входа» через Fetcher и привязки run_id к одному и тому же run в двух системах.

---

## 5. Рекомендации по интеграции

1. **Backend**  
   - Добавить в конфиг URL Fetcher (например `TF_BACKEND_FETCHER_API_URL`) и при необходимости API key.  
   - Реализовать сценарий «анализ по YouTube URL»:  
     - либо канонический `POST /api/runs` (создание run в backend, вызов Fetcher `POST /api/v1/runs` с этим run_id и source_url, при необходимости постановка задачи в очередь Fetcher, если Fetcher ожидает очередь вместо HTTP),  
     - либо создание Video + run/job с последующим вызовом Fetcher API с тем же run_id.  
   - Единый идентификатор: run в backend и run_id в Fetcher должны совпадать (один UUID).

2. **Fetcher**  
   - После успешного finalize реализовать вызов Backend или DataProcessor для запуска обработки:  
     - либо HTTP: Backend `POST /api/runs/{run_id}/trigger-processing` или DataProcessor API приёма run по manifest,  
     - либо постановка задачи в очередь, которую обрабатывает Backend/DataProcessor (например `process_run(run_id)` с параметрами из manifest/артефактов).  
   - Конфигурация: URL Backend или DataProcessor и способ аутентификации.

3. **Данные и пути**  
   - Для сценария «YouTube → Fetcher → DataProcessor» Backend (или DataProcessor) должен получать путь к видео/артефактам из manifest Fetcher (например signed URL или путь в общем storage), а не только из локального `Video.storage_path`.  
   - Либо Fetcher и Backend используют общий object storage (MinIO/S3), и Backend/DataProcessor при запуске `process_run` передаёт в DataProcessor путь/URL из manifest.

4. **События и статусы**  
   - По контракту Fetcher публикует события (run.status_changed, run.stage_changed, log.line). Backend может подписаться на них (Redis/Kafka) и обновлять свой run/ingestion_status и ретранслировать в WebSocket для UI.

5. **E2E и тесты**  
   - Добавить сценарий: Backend создаёт run по YouTube URL → Fetcher получает задачу/API → finalize → Backend/DataProcessor получает process_run → результат появляется в Backend.  
   - Проверить совпадение run_id, формата manifest и ожиданий DataProcessor по артефактам.

---

## 6. План дальнейших шагов

Ниже — поэтапный план интеграции. Фазы можно выполнять последовательно; внутри фазы задачи можно дробить на подзадачи и выносить в тикеты.

### Фаза 0: Подготовка (конфиг и клиент)

**Цель:** Backend умеет вызывать Fetcher по HTTP, без изменения бизнес-потока.

| # | Задача | Где | Результат |
|---|--------|-----|-----------|
| 0.1 | Добавить в Backend конфиг `fetcher_api_url`, опционально `fetcher_api_key` | `backend/app/config.py` | Переменные `TF_BACKEND_FETCHER_API_URL`, `TF_BACKEND_FETCHER_API_KEY` |
| 0.2 | Реализовать HTTP-клиент к Fetcher (POST /api/v1/runs, GET /api/v1/runs/{run_id}, опционально manifest/artifacts) | `backend/app/services/fetcher_client.py` (новый) | Модуль с async/sync вызовами, обработка ошибок и таймаутов |
| 0.3 | Unit-тесты клиента с моками (httpx/responses) | `backend/tests/` | Уверенность, что контракт запрос/ответ соблюдён |

**Критерий готовности:** из Backend можно вызвать создание run в Fetcher по URL и получить статус по run_id (при поднятом Fetcher или моке).

**Статус:** выполнено. Реализовано: `backend/app/config.py` (fetcher_api_url, fetcher_api_key, fetcher_timeout_seconds), `backend/app/services/fetcher_client.py` (create_run, get_run, get_run_manifest, get_run_artifacts; sync и async), `backend/tests/test_fetcher_client.py` (unit-тесты с моками), `backend/docs/FETCHER_INTEGRATION.md` (документация).

---

### Фаза 1: Backend — создание run по YouTube URL и вызов Fetcher

**Цель:** В Backend есть явный сценарий «запуск анализа по YouTube URL» с единым run_id и вызовом Fetcher.

| # | Задача | Где | Результат |
|---|--------|-----|-----------|
| 1.1 | Определить, как связывать run с существующей моделью: новый `POST /api/runs` (legacy Run) или расширение потока Workspace/Channel/Video (создание Video по URL + run/job с тем же run_id) | Дизайн | Выбранный вариант и обновление API.md |
| 1.2 | Реализовать создание run в Backend (запись в БД с run_id, привязка к user/workspace/video или legacy Run) | `backend/app/routers/` + модели | Endpoint создаёт run_id (UUID), сохраняет source_url / platform |
| 1.3 | После создания run вызвать Fetcher API: POST /api/v1/runs с run_id, source_url, при необходимости webhook_url (Backend callback) | Роутер + fetcher_client | Fetcher получает задачу на ingestion для этого run_id |
| 1.4 | Опционально: idempotency для создания run (Idempotency-Key), если контракт требует | Роутер | Повтор запроса с тем же ключом не создаёт дубликат |

**Критерий готовности:** По запросу из Backend создаётся run с run_id и в Fetcher появляется соответствующий run (видно по GET /api/v1/runs/{run_id}).

**Статус:** выполнено. Реализовано: модель `core.ingestion_runs` (IngestionRun), миграция `0003_add_ingestion_runs.py`; роутер `POST /api/runs`, `GET /api/runs`, `GET /api/runs/{run_id}` с вызовом Fetcher после создания записи; поддержка заголовка `Idempotency-Key`; документация в `backend/docs/FETCHER_INTEGRATION.md` и `backend/docs/API.md`.

---

### Фаза 2: Fetcher → Backend/DataProcessor (запуск process_run)

**Цель:** После успешного finalize Fetcher явно инициирует запуск обработки (DataProcessor или через Backend).

| # | Задача | Где | Результат |
|---|--------|-----|-----------|
| 2.1 | Выбрать способ запуска: (A) Fetcher вызывает Backend `POST /api/runs/{run_id}/trigger-processing`, (B) Fetcher вызывает DataProcessor API (если есть endpoint приёма run по manifest), (C) общая очередь Celery — Fetcher ставит задачу, которую обрабатывает Backend | Дизайн | Решение и контракт (формат payload, аутентификация) |
| 2.2 | В Backend: реализовать endpoint «trigger processing» (если выбран A/C). Он по run_id запрашивает у Fetcher manifest/artifacts, формирует payload для DataProcessor и ставит `process_analysis_job` или аналог `process_run(run_id)` | `backend/app/routers/`, `backend/app/tasks.py` | Один run_id ведёт к запуску DataProcessor с правильными путями |
| 2.3 | В Fetcher: в finalize_task после перевода run в COMPLETED вызвать Backend (или DataProcessor) по выбранному в 2.1 способу | `Fetcher/fetcher/tasks.py` | Цепочка Fetcher → Backend/DataProcessor замкнута |
| 2.4 | Конфиг Fetcher: URL Backend (или DataProcessor), API key при необходимости | `Fetcher/fetcher/config.py` | Переменные окружения для production |

**Критерий готовности:** После finalize в Fetcher в Backend/DataProcessor реально запускается обработка для этого run_id (хотя бы до старта пайплайна).

**Статус:** выполнено. Реализовано: (A) Fetcher вызывает Backend: в Backend добавлен `POST /api/runs/{run_id}/trigger-processing` (опциональная аутентификация по `TF_BACKEND_RUN_TRIGGER_API_KEY`); задача `process_ingestion_run(run_id)` забирает manifest и артефакты из Fetcher, скачивает видео во временный файл и запускает DataProcessor. В Fetcher в `finalize_task` после COMPLETED вызывается Backend trigger (конфиг `FETCHER_BACKEND_BASE_URL`, `FETCHER_BACKEND_TRIGGER_API_KEY`). Документация: `backend/docs/FETCHER_INTEGRATION.md`, конфиг Backend и Fetcher.

---

### Фаза 3: Пути к артефактам и manifest

**Цель:** DataProcessor получает входные данные (видео, мета, комментарии) из хранилища Fetcher (MinIO/S3), а не только с локального диска Backend.

| # | Задача | Где | Результат |
|---|--------|-----|-----------|
| 3.1 | Договориться о доступе: общий MinIO/S3 для Backend и Fetcher; либо Fetcher отдаёт signed URL в manifest/artifacts API | Контракт | Как Backend/DataProcessor получают video_path (URL или mount) |
| 3.2 | Backend (или задача process_run): по run_id запрашивать у Fetcher GET manifest и/или artifacts; извлекать путь/URL к video_file | fetcher_client + dataprocessor_adapter | Payload для DataProcessor содержит путь/URL из Fetcher |
| 3.3 | DataProcessor: поддержка входа «видео по URL» или по пути в общем volume (если ещё не поддерживается) | DataProcessor | Запуск по артефактам Fetcher без копирования на локальный диск Backend (по возможности) |
| 3.4 | Обновить dataprocessor_adapter: для run’ов, пришедших из Fetcher, не использовать Video.storage_path, а подставлять путь/URL из manifest | `backend/app/services/dataprocessor_adapter.py` | Единая точка формирования payload под оба сценария (upload и YouTube) |

**Критерий готовности:** Полный цикл YouTube URL → Fetcher → manifest → Backend запускает DataProcessor с корректным путём/URL к видео и результат появляется в Backend.

**Статус:** выполнено. Контракт в `docs/PHASE3_ARTIFACTS_CONTRACT.md` (варианты A/B). Backend: `build_ingestion_payload_from_fetcher(run_id)` в `dataprocessor_adapter.py`; `process_ingestion_run` передаёт в DataProcessor `video_url`. `run_dataprocessor_async` принимает `video_path` или `video_url`. DataProcessor: поле `video_url` в ProcessRequest, скачивание в кэш (`video_url_cache_dir`), конфиг и утилита в `api/utils/video_url_cache.py`. Документация: `backend/docs/FETCHER_INTEGRATION.md` (раздел 6), `docs/PHASE3_ARTIFACTS_CONTRACT.md`.

---

### Фаза 4: События и статусы ingestion в Backend

**Цель:** Backend знает статус ingestion (Fetcher) и отдаёт его в UI (например через WebSocket).

| # | Задача | Где | Результат |
|---|--------|-----|-----------|
| 4.1 | Определить транспорт событий Fetcher → Backend: Redis pubsub (общий канал), Kafka или периодический polling GET /api/v1/runs/{run_id} | Дизайн | Выбран способ и формат |
| 4.2 | Backend: подписка на события Fetcher или polling; обновление полей run (ingestion_status, stage, error_code) | `backend/app/services/events.py` или worker | В БД run отражает этапы Fetcher |
| 4.3 | Ретрансляция в WebSocket тех же типов событий (run.stage_changed, run.status_changed, log.line) для UI | Существующий WS endpoint | Единый поток событий run для фронта |

**Критерий готовности:** В UI виден прогресс Fetcher (стадии, ошибки) до перехода в обработку DataProcessor.

**Статус:** выполнено. Транспорт — **polling**: периодическая задача `sync_ingestion_run_status` (Celery beat, интервал `TF_BACKEND_INGESTION_SYNC_INTERVAL_SECONDS`, по умолчанию 20 с) опрашивает Fetcher `GET /api/v1/runs/{run_id}` для run'ов в статусе pending/running; обновляет в БД поля `ingestion_status`, `fetcher_stage`, `fetcher_error_code`, `fetcher_error_message` (миграция `0004_ingestion_runs_fetcher_fields.py`); публикует в Redis события `run.status_changed` и `run.stage_changed`. WebSocket endpoint `GET /api/runs/{run_id}/events` подписывается на канал `run:{run_id}` и пересылает события клиенту. REST `GET /api/runs/{run_id}` возвращает новые поля. Документация: `backend/docs/FETCHER_INTEGRATION.md` (раздел 7), `backend/docs/EVENTS_AND_LOGGING.md`, `backend/app/worker.py` (beat_schedule).

---

### Фаза 5: E2E, отказоустойчивость и документация

**Цель:** Надёжный сквозной сценарий и актуальная документация.

| # | Задача | Где | Результат |
|---|--------|-----|-----------|
| 5.1 | E2E-тест или ручной сценарий: Backend создаёт run по YouTube URL → Fetcher → finalize → Backend запускает DataProcessor → результат (артефакты/прогнозы) в Backend | `backend/tests/e2e/` или `docs/` | Чеклист и при необходимости автоматический тест |
| 5.2 | Обработка ошибок: Fetcher недоступен, таймаут, run failed; повторные попытки или понятное сообщение пользователю | Backend + при необходимости Fetcher | Предсказуемое поведение при сбоях |
| 5.3 | Обновить GAPS_AND_ALIGNMENT.md, OVERVIEW.md, API.md: отразить поддержку YouTube через Fetcher и новый/изменённый endpoint | `backend/docs/` | Документация соответствует коду |
| 5.4 | Опционально: idempotency для trigger-processing; отмена (cancel) — распространение до Fetcher/DataProcessor | По приоритету | Улучшение надёжности и UX |

**Критерий готовности:** Документированный и проверенный E2E-сценарий; список известных ограничений обновлён.

**Статус:** выполнено. E2E чеклист: `docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md`. Обработка ошибок: при ошибке Fetcher при создании run в БД сохраняются `ingestion_status=failed`, `fetcher_error_code`, `fetcher_error_message`; при 404 в sync — run переводится в failed с RUN_NOT_FOUND. Документация обновлена. **5.4 частично:** idempotency для trigger-processing реализована (при `ingestion_status=processing` повторный POST возвращает 202 без постановки задачи); cancel не реализован.

---

### Порядок и зависимости

- **Фаза 0** — база для всех остальных; её лучше закрыть первой.
- **Фаза 1** зависит от 0; даёт «вход» в систему по YouTube URL.
- **Фаза 2** даёт замкнутую цепочку Fetcher → обработка; можно делать упрощённо (например trigger только через Backend), затем дорабатывать пути (Фаза 3).
- **Фаза 3** критична для полного цикла с реальным видео из Fetcher; может частично пересекаться с 2 (общая настройка storage).
- **Фаза 4** улучшает UX и наблюдаемость; допустимо после 2–3.
- **Фаза 5** — после стабилизации 1–4.

Минимальный MVP для «YouTube → анализ»: **Фазы 0, 1, 2, 3** (с упрощением 3, если временно допустимо копирование артефакта с Fetcher на диск Backend). Затем 4 и 5 для продакшена и поддержки.

---

## 7. Ссылки на ключевые файлы

- Backend: `backend/app/config.py`, `backend/app/tasks.py`, `backend/app/services/dataprocessor_adapter.py`, `backend/docs/GAPS_AND_ALIGNMENT.md`, `backend/docs/OVERVIEW.md`.
- Fetcher: `Fetcher/fetcher/api.py` (POST /api/v1/runs), `Fetcher/fetcher/tasks.py` (finalize_task), `Fetcher/fetcher/config.py`, `Fetcher/docs/BACKEND_CONTRACTS.md`.
- Контракты: `backend/docs/reference/backend_qna_contracts.md` (Fetcher orchestration, run_id, fetch_video, process_run).

---

## 8. Дальнейшие шаги (после фаз 0–5)

- **Опционально (Phase 5.4):** idempotency для trigger-processing реализована; отмена (cancel) ingestion run с распространением до Fetcher/DataProcessor — не реализована.
- **Связывание результата:** автоматическое создание/обновление сущностей Backend (Video, Channel, AnalysisJob) по завершённому ingestion run для отображения артефактов и прогнозов в UI.
- **Наблюдаемость:** метрики (Prometheus), трейсы, алерты при падении Fetcher/DataProcessor.
- **Безопасность:** проверка JWT для WebSocket `/api/runs/{run_id}/events`; сужение CORS для production.
---

## Навигация

[Vault](MAIN_INDEX.md)
