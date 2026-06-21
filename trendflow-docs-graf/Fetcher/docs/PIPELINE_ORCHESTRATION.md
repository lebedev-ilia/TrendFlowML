## Orchestration и state machine Fetcher

Этот документ описывает **целевой дизайн orchestration‑уровня Fetcher**:

- state machine для ingestion‑pipeline’а;
- event‑driven постановку задач;
- поддержку resume после сбоев.

Основан на `Fetcher/docs/plan.md`, раздел 4 (Pipeline и state machine).

---

## 1. Общий pipeline и состояния

Логический pipeline для одного `run_id`:

```text
INIT
  ↓
NORMALIZE_SOURCE      (нормализация URL → platform_video_id)
  ↓
CHECK_CACHE           (глобальный кеш по videos/artifacts)
  ├── CACHE_HIT  → FINALIZE → TRIGGER_DATAPROCESSOR
  └── CACHE_MISS → FETCH_METADATA → FETCH_CHANNEL → FETCH_COMMENTS → DOWNLOAD_VIDEO
                                               ↓
                                         UPLOAD_ARTIFACTS → FINALIZE
                                                                ↓
                                                     TRIGGER_DATAPROCESSOR
```

Целевой список состояний run’а в Fetcher:

```text
PENDING
NORMALIZING_SOURCE
CHECKING_CACHE
FETCHING_METADATA
FETCHING_CHANNEL
FETCHING_COMMENTS
DOWNLOADING_VIDEO
UPLOADING_ARTIFACTS
FINALIZING
COMPLETED
FAILED
```

---

## 2. State machine engine

### 2.1. Цели

- Явно описать **разрешённые переходы** между состояниями.
- Обеспечить **идемпотентность**: повторный вызов шага не ломает артефакты.
- Позволить pipeline безопасно продолжать работу после рестартов воркеров/сервиса.

### 2.2. Модель состояния

Каждый run в Fetcher имеет:

- `status` — одно из состояний, перечисленных выше;
- `current_stage` / `last_stage` (опционально) — для логирования;
- `error_code` / `error_reason` — при `FAILED`.

Хранение:

- источник истины для статуса run’а — таблица `runs` в БД Fetcher (`status` поле);
- при необходимости, дублирование в Redis key (для быстрых чтений/локов) будет описано на этапе реализации очереди.

### 2.3. Разрешённые переходы

Пример таблицы переходов (упрощённо):

- `PENDING` → `NORMALIZING_SOURCE`
- `NORMALIZING_SOURCE` → `CHECKING_CACHE`
- `CHECKING_CACHE` → `FETCHING_METADATA` / `FINALIZING`
- `FETCHING_METADATA` → `FETCHING_CHANNEL`
- `FETCHING_CHANNEL` → `FETCHING_COMMENTS`
- `FETCHING_COMMENTS` → `DOWNLOADING_VIDEO`
- `DOWNLOADING_VIDEO` → `UPLOADING_ARTIFACTS`
- `UPLOADING_ARTIFACTS` → `FINALIZING`
- `FINALIZING` → `COMPLETED` / `FAILED`

Дополнительно:

- любой промежуточный статус может перейти в `FAILED` при фатальной ошибке.

В будущем реализация может использовать enum и таблицу `ALLOWED_TRANSITIONS`, аналогично `RunStatus` в DataProcessor.

---

## 3. Event‑driven постановка задач

### 3.1. Общий подход

Orchestrator Fetcher реализуется как набор **задач очереди** (Celery/Redis/Kafka) + state machine в БД:

- каждая задача:
  - читает текущее состояние run’а из БД;
  - проверяет, имеет ли право выполняться в этом состоянии;
  - выполняет свой шаг (normalize/fetch/download/…);
  - обновляет state и публикует событие;
  - при необходимости ставит следующую задачу.

### 3.2. Основные задачи

Минимальный набор:

- `normalize_source(run_id)`
  - нормализует URL → `(platform, platform_video_id)`;
  - обновляет `video_sources`, `videos` (создаёт/находит запись).
- `check_cache(run_id)`
  - проверяет наличие артефактов в `videos`/`artifacts`;
  - при cache hit → сразу ставит `finalize(run_id)`;
  - при cache miss → ставит `fetch_metadata`, `fetch_comments`, `download_video`.
- `fetch_metadata(run_id)`
- `fetch_channel(run_id)` (может быть частью metadata в MVP)
- `fetch_comments(run_id)`
- `download_video(run_id)`
- `upload_artifacts(run_id)` (если отделено от download)
- `finalize(run_id)`
  - собирает `manifest.json`;
  - помечает run как `COMPLETED` или `FAILED`;
  - инициирует `TRIGGER_DATAPROCESSOR` (через Backend/DP API).

Каждая задача должна быть **идемпотентной** и уметь корректно обрабатывать повторные вызовы (см. `plan.md` и чеклист).

### 3.3. Триггеры задач

На уровне дизайна:

- запуск ingestion:
  - Backend добавляет задание `fetch_video(run_id)` (абстракция над `normalize_source`).
- переходы:
  - завершение `normalize_source` → постановка `check_cache`;
  - `check_cache` при miss → постановка fan‑out задач `fetch_metadata`, `fetch_comments`, `download_video`;
  - `fetch_metadata` / `fetch_comments` / `download_video` по завершении сообщают о готовности артефактов (`artifacts`/БД);
  - `upload_artifacts` (если требуется) и `finalize` выполняют fan‑in (ожидание завершения всех обязательных задач).

Фактический транспорт (Celery/Redis/Kafka) будет выбран и описан в отдельных документах (Phase 1+).

---

## 4. Resume support

### 4.1. Требования

Система должна:

- корректно продолжать выполнение pipeline после рестартов воркеров/сервиса;
- не повторять уже успешно выполненные шаги;
- уметь восстанавливаться из состояний “на полпути” (например, видео скачано, но не загружено в S3).

### 4.2. Механика resume

Основные принципы:

- **источник истины** для статуса шагов — таблицы БД Fetcher (`runs`, `fetch_jobs`, `artifacts`);
- перед выполнением любого шага:
  - задача проверяет, были ли уже созданы необходимые записи/артефакты;
  - если да — пропускает выполнение и просто обновляет state.

Примеры:

- `download_video(run_id)`:
  - если в `artifacts` уже есть `artifact_type='video_file'` со статусом `COMPLETED` → скачивание не выполняется;
  - если есть `PENDING`/`FAILED` → в зависимости от retry‑политики шаг может быть повторён.
- `fetch_comments(run_id)`:
  - если достигнут лимит комментариев / времени в прошлых запусках и `comments` таблица уже заполнена → шаг считается завершённым.

### 4.3. Взаимодействие с state machine

- при старте любой задачи:
  - state machine валидирует переход (например, нельзя скачивать видео из `PENDING`);
  - если состояние не соответствует ожидаемому — задача либо:
    - завершает run с ошибкой (если это неконсистентное состояние),
    - либо просто не делает ничего (если это “запаздывающее” повторное выполнение).

Такой дизайн делает возможным:

- повторное добавление задач без строгих гарантий at‑most‑once;
- восстановление после падений воркеров, не нарушая консистентность артефактов.

---

## 5. Связанные документы

- `Fetcher/docs/plan.md` — разделы о pipeline и state machine.
- `Fetcher/docs/BACKEND_CONTRACTS.md` — run_id lifecycle и события.
- `Fetcher/docs/DATABASE.md` — таблицы `runs`, `fetch_jobs`, `artifacts`, `fetch_logs`.
- `Fetcher/docs/STORAGE_LAYOUT.md` — layout артефактов в S3/MinIO.
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
