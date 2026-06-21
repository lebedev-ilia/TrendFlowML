## Queue & Orchestration для Fetcher

Этот документ описывает дизайн системы очередей и оркестрации для Fetcher на базе Celery + Redis (MVP).

---

## 1. Архитектура очередей

### 1.1. Структура очередей

Fetcher использует **Celery + Redis** для управления задачами с отдельными очередями для каждого типа воркера:

- **`fetch.metadata`** — очередь для metadata worker (high priority)
- **`fetch.video`** — очередь для video download worker (low priority)
- **`fetch.comments`** — очередь для comments worker (medium priority)
- **`fetch.finalize`** — очередь для artifact builder (high priority)

### 1.2. Приоритеты задач

Приоритеты определяют порядок обработки задач в рамках одной очереди:

- **High (9)**: `fetch.metadata`, `fetch.finalize`
  - Быстрое принятие решения (metadata) и завершение run (finalize)
- **Medium (5)**: `fetch.comments`
  - Средняя важность, не блокирует завершение run
- **Low (1)**: `fetch.video`
  - Самая тяжёлая операция, не должна "задавить" остальные

### 1.3. Роутинг задач

Каждая задача маршрутизируется в соответствующую очередь через Celery routing:

```python
CELERY_TASK_ROUTES = {
    'fetcher.tasks.fetch_metadata_task': {'queue': 'fetch.metadata'},
    'fetcher.tasks.download_video_task': {'queue': 'fetch.video'},
    'fetcher.tasks.fetch_comments_task': {'queue': 'fetch.comments'},
    'fetcher.tasks.finalize_task': {'queue': 'fetch.finalize'},
}
```

---

## 2. Оркестратор (Orchestrator)

### 2.1. Основная функция

Оркестратор — это точка входа для запуска ingestion pipeline для конкретного `run_id`:

```python
def fetch_video(run_id: str) -> None:
    """
    Главная функция оркестратора.
    
    Логика:
    1. Нормализация source (URL → platform_video_id)
    2. Проверка глобального кеша
    3. Если cache hit — сразу finalize
    4. Если cache miss — постановка задач в очереди
    """
```

### 2.2. State machine интеграция

Оркестратор управляет state machine через обновление статуса `Run` в БД:

- `PENDING` → `NORMALIZING_SOURCE` → `CHECKING_CACHE`
- Если cache hit: `CHECKING_CACHE` → `FINALIZING` → `COMPLETED`
- Если cache miss: `CHECKING_CACHE` → `FETCHING_METADATA` → (параллельно) `FETCHING_COMMENTS`, `DOWNLOADING_VIDEO` → `FINALIZING` → `COMPLETED`

### 2.3. Постановка задач

После нормализации и проверки кеша оркестратор ставит задачи в очереди:

```python
# Если cache miss
fetch_metadata_task.delay(run_id)
fetch_comments_task.delay(run_id)
download_video_task.delay(run_id)

# Artifact builder ждёт завершения всех задач через fan-in
```

---

## 3. Celery задачи

### 3.1. Metadata Task

```python
@celery_app.task(
    name='fetcher.fetch_metadata',
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    queue='fetch.metadata',
    priority=9,
)
def fetch_metadata_task(self, run_id: str) -> None:
    """
    Celery задача для metadata ingestion.
    
    Args:
        run_id: UUID run'а
        
    Retry policy:
    - RateLimitError: retry с exponential backoff
    - NetworkError: retry с exponential backoff
    - NonRetryableError: не ретраим (video removed, private, etc.)
    """
```

### 3.2. Video Download Task

```python
@celery_app.task(
    name='fetcher.download_video',
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    queue='fetch.video',
    priority=1,
)
def download_video_task(self, run_id: str) -> None:
    """
    Celery задача для video download.
    
    Args:
        run_id: UUID run'а
        
    Особенности:
    - Использует distributed lock для предотвращения дублирующихся скачиваний
    - Проверяет кеш перед скачиванием
    - Retry только для сетевых ошибок
    """
```

### 3.3. Comments Task

```python
@celery_app.task(
    name='fetcher.fetch_comments',
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    queue='fetch.comments',
    priority=5,
)
def fetch_comments_task(self, run_id: str, limit: int = 100) -> None:
    """
    Celery задача для comments ingestion.
    
    Args:
        run_id: UUID run'а
        limit: Максимальное количество комментариев (default: 100)
    """
```

### 3.4. Finalize Task

```python
@celery_app.task(
    name='fetcher.finalize',
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue='fetch.finalize',
    priority=9,
)
def finalize_task(self, run_id: str) -> None:
    """
    Celery задача для artifact builder и завершения pipeline.
    
    Логика:
    1. Ждёт завершения всех обязательных задач (metadata, video, comments)
    2. Строит manifest.json
    3. Загружает manifest в storage
    4. Обновляет статус run на COMPLETED
    5. Enqueue process_run(run_id) в DataProcessor (если нужно)
    """
```

---

## 4. Fan-in для Artifact Builder

### 4.1. Проблема

Artifact Builder должен ждать завершения всех задач (metadata, video, comments) перед построением manifest.

### 4.2. Решение: Polling + State Check

Каждая задача обновляет статус артефакта в БД. Finalize task проверяет готовность всех артефактов:

```python
def finalize_task(self, run_id: str) -> None:
    # Проверяем готовность артефактов
    required_artifacts = ['video_file', 'metadata_file', 'comments_file']
    
    while not all_artifacts_ready(run_id, required_artifacts):
        time.sleep(5)  # Polling каждые 5 секунд
        # Можно добавить timeout (например, 30 минут)
    
    # Строим manifest
    build_manifest(run_id)
```

Альтернатива: использовать Celery `chord` для fan-in (более элегантно, но сложнее).

---

## 5. Retry политика

### 5.1. Retryable ошибки

- **RateLimitError** (429): retry с exponential backoff (60s, 120s, 240s)
- **NetworkError** (timeout, connection error): retry с exponential backoff
- **TemporaryError** (503, 502): retry с exponential backoff

### 5.2. Non-retryable ошибки

- **VideoRemovedError**: видео удалено или недоступно
- **VideoPrivateError**: видео приватное
- **AgeRestrictedError**: возрастное ограничение (если нет cookies)
- **GeoBlockedError**: геоблокировка (если нет прокси)

### 5.3. Max retries

- Metadata: 5 retries
- Video download: 3 retries (тяжёлая операция)
- Comments: 3 retries
- Finalize: 3 retries

---

## 6. Distributed locks

### 6.1. Lock для video download

Перед скачиванием видео устанавливается Redis lock:

```python
lock_key = f"lock:video:{platform}:{platform_video_id}"
lock_timeout = 3600  # 1 час

if acquire_lock(lock_key, lock_timeout):
    try:
        download_video(...)
    finally:
        release_lock(lock_key)
else:
    # Lock уже установлен, ждём завершения или переиспользуем артефакт
    wait_for_artifact_or_reuse(...)
```

### 6.2. Lock для artifact upload

Аналогично для upload артефактов (предотвращение дублирующихся upload'ов).

---

## 7. Мониторинг очередей

### 7.1. Метрики

- **Queue depth**: длина каждой очереди (`fetch.metadata`, `fetch.video`, etc.)
- **Active tasks**: количество активных задач по типам
- **Failed tasks**: количество упавших задач
- **Task latency**: время выполнения задач (P50, P95, P99)

### 7.2. Prometheus метрики

```python
# Будущие метрики (после реализации очереди)
fetcher_queue_length = Gauge(
    "fetcher_queue_length",
    "Current queue length",
    ["queue_name"]
)

fetcher_active_tasks = Gauge(
    "fetcher_active_tasks",
    "Current number of active tasks",
    ["queue_name"]
)

fetcher_task_duration_seconds = Histogram(
    "fetcher_task_duration_seconds",
    "Task duration in seconds",
    ["queue_name", "task_name"]
)
```

---

## 8. Backpressure control

### 8.1. Проблема

Если DataProcessor очередь переполнена, не нужно продолжать ингестию новых видео.

### 8.2. Решение

Перед постановкой задачи `finalize` проверяем размер очереди DataProcessor:

```python
def check_backpressure() -> bool:
    """Проверить, не переполнена ли очередь DataProcessor."""
    processor_queue_size = get_processor_queue_size()  # HTTP запрос к DataProcessor API
    threshold = 1000  # configurable
    
    return processor_queue_size > threshold

def finalize_task(self, run_id: str) -> None:
    if check_backpressure():
        # Pause ingestion, логируем состояние
        log_backpressure_state(run_id)
        # Retry позже (через 5 минут)
        raise self.retry(countdown=300)
    
    # Продолжаем нормальную обработку
    ...
```

---

## 9. Конфигурация Celery

### 9.1. Базовые настройки

```python
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 час для video download
    task_soft_time_limit=3300,  # 55 минут soft limit
    worker_prefetch_multiplier=1,  # Не prefetch много задач
    worker_max_tasks_per_child=50,  # Перезапуск worker после N задач
)
```

### 9.2. Роутинг

```python
celery_app.conf.task_routes = {
    'fetcher.tasks.fetch_metadata_task': {'queue': 'fetch.metadata', 'priority': 9},
    'fetcher.tasks.download_video_task': {'queue': 'fetch.video', 'priority': 1},
    'fetcher.tasks.fetch_comments_task': {'queue': 'fetch.comments', 'priority': 5},
    'fetcher.tasks.finalize_task': {'queue': 'fetch.finalize', 'priority': 9},
}
```

---

## 10. Запуск воркеров

### 10.1. Отдельные воркеры для каждой очереди

```bash
# Metadata worker
celery -A fetcher.celery_app worker \
    --queue=fetch.metadata \
    --concurrency=4 \
    --loglevel=info

# Video download worker
celery -A fetcher.celery_app worker \
    --queue=fetch.video \
    --concurrency=2 \
    --loglevel=info

# Comments worker
celery -A fetcher.celery_app worker \
    --queue=fetch.comments \
    --concurrency=4 \
    --loglevel=info

# Finalize worker
celery -A fetcher.celery_app worker \
    --queue=fetch.finalize \
    --concurrency=2 \
    --loglevel=info
```

### 10.2. Kubernetes deployment

Каждый тип воркера — отдельный deployment:

- `fetcher-metadata-worker`: CPU 0.5, RAM 512MB, concurrency=4
- `fetcher-video-worker`: CPU 2, RAM 2GB, concurrency=2
- `fetcher-comments-worker`: CPU 1, RAM 1GB, concurrency=4
- `fetcher-finalize-worker`: CPU 0.5, RAM 512MB, concurrency=2

---

## 11. Связанные документы

- [PIPELINE_ORCHESTRATION.md](./PIPELINE_ORCHESTRATION.md) — дизайн state machine и pipeline
- [CORE_INGESTION.md](./CORE_INGESTION.md) — логика воркеров
- [RATE_LIMITING_AND_LOCKS.md](./RATE_LIMITING_AND_LOCKS.md) — rate limiting и distributed locks
- [FETCHER_OBSERVABILITY.md](./FETCHER_OBSERVABILITY.md) — метрики и observability

---

## 12. Следующие шаги

1. Реализовать Celery app для Fetcher
2. Создать задачи для каждого типа воркера
3. Реализовать оркестратор с нормализацией source и проверкой кеша
4. Интегрировать distributed locks для video download
5. Реализовать fan-in для artifact builder
6. Добавить метрики для мониторинга очередей
7. Настроить backpressure control
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
