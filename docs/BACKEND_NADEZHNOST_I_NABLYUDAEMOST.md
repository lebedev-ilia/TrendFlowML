# Backend: надёжность и наблюдаемость (задача 5 / P7-code + P9)

Документ описывает изменения в backend по надёжности обработки и структурному
логированию. Все правки аддитивны и не меняют бизнес-логику задач.

## 1. Хардненинг Celery (надёжность) — `app/worker.py`

Обработка одного видео может идти десятки минут — часы. Добавлены настройки:

| Параметр | Значение | Зачем |
|---|---|---|
| `task_acks_late` | `True` | ACK только после выполнения: при падении/вытеснении воркера задача переедет к другому, а не потеряется. |
| `task_reject_on_worker_lost` | `True` | Корректный возврат задачи в очередь при гибели воркера. |
| `worker_prefetch_multiplier` | `1` | Длинные GPU-задачи берём по одной — честное распределение, без «залипания» пачки на одном воркере. |
| `task_track_started` | `True` | Видно статус STARTED (для мониторинга/отладки). |
| `broker_transport_options.visibility_timeout` | `6 ч` | **Критично для Redis-брокера.** Дефолт 1 ч → задача дольше часа повторно раздаётся брокером и **исполняется дважды**. Поднято выше максимальной длительности задачи. |
| `result_backend_transport_options.visibility_timeout` | `6 ч` | То же для backend результатов. |
| `result_expires` | `24 ч` | Результаты задач не хранятся вечно. |
| `task_soft_time_limit` / `task_time_limit` | из Settings (по умолч. выкл.) | Защита от зависших задач, когда понадобится. |

Все значения переопределяются через `Settings` (`celery_visibility_timeout_seconds`,
`celery_result_expires_seconds`, `celery_task_soft_time_limit_seconds`,
`celery_task_time_limit_seconds`); по умолчанию работают безопасные дефолты.

**Требование:** `task_acks_late` корректен только при идемпотентности задач по
`run_id`. У нас выполняется: `process_ingestion_run` повторно выставляет статус и
триггерит DataProcessor, который дедуплицирует по `run_id`.

### Что осознанно НЕ сделано (follow-up)
- `autoretry_for` для тяжёлых задач (`process_ingestion_run`, `process_analysis_job`)
  не включён: повторный запуск нон-стоп мог бы дать двойную обработку на
  неидемпотентных участках. Нужен явный **dead-letter** для «отравленных» run'ов
  (max_attempts → отдельный поток + статус `failed`) — отдельным PR с тестами.
- Разделение очередей GPU/CPU на уровне приложения (для настоящего разделения
  пулов воркеров) — требует маршрутизации задач по типу; пока worker берёт
  `queue:high/normal/low` по приоритету.

## 2. Структурные логи + correlation_id (наблюдаемость) — `app/logging_setup.py`

- **JSON-логи** в stdout без внешних зависимостей (stdlib): поля `ts`, `level`,
  `logger`, `service`, `message`, `correlation_id` (+ любые `extra={...}`).
- **`CorrelationIdMiddleware`**: на каждый HTTP-запрос берёт заголовок
  `X-Request-ID` или генерит UUID, кладёт в `contextvar` и возвращает в ответе.
  Все логи в рамках запроса получают этот id.
- **Celery-задачи**: в начале `process_ingestion_run` и `process_analysis_job`
  выставляется `correlation_id = run_id` / `analysis_job_id` — теперь логи одной
  обработки сквозным образом фильтруются по одному id.

Включается в `app/main.py`: `configure_logging(level, json_format)` + middleware.
Формат и уровень берутся из `Settings` (`log_format`, `log_level`); по умолчанию JSON.

### Связь с метриками
Вместе с метриками backend (`app/metrics.py`, `GET /metrics`: RED по HTTP +
`backend_celery_queue_length`) и зрелыми метриками DataProcessor это даёт сквозную
картину: метрика → алерт → deep-link в логи по `correlation_id`/`run_id`
(см. `docs/LOAD_AND_SCALING_PLAN.md`, `docs/PROD_ARCH_GAP_MAP.md`).

## 3. Проверка

- `python -m py_compile` для всех изменённых файлов — OK.
- Полноценный прогон (поднятый Redis/Postgres) выполняется на твоей машине:
  `celery -A app.worker:celery_app worker -l info` + запрос к API → в stdout
  должны идти JSON-строки с `correlation_id`, а `GET /metrics` отдавать метрики.
