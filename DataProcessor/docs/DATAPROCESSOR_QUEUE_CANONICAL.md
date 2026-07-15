# Очередь обработки DataProcessor — канонический путь (A3)

В репозитории было **два** механизма очереди. Зафиксировано, какой канонический.

## ✅ Канонический (прод): Redis Streams

```
Backend ──HTTP──▶ DataProcessor API  (POST /api/v1/process, api/endpoints/process.py)
                        │  api/services/queue.enqueue_run(...)
                        ▼
             Redis Streams: queue:high | queue:normal | queue:low
                        │  consumer group "workers" (XREADGROUP + ACK + xpending reclaim)
                        ▼
             api/services/worker.py  ──▶  main.py (обработка видео)
```

Свойства: состояние в Redis (не in-memory), idempotency, recovery, state_machine,
consumer groups для горизонтального масштаба, ACK/reclaim для надёжности.
Покрыт тестами (`api/tests/unit/test_queue.py`, `test_worker.py`, `test_recovery.py`, …).
Автоскейл воркера — по длине очереди (KEDA, `k8s/dataprocessor/keda-scaledobject.yaml`).

## ⚠️ Legacy (не использовать): `dp_queue` (Celery)

`DataProcessor/dp_queue` (celery task `dataprocessor.process_video_job`) в текущем
пайплайне **никем не импортируется** (проверено grep'ом по репо). Помечен
deprecated в `dp_queue/__init__.py`. Не подключать в новый код; кандидат на
удаление после подтверждения, что внешние скрипты на него не завязаны.

## Рекомендация
- Весь новый код — только Redis Streams (`api/services/queue`).
- При чистке репо: удалить `dp_queue/` (или перенести в `legacy/`), убедившись,
  что нет внешних (вне репо) вызовов celery-задачи.
