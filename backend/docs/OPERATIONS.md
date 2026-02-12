# Operations (dev)

## 1) Зависимости

- PostgreSQL
- Redis
- ffprobe (из ffmpeg)
- Python packages из `backend/requirements.txt`

## 2) Запуск API

Пример:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

На старте выполняется:

- создание директорий storage (`ensure_dirs`)
- `Base.metadata.create_all` (без миграций)
- seed публичных профилей из `DataProcessor/profiles`

## 3) Запуск Celery worker

```
celery -A app.worker.celery_app worker --loglevel=INFO
```

Worker нужен для `process_run`.

## 4) Что хранить в .env

Минимум:

- `TF_BACKEND_DB_DSN`
- `TF_BACKEND_REDIS_URL`
- `TF_BACKEND_JWT_SECRET`
- `TF_BACKEND_ADMIN_EMAILS`

## 5) Health checks

Отдельных `/health` endpoints пока нет.  
Добавление health‑checks — TODO (см. `GAPS_AND_ALIGNMENT.md`).

