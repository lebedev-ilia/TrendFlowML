# Configuration and env vars

Настройки определены в `backend/app/config.py` и читаются из окружения
с префиксом `TF_BACKEND_`.

## 1) Переменные окружения

Обязательные/часто используемые:

- `TF_BACKEND_DB_DSN`  
  DSN для PostgreSQL.  
  По умолчанию: `postgresql+psycopg://trendflow:trendflow@localhost:5432/trendflow`
- `TF_BACKEND_REDIS_URL`  
  Redis для Celery и WS pubsub.  
  По умолчанию: `redis://localhost:6379/0`
- `TF_BACKEND_JWT_SECRET`  
  Секрет для JWT. По умолчанию `"change-me"`.
- `TF_BACKEND_JWT_ALGORITHM`  
  По умолчанию `HS256`.
- `TF_BACKEND_JWT_EXP_MINUTES`  
  Время жизни токена, по умолчанию 7 дней.
- `TF_BACKEND_ADMIN_EMAILS`  
  Список email через запятую для admin‑доступа.

Пути хранилища (могут быть переопределены):

- `TF_BACKEND_STORAGE_ROOT`
- `TF_BACKEND_RESULT_STORE_BASE`
- `TF_BACKEND_FRAMES_DIR_BASE`
- `TF_BACKEND_RAW_UPLOADS_DIR`
- `TF_BACKEND_EXAMPLE_VIDEOS_DIR`

Интеграция с DataProcessor:

- `TF_BACKEND_DATAPROC_ROOT`
- `TF_BACKEND_VISUAL_CFG_DEFAULT`

## 2) Как разрешаются пути

Если env не задан, пути рассчитываются относительно корня репозитория:

- `storage_root` → `<repo>/storage`
- `result_store_base` → `<storage_root>/result_store`
- `frames_dir_base` → `<storage_root>/frames_dir`
- `raw_uploads_dir` → `<storage_root>/raw`
- `example_videos_dir` → `<repo>/example/example_videos`
- `dataproc_root` → `<repo>/DataProcessor`
- `visual_cfg_default` → `<dataproc_root>/configs/visual_triton_baseline_gpu_local.yaml`

## 3) Где используется

- Создание папок: `backend/app/services/storage.py::ensure_dirs`
- Запуск DataProcessor: `backend/app/tasks.py`
- Seed профилей: `backend/app/routers/profiles.py`

