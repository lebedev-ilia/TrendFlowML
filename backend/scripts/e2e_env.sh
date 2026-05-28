#!/usr/bin/env bash
# Переменные окружения для полного E2E (Backend → Fetcher → DataProcessor).
# Использование (переменные попадут в текущий shell):
#   source backend/scripts/e2e_env.sh     # из корня репо
#   source scripts/e2e_env.sh            # из backend
#
# Для полного E2E включены Backend, Fetcher и DataProcessor. Порты: Backend 8001, Fetcher 8000, DataProcessor API 8002.
# Метрики воркера для Prometheus: не 8001 (занят Backend API), см. monitoring/prometheus.e2e_host.yml и setup_e2e_infra.sh.
export DP_WORKER_METRICS_PORT="${DP_WORKER_METRICS_PORT:-8003}"

E2E_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_REPO_ROOT="$(cd "${E2E_ENV_DIR}/../.." && pwd)"
E2E_STORAGE_ROOT="${E2E_REPO_ROOT}/storage"

# Backend (API, Celery worker, Celery beat)
export TF_BACKEND_DB_DSN='postgresql+psycopg://trendflow:trendflow@localhost:5433/trendflow'
export TF_BACKEND_REDIS_URL='redis://localhost:6379/0'
export TF_BACKEND_FETCHER_API_URL='http://localhost:8000'
export TF_BACKEND_DATAPROCESSOR_API_URL='http://localhost:8002'
export TF_BACKEND_DATAPROCESSOR_API_KEY='dev-key'
export TF_BACKEND_RUN_TRIGGER_API_KEY='dev-key'
# DataProcessor API (internal): API_KEY используется самим DataProcessor
export DATAPROCESSOR_API_KEY='dev-key'
export API_KEY='dev-key'
export FETCHER_BACKEND_TRIGGER_API_KEY="dev-key"
# Никогда не храните реальный API key в репозитории.
# При необходимости экспортируйте FETCHER_YOUTUBE_DATA_API_KEY в shell до source этого файла.
export FETCHER_YOUTUBE_DATA_API_KEY="${FETCHER_YOUTUBE_DATA_API_KEY:-}"
# Без ключа Data API обязательно выключаем (иначе metadata/comments падают на «API key is not configured»).
# С тёплым shell, где уже export FETCHER_YOUTUBE_DATA_ENABLED=true, перезапись нужна явно.
if [ -z "$FETCHER_YOUTUBE_DATA_API_KEY" ]; then
  export FETCHER_YOUTUBE_DATA_ENABLED=false
else
  : "${FETCHER_YOUTUBE_DATA_ENABLED:=true}"
fi
export FETCHER_YOUTUBE_MOCK_VIDEO_DOWNLOAD=true
# Имя файла: {platform_video_id}.mp4 (напр. -Q6fnPIybEI.mp4) или sample_N.mp4 / ровно один *.mp4 в каталоге
export FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_DIR="${E2E_REPO_ROOT}/example/example_videos"
export FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_COUNT=8

# Fetcher (API и worker — один и тот же Redis)
export FETCHER_REDIS_URL='redis://localhost:6379/0'
export CELERY_BROKER_URL="${FETCHER_REDIS_URL}"
export CELERY_RESULT_BACKEND="${FETCHER_REDIS_URL}"
export FETCHER_POSTGRES_DSN='postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db'
export FETCHER_S3_ENDPOINT_URL='http://localhost:9000'
export FETCHER_S3_ACCESS_KEY='minioadmin'
export FETCHER_S3_SECRET_KEY='minioadmin123'
export FETCHER_BUCKET_RAW='video-analytics-raw'
export FETCHER_YOUTUBE_USE_YT_DLP=false
# Для локального E2E: позволяем завершать пайплайн без comments_file
export FETCHER_ALLOW_FINALIZE_WITHOUT_COMMENTS=true
# Полный E2E: Fetcher вызывает Backend trigger-processing после finalize
export FETCHER_BACKEND_BASE_URL='http://localhost:8001'

# DataProcessor (API и worker)
export REDIS_URL='redis://localhost:6379/1'
export STORAGE_TYPE='fs'
# Важно: DataProcessor API должен читать тот же state/result_store, который пишет main.py.
export STORAGE_ROOT="${E2E_STORAGE_ROOT}"
export TREND_STORAGE_BACKEND='fs'
export TREND_FS_ROOT="${STORAGE_ROOT}"
export ALLOWED_VIDEO_PATHS="${STORAGE_ROOT}/videos,${STORAGE_ROOT}/uploads"
export VIDEO_URL_CACHE_DIR="${STORAGE_ROOT}/videos/_url_cache"
# Triton для VisualProcessor: на хосте НЕ используйте :8000/:8001/:8002 — заняты Fetcher/Backend/DataProcessor.
# См. backend/scripts/e2e_triton_docker.sh и флаг e2e_full_max_run.py --with-triton-docker (по умолчанию HTTP на 8010).
export TRITON_E2E_HTTP_PORT="${TRITON_E2E_HTTP_PORT:-8010}"
# Рекомендуется экспортировать в том же shell, где стартуют DataProcessor API/worker (и при ручном uvicorn),
# иначе подпроцессы VisualProcessor могут унаследовать пустой TRITON_HTTP_URL и подставить из YAML заглушку :8000.
# Если Triton ещё не поднят — закомментируйте строку ниже или поднимите контейнер (e2e_triton_docker.sh / --with-triton-docker).
export TRITON_HTTP_URL="${TRITON_HTTP_URL:-http://127.0.0.1:${TRITON_E2E_HTTP_PORT}}"
# Embedding Service (VisualProcessor semantic identity modules → http://localhost:8005 из global_config)
export EMBEDDING_SERVICE_PORT="${EMBEDDING_SERVICE_PORT:-8005}"
export EMBEDDING_SERVICE_URL="${EMBEDDING_SERVICE_URL:-http://127.0.0.1:${EMBEDDING_SERVICE_PORT}}"
# Подключение к тому же Postgres, что и Fetcher (порт 5433 на хосте); БД embeddings — setup_e2e_infra.sh
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5433}"
export POSTGRES_DB="${POSTGRES_DB:-embeddings}"
export POSTGRES_USER="${POSTGRES_USER:-fetcher}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-fetcher_password}"
export TRITON_BASE_URL="${TRITON_BASE_URL:-${TRITON_HTTP_URL}}"
export FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${E2E_STORAGE_ROOT}/embedding_faiss}"
export STORAGE_LOCAL_PATH="${STORAGE_LOCAL_PATH:-${E2E_STORAGE_ROOT}/embedding_service_uploads}"
# Полный E2E с global_config (см. backend/scripts/e2e_full_max_run.py): маркер
#   storage/e2e_full_max/active_global_config → путь к YAML; либо TF_BACKEND_DATAPROCESSOR_GLOBAL_CONFIG.
export TF_BACKEND_DATAPROCESSOR_TIMEOUT_SECONDS="${TF_BACKEND_DATAPROCESSOR_TIMEOUT_SECONDS:-7200}"
# POST /api/v1/process: DP может скачивать video_url в кеш до ответа — дефолт httpx 30s мало для cold cache.
export TF_BACKEND_DATAPROCESSOR_ENQUEUE_TIMEOUT_SECONDS="${TF_BACKEND_DATAPROCESSOR_ENQUEUE_TIMEOUT_SECONDS:-600}"
# PyTorch: меньше фрагментации VRAM на длинных прогонах (CUDA malloc, см. PyTorch docs).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# DataProcessor API: больше параллельных run для сьюта из нескольких роликов (дефолт API — 4).
export MAX_CONCURRENT_RUNS="${MAX_CONCURRENT_RUNS:-8}"
# Backend → POST /api/v1/process: повтор при 503 / Retry-After (backpressure).
export TF_BACKEND_DATAPROCESSOR_ENQUEUE_MAX_RETRIES="${TF_BACKEND_DATAPROCESSOR_ENQUEUE_MAX_RETRIES:-15}"
export TF_BACKEND_DATAPROCESSOR_ENQUEUE_RETRY_AFTER_CAP_SECONDS="${TF_BACKEND_DATAPROCESSOR_ENQUEUE_RETRY_AFTER_CAP_SECONDS:-120}"
