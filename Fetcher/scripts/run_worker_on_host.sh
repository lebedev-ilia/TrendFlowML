#!/usr/bin/env bash
# Запуск Fetcher Celery worker на хосте (не в Docker).
# Используется, когда контейнер fetcher-worker не имеет доступа в интернет (например к YouTube),
# а хост имеет — воркер на хосте подключается к тем же Postgres/Redis/MinIO по проброшенным портам.
#
# Требования:
#   - Docker: postgres, redis, minio (и при необходимости fetcher-api) запущены: docker compose up -d postgres redis minio
#   - Порт хоста 5433 → postgres:5432, 6379 → redis:6379, 9000 → minio:9000
#   - В MinIO созданы бакеты (один раз): PYTHONPATH="$FETCHER_ROOT" python scripts/init_minio_buckets.py
#   - В каталоге Fetcher установлены зависимости (venv активирован)
#
# Использование (из корня Fetcher):
#   source .venv/bin/activate   # или .fetcher_venv
#   ./scripts/run_worker_on_host.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCHER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FETCHER_ROOT"

# Пакет schemas (events, manifest) лежит в Fetcher/schemas/ — нужен корень проекта в PYTHONPATH
export PYTHONPATH="$FETCHER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Подключение к сервисам по портам хоста (как проброшены в docker-compose)
export FETCHER_POSTGRES_DSN="${FETCHER_POSTGRES_DSN:-postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db}"
export FETCHER_REDIS_URL="${FETCHER_REDIS_URL:-redis://localhost:6379/0}"
export FETCHER_S3_ENDPOINT_URL="${FETCHER_S3_ENDPOINT_URL:-http://localhost:9000}"
export FETCHER_S3_ACCESS_KEY="${FETCHER_S3_ACCESS_KEY:-minioadmin}"
export FETCHER_S3_SECRET_KEY="${FETCHER_S3_SECRET_KEY:-minioadmin123}"
export FETCHER_BUCKET_RAW="${FETCHER_BUCKET_RAW:-video-analytics-raw}"
export FETCHER_S3_USE_SSL="${FETCHER_S3_USE_SSL:-false}"
export FETCHER_S3_VERIFY_SSL="${FETCHER_S3_VERIFY_SSL:-false}"
# На хосте обычно есть доступ в интернет — включаем реальные запросы к YouTube
# export FETCHER_YOUTUBE_USE_YT_DLP="${FETCHER_YOUTUBE_USE_YT_DLP:-true}"

# Прокси для доступа к YouTube (если без прокси недоступен):
# export FETCHER_ENABLE_PROXIES=true
# export FETCHER_PROXIES="195.114.209.50:80"   # или socks5://user:pass@host:1080, несколько через запятую
# Скрипт не переопределяет эти переменные — задайте их перед запуском при необходимости.

export CELERY_BROKER_URL="${CELERY_BROKER_URL:-$FETCHER_REDIS_URL}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$FETCHER_REDIS_URL}"

CELERY_BIN="${FETCHER_CELERY_BIN:-}"
if [[ -z "$CELERY_BIN" ]]; then
  for candidate in "$FETCHER_ROOT/.fetcher_venv/bin/celery" "$FETCHER_ROOT/.venv/bin/celery"; do
    if [[ -x "$candidate" ]]; then
      CELERY_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$CELERY_BIN" ]]; then
  CELERY_BIN="$(command -v celery || true)"
fi

if [[ -z "$CELERY_BIN" ]]; then
  echo "FATAL: celery executable not found. Activate Fetcher venv or install it into .fetcher_venv/.venv." >&2
  exit 1
fi

QUEUES="fetcher.high,fetcher.normal,fetcher.low,fetch.metadata,fetch.video,fetch.comments,fetch.finalize,fetch.maintenance"
echo "==> Fetcher worker on host: Postgres=localhost:5433, Redis=localhost:6379, S3=localhost:9000"
if [[ -n "$FETCHER_ENABLE_PROXIES" && "$FETCHER_ENABLE_PROXIES" != "0" && -n "$FETCHER_PROXIES" ]]; then
  echo "==> Proxy: enabled (FETCHER_PROXIES)"
fi
echo "==> QUEUES=$QUEUES"
exec "$CELERY_BIN" -A fetcher.celery_app worker --loglevel=info --concurrency=4 -Q "$QUEUES"
