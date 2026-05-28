#!/usr/bin/env bash
# Подготовка инфраструктуры для полного E2E (Backend → Fetcher → DataProcessor):
#   Postgres + Redis + MinIO (docker compose), БД trendflow, миграции Backend, бакеты MinIO.
#
# Запуск из корня репозитория TrendFlowML:
#   ./backend/scripts/setup_e2e_infra.sh
# Или из backend:
#   ./scripts/setup_e2e_infra.sh
#
# Требования: docker, psql (клиент PostgreSQL), backend/.venv с зависимостями и alembic.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"
FETCHER_DIR="$REPO_ROOT/Fetcher"

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5433}"
PG_USER_FETCHER="${PG_USER_FETCHER:-fetcher}"
PG_PASSWORD_FETCHER="${PG_PASSWORD_FETCHER:-fetcher_password}"
PG_DB_FETCHER="${PG_DB_FETCHER:-fetcher_db}"
TRENDFLOW_USER="${TRENDFLOW_USER:-trendflow}"
TRENDFLOW_PASSWORD="${TRENDFLOW_PASSWORD:-trendflow}"
TRENDFLOW_DB="${TRENDFLOW_DB:-trendflow}"

echo "==> Repo root: $REPO_ROOT"
echo "==> Backend:   $BACKEND_DIR"
echo "==> Fetcher:   $FETCHER_DIR"

# --- 1. Docker: postgres, redis, minio ---
if [[ ! -f "$FETCHER_DIR/docker-compose.yml" ]]; then
  echo "FATAL: $FETCHER_DIR/docker-compose.yml not found." >&2
  exit 1
fi
echo ""
echo "==> [1/6] Starting Postgres, Redis, MinIO (Fetcher docker-compose)..."
(cd "$FETCHER_DIR" && docker compose up -d postgres redis minio)

# Шаги ниже нумеруются как [n/6] (включая БД embeddings, Prometheus/Grafana).

# --- 2. Ждём готовности Postgres ---
echo ""
echo "==> [2/5] Waiting for Postgres on $POSTGRES_HOST:$POSTGRES_PORT..."
if command -v pg_isready &>/dev/null; then
  for i in {1..30}; do
    if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$PG_USER_FETCHER" 2>/dev/null; then
      break
    fi
    if [[ $i -eq 30 ]]; then
      echo "FATAL: Postgres did not become ready in time." >&2
      exit 1
    fi
    sleep 1
  done
else
  echo "Warning: pg_isready not found, waiting 10s for Postgres..."
  sleep 10
fi

# --- 3. Создание пользователя и БД trendflow (идемпотентно) ---
echo ""
echo "==> [3/6] Creating Backend DB user and database ($TRENDFLOW_DB)..."
run_psql() {
  if [[ -n "$USE_HOST_PSQL" ]]; then
    PGPASSWORD="$PG_PASSWORD_FETCHER" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$PG_USER_FETCHER" -d "$PG_DB_FETCHER" -v ON_ERROR_STOP=1 "$@"
  else
    docker exec fetcher-postgres psql -U "$PG_USER_FETCHER" -d "$PG_DB_FETCHER" -v ON_ERROR_STOP=1 "$@"
  fi
}
# Пробуем через docker (если контейнер есть), иначе через psql с хоста
if docker exec fetcher-postgres psql -U "$PG_USER_FETCHER" -d "$PG_DB_FETCHER" -c "SELECT 1" &>/dev/null; then
  USE_HOST_PSQL=""
else
  USE_HOST_PSQL=1
  export PGPASSWORD="$PG_PASSWORD_FETCHER"
fi

run_psql -c "DO \$\$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '$TRENDFLOW_USER') THEN
    CREATE USER $TRENDFLOW_USER WITH PASSWORD '$TRENDFLOW_PASSWORD';
  END IF;
END \$\$;" 2>/dev/null || true

run_psql -c "SELECT 1 FROM pg_database WHERE datname = '$TRENDFLOW_DB'" -t | grep -q 1 || \
  run_psql -c "CREATE DATABASE $TRENDFLOW_DB OWNER $TRENDFLOW_USER;"

unset PGPASSWORD 2>/dev/null || true
echo "    Backend DB: postgresql+psycopg://$TRENDFLOW_USER:***@$POSTGRES_HOST:$POSTGRES_PORT/$TRENDFLOW_DB"

# --- 3.4 БД embeddings + pgvector (Embedding Service на :8005) ---
echo ""
echo "==> [3.4/6] Embedding Service database (embeddings + vector extension)..."
run_psql_embeddings() {
  if [[ -n "${USE_HOST_PSQL:-}" ]]; then
    PGPASSWORD="$PG_PASSWORD_FETCHER" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$PG_USER_FETCHER" -d embeddings -v ON_ERROR_STOP=1 "$@"
  else
    docker exec fetcher-postgres psql -U "$PG_USER_FETCHER" -d embeddings -v ON_ERROR_STOP=1 "$@"
  fi
}
run_psql -c "SELECT 1 FROM pg_database WHERE datname = 'embeddings'" -t | grep -q 1 || \
  run_psql -c "CREATE DATABASE embeddings OWNER $PG_USER_FETCHER;"
run_psql_embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"
# Та же схема, что embedding_service PostgresEmbeddingStore (до первого старта сервиса).
run_psql_embeddings -c "
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY,
    category TEXT NOT NULL,
    name TEXT,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    embedding VECTOR,
    metadata JSONB,
    image_url TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_embeddings_category ON embeddings (category);
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_model ON embeddings (embedding_model);
CREATE INDEX IF NOT EXISTS idx_embeddings_category_model ON embeddings (category, embedding_model);
"
# Семантические головы делают fail-fast при 0 labels в Embedding Service; для E2E — placeholder-строки (512-D, L2-нормированы).
E2E_FR_VEC="$(
  python3 -c "import math,random; random.seed(42); x=[random.gauss(0,1) for _ in range(512)]; n=math.sqrt(sum(v*v for v in x)) or 1.0; x=[v/n for v in x]; print('['+','.join(str(v) for v in x)+']')"
)"
run_psql_embeddings -c "
INSERT INTO embeddings (id, category, name, embedding_model, embedding_dim, embedding, metadata) VALUES
  ('a0000000-0000-0000-0000-000000000001'::uuid, 'franchise', 'e2e_seed_placeholder', 'clip_224', 512, '${E2E_FR_VEC}'::vector, NULL),
  ('a0000000-0000-0000-0000-000000000002'::uuid, 'brand', 'e2e_seed_brand', 'clip_336', 512, '${E2E_FR_VEC}'::vector, NULL),
  ('a0000000-0000-0000-0000-000000000003'::uuid, 'face', 'e2e_seed_face', 'arcface', 512, '${E2E_FR_VEC}'::vector, NULL),
  ('a0000000-0000-0000-0000-000000000004'::uuid, 'place', 'e2e_seed_place', 'clip_448', 512, '${E2E_FR_VEC}'::vector, NULL),
  ('a0000000-0000-0000-0000-000000000005'::uuid, 'car', 'e2e_seed_car', 'clip_336', 512, '${E2E_FR_VEC}'::vector, NULL)
ON CONFLICT (id) DO NOTHING;
"
echo "    embeddings DB ready on $POSTGRES_HOST:$POSTGRES_PORT (schema + semantic seed: franchise/brand/face/place/car)"

# --- 3.5 Схема Fetcher (иначе POST /api/v1/runs падает: relation "runs" does not exist) ---
echo ""
echo "==> [3.5/5] Running Fetcher migrations (alembic upgrade head on fetcher_db)..."
export FETCHER_POSTGRES_DSN="postgresql+psycopg2://${PG_USER_FETCHER}:${PG_PASSWORD_FETCHER}@${POSTGRES_HOST}:${POSTGRES_PORT}/${PG_DB_FETCHER}"
FETCHER_ALEMBIC=""
for v in "$FETCHER_DIR/.fetcher_venv/bin/alembic" "$FETCHER_DIR/.venv/bin/alembic"; do
  if [[ -x "$v" ]]; then
    FETCHER_ALEMBIC="$v"
    break
  fi
done
if [[ -z "$FETCHER_ALEMBIC" ]]; then
  echo "FATAL: Fetcher alembic not found (.fetcher_venv or .venv)." >&2
  exit 1
fi
(cd "$FETCHER_DIR" && "$FETCHER_ALEMBIC" upgrade head)
echo "    Fetcher migrations done."

# --- 4. Миграции Backend ---
echo ""
echo "==> [4/6] Running Backend migrations (alembic upgrade head)..."
export TF_BACKEND_DB_DSN="postgresql+psycopg://$TRENDFLOW_USER:$TRENDFLOW_PASSWORD@$POSTGRES_HOST:$POSTGRES_PORT/$TRENDFLOW_DB"
VENV_ACTIVATE="$BACKEND_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "FATAL: Backend venv not found at $VENV_ACTIVATE. Create it and install dependencies." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"
(cd "$BACKEND_DIR" && alembic upgrade head)
echo "    Migrations done."

# --- 4.5 Embedding Service (:8005): faiss-cpu, insightface, … в DataProcessor .data_venv ---
DATAPROCESSOR_DIR="$REPO_ROOT/DataProcessor"
DATA_PY="$DATAPROCESSOR_DIR/.data_venv/bin/python"
EMBED_REQ="$DATAPROCESSOR_DIR/embedding_service/requirements-e2e.txt"
echo ""
echo "==> [4.5/6] Embedding Service Python deps (DataProcessor .data_venv)..."
if [[ -x "$DATA_PY" && -f "$EMBED_REQ" ]]; then
  if (cd "$DATAPROCESSOR_DIR" && "$DATA_PY" -c "import faiss; from embedding_service.config.settings import EmbeddingServiceConfig") 2>/dev/null; then
    echo "    faiss + embedding_service already importable — skip pip."
  else
    "$DATA_PY" -m pip install -r "$EMBED_REQ"
    echo "    pip install -r embedding_service/requirements-e2e.txt done."
  fi
else
  echo "WARN: Skip Embedding deps (need $DATA_PY and $EMBED_REQ)." >&2
fi

# --- 5. MinIO бакеты (Fetcher) ---
echo ""
echo "==> [5/6] Creating MinIO buckets (Fetcher)..."
export FETCHER_S3_ENDPOINT_URL="${FETCHER_S3_ENDPOINT_URL:-http://localhost:9000}"
export FETCHER_S3_ACCESS_KEY="${FETCHER_S3_ACCESS_KEY:-minioadmin}"
export FETCHER_S3_SECRET_KEY="${FETCHER_S3_SECRET_KEY:-minioadmin123}"
export FETCHER_BUCKET_RAW="${FETCHER_BUCKET_RAW:-video-analytics-raw}"
FETCHER_PYTHON=""
for venv in "$FETCHER_DIR/.venv/bin/python" "$FETCHER_DIR/.fetcher_venv/bin/python"; do
  if [[ -x "$venv" ]]; then
    FETCHER_PYTHON="$venv"
    break
  fi
done
if [[ -z "$FETCHER_PYTHON" ]]; then
  echo "Warning: Fetcher venv not found (.venv or .fetcher_venv). Using current python (boto3 must be installed)." >&2
  FETCHER_PYTHON="python"
fi
(cd "$FETCHER_DIR" && PYTHONPATH="$FETCHER_DIR" "$FETCHER_PYTHON" scripts/init_minio_buckets.py)
echo "    Buckets done."

# --- 6. Prometheus + Grafana: scrape DataProcessor на хосте (E2E: API :8002, worker /metrics :8003) ---
DATAPROCESSOR_DIR="${REPO_ROOT}/DataProcessor"
E2E_STATE_DIR="$BACKEND_DIR/.e2e/state"
mkdir -p "$E2E_STATE_DIR"

# Первый свободный порт с host bind (0.0.0.0), чтобы не стопориться на «address already in use» (часто 9090 занят)
e2e_pick_free_host_port() {
  local start="${1:-9090}"
  local span="${2:-30}"
  python3 - "$start" "$span" <<'PY'
import socket, sys
start, span = int(sys.argv[1]), int(sys.argv[2])
for p in range(start, start + span):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", p))
        print(p)
        sys.exit(0)
    except OSError:
        pass
    finally:
        s.close()
sys.exit(1)
PY
}

echo ""
echo "==> [6/6] Starting Prometheus and Grafana (DataProcessor docker compose, E2E host scrape)..."
if [[ -f "$DATAPROCESSOR_DIR/docker-compose.yml" && -f "$DATAPROCESSOR_DIR/monitoring/docker-compose.prometheus-override-e2e.yml" ]]; then
  if [[ -n "${E2E_PROMETHEUS_HOST_PORT:-}" ]]; then
    E2E_PM_PORT="$E2E_PROMETHEUS_HOST_PORT"
  else
    E2E_PM_PORT="$(e2e_pick_free_host_port 9090 40)" || {
      echo "FATAL: could not find a free host port for Prometheus (40 портов начиная с 9090)." >&2
      exit 1
    }
  fi
  if [[ -n "${E2E_GRAFANA_HOST_PORT:-}" ]]; then
    E2E_GF_PORT="$E2E_GRAFANA_HOST_PORT"
  else
    E2E_GF_PORT="$(e2e_pick_free_host_port 3000 50)" || {
      echo "FATAL: could not find a free host port for Grafana (tried 3000+)." >&2
      exit 1
    }
  fi
  export E2E_PROMETHEUS_HOST_PORT="$E2E_PM_PORT"
  export E2E_GRAFANA_HOST_PORT="$E2E_GF_PORT"
  # После «Created» + ошибки bind (например 9090 занят) контейнеры остаётся снять, иначе up может не сменить порт
  docker rm -f dataprocessor-prometheus dataprocessor-grafana 2>/dev/null || true
  (cd "$DATAPROCESSOR_DIR" && docker compose -f docker-compose.yml -f monitoring/docker-compose.prometheus-override-e2e.yml up -d prometheus grafana)
  {
    echo "# Автогенерация: setup_e2e_infra.sh — для start_e2e_stack.sh (URL мониторинга)"
    echo "export E2E_PROMETHEUS_HOST_PORT=$E2E_PM_PORT"
    echo "export E2E_GRAFANA_HOST_PORT=$E2E_GF_PORT"
  } >"$E2E_STATE_DIR/monitoring_ports.env"
  echo "    Prometheus: http://localhost:$E2E_PM_PORT  Grafana: http://localhost:$E2E_GF_PORT  (user/password: admin/admin)"
  echo "    Scrape: DP API host.docker.internal:8002, worker host.docker.internal:8003 (export DP_WORKER_METRICS_PORT in same shell as start_e2e_stack.sh)"
  echo "    Жёстко задать порты: E2E_PROMETHEUS_HOST_PORT / E2E_GRAFANA_HOST_PORT перед setup_e2e_infra.sh"
else
  echo "WARN: DataProcessor compose/override not found, skip Prometheus/Grafana." >&2
fi

echo ""
echo "==> E2E infrastructure ready."
echo "    Postgres: $POSTGRES_HOST:$POSTGRES_PORT (fetcher_db + $TRENDFLOW_DB + embeddings)"
echo "    Redis:    localhost:6379 (from compose)"
echo "    MinIO:    $FETCHER_S3_ENDPOINT_URL"
if [[ -f "$E2E_STATE_DIR/monitoring_ports.env" ]]; then
  # shellcheck source=/dev/null
  source "$E2E_STATE_DIR/monitoring_ports.env"
  echo "    Metrics:  http://localhost:${E2E_PROMETHEUS_HOST_PORT:-9090} (Prometheus)  http://localhost:${E2E_GRAFANA_HOST_PORT:-3000} (Grafana) — шаг [6/6]"
else
  echo "    Metrics:  (шаг [6/6] пропущен — см. выше)"
fi
echo ""
echo "Next: start Backend, Fetcher, DataProcessor (see docs/E2E_FULL_CHECKLIST.md), then run:"
echo "  python scripts/e2e_run_to_complete.py --source-url '...' --with-dataprocessor --fetcher-url http://localhost:8000 --verbose"
echo ""
