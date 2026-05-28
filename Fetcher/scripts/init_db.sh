#!/usr/bin/env bash
# Запуск Postgres в Docker и применение первой миграции (создание таблиц).
# Использование: из корня Fetcher выполнить:
#   source .fetcher_venv/bin/activate   # при необходимости
#   ./scripts/init_db.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCHER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$FETCHER_ROOT"
if [[ -d .fetcher_venv ]]; then
  source .fetcher_venv/bin/activate
fi

echo "==> Запуск Postgres (порт 5433 на хосте, чтобы не конфликтовать с CVAT)..."
docker compose up -d postgres

echo "==> Ожидание готовности Postgres..."
for i in {1..30}; do
  if docker compose exec -T postgres pg_isready -U fetcher -d fetcher_db >/dev/null 2>&1; then
    echo "    Postgres готов."
    break
  fi
  if [[ $i -eq 30 ]]; then
    echo "Ошибка: Postgres не поднялся за 30 попыток." >&2
    exit 1
  fi
  sleep 1
done

# DSN для подключения с хоста: порт 5433, пароль и БД из docker-compose (config использует FETCHER_*)
export FETCHER_POSTGRES_DSN="postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db"

echo "==> Применение миграций (alembic upgrade head)..."
python -m alembic upgrade head

echo "==> Готово. Таблицы созданы. Проверка:"
python -c "
from fetcher.db import engine
from sqlalchemy import text
with engine.connect() as c:
    r = c.execute(text(\"SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename\"))
    tables = [row[0] for row in r]
    print('   Таблицы:', ', '.join(tables))
"
