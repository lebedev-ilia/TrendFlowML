# Быстрый старт Embedding Service

## Настройка переменных окружения

Создайте файл `.env` в корне `DataProcessor/` или экспортируйте переменные:

### Для Docker контейнера (порт 5433):
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433  # <-- ВАЖНО: порт 5433 для Docker!
export POSTGRES_DB=embeddings
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=123
export TRITON_BASE_URL=http://localhost:8000
export FAISS_INDEX_PATH=./faiss_indices
```

### Для локального PostgreSQL (порт 5432):
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=embeddings
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password
export TRITON_BASE_URL=http://localhost:8000
export FAISS_INDEX_PATH=./faiss_indices
```

## Проверка подключения к базе данных

### Для Docker контейнера:
```bash
# Проверить, что контейнер запущен
docker ps | grep postgres-embeddings

# Проверить подключение
docker exec -it postgres-embeddings psql -U postgres -d embeddings -c "SELECT version();"

# Установить расширение pgvector (если еще не установлено)
docker exec -it postgres-embeddings psql -U postgres -d embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Для локального PostgreSQL:
```bash
# Проверить подключение
sudo -u postgres psql -d embeddings -c "SELECT version();"

# Установить расширение pgvector
sudo -u postgres psql -d embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Запуск сервиса

```bash
cd DataProcessor

# Установить переменные окружения (выберите нужный вариант выше)
export POSTGRES_PORT=5433  # или 5432 для локального

# Запустить сервис
python embedding_service/run_server.py
```

## Проверка работы

После запуска откройте в браузере:
- API документация: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Список категорий: http://localhost:8000/categories

## Решение проблем

### Ошибка "Connection refused" на порту 5432:
- Проверьте, что PostgreSQL запущен: `docker ps` или `sudo systemctl status postgresql`
- Если используете Docker на порту 5433, установите `POSTGRES_PORT=5433`
- Проверьте подключение вручную (команды выше)

### Ошибка "database does not exist":
```bash
# Для Docker:
docker exec -it postgres-embeddings psql -U postgres -c "CREATE DATABASE embeddings;"

# Для локального:
sudo -u postgres createdb embeddings
```

