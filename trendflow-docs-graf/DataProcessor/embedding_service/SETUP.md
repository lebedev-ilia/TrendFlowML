# Инструкция по настройке Embedding Service

## 1. Установка PostgreSQL с pgvector

### Проверка версии PostgreSQL

Перед установкой pgvector проверьте версию PostgreSQL:

```bash
# Способ 1: Через psql (самый простой)
psql --version

# Способ 2: Через SQL запрос (от имени пользователя postgres)
sudo -u postgres psql -c "SELECT version();"

# Способ 3: Через систему (Ubuntu/Debian)
dpkg -l | grep postgresql | grep server

# Способ 4: Через systemctl (если PostgreSQL запущен)
sudo systemctl status postgresql
```

**Если получаете ошибку аутентификации:**
```bash
psql: error: connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: 
FATAL: Peer authentication failed for user "postgres"
```

**Решение:**
Используйте `sudo -u postgres` для выполнения команд от имени системного пользователя postgres:

```bash
# Правильно:
sudo -u postgres psql -c "SELECT version();"

# Или подключитесь интерактивно:
sudo -u postgres psql

# Затем выполните запрос:
SELECT version();
```

**Альтернативные способы:**
```bash
# 1. Через пароль (если установлен пароль для postgres)
PGPASSWORD=your_password psql -U postgres -c "SELECT version();"

# 2. Через локального пользователя (если есть)
psql -U your_username -d postgres -c "SELECT version();"

# 3. Через TCP/IP подключение (если настроен)
psql -h localhost -U postgres -c "SELECT version();"
```

Пример вывода `psql --version`:
```
psql (PostgreSQL) 14.12
```
В этом случае нужно установить `postgresql-14-pgvector`.

### Ubuntu/Debian:
```bash
# Установить PostgreSQL (если еще не установлен)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Установить pgvector (выберите версию по номеру вашей PostgreSQL)
# Для PostgreSQL 14:
sudo apt install postgresql-14-pgvector
# Для PostgreSQL 15:
# sudo apt install postgresql-15-pgvector
# Для PostgreSQL 16:
# sudo apt install postgresql-16-pgvector

# Или через исходники (универсально для любой версии):
# git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
# cd pgvector
# make
# sudo make install
```

### macOS:
```bash
brew install postgresql
brew install pgvector
```

### Docker (рекомендуется):
```bash
docker run -d \
  --name postgres-embeddings \
  -e POSTGRES_DB=embeddings \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  pgvector/pgvector:pg14
```

**Если получаете ошибку "address already in use" (порт 5432 занят):**

Это означает, что PostgreSQL уже запущен локально. У вас есть два варианта:

#### Вариант 1: Использовать существующий локальный PostgreSQL (проще)
Просто используйте ваш локальный PostgreSQL:
```bash
# Проверить, запущен ли PostgreSQL
sudo systemctl status postgresql

# Если запущен, используйте его - не нужен Docker!
# Просто создайте базу и установите расширение (см. раздел "2. Настройка базы данных")
```

#### Вариант 2: Остановить локальный PostgreSQL и использовать Docker
```bash
# Остановить локальный PostgreSQL
sudo systemctl stop postgresql
sudo systemctl disable postgresql  # Отключить автозапуск

# Затем запустить Docker контейнер (команда выше)
```

#### Вариант 3: Использовать другой порт для Docker
```bash
# Запустить Docker на другом порту (например, 5433)
docker run -d \
  --name postgres-embeddings \
  -e POSTGRES_DB=embeddings \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=your_password \
  -p 5433:5432 \
  pgvector/pgvector:pg14

# Затем в .env укажите:
# POSTGRES_PORT=5433
```

**Если получаете ошибку "container name already in use":**

Контейнер с таким именем уже существует. Решения:

```bash
# Способ 1: Удалить старый контейнер
docker rm -f postgres-embeddings

# Затем запустить заново
docker run -d \
  --name postgres-embeddings \
  -e POSTGRES_DB=embeddings \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=your_password \
  -p 5433:5432 \
  pgvector/pgvector:pg15

# Способ 2: Проверить существующий контейнер
docker ps -a | grep postgres-embeddings

# Если контейнер остановлен, можно перезапустить:
docker start postgres-embeddings

# Если контейнер запущен, проверьте его статус:
docker ps | grep postgres-embeddings

# Способ 3: Использовать другое имя контейнера
docker run -d \
  --name postgres-embeddings-new \
  -e POSTGRES_DB=embeddings \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=your_password \
  -p 5433:5432 \
  pgvector/pgvector:pg15
```

**Управление Docker контейнерами:**
```bash
# Список всех контейнеров
docker ps -a

# Остановить контейнер
docker stop postgres-embeddings

# Запустить контейнер
docker start postgres-embeddings

# Удалить контейнер
docker rm postgres-embeddings

# Удалить контейнер и образ (полная очистка)
docker rm -f postgres-embeddings
docker rmi pgvector/pgvector:pg15

# Просмотр логов контейнера
docker logs postgres-embeddings

# Войти в контейнер (для отладки)
docker exec -it postgres-embeddings psql -U postgres -d embeddings
```

#### Проверка, что порт занят:
```bash
# Проверить, что использует порт 5432
sudo lsof -i :5432
# или
sudo netstat -tlnp | grep 5432
# или
sudo ss -tlnp | grep 5432
```

**Рекомендация:** Если PostgreSQL уже установлен локально, используйте его (Вариант 1) - проще и быстрее.

## 2. Настройка базы данных

### Для локального PostgreSQL (порт 5432):

```bash
# Подключиться к PostgreSQL (от имени пользователя postgres)
sudo -u postgres psql

# Создать базу данных
CREATE DATABASE embeddings;

# Подключиться к базе
\c embeddings

# Установить расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

# Проверить установку расширения
\dx vector

# Выйти
\q
```

**Или через одну команду:**
```bash
# Создать базу данных
sudo -u postgres createdb embeddings

# Установить расширение pgvector
sudo -u postgres psql -d embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Проверить
sudo -u postgres psql -d embeddings -c "\dx vector"
```

### Для Docker контейнера (порт 5433):

**Важно:** Docker контейнер работает через TCP/IP на порту 5433, а не через локальный Unix socket.

```bash
# Способ 1: Подключиться через docker exec (рекомендуется)
docker exec -it postgres-embeddings psql -U postgres -d embeddings

# Или создать базу данных через docker exec
docker exec -it postgres-embeddings psql -U postgres -c "CREATE DATABASE embeddings;"
docker exec -it postgres-embeddings psql -U postgres -d embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker exec -it postgres-embeddings psql -U postgres -d embeddings -c "\dx vector"

# Способ 2: Подключиться через TCP/IP (если psql установлен локально)
# Внимание: используйте порт 5433, НЕ 5432!
psql -h localhost -p 5433 -U postgres -d embeddings

# Или через одну команду (с переменной PGPASSWORD):
PGPASSWORD=123 psql -h localhost -p 5433 -U postgres -d embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Способ 3: Через одну команду docker exec
docker exec -it postgres-embeddings psql -U postgres -c "
CREATE DATABASE IF NOT EXISTS embeddings;
\c embeddings
CREATE EXTENSION IF NOT EXISTS vector;
\dx vector
"
```

**Разница между локальным PostgreSQL и Docker:**

| Аспект | Локальный PostgreSQL | Docker контейнер |
|--------|----------------------|------------------|
| Порт | 5432 | 5433 (если использовали `-p 5433:5432`) |
| Подключение | `sudo -u postgres psql` | `docker exec -it postgres-embeddings psql -U postgres` |
| TCP/IP | `psql -h localhost -p 5432` | `psql -h localhost -p 5433` |
| Пароль | Не нужен (peer auth) | Нужен (указан в `-e POSTGRES_PASSWORD`) |

**Если получаете ошибку "could not change directory" или "нет такого файла":**
- Это нормально - это предупреждение о домашней директории, не критично
- Важно: если используете Docker, используйте команды через `docker exec`, а не `sudo -u postgres`

## 3. Настройка переменных окружения

Создайте файл `.env` в корне `embedding_service/`:

```bash
cd DataProcessor/embedding_service
cp .env.example .env  # если есть пример
```

Или создайте `.env` вручную:

```bash
# PostgreSQL настройки
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=embeddings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here

# Triton Inference Server (если используется)
TRITON_BASE_URL=http://localhost:8000
TRITON_TIMEOUT_SEC=30.0

# Хранилище файлов
STORAGE_TYPE=local
STORAGE_LOCAL_PATH=./embedding_storage

# FAISS индексы
FAISS_INDEX_PATH=./faiss_indices
FAISS_SYNC_INTERVAL_SEC=300.0
```

## 4. Проверка Triton сервера (для CLIP моделей)

Если используете CLIP модели через Triton:

```bash
# Проверить, что Triton работает
curl http://localhost:8000/v2/health/ready

# Или через Python
python -c "
from dp_triton.http_client import TritonHttpClient
client = TritonHttpClient(base_url='http://localhost:8000')
print('Triton ready:', client.ready())
"
```

Убедитесь, что модели CLIP доступны:
- `clip_224`
- `clip_336`
- `clip_448`

## 5. Создание необходимых директорий

```bash
cd DataProcessor/embedding_service

# Создать директории для хранения
mkdir -p embedding_storage
mkdir -p faiss_indices
```

## 6. Запуск сервиса

### Вариант 1: Через Python модуль
```bash
cd DataProcessor
python -m embedding_service.api.main
```

### Вариант 2: Через uvicorn напрямую
```bash
cd DataProcessor
uvicorn embedding_service.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Вариант 3: Создать скрипт запуска

Создайте `embedding_service/run_server.py`:

```python
#!/usr/bin/env python3
"""Скрипт запуска Embedding Service"""

import uvicorn
from embedding_service.api.main import create_app
from embedding_service.config.settings import EmbeddingServiceConfig

if __name__ == "__main__":
    config = EmbeddingServiceConfig()
    app = create_app(config)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,  # Убрать в production
    )
```

Затем:
```bash
python embedding_service/run_server.py
```

## 7. Проверка работоспособности

### Health check:
```bash
curl http://localhost:8000/health
# Должен вернуть: {"status":"healthy"}
```

### Проверить категории:
```bash
curl http://localhost:8000/categories
```

### Проверить документацию API:
Откройте в браузере: http://localhost:8000/docs

## 8. Тестирование API

### Добавить объект (face):
```bash
curl -X POST "http://localhost:8000/objects/add" \
  -H "accept: application/json" \
  -F "category=face" \
  -F "name=test_person" \
  -F "image=@path/to/face.jpg"
```

### Извлечь embedding:
```bash
curl -X POST "http://localhost:8000/embed" \
  -H "accept: application/json" \
  -F "category=place" \
  -F "image=@path/to/image.jpg"
```

### Поиск похожих:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "accept: application/json" \
  -F "category=face" \
  -F "top_k=5" \
  -F "similarity_threshold=0.7" \
  -F "image=@path/to/query_face.jpg"
```

## 9. Возможные проблемы

### Ошибка подключения к PostgreSQL:
```
psycopg2.OperationalError: could not connect to server
```
**Решение:** Проверьте, что PostgreSQL запущен и настройки в `.env` корректны.

### Ошибка pgvector:
```
ERROR: could not open extension control file
```
**Решение:** Установите расширение pgvector в PostgreSQL.

### Triton недоступен:
```
EmbeddingServiceError: Triton server not ready
```
**Решение:** 
- Запустите Triton сервер
- Проверьте `TRITON_BASE_URL` в `.env`
- Убедитесь, что CLIP модели загружены в Triton

### Ошибка ArcFace:
```
ImportError: No module named 'insightface'
```
**Решение:** Установите `insightface` и его зависимости.

## 10. Следующие шаги

1. ✅ Зависимости установлены
2. ✅ PostgreSQL настроен с pgvector
3. ✅ Переменные окружения настроены
4. ✅ База данных создана
5. ✅ Сервис запущен
6. ⏭️ Начать использовать API для добавления и поиска объектов

## Полезные команды

```bash
# Проверить подключение к БД
python -c "
from embedding_service.core.database.postgres import PostgresEmbeddingStore
from embedding_service.config.settings import EmbeddingServiceConfig
config = EmbeddingServiceConfig()
store = PostgresEmbeddingStore(
    host=config.postgres_host,
    port=config.postgres_port,
    database=config.postgres_db,
    user=config.postgres_user,
    password=config.postgres_password
)
print('Database connected successfully!')
store.close()
"

# Проверить статистику индексов
python -c "
from embedding_service.core.database.faiss_index import FaissIndexManager
from embedding_service.config.settings import EmbeddingServiceConfig
config = EmbeddingServiceConfig()
manager = FaissIndexManager(config.faiss_index_path)
for model in ['arcface', 'clip_224', 'clip_336', 'clip_448']:
    stats = manager.get_stats(model)
    print(f'{model}: {stats}')
"
```
---

## Навигация

[README](README.md) · [DataProcessor](../docs/MAIN_INDEX.md) · [embedding_service](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
