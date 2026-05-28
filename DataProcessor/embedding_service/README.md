# Unified Embedding Service

Единый Embedding Service для управления эмбеддингами разных категорий.

## 🎯 Быстрый старт

```bash
# 1. Установить зависимости
pip install -r embedding_service/requirements.txt

# 2. Настроить базу данных (см. SETUP.md)
# PostgreSQL с pgvector или Docker контейнер

# 3. Настроить .env файл (опционально)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433  # или 5432 для локального PostgreSQL
export POSTGRES_DB=embeddings
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=123

# 4. Запустить сервис
cd DataProcessor
python embedding_service/run_server.py
```

Сервис будет доступен на http://localhost:8001 (или порт из `EMBEDDING_SERVICE_PORT`)

## 📋 Архитектура

```
┌─────────────────────────────────────────┐
│      Embedding Service API (FastAPI)    │
│  /embed, /search, /add, /delete, etc.   │
└──────────────────┬──────────────────────┘
                   │
                   V
┌─────────────────────────────────────────┐
│      Embedding Manager (Python)         │
│  Хранит meta-инфу и правила             │
└──────────────────┬──────────────────────┘
                   │
                   V
┌─────────────────────────────────────────┐
│   Embeddings Storage                    │
│   PostgreSQL (pgvector) + FAISS         │
└─────────────────────────────────────────┘
```

## 📁 Структура проекта

```
embedding_service/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI приложение со всеми эндпоинтами
├── config/
│   ├── __init__.py
│   └── settings.py          # Конфигурация сервиса
├── core/
│   ├── __init__.py
│   ├── embedding_manager.py # Главный менеджер
│   ├── errors.py            # Исключения
│   ├── database/
│   │   ├── __init__.py
│   │   ├── postgres.py      # PostgreSQL с pgvector
│   │   └── faiss_index.py   # FAISS индексы
│   ├── embedding_extractors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── arcface_extractor.py  # ArcFace extractor
│   │   ├── clip_extractor.py     # CLIP extractor (224/336/448)
│   │   └── factory.py
│   └── managers/
│       ├── __init__.py
│       ├── base_manager.py   # Базовый менеджер
│       ├── face_manager.py   # Менеджер для лиц
│       ├── brand_manager.py  # Менеджер для брендов
│       ├── car_manager.py    # Менеджер для машин
│       ├── place_manager.py  # Менеджер для мест
│       └── factory.py        # Фабрика менеджеров
├── scripts/
│   └── check_setup.py       # Скрипт проверки настройки
├── __init__.py
├── run_server.py            # Скрипт запуска сервера
├── requirements.txt         # Зависимости
├── requirements-dev.txt     # Зависимости для разработки
├── README.md                # Эта документация
├── SETUP.md                 # Детальная инструкция по настройке
└── QUICK_START.md           # Быстрый старт
```

## 🔌 API Эндпоинты

### Основные операции:

- `POST /objects/add` - Добавить объект
- `GET /objects/{id}` - Получить объект
- `DELETE /objects/{id}` - Удалить объект
- `PATCH /objects/{id}` - Обновить объект
- `POST /search` - Поиск похожих объектов
- `POST /embed` - Извлечь embedding из изображения
- `POST /objects/batch_add` - Batch добавление объектов

### Служебные:

- `GET /categories` - Список категорий
- `GET /categories/{category}/count` - Количество объектов в категории
- `GET /categories/{category}/embeddings` - Все embeddings категории (bulk, для локального сравнения)
- `GET /categories/{category}/labels` - Label-space без embeddings (bulk, для db_digest/маппинга id)
- `GET /health` - Health check

**Документация API**: http://localhost:8001/docs (Swagger UI)

## 🏷️ Категории и модели

| Категория | Модель | Размерность | Разрешение |
|-----------|--------|-------------|------------|
| face / face_semantic | arcface | 512 | - |
| brand / brand_semantic | clip_336 | 512 | 336x336 |
| car / car_semantic | clip_336 | 512 | 336x336 |
| place / place_semantic | clip_448 | 512 | 448x448 |
| person / object / logo | clip_224 | 512 | 224x224 |

## ⚙️ Конфигурация

### Переменные окружения:

Создайте `.env` файл в `embedding_service/` или экспортируйте переменные:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5433  # или 5432 для локального
POSTGRES_DB=embeddings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Triton Inference Server (для CLIP)
TRITON_BASE_URL=http://localhost:8000
TRITON_TIMEOUT_SEC=30.0

# Хранилище
STORAGE_TYPE=local  # local или s3
STORAGE_LOCAL_PATH=./embedding_storage

# FAISS индексы
FAISS_INDEX_PATH=./faiss_indices
FAISS_SYNC_INTERVAL_SEC=300.0

# Сервер
EMBEDDING_SERVICE_PORT=8001  # Порт по умолчанию: 8001
EMBEDDING_SERVICE_HOST=0.0.0.0
```

### Автоматическая загрузка .env:

Сервис автоматически ищет `.env` файл в следующих местах:
1. `embedding_service/.env`
2. `DataProcessor/.env`
3. Текущая рабочая директория `.env`

## 💻 Примеры использования

### Добавить объект (face):

```python
import requests

files = {"image": open("face.jpg", "rb")}
data = {
    "category": "face",
    "name": "ASATAchannel",
    "metadata": '{"source": "video"}'
}

response = requests.post(
    "http://localhost:8001/objects/add",
    files=files,
    data=data
)

print(response.json())
# {"id": "...", "status": "success"}
```

### Поиск похожих:

```python
import requests

files = {"image": open("query_face.jpg", "rb")}
data = {
    "category": "face",
    "top_k": 5,
    "similarity_threshold": 0.7
}

response = requests.post(
    "http://localhost:8001/search",
    files=files,
    data=data
)

results = response.json()["results"]
for r in results:
    print(f"{r['name']}: {r['similarity']:.3f}")
```

### Извлечь embedding:

```python
import requests

files = {"image": open("image.jpg", "rb")}
data = {"category": "place"}

response = requests.post(
    "http://localhost:8001/embed",
    files=files,
    data=data
)

embedding = response.json()["embedding"]
print(f"Embedding dimension: {len(embedding)}")
```

## 🗄️ База данных

### PostgreSQL с pgvector

Таблица создается автоматически при первом запуске:

```sql
CREATE TABLE embeddings (
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
```

**Индексы:**
- `idx_embeddings_category` - на category
- `idx_embeddings_embedding_model` - на embedding_model
- `idx_embeddings_category_model` - на (category, embedding_model)

**Примечание:** Векторный индекс `ivfflat` не создается автоматически, так как требуется знание размерности. Для больших объемов данных можно создать его вручную для конкретной размерности.

### FAISS индексы

Каждая модель имеет отдельный FAISS индекс:
- `{model_name}.faiss` - FAISS индекс
- `{model_name}_ids.npy` - Соответствие индексов к UUID объектов

## 🚀 Запуск

### Вариант 1: Через скрипт (рекомендуется):
```bash
cd DataProcessor
python embedding_service/run_server.py
```

### Вариант 2: Через Python модуль:
```bash
cd DataProcessor
python -m embedding_service.api.main
```

### Вариант 3: Через uvicorn напрямую:
```bash
cd DataProcessor
uvicorn embedding_service.api.main:app --host 0.0.0.0 --port 8001 --reload
```

## 📚 Документация

- **SETUP.md** - Детальная инструкция по настройке (PostgreSQL, Docker, pgvector)
- **QUICK_START.md** - Быстрый старт с примерами
- **API документация** - http://localhost:8001/docs (Swagger UI)

## 🔧 Требования

- Python 3.8+
- PostgreSQL 14+ с pgvector
- Triton Inference Server (для CLIP моделей, опционально)
- FastAPI, uvicorn
- faiss-cpu или faiss-gpu
- opencv-python
- insightface (для ArcFace)
- psycopg2-binary
- numpy<2.0.0 (FAISS требует NumPy 1.x)

## 🛠️ Решение проблем

### Проверка настройки:
```bash
cd DataProcessor
python embedding_service/scripts/check_setup.py
```

### Частые проблемы:

1. **Порт занят**: Измените `EMBEDDING_SERVICE_PORT` в `.env`
2. **Ошибка подключения к БД**: Проверьте `POSTGRES_PORT` (5432 для локального, 5433 для Docker)
3. **NumPy 2.x ошибки**: Установите `numpy<2.0.0` (требуется для FAISS)
4. **pgvector не работает**: Установите расширение `CREATE EXTENSION vector;`

См. **SETUP.md** для детальных инструкций.

## 🔄 Расширение

### Добавление новой категории:

1. Добавить категорию в `config/settings.py`:
```python
category_model_mapping = {
    ...
    "new_category": "clip_224",
}
```

2. (Опционально) Создать категорийный менеджер в `core/managers/`

3. Добавить в `ManagerFactory._manager_classes`

### Добавление новой модели:

1. Создать extractor в `core/embedding_extractors/`
2. Зарегистрировать в `EmbeddingExtractorFactory`

## 🚀 Дальнейшие улучшения

Ниже перечислены улучшения, которые могут повысить стабильность, скорость, масштабируемость и удобство разработки в следующих версиях системы.

### 🔧 Архитектура и производительность

#### 1. Асинхронная обработка и очереди задач

Вынести тяжёлые операции (батчевые вставки, удаление, перестройка индексов, ребалансировка моделей) в очереди:

- Redis Streams
- RabbitMQ
- Kafka

Это значительно снизит нагрузку на API и улучшит отзывчивость.

#### 2. Кеширование результата embedding-ов

Хранить эмбеддинги в Redis с TTL, чтобы повторные запросы на ту же картинку обрабатывались мгновенно.
Особенно полезно при больших нагрузках.

#### 3. Incremental FAISS Index Refresh

Реализовать механизм обновления FAISS без полной перестройки:

- добавление в real-time
- периодическое сжатие/оптимизация
- поддержка HNSW / IVF / Scalar quantization

#### 4. Горизонтальное масштабирование Embedding-Service

Вынести модели (Clip, ArcFace, PlaceNet) в отдельные микросервисы:

- 1 CLIP = N входных разрешений (224/336/448)
- автоскейлинг через Kubernetes / Docker Swarm
- балансировка через Nginx / Envoy

Возможность подключать новые модели без изменений в коде ядра.

### 🗄️ Хранилище и индексы

#### 5. Раздельные namespaces в pgvector

Для разных типов объектов:

- `face`
- `brand`
- `car`
- `place`

Это упростит миграции, а также позволит независимо изменять размерность и параметры.

#### 6. История изменений и soft-delete

Хранить удалённые записи как `is_deleted = true`, чтобы:

- откатывать систему
- синхронизировать FAISS
- отслеживать статистику использования

#### 7. Микрошардирование FAISS

При росте данных более чем на миллионы объектов:

- сегментировать индексы на shards
- использовать meta-index для роутинга запросов

### 📦 Модульность и расширяемость

#### 8. Plug-and-Play экстракторы (модель → адаптер)

Вынести логику добавления моделей в систему:

```
extractors/
    clip_336.py
    clip_224.py
    arcface.py
    place_semantic_v2.py
```

Добавление новой модели → просто новый файл.

#### 9. Автоматическая валидация модели при старте

- Проверка размерности embedding-ов
- Проверка совместимости с pgvector / FAISS
- Тестовый прогон на 1–2 изображениях

Предотвращает ошибки несовпадения размерностей.

### 📊 Мониторинг и логирование

#### 10. Метрики (Prometheus + Grafana)

Добавить:

- время извлечения embedding-а
- latency FAISS поиска
- latency PostgreSQL поиска
- количество ошибок Triton
- уникальные пользователи/клиенты
- размер индексов

Позволит быстро находить bottleneck-и.

#### 11. Логи запросов (ELK stack или Loki)

Хранить:

- входящие запросы
- параметры поиска
- latency
- ошибки модели

### 🔐 Безопасность

#### 12. Rate Limiting / Throttling

Защищает от DoS и слишком частых запросов:

- Redis-based token bucket
- ограничение на IP/ключ пользователя

#### 13. Поддержка API-ключей

Выдача ключей пользователям с ограничениями:

- RPS
- namespace
- время жизни

### 🤖 ML-улучшения

#### 14. Перекалибровка сходства (Re-ranking)

После FAISS сделать ML-пересортировку топ-результатов.

#### 15. Distillation

Сжать модели (Clip, PlaceSemantic) через distillation для:

- ускорения
- уменьшения VRAM
- роста throughput

### ☁️ Инфраструктура

#### 16. Migration Management (GitOps style)

Хранить pgvector/SQL миграции в отдельной директории:

```
migrations/
   001_init.sql
   002_add_place_embedding.sql
```

и применять автоматически при деплое.

## 📝 Лицензия

Внутренний проект TrendFlowML.
