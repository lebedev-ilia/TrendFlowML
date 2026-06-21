# Главный индекс документации Embedding Service

Этот документ служит единой точкой входа для навигации по всей документации и структуре Embedding Service. Каждый раздел содержит краткое описание файлов, папок и документов.

**Главный индекс DataProcessor**: [../docs/MAIN_INDEX.md](../docs/MAIN_INDEX.md) · **Vault root**: [../../docs/MAIN_INDEX.md](../../docs/MAIN_INDEX.md)

---

## Документация

### README.md
**Краткое описание**: Основная документация Embedding Service — единого сервиса для управления эмбеддингами разных категорий. Описывает архитектуру (FastAPI → Embedding Manager → PostgreSQL + FAISS), структуру проекта, API эндпоинты (add, search, embed, delete, update, batch_add), категории и модели (face/arcface, brand/car/place/clip), конфигурацию через переменные окружения, примеры использования, структуру базы данных (PostgreSQL с pgvector, FAISS индексы), способы запуска, требования (Python 3.8+, PostgreSQL 14+ с pgvector, Triton для CLIP, FastAPI, faiss, insightface), решение проблем, расширение функционала (добавление категорий и моделей), roadmap улучшений (асинхронная обработка, кеширование, incremental FAISS refresh, горизонтальное масштабирование, раздельные namespaces, soft-delete, микрошардирование, plug-and-play экстракторы, мониторинг, rate limiting, API-ключи, re-ranking, distillation, migration management).

**Расположение**: `embedding_service/README.md`

### QUICK_START.md
**Краткое описание**: Быстрый старт для Embedding Service. Содержит инструкции по настройке переменных окружения (для Docker контейнера на порту 5433 и локального PostgreSQL на порту 5432), проверке подключения к базе данных (через docker exec или локальный psql), установке расширения pgvector, запуску сервиса, проверке работы (API документация, health check, список категорий), решению проблем (ошибки подключения, отсутствие базы данных).

**Расположение**: `embedding_service/QUICK_START.md`

### SETUP.md
**Краткое описание**: Детальная инструкция по настройке Embedding Service. Описывает установку PostgreSQL с pgvector (проверка версии, установка для Ubuntu/Debian/macOS/Docker, решение проблем с занятым портом, управление Docker контейнерами), настройку базы данных (создание БД, установка расширения pgvector для локального PostgreSQL и Docker контейнера), настройку переменных окружения (.env файл), проверку Triton сервера (для CLIP моделей), создание необходимых директорий, запуск сервиса (через Python модуль, uvicorn, скрипт), проверку работоспособности (health check, категории, API документация), тестирование API (добавление объектов, извлечение embedding, поиск), возможные проблемы и их решения, полезные команды для проверки подключения к БД и статистики индексов.

**Расположение**: `embedding_service/SETUP.md`

---

## Основные модули

### api/main.py
**Краткое описание**: FastAPI приложение Embedding Service со всеми эндпоинтами. Содержит функции создания приложения (`create_app()`), эндпоинты для работы с объектами (`POST /objects/add`, `GET /objects/{id}`, `DELETE /objects/{id}`, `PATCH /objects/{id}`, `POST /objects/batch_add`), поиск (`POST /search` с поддержкой изображения или embedding вектора), извлечение embedding (`POST /embed`), управление категориями (`GET /categories`, `GET /categories/{category}/count`, `GET /categories/{category}/embeddings`), health check (`GET /health`). Обрабатывает декодирование изображений из bytes, парсинг JSON метаданных, валидацию категорий, обработку ошибок (InvalidCategoryError, EmbeddingNotFoundError, EmbeddingServiceError), конвертацию numpy массивов в списки для JSON сериализации. Использует EmbeddingManager для бизнес-логики.

**Расположение**: `embedding_service/api/main.py`

### config/settings.py
**Краткое описание**: Конфигурация Embedding Service. Класс `EmbeddingServiceConfig` (dataclass) загружает настройки из переменных окружения или .env файла (ищет в embedding_service/.env, DataProcessor/.env, текущей директории). Содержит настройки PostgreSQL (host, port, db, user, password), Triton Inference Server (base_url, timeout_sec), хранилища (storage_type: local/s3, storage_local_path, storage_s3_bucket, storage_s3_region), FAISS индексов (faiss_index_path, faiss_sync_interval_sec), сервера (server_port, server_host), маппинг категорий на модели (category_model_mapping: face→arcface, brand/car→clip_336, place→clip_448, person/object/logo/franchise→clip_224). Автоматически инициализирует дефолтный маппинг категорий в `__post_init__()`.

**Расположение**: `embedding_service/config/settings.py`

### core/embedding_manager.py
**Краткое описание**: Главный менеджер Embedding Service — единый интерфейс для всех категорий. Класс `EmbeddingManager` инициализирует PostgreSQL store (`PostgresEmbeddingStore`) и FAISS manager (`FaissIndexManager`), кеширует менеджеры по категориям (`_get_manager()`). Методы: `add()` для добавления объекта из изображения, `add_from_embedding()` для добавления с предвычисленным embedding (offline-сценарии, миграция баз), `get()` для получения объекта по ID, `delete()` для удаления, `update()` для обновления, `search()` для поиска похожих объектов (через изображение или embedding вектор), `list_categories()` для списка категорий, `count_by_category()` для подсчета объектов, `get_all_embeddings()` для получения всех embedding категории (для локального сравнения). Использует фабрику менеджеров (`ManagerFactory`) для создания категорийных менеджеров, делегирует операции соответствующим менеджерам.

**Расположение**: `embedding_service/core/embedding_manager.py`

### core/errors.py
**Краткое описание**: Исключения Embedding Service. Определяет базовый класс `EmbeddingServiceError` и специализированные исключения: `InvalidCategoryError` (неверная категория), `EmbeddingNotFoundError` (объект не найден), `InvalidEmbeddingError` (неверный формат embedding), `DatabaseError` (ошибки БД), `FaissIndexError` (ошибки FAISS индекса). Используются для обработки ошибок в API и бизнес-логике.

**Расположение**: `embedding_service/core/errors.py`

---

## База данных и индексы

### core/database/postgres.py
**Краткое описание**: PostgreSQL хранилище с pgvector для эмбеддингов. Класс `PostgresEmbeddingStore` управляет подключением к PostgreSQL, создает схему БД (таблица `embeddings` с полями: id UUID, category TEXT, name TEXT, embedding_model TEXT, embedding_dim INTEGER, embedding VECTOR, metadata JSONB, image_url TEXT, added_at/updated_at TIMESTAMP), индексы (idx_embeddings_category, idx_embeddings_embedding_model, idx_embeddings_category_model). Методы: `_init_schema()` для инициализации схемы, `add()` для добавления embedding, `get()` для получения по ID, `delete()` для удаления, `update()` для обновления, `search()` для векторного поиска через pgvector (cosine similarity), `get_all()` для получения всех embedding категории, `count()` для подсчета. Использует psycopg2 для подключения, автоматически создает таблицу при первом использовании.

**Расположение**: `embedding_service/core/database/postgres.py`

### core/database/faiss_index.py
**Краткое описание**: Менеджер FAISS индексов для быстрого векторного поиска. Класс `FaissIndexManager` управляет FAISS индексами для каждой модели (arcface, clip_224, clip_336, clip_448). Хранит индексы в файлах: `{model_name}.faiss` (FAISS индекс) и `{model_name}_ids.npy` (соответствие индексов к UUID объектов). Методы: `get_index()` для получения/создания индекса, `add()` для добавления embedding, `search()` для поиска (top-k с similarity threshold), `remove()` для удаления, `rebuild()` для перестройки индекса, `get_stats()` для статистики (количество векторов, размерность), `sync_from_db()` для синхронизации с PostgreSQL. Поддерживает автоматическую синхронизацию с БД по интервалу (faiss_sync_interval_sec), инкрементальное обновление индексов.

**Расположение**: `embedding_service/core/database/faiss_index.py`

---

## Извлечение эмбеддингов

### core/embedding_extractors/base.py
**Краткое описание**: Базовый класс для extractors эмбеддингов. Абстрактный класс `BaseEmbeddingExtractor` определяет интерфейс: `extract()` для извлечения embedding из изображения (numpy array), `get_dimension()` для размерности embedding, `get_model_name()` для имени модели. Используется как базовый класс для всех extractors (ArcFace, CLIP).

**Расположение**: `embedding_service/core/embedding_extractors/base.py`

### core/embedding_extractors/arcface_extractor.py
**Краткое описание**: ArcFace extractor для извлечения эмбеддингов лиц. Класс `ArcFaceExtractor` использует библиотеку `insightface` для извлечения 512-мерных эмбеддингов лиц. Загружает предобученную модель ArcFace при инициализации, обрабатывает изображения (детекция лица, выравнивание, нормализация), возвращает embedding вектор. Используется для категорий `face` и `face_semantic`.

**Расположение**: `embedding_service/core/embedding_extractors/arcface_extractor.py`

### core/embedding_extractors/clip_extractor.py
**Краткое описание**: CLIP extractor для извлечения эмбеддингов изображений через Triton Inference Server. Класс `CLIPExtractor` поддерживает несколько разрешений (224x224, 336x336, 448x448) через разные модели Triton (`clip_224`, `clip_336`, `clip_448`). Использует `dp_triton.http_client.TritonHttpClient` для запросов к Triton, обрабатывает изображения (препроцессинг, нормализация), возвращает 512-мерные embedding векторы. Используется для категорий: brand/car (clip_336), place (clip_448), person/object/logo/franchise (clip_224).

**Расположение**: `embedding_service/core/embedding_extractors/clip_extractor.py`

### core/embedding_extractors/factory.py
**Краткое описание**: Фабрика extractors для создания экземпляров по имени модели. Класс `EmbeddingExtractorFactory` регистрирует extractors (ArcFace, CLIP), создает экземпляры через `create()` по имени модели (arcface, clip_224, clip_336, clip_448). Используется менеджерами категорий для получения нужного extractor'а.

**Расположение**: `embedding_service/core/embedding_extractors/factory.py`

---

## Менеджеры категорий

### core/managers/base_manager.py
**Краткое описание**: Базовый менеджер для категорий объектов. Абстрактный класс `BaseManager` определяет интерфейс для работы с категорией: `add()` для добавления объекта из изображения, `add_from_embedding()` для добавления с предвычисленным embedding, `search()` для поиска похожих, `extract_embedding()` для извлечения embedding. Использует extractor для извлечения embedding, хранилище (PostgreSQL + FAISS) для сохранения и поиска. Используется как базовый класс для категорийных менеджеров (FaceManager, BrandManager, CarManager, PlaceManager).

**Расположение**: `embedding_service/core/managers/base_manager.py`

### core/managers/face_manager.py
**Краткое описание**: Менеджер для категории лиц (face, face_semantic). Класс `FaceManager` наследует `BaseManager`, использует `ArcFaceExtractor` для извлечения 512-мерных эмбеддингов лиц. Специализированная логика для работы с лицами: валидация изображений, обработка детекции лиц, сохранение в БД и FAISS индекс arcface.

**Расположение**: `embedding_service/core/managers/face_manager.py`

### core/managers/brand_manager.py
**Краткое описание**: Менеджер для категории брендов (brand, brand_semantic). Класс `BrandManager` наследует `BaseManager`, использует `CLIPExtractor` с разрешением 336x336 (clip_336) для извлечения 512-мерных эмбеддингов брендов. Специализированная логика для работы с брендами: валидация изображений, сохранение в БД и FAISS индекс clip_336.

**Расположение**: `embedding_service/core/managers/brand_manager.py`

### core/managers/car_manager.py
**Краткое описание**: Менеджер для категории машин (car, car_semantic). Класс `CarManager` наследует `BaseManager`, использует `CLIPExtractor` с разрешением 336x336 (clip_336) для извлечения 512-мерных эмбеддингов машин. Специализированная логика для работы с машинами: валидация изображений, сохранение в БД и FAISS индекс clip_336.

**Расположение**: `embedding_service/core/managers/car_manager.py`

### core/managers/place_manager.py
**Краткое описание**: Менеджер для категории мест (place, place_semantic). Класс `PlaceManager` наследует `BaseManager`, использует `CLIPExtractor` с разрешением 448x448 (clip_448) для извлечения 512-мерных эмбеддингов мест. Специализированная логика для работы с местами: валидация изображений, сохранение в БД и FAISS индекс clip_448.

**Расположение**: `embedding_service/core/managers/place_manager.py`

### core/managers/factory.py
**Краткое описание**: Фабрика менеджеров для создания категорийных менеджеров. Класс `ManagerFactory` регистрирует менеджеры (FaceManager, BrandManager, CarManager, PlaceManager, BaseManager для остальных категорий), создает экземпляры через `create()` по категории. Использует маппинг категорий на модели из конфигурации, создает соответствующий extractor, инициализирует менеджер с хранилищем и FAISS manager. Используется `EmbeddingManager` для получения менеджеров по категориям.

**Расположение**: `embedding_service/core/managers/factory.py`

---

## Утилиты и скрипты

### run_server.py
**Краткое описание**: Скрипт запуска Embedding Service. Функция `main()` загружает конфигурацию (`EmbeddingServiceConfig`), создает FastAPI приложение через `create_app()`, выводит информацию о настройках (база данных, Triton, FAISS индексы), обрабатывает ошибки инициализации (подключение к PostgreSQL, создание БД, установка pgvector), запускает uvicorn сервер на указанном хосте и порту. Используется как основной способ запуска сервиса в production и development.

**Расположение**: `embedding_service/run_server.py`

### scripts/check_setup.py
**Краткое описание**: Скрипт проверки настройки Embedding Service. Функции: `check_dependencies()` проверяет установленные зависимости (fastapi, uvicorn, psycopg2, faiss, opencv, numpy, insightface), `check_postgres()` проверяет подключение к PostgreSQL и доступность pgvector, `check_triton()` проверяет доступность Triton Inference Server (опционально), `check_directories()` проверяет существование директорий (FAISS индексы, storage) и создает их при необходимости. Функция `main()` запускает все проверки и выводит итоги. Используется для диагностики проблем перед запуском сервиса.

**Расположение**: `embedding_service/scripts/check_setup.py`

---

## Конфигурация и зависимости

### requirements.txt
**Краткое описание**: Зависимости Embedding Service. Содержит FastAPI и веб-сервер (fastapi, uvicorn, python-multipart, pydantic), базу данных (psycopg2-binary для PostgreSQL с pgvector), FAISS для векторного поиска (faiss-cpu или faiss-gpu), обработку изображений (opencv-python, Pillow, numpy<2.0.0 для совместимости с FAISS), ArcFace для лиц (insightface, onnxruntime), утилиты (python-dotenv для .env файлов). Примечания: pgvector устанавливается отдельно в PostgreSQL, tritonclient опционален (используется dp_triton из проекта), numpy должен быть <2.0.0 из-за несовместимости FAISS с NumPy 2.x.

**Расположение**: `embedding_service/requirements.txt`

### requirements-dev.txt
**Краткое описание**: Зависимости для разработки Embedding Service (опционально). Содержит инструменты для тестирования (pytest, pytest-asyncio), форматирования кода (black), проверки типов (mypy). Используется для разработки и CI/CD.

**Расположение**: `embedding_service/requirements-dev.txt`

---

## Структура данных

### faiss_indices/
**Краткое описание**: Директория для хранения FAISS индексов. Содержит файлы индексов для каждой модели: `{model_name}.faiss` (FAISS индекс с векторами), `{model_name}_ids.npy` (NumPy массив с UUID объектов, соответствующих индексам в FAISS). Примеры файлов: `arcface.faiss` + `arcface_ids.npy` для ArcFace, `clip_224.faiss` + `clip_224_ids.npy` для CLIP 224, `clip_336.faiss` + `clip_336_ids.npy` для CLIP 336, `clip_448.faiss` + `clip_448_ids.npy` для CLIP 448. Индексы создаются автоматически при первом добавлении объекта, синхронизируются с PostgreSQL по интервалу (faiss_sync_interval_sec).

**Расположение**: `embedding_service/faiss_indices/`

---

## Инициализация модулей

### __init__.py
**Краткое описание**: Инициализация модуля `embedding_service`. Экспортирует публичный API модуля для использования в других частях проекта.

**Расположение**: `embedding_service/__init__.py`

### api/__init__.py
**Краткое описание**: Инициализация модуля `api`. Экспортирует FastAPI приложение и функции создания приложения.

**Расположение**: `embedding_service/api/__init__.py`

### config/__init__.py
**Краткое описание**: Инициализация модуля `config`. Экспортирует класс конфигурации `EmbeddingServiceConfig`.

**Расположение**: `embedding_service/config/__init__.py`

### core/__init__.py
**Краткое описание**: Инициализация модуля `core`. Экспортирует основные классы: `EmbeddingManager`, исключения из `errors`, базовые классы для менеджеров и extractors.

**Расположение**: `embedding_service/core/__init__.py`

### core/database/__init__.py
**Краткое описание**: Инициализация модуля `database`. Экспортирует классы хранилищ: `PostgresEmbeddingStore`, `FaissIndexManager`.

**Расположение**: `embedding_service/core/database/__init__.py`

### core/embedding_extractors/__init__.py
**Краткое описание**: Инициализация модуля `embedding_extractors`. Экспортирует базовый класс `BaseEmbeddingExtractor`, фабрику `EmbeddingExtractorFactory`, конкретные extractors (ArcFace, CLIP).

**Расположение**: `embedding_service/core/embedding_extractors/__init__.py`

### core/managers/__init__.py
**Краткое описание**: Инициализация модуля `managers`. Экспортирует базовый класс `BaseManager`, фабрику `ManagerFactory`, конкретные менеджеры (FaceManager, BrandManager, CarManager, PlaceManager).

**Расположение**: `embedding_service/core/managers/__init__.py`

---
---

## Навигация

[README](README.md) · [DataProcessor](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
