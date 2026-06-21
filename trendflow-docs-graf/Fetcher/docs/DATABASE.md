## База данных Fetcher

Этот документ описывает **целевую схему PostgreSQL для Fetcher** и её роль в общей архитектуре (см. `plan.md`, раздел 3).

Цель БД Fetcher:

- хранить лёгкие метаданные и статусы ingestion‑pipeline’а;
- обеспечивать глобальный кеш по `(platform, platform_video_id)` и артефактам;
- хранить event‑лог работы Fetcher.

Тяжёлые данные (видео, крупные JSON) живут в object storage (S3/MinIO) и ссылаются через таблицу `artifacts`.

---

## 0. Первый запуск: Postgres и миграции

Если таблиц ещё нет:

1. **Запустить Postgres в Docker** (порт **5433** на хосте, чтобы не конфликтовать с CVAT на 5432):
   ```bash
   cd Fetcher
   docker compose up -d postgres
   ```
2. **Применить миграции** (создание всех таблиц). Подключение — к `localhost:5433`, БД `fetcher_db` (настройки читают переменные с префиксом `FETCHER_`):
   ```bash
   export FETCHER_POSTGRES_DSN="postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db"
   python -m alembic upgrade head
   ```
   Либо использовать скрипт (поднимает Postgres и запускает миграции):
   ```bash
   source .fetcher_venv/bin/activate
   ./scripts/init_db.sh
   ```
3. Для **приложения на хосте** задайте `FETCHER_POSTGRES_DSN` с портом **5433** и БД **fetcher_db**. Для сервисов внутри Docker (docker-compose) используется сервис `postgres` и порт 5432 внутри сети.

---

## 1. Основные сущности

Fetcher использует **свою БД**, отделённую от backend и DataProcessor.  
Ключевые таблицы:

- `runs` — связка с backend‑run по `run_id` (UUID из backend).
- `video_sources` — исходные URL и нормализованные идентификаторы.
- `videos` — глобальный кеш по `(platform, platform_video_id)`.
- `video_metadata` — сырые метаданные видео (включая `raw_json`).
- `channel_metadata` — метаданные каналов.
- `video_snapshots` — временные снэпшоты метрик (views/likes/comments/subs).
- `comments` — комментарии (одна строка на комментарий, + `raw_json`).
- `artifacts` — артефакты в S3/MinIO (video/meta/comments/manifest/thumbnail/...).
- `fetch_jobs` — шаги pipeline (`fetch_metadata`, `download_video`, ...).
- `fetch_logs` — event‑лог работы pipeline.

---

## 2. Схема таблиц (MVP)

Ниже — **целевой SQL‑контракт** (MVP‑вариант). Фактические миграции будут добавлены позже.

### 2.1. Таблицы run’ов и источников

```sql
CREATE TABLE runs (
    id UUID PRIMARY KEY,                 -- run_id из backend
    source_type VARCHAR(20) NOT NULL,    -- youtube / tiktok / ...
    source_url  TEXT        NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMP   NOT NULL DEFAULT NOW(),
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    error       TEXT
);

CREATE TABLE video_sources (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(id),
    platform VARCHAR(20) NOT NULL,          -- 'youtube'
    url TEXT         NOT NULL,
    normalized_video_id VARCHAR(50),        -- 'abc123'
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2.2. Глобальный кеш видео

```sql
CREATE TABLE videos (
    id UUID PRIMARY KEY,
    platform VARCHAR(20) NOT NULL,
    platform_video_id VARCHAR(100) NOT NULL,
    channel_id VARCHAR(100),
    duration_seconds INT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(platform, platform_video_id)
);
```

### 2.3. Метаданные видео и каналов

```sql
CREATE TABLE video_metadata (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    title TEXT,
    description TEXT,
    language VARCHAR(10),
    duration_seconds INT,
    published_at TIMESTAMP,
    raw_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE channel_metadata (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    channel_id VARCHAR(100),
    channel_title TEXT,
    subscriber_count BIGINT,
    video_count INT,
    view_count BIGINT,
    raw_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 2.4. Временные снэпшоты и комментарии

```sql
CREATE TABLE video_snapshots (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    snapshot_index INT,          -- 0,1,2,3 ...
    view_count    BIGINT,
    like_count    BIGINT,
    comment_count BIGINT,
    subscriber_count BIGINT,
    collected_at TIMESTAMP
);

CREATE TABLE comments (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    author TEXT,
    text   TEXT,
    like_count  INT,
    reply_count INT,
    published_at TIMESTAMP,
    raw_json JSONB
);
```

### 2.5. Артефакты и pipeline‑jobs

```sql
CREATE TABLE artifacts (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES videos(id),
    artifact_type VARCHAR(30),   -- video_file / metadata_file / comments_file / manifest / thumbnail
    storage_path TEXT,           -- S3/MinIO key
    size_bytes BIGINT,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',  -- PENDING / UPLOADING / COMPLETED / FAILED
    checksum TEXT,                                  -- SHA256 для валидации целостности
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE fetch_jobs (
    id UUID PRIMARY KEY,
    run_id UUID REFERENCES runs(id),
    job_type VARCHAR(30),        -- fetch_metadata / download_video / fetch_comments / finalize
    status  VARCHAR(20),         -- pending / running / completed / failed
    retries INT DEFAULT 0,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    error TEXT
);
```

### 2.6. Логи pipeline

```sql
CREATE TABLE fetch_logs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID,
    stage  VARCHAR(50),
    level  VARCHAR(10),
    message   TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 3. Инварианты и использование

- **Глобальный кеш видео**:
  - таблица `videos` гарантирует единственность (`UNIQUE(platform, platform_video_id)`);
  - все артифакты (`artifacts`) и метаданные (`video_metadata`, `comments`, ...) ссылаются на один `video_id`.
- **Idempotency**:
  - перед созданием новых артефактов Fetcher обязан проверять наличие записей в `artifacts` для данного `video_id`/`artifact_type`;
  - повторные run’ы для одного и того же URL используют существующие записи `videos` и `artifacts`.
- **Разделение ответственности**:
  - Backend не пишет в эти таблицы напрямую, только через Fetcher;
  - DataProcessor не зависит от схемы Fetcher БД, работает только с `manifest.json`.

---

## 4. Следующие шаги

- Подготовить Alembic‑миграции на основе этой схемы (отдельный implementation‑этап).
- Описать индексы и возможный partitioning для крупных таблиц (`comments`, `fetch_logs`) для production‑нагрузки.
- Согласовать поля, которые будут отображаться в Backend (например, агрегация статусов run’а и счётчики артефактов).
---

## Навигация

[Fetcher](INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
