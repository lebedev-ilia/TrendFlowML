# Fetcher tests: quick guide

Этот файл — **короткий README по запуску тестов Fetcher**. За детальной стратегией и чеклистом см. `docs/TESTING_PLAN.md`.

---

## 1. Базовое окружение

- Рабочая директория: `Fetcher/`.
- Python: используется локальный venv `.fetcher_venv`.
- PostgreSQL: контейнер `fetcher-postgres` (image `postgres:15-alpine`, порт `5433` → `5432`).

### 1.1. Запуск окружения

```bash
cd /media/ilya/Новый\ том/TrendFlowML/Fetcher

# Активировать venv
. .fetcher_venv/bin/activate

# Убедиться, что Postgres для Fetcher запущен
docker start fetcher-postgres 2>/dev/null || true

# DSN для тестов Fetcher
export FETCHER_POSTGRES_DSN="postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db"
```

Если DSN не задан или Postgres недоступен, все тесты с маркерами `integration`, `chaos`, `e2e` будут **пропущены** (см. `tests/conftest.py` и функцию `_postgres_available()`).

---

## 2. Маркеры и уровни тестов

Маркеры объявлены в `pytest.ini` и используются так:

- `@pytest.mark.unit` — юнит‑тесты (не требуют живых сервисов).
- `@pytest.mark.integration` — интеграционные тесты (реальная БД, иногда Redis/Storage).
- `@pytest.mark.chaos` — chaos‑тесты (отказы зависимостей, падения воркеров).
- `@pytest.mark.e2e` — сквозные сценарии (API + pipeline до manifest).
- Дополнительные:
  - `@pytest.mark.database` — требует доступности PostgreSQL.
  - `@pytest.mark.redis`, `@pytest.mark.storage`, `@pytest.mark.slow` — узкоспециализированные.

---

## 3. Быстрый запуск

### 3.1. Только unit‑тесты (без БД)

```bash
cd Fetcher
. .fetcher_venv/bin/activate
pytest tests/unit/ -v --no-cov --tb=short
```

Эти тесты не трогают реальные PostgreSQL/Redis/S3 и проходят даже без поднятых контейнеров.

### 3.2. Unit + integration + chaos (с Postgres)

```bash
cd Fetcher
. .fetcher_venv/bin/activate
export FETCHER_POSTGRES_DSN="postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db"

pytest tests/unit/ tests/integration/ tests/chaos/ -v --no-cov --tb=short
```

Ожидаемый результат на текущем состоянии:

- **Unit**: 25 тестов зелёные (`test_idempotency.py`, `test_resume.py`, `test_state_machine.py`, `test_youtube_adapter.py`).
- **Integration**: 9 тестов зелёные (`test_full_pipeline.py`, `test_idempotency.py`, `test_resume.py`).
- **Chaos**: 9 тестов зелёные (`test_network_failures.py`, `test_worker_failures.py`).

---

## 4. Детали по слоям

### 4.1. Integration tests

Папка: `tests/integration/`.

Основные файлы:

- `test_full_pipeline.py` — полный pipeline (metadata → video → comments → finalize) с реальной БД:
  - `integration_test_run` создаёт `Run` + `VideoSource` через `session_scope`.
  - Celery‑задачи (`fetch_metadata_task`, `download_video_task`, `fetch_comments_task`, `finalize_task`) патчатся на синхронные воркеры (`run_metadata_worker`, `run_video_worker`, `run_comments_worker`) без Redis.
  - `yt_dlp.YoutubeDL` и файловые операции (`Path.write_text`, `Path.stat`, `Path.mkdir`, `Path.unlink`, `open`) замоканы, чтобы не ходить в сеть/файловую систему.
- `test_idempotency.py` — интеграционные проверки идемпотентности worker’ов (metadata/video/comments) на реальной БД.
- `test_resume.py` — интеграционный тест механизма resume.

Запуск только integration:

```bash
pytest tests/integration/ -v --no-cov --tb=short
```

### 4.2. Chaos tests

Папка: `tests/chaos/`.

- `test_network_failures.py` — имитация сетевых отказов:
  - Redis (`test_redis_connection_loss`),
  - Storage (`test_storage_connection_loss`),
  - БД (`test_database_connection_loss`),
  - YouTube API timeout/429 (через `DownloadError`).
- `test_worker_failures.py` — падения воркеров metadata/video/comments/finalize и восстановление.

Большинство chaos‑тестов помечены `@pytest.mark.database` и требуют живой БД. При недоступном Postgres они будут автоматически `skip`.

Запуск только chaos:

```bash
pytest tests/chaos/ -v --no-cov --tb=short
```

---

## 5. E2E тесты

Папка: `tests/e2e/`.

### 5.1. Happy‑path Fetcher API

Файл: `tests/e2e/test_happy_path.py`.

Сценарий:

1. `POST /api/v1/runs` с заранее сгенерированным `run_id` и YouTube URL.
2. Синхронный прогон pipeline в тесте (через вызовы `run_metadata_worker`, `run_video_worker`, `run_comments_worker`, `run_artifact_builder`).
3. `GET /api/v1/runs/{run_id}` и `GET /api/v1/runs/{run_id}/manifest` с проверкой ключевых полей (`manifest_version`, `run_id`, `platform`, `video_id`, `artifacts`).

Особенности:

- Используется in-memory storage (`tests/e2e/conftest.py`, класс `InMemoryStorage`), который подставляется вместо реального `fetcher.storage.storage_client`.
- `fetcher.api.fetch_metadata_task.apply_async` замокан: во время `POST /api/v1/runs` задача в очередь **не ставится**, а pipeline запускается синхронно уже после успешного создания run.
- `yt_dlp.YoutubeDL` и файловые операции (`Path.write_text`, `Path.stat`, `Path.mkdir`, `Path.unlink`, `open`) замоканы, чтобы не требовать сеть и диск.
- Класс `TestE2EHappyPath` помечен:
  
  ```python
  @pytest.mark.e2e
  @pytest.mark.slow
  @pytest.mark.database
  @pytest.mark.skip("E2E happy-path: отключён по умолчанию, запускать вручную при необходимости")
  class TestE2EHappyPath:
      ...
  ```

Это означает, что **по умолчанию** E2E‑тест пропускается (skip), чтобы не замедлять обычные прогоны и CI.

### 5.2. Ручной запуск E2E

Если нужно явно прогнать E2E:

1. Временно убрать/закомментировать `@pytest.mark.skip(...)` над `TestE2EHappyPath`.
2. Запустить с поднятым Postgres:

   ```bash
   export FETCHER_POSTGRES_DSN="postgresql+psycopg2://fetcher:fetcher_password@localhost:5433/fetcher_db"
   pytest tests/e2e/test_happy_path.py::TestE2EHappyPath::test_post_runs_to_manifest -v --no-cov --tb=short
   ```

3. После проверки можно вернуть `skip`, чтобы не включать E2E в каждодневный прогон.

---

## 6. CI

Файл: `.github/workflows/fetcher-ci.yml`.

Основные job’ы:

- `lint` — Ruff.
- `alembic` — проверка миграций.
- `unit-tests` — прогон `tests/unit/`.
- `integration-tests` — поднимает Postgres и Redis как services, делает `alembic upgrade head` и запускает `pytest tests/integration/ -m integration ...`.

Важно:

- Для `integration-tests` **снят** `continue-on-error: true` — падение интеграционных тестов теперь роняет pipeline.
- Конфигурация Postgres в CI совпадает с локальной (DSN `postgresql+psycopg2://fetcher:fetcher@localhost:5432/fetcher_test`).

---

## 7. Полезные ссылки

- Общий план и прогресс тестирования: `docs/TESTING_PLAN.md`.
- Интеграция с Backend: `docs/BACKEND_CONTRACTS.md`, `docs/BACKEND_FETCHER_INTEGRATION_ANALYSIS.md`.
- Структура Fetcher и pipeline: `docs/CORE_INGESTION.md`, `docs/PIPELINE_ORCHESTRATION.md`.
---

## Навигация

[Fetcher](../docs/INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
