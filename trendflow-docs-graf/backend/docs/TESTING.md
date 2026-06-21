# Тестирование Backend

Документ описывает текущее состояние тестов backend, как их запускать и куда смотреть для расширения покрытия.

---

## 1. Обзор

- **Команды и окружение (.venv, Docker, выборочные маркеры):** [README.md](../README.md).
- **План и чеклист** по всем областям тестирования backend: [TESTING_PLAN.md](TESTING_PLAN.md).
- **Реализовано на данный момент**: интеграция Backend ↔ DataProcessor (клиент, адаптер, контракт, SSE, webhook, task); unit-тесты (конфиг, JWT/пароли, события); API-тесты с dependency overrides; в CI дополнительно **Ruff** и отчёт **coverage.xml** (см. §3).
- **Расположение тестов**: каталог [backend/tests/](../tests/) — подкаталоги `unit/`, `integration/`, `api/`; корневые тесты DataProcessor. Краткое описание: [backend/tests/README.md](../tests/README.md).
- **Маркеры pytest**: `unit`, `integration`, `contract`, `e2e` — выборочный запуск: `pytest -m unit`, `pytest -m integration`, `pytest -m e2e`.
- **E2E ingestion (Phase 5):** тесты в [tests/e2e/](../tests/e2e/) (мок Fetcher); ручной чеклист: [docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md](../../docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md).

---

## 2. Что реализовано

### 2.1 Unit-тесты (tests/unit/)

| Файл | Описание |
|------|----------|
| [test_config.py](../tests/unit/test_config.py) | Settings: значения по умолчанию, env prefix, resolve_paths (все атрибуты, явный storage_root). |
| [test_auth_jwt.py](../tests/unit/test_auth_jwt.py) | hash_password, verify_password, create_access_token, decode_token (истёкший токен → None). |
| [test_events.py](../tests/unit/test_events.py) | run_channel, publish_run_event (мок Redis: канал, сериализация JSON, unicode). |
| [test_profiles.py](../tests/unit/test_profiles.py) | build_profile_yaml (YAML roundtrip); compute_config_hash (детерминированность); seed_public_profiles из YAML (мок БД: создание по имени, пропуск существующих, отсутствие директории). |
| [test_storage.py](../tests/unit/test_storage.py) | ensure_dirs, move_upload_to_storage, sha256_file (пути через мок Settings). |
| [test_quality.py](../tests/unit/test_quality.py) | discover_quality_scripts, find_component_npz, build_quality_command, run_quality_reports (мок subprocess). |
| [test_db_models_sqlite.py](../tests/unit/test_db_models_sqlite.py) | core v2 модели (User, Workspace, Channel, Video, AnalysisJob, Prediction) на in-memory SQLite: базовые create/read и relationships. |
| [test_admin.py](../tests/unit/test_admin.py) | admin_email_set (Settings): пустой список, один/несколько email, нормализация; WorkspaceRole.admin. |

### 2.2 Интеграция Backend ↔ DataProcessor

| Компонент | Описание | Файл тестов |
|-----------|----------|-------------|
| **Клиент DataProcessor API** | POST /api/v1/process, GET /api/v1/runs/{id}/status, GET /api/v1/runs/{id}/events (SSE). Payload, X-API-Key, 202, 4xx/5xx, timeout, поток SSE до complete. | [tests/test_dataprocessor_client.py](../tests/test_dataprocessor_client.py) |
| **Адаптер v2 → legacy** | Преобразование AnalysisJob (и связанных Video, Channel, AnalysisProfile) в формат DataProcessor. Тесты с моками БД: обязательные поля payload, fallback video_id, ошибки при отсутствии Video/Channel. | [tests/test_dataprocessor_adapter.py](../tests/test_dataprocessor_adapter.py) |
| **Контракт payload** | Проверка, что payload для POST /api/v1/process содержит обязательные поля и форматы (run_id UUID, platform_id youtube|upload, profile_config.processors и т.д.), а пути результатов (manifest, state_events) соответствуют соглашению с DataProcessor. | [tests/test_backend_dataprocessor_contract.py](../tests/test_backend_dataprocessor_contract.py) |
| **Симметричный контракт в DataProcessor** | DataProcessor API принимает payload в формате Backend и возвращает 202; валидация через ProcessRequest (Pydantic). | [DataProcessor/api/tests/integration/test_backend_integration_contract.py](../../DataProcessor/api/tests/integration/test_backend_integration_contract.py) |

Используются моки: **httpx** для HTTP, **Session/ORM-объекты** для БД. Реальный DataProcessor и реальная БД в этих тестах не требуются.

### 2.3 Интеграционные тесты (tests/integration/)

| Файл | Описание |
|------|----------|
| [test_webhooks.py](../tests/integration/test_webhooks.py) | POST /api/webhooks/dataprocessor: успех (200, обновление AnalysisJob), 401 при неверной подписи, 404 при отсутствии job, `status=error` → failed, **`status=cancelled` → `AnalysisJob` в `canceled`**. Патч `session_scope` — `@contextmanager`, иначе `with session_scope() as db` не работает. |
| [test_tasks.py](../tests/integration/test_tasks.py) | process_analysis_job: выход при отсутствии job / уже canceled, ValueError из prepare_dataprocessor_payload (патчи `app.tasks.analysis`). |
| [test_manifest_artifacts.py](../tests/integration/test_manifest_artifacts.py) | Парсинг manifest.json (контракт run/components/predictions), _sync_from_manifest_v2, _register_artifact, _scan_and_register_artifacts (мок БД). |
| [test_ws_events.py](../tests/integration/test_ws_events.py) | WebSocket /api/runs/{id}/events: обязательный `?token=`, отказ 1008 без/с плохим JWT, приём событий из subscribe_run_events, закрытие 1011 при внутренней ошибке. |

### 2.4 API-тесты (tests/api/)

| Файл | Описание |
|------|----------|
| [test_auth.py](../tests/api/test_auth.py) | GET /api/auth/me без токена и с невалидным токеном → 401; register/login: валидация 422; при доступной БД — 201/409/500. **Без БД** Starlette `TestClient` может **пробросить `sqlalchemy.exc.OperationalError`** при register — тест это учитывает (не требует ответа с кодом). |
| [test_workspaces.py](../tests/api/test_workspaces.py) | GET /api/workspaces без токена → 401; с override: список 200, GET по id → 404. Мок `query(Workspace)` должен имитировать цепочку **`.join(WorkspaceMember).filter(...).first()`** / **`.order_by(...).all()`** (как в роутере). |
| [test_channels.py](../tests/api/test_channels.py) | GET /api/workspaces/{id}/channels: 401, 403 (не член), 404 (workspace не найден), 200. |
| [test_videos.py](../tests/api/test_videos.py) | GET /api/channels/{id}/videos: 401, 404 (channel не найден), 200. |
| [test_analysis.py](../tests/api/test_analysis.py) | POST job (201, мок Celery), GET список jobs, GET predictions; **POST cancel** (canceled / cancel_requested / noop); 401, 404 (video/job не найден). |
| [test_errors.py](../tests/api/test_errors.py) | Сводка 401 без токена по всем защищённым эндпоинтам; невалидный JWT → 401. |
| [test_runs.py](../tests/api/test_runs.py) | Runs (ingestion по URL): POST/GET /api/runs, GET /api/runs/{id}, POST trigger-processing; 401, 404, 201, 202; мок **`fetcher_create_run_async`** (`AsyncMock`), Celery. **Идемпотентность `Idempotency-Key`:** существующий run → 201, тот же `run_id`, Fetcher не вызывается. Для мок-ORM: **`refresh` заполняет `created_at`/`updated_at`** у `IngestionRun`. |
| [test_admin.py](../tests/api/test_admin.py) | GET /api/auth/admin-check: 200 при email в admin_emails, 403 вне списка и при пустом списке (require_admin_user). |
| [test_health.py](../tests/api/test_health.py) | GET /health, /health/live — 200; GET /health/ready — 200 с моками БД/Redis, 503 при падении проверки БД. |

### 2.5 E2E ingestion (tests/e2e/)

| Файл | Описание |
|------|----------|
| [test_ingestion_e2e.py](../tests/e2e/test_ingestion_e2e.py) | Сценарии создания run по URL: успех (201, run_id); ошибка Fetcher (502 при Exception/Timeout). Мок **`fetcher_create_run_async`** (`AsyncMock`). На мок-сессии **`refresh`** выставляет `created_at`/`updated_at` у новой `IngestionRun`. Запуск: `pytest tests/e2e/ -v` или `pytest -m e2e -v`. |

Полный ручной E2E чеклист (YouTube URL → результат): [docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md](../../docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md).

### 2.6 Фикстуры и конфигурация

- **tests/conftest.py**: `mock_settings`, `sample_profile_config`, `sample_run_id`, `sample_process_payload`.
- **tests/api/conftest.py**: `mock_user`, `client`, `client_with_user`; в отдельных модулях API — свои фикстуры.
- **pytest.ini**: `asyncio_mode = auto`, `testpaths = tests`, маркеры `unit`, `integration`, `contract`.
- **backend/pyproject.toml**: конфигурация **Ruff** (`[tool.ruff]`, `[tool.ruff.lint]`; для длинных строк комментариев задано `ignore = ["E501"]`).

Зависимости: [backend/requirements.txt](../requirements.txt) — pytest, pytest-asyncio, pytest-cov, **ruff** и зависимости приложения.

---

## 3. Как запускать тесты

Из каталога **backend**:

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

По маркерам:

```bash
pytest -m unit -v
pytest -m integration -v
pytest -m contract -v
```

Только интеграция с DataProcessor:

```bash
pytest tests/test_dataprocessor_client.py tests/test_dataprocessor_adapter.py tests/test_backend_dataprocessor_contract.py -v
```

Запуск контрактных тестов со стороны DataProcessor (из корня репозитория или из `DataProcessor/api`):

```bash
cd DataProcessor/api && pytest tests/integration/test_backend_integration_contract.py -v
```

### CI (GitHub Actions)

При push или pull request в ветки `main`/`master`, затронувших каталог `backend/`, запускается workflow **Backend CI** (файл [.github/workflows/backend-ci.yml](../../.github/workflows/backend-ci.yml)):

- установка Python 3.11 и зависимостей из `backend/requirements.txt`;
- проверка стиля и импорта: `ruff check app` (конфиг — `backend/pyproject.toml`);
- запуск `pytest tests/ -v --tb=short --cov=app --cov-report=term-missing --cov-report=xml --no-cov-on-fail` в каталоге `backend/`;
- публикация `backend/coverage.xml` как артефакта workflow (имя артефакта **`coverage-xml`**, шаг с `if: always()`).

Сервисы (PostgreSQL, Redis) в CI не поднимаются — текущие тесты используют моки. При необходимости интеграционных тестов с реальной БД можно добавить отдельный job с services (по аналогии с Fetcher CI).

**Локально как в CI** (из каталога `backend/`, после `pip install -r requirements.txt`):

```bash
.venv/bin/ruff check app
TF_BACKEND_JWT_SECRET=ci-secret-do-not-use-in-production \
TF_BACKEND_DB_DSN=postgresql+psycopg://u:p@localhost/db \
TF_BACKEND_REDIS_URL=redis://localhost:6379/0 \
.venv/bin/python -m pytest tests/ -v --tb=short --cov=app --cov-report=term-missing --cov-report=xml --no-cov-on-fail
```

Файл **`coverage.xml`** появляется в `backend/coverage.xml` (для сравнения с артефактом Actions).

---

## 4. План дальнейшего развития

Полный чеклист по всем областям (REST API, БД, события, безопасность, профили, хранилище, CI и т.д.) и приоритеты реализации приведены в **[TESTING_PLAN.md](TESTING_PLAN.md)**. Кратко:

- **Уже есть**: интеграция с DataProcessor (клиент, адаптер, контракт, SSE), webhook (в т.ч. финальный **cancelled**), Celery task (мок), unit (конфиг, JWT, события), API (в т.ч. **идемпотентность POST /api/runs**).
- **Далее по приоритету**: upload, profiles API; БД/модели; WebSocket endpoint. Хранилище и артефакты (§ 3.8) покрыты: storage, quality, manifest/артефакты.

---

## 5. Связанные документы

- [TESTING_PLAN.md](TESTING_PLAN.md) — план и чеклист всех тестов backend.
- [tests/README.md](../tests/README.md) — содержимое каталога тестов и примеры команд.
- [reference/DATAPROCESSOR_CONTRACT.md](reference/DATAPROCESSOR_CONTRACT.md) — контракт Backend ↔ DataProcessor.
- [RUNS_AND_WORKERS.md](RUNS_AND_WORKERS.md) — lifecycle analysis job и вызов DataProcessor.
- [CONFIGURATION.md](CONFIGURATION.md) — переменные окружения (в т.ч. DataProcessor API).
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
