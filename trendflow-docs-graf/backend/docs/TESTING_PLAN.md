# План и чеклист тестирования Backend

Документ задаёт стратегию тестирования backend-сервиса, приоритеты и чеклист для поэтапной реализации тестов.

---

## 1. Цели и границы

- **Цель**: обеспечить регрессионную защиту и проверку контрактов (API, DataProcessor, БД, события).
- **Границы**: тесты в репозитории `backend/`; интеграция с внешними сервисами (DataProcessor, Redis, PostgreSQL) — через моки или тестовые инстансы по выбору этапа.
- **Инструменты**: pytest, pytest-asyncio; при необходимости — TestClient (FastAPI), фабрики БД (Alembic + test DB), моки (unittest.mock, respx/httpx).
- **Практические команды и CI**: [README.md](../README.md), [.github/workflows/backend-ci.yml](../../.github/workflows/backend-ci.yml).

---

## 2. Уровни тестирования

| Уровень | Описание | Где живут |
|--------|----------|-----------|
| **Unit** | Изолированные модули/функции с моками зависимостей | `backend/tests/unit/` |
| **Integration** | Взаимодействие компонентов (API ↔ сервисы, сервисы ↔ БД, Backend ↔ DataProcessor) | `backend/tests/integration/` |
| **Contract** | Совпадение payload/схем с контрактами (DataProcessor, API) | `backend/tests/` или `tests/contract/` |
| **E2E** (опционально) | Полный сценарий с реальными сервисами (по необходимости) | `backend/tests/e2e/` |

---

## 3. Чеклист по областям

### 3.1 Интеграция Backend ↔ DataProcessor

**Статус**: реализовано.

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.1.1 | Клиент DataProcessor API: POST /api/v1/process (payload, headers, 202) | ✅ | `tests/test_dataprocessor_client.py` |
| 3.1.2 | Клиент: GET /api/v1/runs/{id}/status (polling, финальный статус, timeout) | ✅ | `tests/test_dataprocessor_client.py` |
| 3.1.3 | Клиент: GET /api/v1/runs/{id}/events (SSE), опционально | ✅ | `tests/test_dataprocessor_client.py` (TestStreamRunEventsSse) |
| 3.1.4 | Адаптер: prepare_dataprocessor_payload (v2 → legacy, мок БД) | ✅ | `tests/test_dataprocessor_adapter.py` |
| 3.1.5 | Адаптер: resolve_run_paths_v2, resolve_run_paths (структура путей) | ✅ | `tests/test_dataprocessor_adapter.py`, `test_dataprocessor_client.py` |
| 3.1.6 | Контракт payload: обязательные поля и форматы для DataProcessor | ✅ | `tests/test_backend_dataprocessor_contract.py` |
| 3.1.7 | Webhook DataProcessor: приём POST /api/webhooks/dataprocessor, обновление AnalysisJob (в т.ч. **success, error, cancelled** → `AnalysisStatus`) | ✅ | `tests/integration/test_webhooks.py`; подмена `session_scope` — `@contextmanager` |
| 3.1.8 | Celery task process_analysis_job: ранние выходы, ValueError из prepare_payload | ✅ | `tests/integration/test_tasks.py` (патчи `app.tasks.analysis.*`) |

**Документация**: [TESTING.md](TESTING.md) (раздел «Интеграция с DataProcessor»), [backend/tests/README.md](../tests/README.md).

---

### 3.2 REST API (FastAPI)

**Статус**: в основном реализовано (остались upload, profiles API).

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.2.1 | Auth: register, login, me (успех, неверные данные, JWT) | ✅ | `tests/api/test_auth.py` (me 401, register/login валидация) |
| 3.2.2 | Workspaces: создание, список, получение, права доступа | ✅ | `tests/api/test_workspaces.py` (401 без токена, список/404 с override) |
| 3.2.3 | Channels: создание, список по workspace | ✅ | `tests/api/test_channels.py` (401, 403, 404, 200) |
| 3.2.4 | Videos: список по channel, метаданные | ✅ | `tests/api/test_videos.py` (401, 404, 200) |
| 3.2.5 | Analysis: POST job, список jobs, predictions, **POST cancel** (queued/processing/noop) | ✅ | `tests/api/test_analysis.py` |
| 3.2.6 | Upload: init, complete (и при необходимости multipart), дедуп | ⬜ | `tests/api/test_uploads.py` |
| 3.2.7 | Profiles: список публичных, получение по id | ⬜ | `tests/api/test_profiles.py` (API; логика профиля — unit) |
| 3.2.8 | Ошибки: 401 без токена, 403 при недостаточных правах, 404 | ✅ | `tests/api/test_errors.py` + в test_* (401/403/404 по эндпоинтам) |
| 3.2.9 | Runs (ingestion по URL): create, list, get, trigger-processing, **идемпотентность `Idempotency-Key`**, мок **`fetcher_create_run_async`** | ✅ | `tests/api/test_runs.py`; E2E: `tests/e2e/test_ingestion_e2e.py` |

**Зависимости**: фикстуры БД (session, тестовые пользователи/workspaces), JWT-токены в заголовках.

---

### 3.3 База данных и модели

**Статус**: частично реализовано.

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.3.1 | Миграции Alembic: применение up/down на тестовой БД | ⬜ | `tests/integration/test_migrations.py` или скрипт |
| 3.3.2 | Модели core.*: создание/чтение User, Workspace, Channel, Video, AnalysisJob | ✅ | `tests/unit/test_db_models_sqlite.py` (in-memory SQLite, базовые create/read и relationships) |
| 3.3.3 | Legacy-модели (при использовании): Run, VideoFile, AnalysisProfile | ⬜ | По необходимости |
| 3.3.4 | Связи и ограничения: уникальность, FK, каскады | ⬜ | В рамках тестов моделей или интеграции |

**Зависимости**: тестовая БД (например, PostgreSQL в Docker или SQLite для части тестов), Alembic env.

---

### 3.4 События и WebSocket

**Статус**: реализовано.

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.4.1 | publish_run_event: публикация в Redis, формат payload | ✅ | `tests/unit/test_events.py` (мок Redis) |
| 3.4.2 | WebSocket GET /api/runs/{id}/events: `?token=` (JWT), отказ без токена/с неверным JWT, подключение и события | ✅ | `tests/integration/test_ws_events.py` |
| 3.4.3 | Формат событий: run.status_changed, run.stage_changed, component.* | ✅ | `tests/integration/test_ws_events.py` (передача payload как JSON по WebSocket) |

**Ссылки**: [EVENTS_AND_LOGGING.md](EVENTS_AND_LOGGING.md).

---

### 3.5 Конфигурация и пути

**Статус**: реализовано.

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.5.1 | Settings: загрузка из env (TF_BACKEND_*), значения по умолчанию | ✅ | `tests/unit/test_config.py` |
| 3.5.2 | resolve_paths: storage_root, result_store_base, dataproc_root при разных env | ✅ | `tests/unit/test_config.py` |

**Ссылки**: [CONFIGURATION.md](CONFIGURATION.md).

---

### 3.6 Безопасность

**Статус**: реализовано.

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.6.1 | JWT: создание и верификация токена, истечение срока | ✅ | `tests/unit/test_auth_jwt.py` |
| 3.6.2 | get_current_user: извлечение пользователя из токена, 401 при невалидном | ✅ | `tests/api/test_auth.py` (me без/с невалидным токеном) |
| 3.6.3 | Webhook: проверка подписи X-Webhook-Signature | ✅ | `tests/integration/test_webhooks.py` |
| 3.6.4 | Admin: проверка admin_emails / роли | ✅ | `tests/unit/test_admin.py` (admin_email_set, WorkspaceRole.admin); `tests/api/test_admin.py` (GET /api/auth/admin-check: 200 в списке, 403 вне списка и при пустом списке); `app/deps.require_admin_user`, GET /api/auth/admin-check |

**Ссылки**: [SECURITY.md](SECURITY.md).

---

### 3.7 Профили анализа

**Статус**: реализовано.

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.7.1 | Нормализация профиля: visual.cfg_path, processors по умолчанию | ✅ | В adapter: `test_dataprocessor_adapter.py`; запись YAML: `tests/unit/test_profiles.py` |
| 3.7.2 | Вычисление config_hash (детерминированность, формат) | ✅ | Через adapter (AnalysisProfile.config_hash) в `test_dataprocessor_adapter.py` |
| 3.7.3 | Seed публичных профилей из YAML | ✅ | `app/services/profiles.py`: compute_config_hash, seed_public_profiles; вызов на startup (main.py). Unit: `tests/unit/test_profiles.py` (TestComputeConfigHash, TestSeedPublicProfiles — мок БД и ФС) |

**Ссылки**: [PROFILES.md](PROFILES.md).

---

### 3.8 Хранилище и артефакты

**Статус**: реализовано.

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.8.1 | Разрешение путей: raw_uploads_dir, result_store, frames_dir | ✅ | `tests/unit/test_config.py` (resolve_paths), `tests/unit/test_storage.py` (ensure_dirs, move_upload) |
| 3.8.2 | Чтение manifest.json, регистрация артефактов в БД | ✅ | `tests/integration/test_manifest_artifacts.py` (_sync_from_manifest_v2, _register_artifact, _scan_and_register_artifacts) |
| 3.8.3 | Quality reports: вызов скриптов, регистрация артефактов | ✅ | `tests/unit/test_quality.py` (discover_quality_scripts, find_component_npz, build_quality_command, run_quality_reports с моком subprocess) |

**Ссылки**: [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md), [RUNS_AND_WORKERS.md](RUNS_AND_WORKERS.md).

---

### 3.9 Прочее и инфраструктура

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.9.1 | CI: запуск pytest в pipeline (GitHub Actions / другой CI) | ✅ | [.github/workflows/backend-ci.yml](../../.github/workflows/backend-ci.yml) — при push/PR в `backend/**` |
| 3.9.2 | Покрытие кода (coverage): отчёт, порог (опционально) | ✅ | pytest-cov в CI (--cov=app, term-missing + xml); порог не задан |
| 3.9.3 | Маркировка тестов: unit / integration / contract (pytest markers) | ✅ | pytest.ini, маркеры unit, integration, contract |
| 3.9.4 | Документирование: обновление TESTING.md при добавлении новых групп тестов | ✅ | По мере реализации (актуализируется) |

---

### 3.10 E2E: ingestion run до completed

**Статус**: реализовано (скрипт + документация).

| # | Задача | Статус | Файл/примечание |
|---|--------|--------|------------------|
| 3.10.1 | Скрипт e2e_run_to_complete: регистрация/логин, создание run, опрос до completed/failed | ✅ | `backend/scripts/e2e_run_to_complete.py` |
| 3.10.2 | Опциональный опрос Fetcher (--fetcher-url) для прогресса по этапам | ✅ | Там же |
| 3.10.3 | Документация: переменные окружения, команды запуска, типичные проблемы | ✅ | [E2E_RUNBOOK.md](E2E_RUNBOOK.md) |
| 3.10.4 | Полный E2E (Fetcher + DataProcessor): --with-dataprocessor, ожидание processing → completed | ✅ | `e2e_run_to_complete.py --with-dataprocessor`; runbook п. 1.2, 4.6–4.7, 5 |

**Полный runbook** (что запускать, какие env, команды, что мы правили): [E2E_RUNBOOK.md](E2E_RUNBOOK.md). Полный E2E с DataProcessor: запуск DataProcessor API + worker, `TF_BACKEND_DATAPROCESSOR_API_URL`, скрипт с `--with-dataprocessor`.

---

## 4. Приоритеты реализации

1. **Уже есть**: интеграция Backend ↔ DataProcessor (клиент, адаптер, контракт payload) — см. [TESTING.md](TESTING.md).
2. **Высокий приоритет**: REST API (auth, workspaces, analysis, upload) — защита основного сценария.
3. **Средний**: Celery task с моком DataProcessor, webhook, события/WebSocket.
4. **Средний**: Конфиг, профили, БД/модели.
5. **Низкий**: E2E, coverage, расширенные сценарии безопасности и хранилища.

---

## 5. Запуск и конфигурация тестов

- **Команда**: из каталога `backend`: `pytest tests/ -v`.
- **Ruff** (как в CI): из каталога `backend`: `ruff check app`.
- **Только unit**: `pytest tests/ -v -m unit` (после введения маркеров).
- **Только интеграция с DataProcessor**:  
  `pytest tests/test_dataprocessor_client.py tests/test_dataprocessor_adapter.py tests/test_backend_dataprocessor_contract.py -v`.
- **Зависимости**: см. [backend/requirements.txt](../requirements.txt) (pytest, pytest-asyncio, pytest-cov, ruff).
- **Переменные окружения**: для тестов без реальных сервисов достаточно моков; CI задаёт `TF_BACKEND_JWT_SECRET`, `TF_BACKEND_DB_DSN`, `TF_BACKEND_REDIS_URL` (см. workflow); при **реальном** вызове register без БД возможен проброс `OperationalError` из TestClient — см. `tests/api/test_auth.py`.

---

## 6. Связанные документы

- [TESTING.md](TESTING.md) — что реализовано, как запускать, ссылки на тесты Backend ↔ DataProcessor.
- [E2E_RUNBOOK.md](E2E_RUNBOOK.md) — полный запуск E2E (Backend + Fetcher до completed): переменные окружения, команды, фиксы, типичные проблемы.
- [backend/tests/README.md](../tests/README.md) — содержимое каталога тестов и примеры команд.
- [reference/DATAPROCESSOR_CONTRACT.md](reference/DATAPROCESSOR_CONTRACT.md) — контракт Backend ↔ DataProcessor.
- [API.md](API.md), [RUNS_AND_WORKERS.md](RUNS_AND_WORKERS.md) — контекст API и воркеров.
- [CONFIGURATION.md](CONFIGURATION.md) — переменные окружения Backend, один Postgres с Fetcher, один Redis для Fetcher API и worker.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
