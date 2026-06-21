# Тесты Backend

Команды (`pytest`, `.venv`, маркеры, Docker): [README.md](../README.md). План и чеклист — [docs/TESTING_PLAN.md](docs/TESTING_PLAN.md), текущее состояние — [docs/TESTING.md](docs/TESTING.md).

**CI** (GitHub Actions, [.github/workflows/backend-ci.yml](../../.github/workflows/backend-ci.yml)) при push/PR в `backend/**`:

1. Установка зависимостей из `requirements.txt` (включая **ruff**, **pytest-cov**).
2. **`ruff check app`** (конфиг: `backend/pyproject.toml`).
3. **`pytest tests/ ... --cov=app --cov-report=xml`** — как в workflow (см. [docs/TESTING.md](../docs/TESTING.md) §3).
4. Загрузка артефакта **`coverage-xml`** (`backend/coverage.xml` относительно корня репозитория).

В тестах ingestion: патч **`app.routers.runs.fetcher_create_run_async`** (`AsyncMock`), не синхронный `fetcher_create_run`.

---

## Структура каталогов

| Каталог / файл | Назначение |
|----------------|------------|
| **conftest.py** | Общие фикстуры: mock_settings, sample_profile_config, sample_run_id, sample_process_payload. |
| **unit/** | Unit-тесты (моки, без внешних сервисов). Маркер: `unit`. |
| **unit/test_config.py** | Settings, resolve_paths (env, пути). |
| **unit/test_auth_jwt.py** | JWT (create/decode token), hash/verify password. |
| **unit/test_events.py** | run_channel, publish_run_event (мок Redis). |
| **integration/** | Интеграционные тесты (API + сервисы, webhooks, tasks). Маркер: `integration`. |
| **integration/test_webhooks.py** | POST /api/webhooks/dataprocessor: подпись, обновление AnalysisJob. |
| **integration/test_tasks.py** | process_analysis_job: отсутствующий job, уже canceled, ValueError при подготовке payload (`app.tasks.analysis`). |
| **integration/test_ws_events.py** | WebSocket /api/runs/{id}/events: JWT в `token`, отказ без/с невалидным токеном (1008), поток событий, ошибка subscribe (1011). |
| **api/** | REST API с dependency overrides (мок user/БД). Маркер: `api`. |
| **api/conftest.py** | mock_user, client, client_with_user. |
| **api/test_auth.py** | /api/auth/me (401), register/login (валидация). |
| **api/test_workspaces.py** | /api/workspaces: 401, список, GET по id → 404. |
| **api/test_channels.py** | /api/workspaces/{id}/channels: 401, 403, 404, 200. |
| **api/test_videos.py** | /api/channels/{id}/videos: 401, 404, 200. |
| **api/test_analysis.py** | Создание job, список jobs, predictions; 401, 404, 201, 200. |
| **api/test_errors.py** | Сводка 401 по защищённым эндпоинтам, невалидный токен. |
| **api/test_runs.py** | Runs (ingestion): create/list/get, trigger-processing; мок **fetcher_create_run_async**; **Idempotency-Key**; `refresh` выставляет timestamps у `IngestionRun`. |
| **api/test_admin.py** | GET /api/auth/admin-check: доступ по admin_emails (200/403). |
| **unit/test_profiles.py** | build_profile_yaml; compute_config_hash; seed_public_profiles (YAML, мок БД). |
| **unit/test_storage.py** | ensure_dirs, move_upload_to_storage, sha256_file (мок путей). |
| **unit/test_quality.py** | discover_quality_scripts, find_component_npz, build_quality_command, run_quality_reports (мок subprocess). |
| **unit/test_db_models_sqlite.py** | core v2 модели (User, Workspace, Channel, Video, AnalysisJob, Prediction) на SQLite (create/read, relationships). |
| **unit/test_admin.py** | admin_email_set, WorkspaceRole.admin. |
| **integration/test_manifest_artifacts.py** | manifest.json (парсинг), _sync_from_manifest_v2, _register_artifact, _scan_and_register_artifacts. |
| **test_dataprocessor_client.py** | Клиент DataProcessor API: POST /process, GET /status, GET /events (SSE). Маркер: `integration`. |
| **test_dataprocessor_adapter.py** | Адаптер v2→legacy: prepare_dataprocessor_payload, resolve_run_paths_v2. Маркер: `integration`. |
| **test_backend_dataprocessor_contract.py** | Контракт payload для DataProcessor. Маркер: `contract`. |

---

## Запуск

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

---

## Контракт с DataProcessor

Симметричные тесты в DataProcessor:  
`DataProcessor/api/tests/integration/test_backend_integration_contract.py`

Запуск (из корня репо или из `DataProcessor/api`):

```bash
cd DataProcessor/api && pytest tests/integration/test_backend_integration_contract.py -v
```
---

## Навигация

[Backend](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
