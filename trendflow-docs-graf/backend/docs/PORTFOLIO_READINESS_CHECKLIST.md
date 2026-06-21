# Чеклист: готовность Backend к портфолио и «нормальному» прод-виду

Документ фиксирует цель (**отдельный репозиторий / понятный для работодателя сервис**), критерии готовности и статус работ. Обновляйте колонку «Статус» по мере выполнения.

**Сводный гайд для демонстрации (питч, чеклист показа, карта документов, Q&A):** [DEMO_AND_PORTFOLIO.md](DEMO_AND_PORTFOLIO.md) — после правок здесь имеет смысл сверяться с ним.

**Легенда:** `[ ]` не начато · `[~]` в работе · `[x]` готово · `[-]` сознательно отложено / не в скоупе

---

## 1. Репозиторий и первое впечатление


| #   | Задача                                                                                | Статус | Примечание                                                                                         |
| --- | ------------------------------------------------------------------------------------- | ------ | -------------------------------------------------------------------------------------------------- |
| 1.1 | Корневой `backend/README.md`: назначение сервиса, стек, быстрый старт                 | [x]    | Ссылка на чеклист, `MAIN_INDEX`, `DEMO_AND_PORTFOLIO`                                              |
| 1.2 | `backend/.env.example`: все основные `TF_BACKEND`_* без секретов                      | [x]    | Копировать в `.env`, подставить значения                                                           |
| 1.3 | Диаграмма потоков (mermaid) в README или в `docs/OVERVIEW.md`                         | [x]    | `backend/README.md` (mermaid)                                                                      |
| 1.4 | Вынос из монорепы: список артефактов, замена путей (`dataproc_root`, `resolve_paths`) | [x]    | [STANDALONE_REPOSITORY.md](STANDALONE_REPOSITORY.md), `docker-compose.standalone.yml`, `profiles/` |
| 1.5 | Docker Compose: Postgres + Redis + API + worker (+ beat при необходимости)            | [x]    | `backend/docker-compose.yml`, `backend/Dockerfile`; beat — вручную при необходимости               |


---

## 2. Безопасность и снижение «красных флагов»


| #   | Задача                                                         | Статус | Примечание                                                                      |
| --- | -------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------- |
| 2.1 | CORS: `allow_origins` из env (список URL), в prod без `*`      | [x]    | `TF_BACKEND_CORS_ORIGINS`, `config.cors_allow_origins()`                        |
| 2.2 | WebSocket `/api/runs/.../events`: проверка JWT (или ticket)    | [x]    | `?token=`, владелец run; `SECURITY.md` §3, `runs.py`                           |
| 2.3 | Запрет дефолтного `jwt_secret` в prod (fail-fast или warning)  | [x]    | `TF_BACKEND_DEPLOYMENT_ENV` production/staging + `validate_security_at_startup` |
| 2.4 | Унификация секретов для webhook/trigger (`X-API-Key`)          | [x]    | Задокументировано в `SECURITY.md` §5 (trigger + webhooks)                       |
| 2.5 | Краткий раздел «Threat model / известные ограничения» в README | [x]    | Секция «Известные ограничения» + `GAPS`; расширено в `DEMO_AND_PORTFOLIO.md` §4 |


---

## 3. Надёжность и операционка


| #   | Задача                                                              | Статус | Примечание                        |
| --- | ------------------------------------------------------------------- | ------ | --------------------------------- |
| 3.1 | Liveness: `GET /health`, `GET /health/live`                         | [x]    | Процесс жив                       |
| 3.2 | Readiness: `GET /health/ready` (PostgreSQL + Redis)                 | [x]    | 503 при недоступности зависимости |
| 3.3 | Документация health в `OPERATIONS.md`, CI не ломается без сервисов  | [x]    | Тесты с моками для `ready`        |
| 3.4 | Cancel analysis: договориться о семантике (флаг vs kill subprocess) | [x]    | `POST .../cancel`, DP `.../runs/{id}/cancel`; `OPERATIONS` §6, `GAPS` §5 |
| 3.5 | Идемпотентность `POST .../upload/complete` (если нужна для истории) | [x]    | Контракт в `UPLOADS_AND_VIDEOS.md` §4, `GAPS` §3                        |


---

## 4. Код и сопровождаемость


| #   | Задача                                                                | Статус | Примечание                                                                                |
| --- | --------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------- |
| 4.1 | Стартап: не глотать `seed_public_profiles` без следа — логировать     | [x]    | `main.py`, уровень warning (+ traceback при `debug`)                                      |
| 4.2 | Разнести `tasks.py` на модули (ingestion / analysis / события)        | [x]    | Пакет `app/tasks/`: `analysis`, `ingestion`, `events`, `manifest`; точка входа `__init__.py` |
| 4.3 | Убрать или локализовать legacy-модели (`models.py` vs `dbv2`) в доках | [x]    | `DATABASE.md`, `OVERVIEW.md`, `MAIN_INDEX`, `API.md`; см. `DATAPROCESSOR_CONTRACT` вверху |


---

## 5. Тесты и CI


| #   | Задача                                               | Статус | Примечание                   |
| --- | ---------------------------------------------------- | ------ | ---------------------------- |
| 5.1 | Тесты для `/health`, `/health/live`, `/health/ready` | [x]    | `tests/api/test_health.py`   |
| 5.2 | Покрытие критичных веток из `GAPS` (по приоритету)   | [x]    | `tests/api/test_runs.py` (`TestRunsCreateIdempotency`); `tests/integration/test_webhooks.py` (`status=cancelled` → `AnalysisStatus.canceled`). Зафиксировано в `TESTING.md`, `TESTING_PLAN.md`, `GAPS_AND_ALIGNMENT.md` §3. |
| 5.3 | Линтер (ruff) в CI                                   | [x]    | Шаг `ruff check app` в workflow; конфиг `[tool.ruff]` в `backend/pyproject.toml` (`ignore = ["E501"]` для длинных комментариев); `ruff` в `requirements.txt`. Локально: `.venv/bin/ruff check app`. |
| 5.4 | Публикация `coverage.xml` в артефакты Actions        | [x]    | `actions/upload-artifact@v4`, имя артефакта `coverage-xml`, путь с корня репо: `backend/coverage.xml`, `if: always()`. |


---

## 6. Документация


| #   | Задача                                                                        | Статус | Примечание                                                    |
| --- | ----------------------------------------------------------------------------- | ------ | ------------------------------------------------------------- |
| 6.1 | Блок «Portfolio» в README (5–7 bullets: стек, интеграции, тесты)              | [x]    | Секция «Для портфолио» в `backend/README.md`                  |
| 6.2 | Обновлять `GAPS_AND_ALIGNMENT.md` после закрытия пунктов; перекрёстные ссылки | [x]    | Health, CORS; ссылка на `DEMO_AND_PORTFOLIO.md`               |
| 6.3 | ADR: Celery + Redis pub/sub + manifest как source of truth                    | [x]    | [docs/adr/0001-celery-redis-pubsub-manifest-source-of-truth.md](adr/0001-celery-redis-pubsub-manifest-source-of-truth.md), индекс [docs/adr/README.md](adr/README.md) |
| 6.4 | Единый документ «демо + портфолио» (чеклист, карта, Q&A)                      | [x]    | `DEMO_AND_PORTFOLIO.md`, индексы `MAIN_INDEX` / `docs/README` |


---

## 7. Подготовка к собеседованиям


| #   | Задача                                                           | Статус | Примечание                                                                |
| --- | ---------------------------------------------------------------- | ------ | ------------------------------------------------------------------------- |
| 7.1 | Шпаргалка Q&A по своему коду (очереди, статусы, идемпотентность) | [x]    | `DEMO_AND_PORTFOLIO.md` §5                                                |
| 7.2 | Один сценарий E2E «снять на экран» (run → события → артефакт)    | [ ]    | [E2E_PIPELINE_NO_TEXT.md](E2E_PIPELINE_NO_TEXT.md), чеклисты в корне репо |


---

## История изменений чеклиста


| Дата       | Изменение                                                                                                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-04-02 | Первая версия; закрыты 3.1–3.3, 4.1, 5.1, частично 1.1–1.2, 6.2                                                                                                                                    |
| 2026-04-02 | Docker Compose + Dockerfile; CORS из env; mermaid в README; правка теста analysis (`refresh` / datetime); корневой `.dockerignore`                                                                 |
| 2026-04-02 | Синхронизация `MAIN_INDEX.md`, `docs/README.md`, `RUNS_AND_WORKERS.md`, `DATABASE.md` (ingestion_runs), `FETCHER_INTEGRATION` (Celery), `API`/`TESTING` ссылками на README и legacy/core           |
| 2026-04-02 | **Финализация документации для демо:** `DEMO_AND_PORTFOLIO.md`, вступление в `DATAPROCESSOR_CONTRACT.md`, `SECURITY.md` §5, ссылки из `GAPS`, пункты 2.4, 4.3, 6.2, 6.4, 7.1 отмечены выполненными |
| 2026-04-02 | П. 1.4: fallback `dataproc_root` в `config.resolve_paths`, `STANDALONE_REPOSITORY.md`, `docker-compose.standalone.yml`, каталог `profiles/` с README                                               |
| 2026-04-02 | П. 2.2–2.3: WS `?token=` + владелец run; `TF_BACKEND_DEPLOYMENT_ENV` + слабый JWT; тесты `test_ws_events`, `test_security_startup`; SECURITY/GAPS/API/EVENTS/DEMO                                              |
| 2026-04-02 | П. 3.4–3.5: отмена analysis + вызов DP cancel; контракт idempotency upload/complete; тесты `test_analysis` (cancel); правка `AnalysisStatus.canceled` в tasks/webhooks                                     |
| 2026-04-02 | П. 4.2: `app/tasks.py` → пакет `app/tasks/` (`analysis`, `ingestion`, `events`, `manifest`); патчи в `test_tasks` на `app.tasks.analysis.*`; обновлены README, RUNS_AND_WORKERS, ссылки в доках                |
| 2026-04-02 | П. 5.2–5.4: тесты идемпотентности POST `/api/runs` и webhook `cancelled`; Ruff в CI + autofix по `app/`; артефакт `coverage.xml`; правка патчей `fetcher_create_run_async` в тестах runs/e2e                             |
| 2026-04-02 | E2E/DataProcessor: сводка фиксов (таймаут enqueue, порт 300 s, CLI/config_parser, аудио NPZ) в [E2E_DP_FIXES_2026-04.md](E2E_DP_FIXES_2026-04.md); донастройка AudioProcessor (mel/chroma/схемы/pitch/key/hpss/GPU batch, рендеры pitch/band_energy) для `state_audio` без ложных validation fail |
| 2026-04-02 | П. 5.2–5.4 в документах: детализация в `TESTING.md`, `TESTING_PLAN.md`, `tests/README.md`, `README.md`, `DEMO_AND_PORTFOLIO.md`, `MAIN_INDEX.md`; в `GAPS_AND_ALIGNMENT.md` §3 — ссылка на тест идемпотентности `POST /api/runs`; примечания по мокам (`session_scope`, `join`, Fetcher async). |
| 2026-04-02 | П. 6.3: ADR 0001 — Celery, Redis (брокер + pub/sub `run:{id}`), manifest/файлы как source of truth; каталог `docs/adr/`, ссылки в `MAIN_INDEX`, `docs/README`, `DEMO_AND_PORTFOLIO` §3. |
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
