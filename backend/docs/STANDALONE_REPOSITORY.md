# Автономный репозиторий backend (вынос из монорепозитория TrendFlowML)

Цель: один git-репозиторий, в корне которого лежит то, что сейчас в каталоге `backend/` монорепо (приложение, тесты, Docker, доки).

## Поведение путей (`resolve_paths`)

- **Монорепозиторий:** `repo_root` — корень TrendFlowML (родитель `backend/app/config.py` поднять на два уровня). Если существует каталог `repo_root/DataProcessor`, то **`dataproc_root`** по умолчанию = этот каталог (профили: `DataProcessor/profiles/*.yaml`).
- **Автономный репо:** того же уровня нет — **`dataproc_root`** = **`repo_root`**, профили: **`./profiles/*.yaml`**.

Переопределение: переменная окружения **`TF_BACKEND_DATAPROC_ROOT`** (абсолютный или относительный путь).

Остальные пути (`storage`, `result_store`, …) строятся от `repo_root`, как и раньше.

## Что положить в корень нового репозитория

| Артефакт | Описание |
| -------- | -------- |
| Содержимое `backend/` | Весь каталог из монорепо без префикса `backend/`: `app/`, `tests/`, `alembic/`, `docs/`, `requirements.txt`, `Dockerfile`, compose-файлы … |
| `profiles/*.yaml` | Скопировать из `DataProcessor/profiles/` (см. `profiles/README.md`) — нужны для seed профилей на старте API |
| `.github/workflows/` | Перенести job из `.github/workflows/backend-ci.yml` в корень репо: убрать префикс `backend/` из `paths` и из `working-directory` (пример ниже) |

Не обязательно копировать: весь `DataProcessor/`, корневые скрипты монорепо, фронт.

## Docker / Compose

- В монорепо: `docker-compose.yml` использует контекст профилей `../DataProcessor/profiles` и volume `../storage`.
- После выноса: **`docker-compose.standalone.yml`** — контекст **`./profiles`**, volume **`./storage`**.

```bash
docker compose -f docker-compose.standalone.yml up --build
```

Сборка образа вручную (если не Compose):

```bash
docker build -f Dockerfile --build-context profiles=./profiles .
```

## Пример GitHub Actions для корня отдельного репо

Скопируйте содержимое текущего `backend-ci.yml` и замените фильтры путей и каталог работы, например:

- `on.push.paths` / `pull_request.paths`: убрать префикс `backend/` (или использовать `**` по всему репо).
- `defaults.run.working-directory`: удалить или оставьте `.`.
- Шаги `cd backend` убрать; если был установлен Python с `working-directory: backend`, перенесите команды в корень.

Точное содержимое workflow смотрите в актуальном файле монорепо: `.github/workflows/backend-ci.yml` — шаги **Ruff** (`ruff check app`), **pytest** с **coverage** и публикация артефакта **`coverage-xml`** (`coverage.xml`); при переносе в корень отдельного репо скорректируйте `paths` к артефакту.

## Варианты синхронизации с монорепо

- **Разовая копия** `DataProcessor/profiles/` → `profiles/` перед релизом.
- **git subtree** или **submodule** на каталог профилей — если нужна авто-синхронизация без дублирования коммитов всего монорепо.

Отдельное имя организации/репозитория в этом документе не фиксируется — выбираете сами при создании репо на GitHub/GitLab.
