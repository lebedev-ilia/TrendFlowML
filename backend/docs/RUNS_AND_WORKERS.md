# Runs and workers

## 1) Run lifecycle

`POST /api/runs`:

- проверяет, что видео принадлежит пользователю (`user_video_links`)
- создаёт `runs` со статусом `queued`
- создаёт минимальные `run_components` для UI (`segmenter`, `visual`)
- запускает Celery task `process_run(run_id)`

## 2) Celery task `process_run`

Код: `backend/app/tasks.py`

Основные шаги:

1. Загружает `Run`, `Video`, `VideoSource`, `VideoFile`.
2. Строит профиль (если нет `visual.cfg_path` и `processors`).
3. Переводит run в `running`, ставит `stage="segmenter"`.
4. Стартует tailer `state_events.jsonl` (DataProcessor пишет progress).
5. Запускает `DataProcessor/main.py` через subprocess.
6. Стримит stdout/stderr в `run_logs` и WebSocket события.
7. После завершения:
   - читает `manifest.json`
   - регистрирует артефакты
   - запускает demo quality scripts
   - финализирует run со статусом `succeeded`/`failed`

## 3) DataProcessor запуск

Команда запуска:

- `--video-path <raw_video_path>`
- `--output <frames_dir_base>`
- `--visual-cfg-path <visual_cfg_default>`
- `--profile-path <profiles_cache/run_id/profile.yaml>`
- `--dag-path <DataProcessor/docs/reference/component_graph.yaml>`
- `--dag-stage baseline`
- `--platform-id`, `--video-id`, `--run-id`
- `--sampling-policy-version v1`
- `--dataprocessor-version dev`
- `--rs-base <result_store_base>`

## 4) Cancel semantics (текущая реализация)

`POST /api/runs/{run_id}/cancel`:

- ставит `cancel_requested_at`
- **не останавливает** текущий DataProcessor процесс

Если нужна строгая отмена — это TODO в `GAPS_AND_ALIGNMENT.md`.

## 5) Quality reports

После успешного run backend ищет
`**/quality_report/demo_*_quality.py` и запускает их.

HTML отчёты регистрируются как `artifacts` и доступны через
`GET /api/runs/{run_id}/artifact`.

