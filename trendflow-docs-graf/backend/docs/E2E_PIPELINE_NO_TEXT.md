# E2E: Backend → Fetcher → Segmenter → Audio → Visual (без TextProcessor)

Краткий **указатель** на уже существующие команды и файлы для прогона цепочки **без текстового процессора**.

## Основные документы (где искать команды)

| Документ | Содержание |
|----------|------------|
| [`E2E_DP_FIXES_2026-04.md`](E2E_DP_FIXES_2026-04.md) | Фиксы после ребута: таймаут `POST /process`, `wait_for_port`, `config_parser`→Audio CLI, аудио/NPZ |
| [`E2E_RUNBOOK.md`](E2E_RUNBOOK.md) | Полный runbook: сервисы, env, `e2e_run_to_complete.py`, расширенный смок DataProcessor |
| [`E2E_FULL_CHECKLIST.md`](E2E_FULL_CHECKLIST.md) | Чеклист: порты 8001/8000/8002, запуск стека, те же скрипты |
| [`E2E_WORKLOG_2026-03-13.md`](E2E_WORKLOG_2026-03-13.md) | История правок; рекомендованная процедура `start_e2e_stack.sh`, короткий YouTube URL |
| [../../docs/E2E_MANUAL_SETUP_AND_FIXES.md](../../docs/E2E_MANUAL_SETUP_AND_FIXES.md) | Ручная настройка Backend + Fetcher Docker |
| [../../docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md](../../docs/E2E_YOUTUBE_INGESTION_CHECKLIST.md) | Чеклист ingestion по API |
| [../../Fetcher/docs/E2E_FETCHER_DATAPROCESSOR.md](../../Fetcher/docs/E2E_FETCHER_DATAPROCESSOR.md) | Связка Fetcher ↔ DataProcessor |

## Скрипты (репозиторий)

| Скрипт | Назначение |
|--------|------------|
| `backend/scripts/e2e_env.sh` | Экспорт `TF_BACKEND_*`, Fetcher, DataProcessor env для локального E2E |
| `backend/scripts/setup_e2e_infra.sh` | Docker Postgres/Redis/MinIO (если используете) |
| `backend/scripts/start_e2e_stack.sh` | Поднимает Backend API + Celery + Fetcher + DataProcessor (см. `--help`) |
| `backend/scripts/e2e_run_to_complete.py` | **Backend → Fetcher** (+ опция **--with-dataprocessor**) до статуса run |
| `backend/scripts/e2e_dataprocessor_audio_smoke.py` | После готового Fetcher run: **DataProcessor** = segmenter + **audio**; флаг **`--with-visual`** добавляет **Visual** (text выключен) |
| `backend/scripts/e2e_full_max_run.py` | **Полный E2E** с `DataProcessor/configs/global_config.yaml` (audio + **text** из `example/example_text_documents/*.json` + visual): артефакты в **`storage/e2e_full_max/<run_tag>/`**, указатель `storage/e2e_full_max/active_global_config`. По умолчанию **кеш Fetcher не чистится** (удобно при таймаутах YouTube / без API key); **`--cold-ingestion`** — холодный ingest (нужна сеть/прокси). |

Короткое видео для локального прогона без скачивания с YouTube: в `e2e_env.sh` задаётся **`FETCHER_YOUTUBE_MOCK_SAMPLE_VIDEO_DIR`** (каталог с примерами, например `example/example_videos`). Иначе используйте короткий ролик, например в worklog встречается `DYor3e2effY`.

## Рекомендованный порядок (локально)

1. **Инфра и стек** (из корня репо, пути подправьте под свой диск):

   ```bash
   ./backend/scripts/start_e2e_stack.sh --with-infra
   ```

   Либо по шагам как в `E2E_RUNBOOK.md` §1–2.

2. **Получить завершённый Fetcher run** (видео в артефактах):

   ```bash
   cd backend
   source scripts/e2e_env.sh
   .venv/bin/python -u scripts/e2e_run_to_complete.py \
     --source-url "https://www.youtube.com/watch?v=DYor3e2effY" \
     --fetcher-url http://localhost:8000 \
     --verbose
   ```

   Без `--with-dataprocessor` ingestion дойдёт до **`completed`** после Fetcher (легче отлаживать). **`fetcher_run_id`** возьмите из вывода скрипта или из Backend/Fetcher API.

3. **DataProcessor: Segmenter + Audio + Visual, без Text**:

   ```bash
   cd backend
   source scripts/e2e_env.sh
   .venv/bin/python -u scripts/e2e_dataprocessor_audio_smoke.py \
     --fetcher-run-id <FETCHER_RUN_UUID> \
     --with-visual \
     --dataprocessor-url http://localhost:8002
   ```

   - По умолчанию **`--with-visual`** использует `DataProcessor/configs/audit_v3/visual/visual_core_5_only.yaml` (часть провайдеров ожидает **Triton** и GPU‑модели — окружение должно их предоставлять).
   - Другой профиль: **`--visual-cfg-path /abs/.../visual_*.yaml`** (см. `DataProcessor/configs/audit_v3/visual/`).

   Только **segmenter + audio** (как раньше): запуск **без** `--with-visual`.

## Поведение Backend → DataProcessor по умолчанию

В **`dataprocessor_adapter._default_ingestion_profile_config`** для ingestion-ранов по умолчанию включён **только segmenter**; **audio / visual / text** выключены. Поэтому полная цепочка **с Audio и Visual без правки Backend** после Fetcher обычно делается отдельным вызовом **`e2e_dataprocessor_audio_smoke.py`** (или своим `POST /api/v1/process` с нужным `profile_config`).

## Где смотреть результаты

- **Result store** (по умолчанию из `e2e_env.sh`): `storage/result_store/<platform_id>/<video_id>/<dataprocessor_run_id>/`
- Запись в dev: [`DataProcessor/docs/audit_v3/RUN_LOG.md`](../../DataProcessor/docs/audit_v3/RUN_LOG.md)

## Связанный run-log

Прогонки смоков DataProcessor фиксируйте в **`DataProcessor/docs/audit_v3/RUN_LOG.md`**.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
