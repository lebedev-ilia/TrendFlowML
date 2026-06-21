# E2E example-suite-7: стабильность, VRAM, Embedding Service

Документ описывает полный прогон семи офлайн-видео (`e2e_full_max_run.py --example-suite-7`), типичные сбои (OOM, `model_load_failed`, Embedding :8005) и порядок действий.

## Быстрый старт

Из корня репозитория:

```bash
./backend/scripts/e2e_suite7_full_cycle.sh
```

Эквивалент вручную (как раньше):

```bash
source ./backend/scripts/e2e_env.sh && source ./backend/.venv/bin/activate && cd backend \
  && ./scripts/stop_e2e_stack.sh \
  && ./scripts/start_e2e_stack.sh --with-infra \
  && python -u scripts/e2e_full_max_run.py --example-suite-7 --with-triton-docker --example-suite-force-all
```

`--with-infra` поднимает Docker (Postgres, Redis, MinIO) и через `setup_e2e_infra.sh` создаёт БД `embeddings`, миграции и **ставит зависимости Embedding Service** в `DataProcessor/.data_venv` (faiss-cpu, insightface, …).

## Что делается без смены `device` (по умолчанию)

1. **`e2e_env.sh`**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_MODULE_LOADING=LAZY` — меньше фрагментации VRAM.
2. **Патч `global_config_e2e.yaml`**: снижены батчи аудио — CLAP до ≤8, `source_separation.batch_size=1`, emotion batch ≤2.
3. **`DataProcessor/main.py`**: после завершения subprocess **AudioProcessor** и **TextProcessor** в родительском процессе вызывается `torch.cuda.empty_cache()` (+ `ipc_collect` при наличии), чтобы освободить кэш аллокатора до следующей стадии.
4. **Между роликами сьюта** (если не передан `--no-e2e-host-scrub`): скрипт `scripts/e2e_host_memory_scrub.sh` — sync, опционально `drop_caches`, снимок `nvidia-smi`, `gc` + `empty_cache` в `.data_venv`.

Ограничение scrub: **память GPU, занятая Triton и долгоживущими воркерами**, из пользовательского скрипта не обнуляется; для «абсолютно чистой» карты нужен перезапуск стека (`stop_e2e_stack.sh` / полный цикл).

## Когда включать `--e2e-low-vram`

Если на карте ~6 GiB после оптимизаций всё ещё:

- `source_separation_extractor` — CUDA OOM;
- `TitleEmbedder` — `model_load_failed` (часто маскирует OOM при загрузке e5-large на CUDA),

запускайте:

```bash
python -u scripts/e2e_full_max_run.py --example-suite-7 --with-triton-docker --example-suite-force-all --e2e-low-vram
```

Тогда в патче YAML: `source_separation.device=cpu` и текстовые экстракторы с `device: cuda` переводятся на CPU.

## Embedding Service (:8005)

- Конфигурация: `e2e_env.sh` (`EMBEDDING_SERVICE_URL`, Postgres `embeddings`, `FAISS_INDEX_PATH`, `STORAGE_LOCAL_PATH`).
- Старт: `start_e2e_stack.sh` поднимает процесс, если в `.data_venv` импортируются `faiss` и `embedding_service`.
- Установка: шаг **`[4.5/6]`** в `setup_e2e_infra.sh` — `pip install -r DataProcessor/embedding_service/requirements-e2e.txt`.

Модули **content_domain** / semantic identity ходят в этот сервис; если сервис не поднят, в логах VisualProcessor возможны ошибки подключения и деградация (или долгие таймауты — отсюда «странные» минуты на `content_domain`).

## Почему `content_domain` / `core_depth_midas` могут занимать много минут

- **content_domain**: CLIP-подобная головка по кадрам + обращения к embedding/BД; холодный старт insightface/faiss, первичное наполнение индексов, сетевые таймауты к :8005 удлиняют стадию.
- **core_depth_midas**: inference через **Triton**; первая загрузка модели в GPU, очередь при конкурирующей нагрузке.

Строка **«DP updated N ago»** в `e2e_run_to_complete` отражает `run_state.updated_at`; пока живёт дочерний **VisualProcessor**, поле может долго не меняться — это не обязательно зависание (см. комментарий в `e2e_run_to_complete.py`).

## Остановка процессов между прогонами

- `./scripts/stop_e2e_stack.sh` — останавливает backend, fetcher, dataprocessor, **embedding-service**, пишет PID из `backend/.e2e/pids/`.
- Полный цикл `e2e_suite7_full_cycle.sh` всегда делает **stop** перед **start**.

## Жёсткий сброс page cache (опционально)

Только если нужен «честный» объём MemAvailable между роликами (нужны права root):

```bash
sudo E2E_SCRUB_DROP_CACHES=1 ./backend/scripts/e2e_host_memory_scrub.sh
```

## См. также

- `docs/E2E_FULL_CHECKLIST.md` (если есть в репозитории) — общий чеклист E2E.
- `backend/scripts/e2e_env.sh` — все переменные окружения для стека.
---

## Навигация

[README](README.md) · [Module README](../README.md) · [Backend](MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
