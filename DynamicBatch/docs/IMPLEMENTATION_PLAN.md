## DynamicBatch — implementation plan (MVP → v1)

Основание (source-of-truth):
- `DynamicBatch/docs/DynamicBatching_Q_A.md`
- `DynamicBatch/docs/DYNAMIC_BATCHING_CHECKLIST.md`
- `DynamicBatch/docs/BENCHMARK_REGISTRY_CONTRACT.md`
- `DataProcessor/docs/contracts/*` (контракты артефактов, segmenter, модельная система)
- `DataProcessor/main.py` и `DataProcessor/VisualProcessor/main.py` (текущий runtime запуск, state-files)

---

## 1) Цели MVP (что “полностью реализовать” в первой итерации)

### 1.1 What MVP must do
- **Batch level 1**: принимать список видео (локальные файлы в MVP), формировать пул задач и запускать анализ максимально параллельно без OOM.
- **Resource-aware**: перед запуском задач опрашивать систему (RAM/VRAM) и выбирать:
  - сколько задач можно запустить параллельно,
  - какие `batch_size` проставить в runtime configs (scheduler-controlled) для GPU компонентов.
- **OOM feedback-loop**: если run/компонент ловит OOM → retry до 3 раз с уменьшением `batch_size` (последняя попытка = 1), backoff=2s; если на `1` OOM → fail-fast.
- **State level 1**: писать `state_level1_global.json` + journal `state_level1_events.jsonl` (durable очередь/история событий).
- **Idempotency**: если `run_id/config_hash` совпадает и есть успешный run → пропуск (MVP: best-effort).
- **Seed costs provider**: в MVP scheduler читает seed costs из `DataProcessor/docs/models_docs/resource_costs/*.json`,
  но дизайн должен быть совместим с будущим `DbCostProvider` (см. `BENCHMARK_REGISTRY_CONTRACT.md`).

### 1.2 What MVP will NOT do (сразу фиксируем)
- Авто-управление жизненным циклом `tritonserver` (start/stop/relocate) — в MVP **assume already running**.
- Поддержка URL ingestion (Fetcher) — отдельным шагом после локального batch runner.

---

## 2) Контракт интеграции (между DynamicBatch и DataProcessor)

### 2.1 Как DynamicBatch запускает DataProcessor
- Запуск через `python DataProcessor/main.py ...`.
- DynamicBatch генерирует **runtime VisualProcessor config YAML** и передаёт в DataProcessor через `--visual-cfg-path`.
  - Это работает, потому что `DataProcessor/main.py` читает `--visual-cfg-path`, затем передаёт config в `VisualProcessor/main.py`,
    а `VisualProcessor/main.py` прокидывает примитивные поля YAML в компонентные CLI args (включая `--batch-size`).

### 2.2 Scheduler-controlled batch_size (MVP rule)
Используем правило из Q&A:
- `free_vram_mb = gpu_total_mb - gpu_used_mb`
- `headroom_mb = max(1024, round(free_vram_mb * 0.25))`
- `effective_budget_mb = max(0, free_vram_mb - headroom_mb)`
- `batch_size = clamp(floor(effective_budget_mb / gpu_memory_per_task_mb), 1, max_batch_size_component)`

Где `gpu_memory_per_task_mb` берём из `resource_costs/*.json` как:
- `gpu_memory_per_task_mb = max(1, vram_triton_delta_run_mb)` (если есть),
- иначе fallback (MVP): `gpu_memory_per_task_mb = 64`.

### 2.3 OOM detection (MVP)
OOM считается, если:
- subprocess завершился non-zero, и в stderr/stdout есть подстроки `CUDA out of memory`, `CUDART`, `OOM`, `out of memory`.
План: строгий парсер + расширяемый список паттернов.

---

## 3) Структура кода (папка `DynamicBatch/`)

### 3.1 Python package
```
DynamicBatch/
  dynamicbatch/
    __init__.py
    cli.py                  # entrypoints
    plan.py                 # plan data-structures + planner
    resource_costs.py       # load/normalize resource_costs json
    runtime_cfg.py          # build VisualProcessor runtime config overrides
    system_probe.py         # RAM/VRAM probe (nvidia-smi + /proc)
    subprocess_runner.py    # run DataProcessor jobs, capture logs, OOM detection
    state_level1.py         # state_level1_global.json + state_level1_events.jsonl
    types.py                # TypedDict/dataclasses
  run_scheduler.py          # thin wrapper: python -m DynamicBatch.run_scheduler ...
  docs/
```

### 3.2 CLI (MVP)
- `DynamicBatch/run_scheduler.py`
  - `--video-path` (repeatable) или `--input-json` (list of local paths)
  - `--dp-root` (default: `../DataProcessor`)
  - `--rs-base` (куда класть runs/result_store)
  - `--visual-cfg-template` (base cfg, default: `DataProcessor/VisualProcessor/config.yaml`)
  - `--dag-stage baseline|v1|v2`
  - `--max-parallel` (hard cap)
  - `--dry-run`

---

## 4) План реализации по шагам (task breakdown)

### Step A — Docs & links (done)
- Перенос Q&A + checklist в `DynamicBatch/docs/`
- Redirect stubs в `DataProcessor/docs/models_docs/`
- Обновление `site` для чтения `DynamicBatch/docs/*`

### Step B — MVP core (код)
- `system_probe.py`: RAM/VRAM (nvidia-smi, /proc/meminfo)
- `resource_costs.py`: загрузка и нормализация `DataProcessor/docs/models_docs/resource_costs/*.json`
  - должно поддерживать **component parts/substeps** как отдельные entities (для объяснимости и safety)
- `runtime_cfg.py`: генерация runtime Visual cfg (переопределение `batch_size` для core providers и модулей)
- `subprocess_runner.py`: запуск `DataProcessor/main.py`, сбор stdout/stderr, OOM parsing
- `plan.py`: simple planner (queue → start jobs respecting `max_parallel` + VRAM gate)
- `state_level1.py`: durable events + current snapshot

### Step C — OOM loop + adaptive batch_size
- При OOM: уменьшаем batch_size (например делим на 2, min=1), ждём 2s, retry (max 3).
- Для следующего запуска/видео сохраняем “pessimistic override” (чтобы не OOM’иться снова).

### Step D — Smoke run
- `--dry-run` должен печатать план (jobs, batch sizes).
- `--video-path DataProcessor/NSumhkOwSg.mp4` должен стартовать DataProcessor с валидным runtime config.

---

## 5) Следующий шаг после MVP (DB-backed registry)

После того как seed‑вариант стабилен:
- добавить `DbCostProvider` (Postgres) + кеширование
- raw artifacts хранить в object storage (MinIO/S3) и ссылаться из DB (`artifact_uri`)
- включить обязательные поля: `device_profile`, `producer_version`, `model_signature`, `git_commit`
- добавить “active selection” (valid_from/valid_to), append-only


