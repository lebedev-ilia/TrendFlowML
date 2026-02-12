# DynamicBatch (global scheduler / orchestrator)

Цель: внешний scheduler, который **выше DataProcessor** и управляет:
- распределением задач по CPU/GPU,
- уровневым batching (Level‑1 batch для DataProcessor и других подсистем),
- выбором `batch_size` на основе **текущих ресурсов системы** + `resource_costs_*`,
- retry loop при OOM (уменьшение batch_size, backoff),
- прогрессом через state-files (Level‑1 global state).

Документация и контракты переехали в `DynamicBatch/docs/`.

## Quickstart (MVP)

Запуск на локальном видео (dry-run, без выполнения):

```bash
python3 DynamicBatch/run_scheduler.py \
  --video-path "DataProcessor/NSumhkOwSg.mp4" \
  --dag-stage baseline \
  --dry-run
```

Реальный запуск (нужно, чтобы окружение DataProcessor было готово + Triton поднят, если профиль triton):

```bash
python3 DynamicBatch/run_scheduler.py \
  --video-path "DataProcessor/NSumhkOwSg.mp4" \
  --dag-stage baseline \
  --dp-models-root "DataProcessor/dp_models/bundled_models" \
  --max-parallel 1
```

## Benchmark costs providers

По умолчанию scheduler читает seed‑цифры из:
- `DataProcessor/docs/models_docs/resource_costs/*.json` (**file provider**)

Подключение Postgres registry (**db provider**):

```bash
python3 DynamicBatch/run_scheduler.py \
  --video-path "DataProcessor/NSumhkOwSg.mp4" \
  --costs-provider db \
  --db-dsn "postgresql://user:pass@host:5432/dbname" \
  --db-table "benchmark_costs_v1" \
  --dry-run
```

DDL: `DynamicBatch/docs/BENCHMARK_REGISTRY_DDL_POSTGRES.sql`

Postgres setup (local): `DynamicBatch/docs/POSTGRESQL_LOCAL_SETUP.md`



