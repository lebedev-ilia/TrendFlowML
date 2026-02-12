## PostgreSQL (local) — установка и запуск для Benchmark Registry + DynamicBatch

Цель: поднять локальный Postgres, создать таблицу `benchmark_costs_v1`, импортировать seed costs из репозитория и запустить `DynamicBatch` в режиме `--costs-provider db`.

Все команды ниже рассчитаны на Ubuntu/Debian (apt).

---

## 1) Установка PostgreSQL

```bash
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
systemctl status postgresql --no-pager
```

Проверка:

```bash
sudo -u postgres psql -c "select version();"
```

---

## 2) Создание пользователя и базы

Открой psql под системным пользователем `postgres`:

```bash
sudo -u postgres psql
```

Внутри psql:

```sql
CREATE USER trendflow WITH PASSWORD 'trendflow';
CREATE DATABASE trendflow_bench OWNER trendflow;
GRANT ALL PRIVILEGES ON DATABASE trendflow_bench TO trendflow;
\\q
```

DSN для подключения:

```text
postgresql://trendflow:trendflow@localhost:5432/trendflow_bench
```

---

## 3) Создание таблиц (DDL)

Выполни DDL:

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML"
psql "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  -f "DynamicBatch/docs/BENCHMARK_REGISTRY_DDL_POSTGRES.sql"
```

Проверка:

```bash
psql "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  -c "\\dt"
```

---

## 4) Python deps для DB режима (с учётом иерархии venv)

DynamicBatch в DB режиме требует `psycopg2` (или `psycopg2-binary`).

В этом репозитории окружения разделены по уровням:
- **Global orchestrator**: `/home/ilya/Рабочий стол/TrendFlowML/.global_venv` (сюда относится DynamicBatch)
- **DataProcessor**: отдельный venv (запускает Segmenter/Visual/Text/Audio orchestration)
- **VisualProcessor components**: `.vp_venv` и/или специальные venv для отдельных компонентов (`core_face_landmarks`)

Правило: **DB доступ (psycopg2) нужен только Global уровню**, т.к. именно DynamicBatch читает costs из Postgres.

Установка в глобальную среду (рекомендуется):

```bash
"/home/ilya/Рабочий стол/TrendFlowML/.global_venv/bin/python" -m pip install -U psycopg2-binary pyyaml
```

Примечание:
- DataProcessor/VisualProcessor не обязаны иметь `psycopg2`, если они не читают Postgres напрямую.
- Если ты всё же запускаешь импортер не из `.global_venv`, тогда `psycopg2-binary` должен быть в том python, которым запускаешь импортер.

```bash
"/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/.data_venv/bin/python" -m pip install -U psycopg2-binary pyyaml
```

---

## 5) Импорт seed costs из репозитория в DB

Импортирует `DataProcessor/docs/models_docs/resource_costs/*.json` в `benchmark_costs_v1` как `component_part='whole'`.

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML"
"/home/ilya/Рабочий стол/TrendFlowML/.global_venv/bin/python" DynamicBatch/tools/import_resource_costs_to_db.py \
  --db-dsn "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  --db-table "benchmark_costs_v1" \
  --resource-costs-dir "DataProcessor/docs/models_docs/resource_costs" \
  --owner dataprocessor \
  --stage baseline \
  --git-commit "unknown"
```

Проверка количества:

```bash
psql "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  -c "select count(*) from benchmark_costs_v1 where valid_to is null;"
```

---

## 6) Импорт свежих benchmark результатов (benchmarks/out → DB)

Seed costs — это “стартовые” оценки. Дальше правильный поток такой:
- прогнал `DataProcessor/benchmarks/run_bench.py` (или другой harness)
- получил `DataProcessor/benchmarks/out/<run_id>/results.jsonl`
- импортировал агрегаты в `benchmark_costs_v1` (append-only + закрытие предыдущих active версий)

Импортёр:
- читает `results.jsonl`
- берёт только `status=ok`
- агрегирует метрики по ключу Registry + `knobs.model_batch_size` (batch)
- по умолчанию импортирует только `batch ∈ {1,8}` (scheduler-facing)

Пример (dry-run):

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML"
"/home/ilya/Рабочий стол/TrendFlowML/.global_venv/bin/python" DynamicBatch/tools/import_benchmarks_results_to_db.py \
  --db-dsn "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  --db-table "benchmark_costs_v1" \
  --bench-out-dir "DataProcessor/benchmarks/out/REPLACE_ME" \
  --keep-batches "1,8" \
  --dry-run
```

Реальный импорт (убери `--dry-run`):

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML"
"/home/ilya/Рабочий стол/TrendFlowML/.global_venv/bin/python" DynamicBatch/tools/import_benchmarks_results_to_db.py \
  --db-dsn "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  --db-table "benchmark_costs_v1" \
  --bench-out-dir "DataProcessor/benchmarks/out/REPLACE_ME" \
  --keep-batches "1,8"
```

Проверка (сколько active записей):

```bash
psql "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  -c "select count(*) as active_rows from benchmark_costs_v1 where valid_to is null;"
```

---

## 7) Запуск DynamicBatch (DB provider)

Dry-run:

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML"
"/home/ilya/Рабочий стол/TrendFlowML/.global_venv/bin/python" DynamicBatch/run_scheduler.py \
  --video-path "DataProcessor/NSumhkOwSg.mp4" \
  --dag-stage baseline \
  --costs-provider db \
  --db-dsn "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  --db-table "benchmark_costs_v1" \
  --dp-python "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/.data_venv/bin/python" \
  --dry-run
```

Реальный запуск (убери `--dry-run`):

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML"
"/home/ilya/Рабочий стол/TrendFlowML/.global_venv/bin/python" DynamicBatch/run_scheduler.py \
  --video-path "DataProcessor/NSumhkOwSg.mp4" \
  --dag-stage baseline \
  --costs-provider db \
  --db-dsn "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  --db-table "benchmark_costs_v1" \
  --dp-python "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/.data_venv/bin/python" \
  --max-parallel 1
```

### Важно про `--dp-python`
- Это **python DataProcessor уровня**, который запускает `DataProcessor/main.py`.
- Дальше VisualProcessor сам будет запускать компоненты в своих venv (`.vp_venv` и спец‑venv), поэтому их менять через DynamicBatch не нужно.


