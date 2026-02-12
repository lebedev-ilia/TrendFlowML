## Benchmark harness (baseline GPU / Triton)

Цель: прогонять **ветки моделей** (фиксированные формы, без dynamic axes) и собирать:
- latency (warmup + steady-state, p50/p90/p99 по итогам)
- RSS (CPU)
- (опционально) GPU VRAM через NVML (если доступно)

Контракт baseline GPU I/O:
- image models: **UINT8 NHWC** (сырой RGB кадр), preprocess внутри Triton graph/ensemble
- text models: **INT64 tokens**

### Запуск

1) Укажи Triton URL (или экспортируй переменную окружения):

```bash
export TRITON_HTTP_URL="http://triton:8000"
```

2) Запусти бенч:

```bash
"/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/.data_venv/bin/python" -m benchmarks.run_bench \
  --spec benchmarks/specs/baseline_gpu.yaml
```

### Важно для 6GB GPU (local)

Если прогонять **все модели подряд** в одном Triton процессе, ORT CUDA может начать возвращать OOM
для последующих моделей (особенно MiDaS/RAFT/YOLO960) из-за удержания/фрагментации VRAM.

Рекомендация: гонять “baseline_gpu_local” **по группам** и делать restart Triton между группами:
- `benchmarks/specs/baseline_gpu_local_group_clip_places.yaml`
- `benchmarks/specs/baseline_gpu_local_group_midas_raft.yaml`
- `benchmarks/specs/baseline_gpu_local_group_yolo.yaml`

Опции:
- `--out-dir benchmarks/out/<name>` (по умолчанию timestamp)
- `--warmup 5` / `--repeats 30`
- `--filter clip_image_` (прогон только части моделей)
- `--dry-run` (только план прогонов)

### Выход

Пишется:
- `results.jsonl` — одна строка на один прогон
- `summary.json` — агрегаты по (model_variant × batch)

---

## Benchmark Registry (DB) — правила для новых бенчмарков (обязательно)

DynamicBatch scheduler опирается на **удалённый Benchmark Registry** (Postgres + raw artifacts в object storage).
Источник истины по формату:
- `DynamicBatch/docs/BENCHMARK_REGISTRY_CONTRACT.md`
- `DynamicBatch/docs/BENCHMARK_REGISTRY_DDL_POSTGRES.sql`

Дополнение (самое важное для планирования):  
- `DataProcessor/benchmarks/SCHEDULER_BENCHMARK_CONTRACT.md`

### 1) Что именно бенчмаркуем

Мы бенчмаркуем не только “компонент целиком”, но и его части:

- **Whole component**: end-to-end стоимость на unit (scheduler planning)
  - пример: `dataprocessor.visual.core_clip.clip_image`
- **Component substeps**: ключевые части/стадии, которые могут дать пик памяти/времени
  - пример: `dataprocessor.visual.cut_detection.feature_ssim_only`

Это критично: иначе scheduler будет планировать по “средней цифре” и не объяснит реальный пик.

### 2) Каноническое имя (component_id) и substeps

Правило:
- `component_id` должен быть **стабильным** и каноническим (без timestamp/paths).
- substeps фиксируем через поле `component_part`:
  - `whole`
  - `substep:<name>` (например `substep:feature_ssim_only`)

### 3) Обязательные identity поля (в любом экспорте результатов)

Чтобы запись попала в DB и правильно выбиралась scheduler’ом, каждая “единица cost” должна иметь:
- `component_id`
- `component_part` (`whole|substep:<name>`)
- `owner` (`dataprocessor|fetcher|models`)
- `stage` (`baseline|v1|v2`)
- `unit` (`frame|frame_pair|segment|prompt|url|...`)
- `runtime` (`triton|inprocess|...`)
- `model_signature` (если есть модель; иначе `null`)
- `model_branch` (если влияет на cost: 224/336/448, midas_384, raft_256, etc.)
- `device_profile` (GPU/CPU/VRAM/RAM/driver/CUDA/OS)
- `input_bucket` (resolution/fps/duration bucket, минимум для сравнимости)
- `knobs` (preset/flags/thresholds, которые меняют cost)
- `producer_version` + `git_commit` (+ `git_dirty`)

### 4) Обязательные метрики (scheduler-facing)

- `latency_ms_mean_stable_per_unit`
- `latency_ms_p95` (желательно)
- `cpu_rss_peak_mb`
- для Triton (GPU): `vram_triton_peak_mb`, `vram_triton_delta_run_mb`, `vram_triton_drift_mb`, `restart_recommended`

### 5) Raw artifacts (append-only)

Каждый бенчмарковый прогон обязан сохранять raw артефакт (JSON/логи) и давать `artifact_uri`
(в MVP допустим `file://...`, в проде — `s3://...` / `minio://...`).

### 6) Требование к формату `results.jsonl` в harness (рекомендация для совместимости)

Текущий harness пишет `results.jsonl` с полями `variant/batch/latency_ms/...`.
Для совместимости с Benchmark Registry рекомендуем добавлять в каждую строку:
- `component_id`, `component_part`, `owner`, `stage`, `unit`, `runtime`, `model_signature`, `model_branch`
- `device_profile`, `input_bucket`, `knobs`
- `artifact_uri` (или `out_dir` + имя файла как proxy)

Идея: `variant` остаётся “человеческим”, но scheduler и DB работают по каноническим ключам.

---

## Scheduler-aware benchmarks (коротко)

Если компонент/процессор должен быть "управляемым" для scheduler, бенчмарки обязаны покрывать:
- **оси параллелизма и батчинга** (inter-video, processor-level, component-level),
- **CPU/RAM/VRAM** (пики) и latency (mean + p95),
- **interference/co-scheduling** режимы для GPU-heavy параллельного запуска (если мы хотим это разрешать).

Полный контракт и матрица бенчей: `DataProcessor/benchmarks/SCHEDULER_BENCHMARK_CONTRACT.md`.

---

## Component-level benchmarks

Для детального анализа производительности компонентов доступны два скрипта:

### 1. Single-run component benchmark (`run_component_bench.py`)

Запускает компонент один раз и детально измеряет ресурсы на разных этапах:
- Ресурсы до запуска Triton
- Ресурсы после запуска Triton
- Ресурсы во время выполнения компонента (time series)
- Ресурсы после выполнения компонента

**Использование:**
```bash
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --batch-size 1
```

**Документация:** `benchmarks/README_COMPONENT_BENCH.md`

### 2. Multi-threaded component benchmark (`run_component_parallel_bench.py`)

Запускает компонент в нескольких потоках одновременно и измеряет системные ресурсы во время параллельного выполнения:
- Параллельный запуск нескольких экземпляров компонента
- Мониторинг ресурсов системы (CPU, GPU, RAM, VRAM)
- Статистика выполнения (средняя, минимальная, максимальная длительность)
- Анализ пропускной способности и масштабируемости

**Использование:**
```bash
python benchmarks/run_component_parallel_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --threads 4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --batch-size 1
```

**Документация:** `benchmarks/README_PARALLEL_BENCH.md`

**Когда использовать:**
- `run_component_bench.py`: для детального анализа одного запуска, понимания этапов выполнения и пиков ресурсов
- `run_component_parallel_bench.py`: для анализа масштабируемости, параллельной производительности и определения оптимального количества параллельных потоков

---

## Импорт benchmark результатов в Postgres (benchmarks/out → DB)

После прогона harness (например `run_bench.py`) у тебя появляется папка:
- `DataProcessor/benchmarks/out/<run_id>/results.jsonl`

Импорт в Benchmark Registry делается на **Global уровне** (в `.global_venv`), потому что именно там живёт DynamicBatch и DB access.

Dry-run:

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML"
"/home/ilya/Рабочий стол/TrendFlowML/.global_venv/bin/python" DynamicBatch/tools/import_benchmarks_results_to_db.py \
  --db-dsn "postgresql://trendflow:trendflow@localhost:5432/trendflow_bench" \
  --db-table "benchmark_costs_v1" \
  --bench-out-dir "DataProcessor/benchmarks/out/REPLACE_ME" \
  --keep-batches "1,8" \
  --dry-run
```

Реальный импорт: убери `--dry-run`.


