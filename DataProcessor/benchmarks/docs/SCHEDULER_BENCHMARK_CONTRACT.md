## Scheduler-aware Benchmarks — контракт для компонентов (DynamicBatch / Benchmark Registry)

Цель: чтобы **scheduler мог перебрать варианты планирования** (параллелизм/батчинг на разных уровнях) и выбрать оптимальный план, опираясь на **воспроизводимые и версионированные** бенчмарки по **времени + памяти** (VRAM/RAM/CPU) для **всех стадий** системы.

Этот документ дополняет и конкретизирует общий контракт registry:
- `DynamicBatch/docs/BENCHMARK_REGISTRY_CONTRACT.md`
- `DynamicBatch/docs/BENCHMARK_REGISTRY_DDL_POSTGRES.sql`

---

## 0) Зафиксированные требования (что scheduler должен уметь)

### 0.1 Что оптимизируем
- **Общее время обработки всех видео (makespan)**.
- **Средняя и p95 latency на видео** (per-video completion time).

### 0.2 Какие ограничения считаем hard
- **VRAM budget**, **RAM budget**, **CPU budget** (ядра/threads).
- Допускается запускать **несколько GPU-heavy компонентов параллельно на одной GPU**, если:
  - бюджет памяти позволяет,
  - и **есть бенчмарки**, покрывающие этот режим (см. interference-бенчи ниже).

### 0.3 Что scheduler НЕ имеет права менять
- Scheduler **не выбирает** `model_branch/model_variant` и вообще не трогает параметры, влияющие на качество.
- Эти параметры задаются конфигом/бекендом и приходят как входные условия планирования.
- Scheduler **может менять только knobs**, влияющие на ресурсы/время, но не на качество (например `batch_size`, `worker counts`, очередность выполнения стадий).

### 0.4 Онлайн-адаптация
- В MVP **не делаем** online replanning; план строится заранее по registry.
- Позже можем добавить (не в этом контракте).

---

## 1) Термины

- **Benchmark entity**: “что измеряем” — `component_id` + `component_part`.
- **Whole**: end-to-end стоимость компонента на единицу работы (`unit`).
- **Substep**: часть компонента (`component_part = substep:<name>`), которая может давать **пики**.
- **Unit**: атом работы для scheduler’а: `frame`, `frame_pair`, `audio_segment`, `prompt`, `video`, ...
- **Knobs**: параметры, которыми scheduler управляет (или перебирает варианты).
- **Input bucket**: bucket входа, чтобы сравнивать сравнимое (fps/resolution/duration/segments_count/...).

---

## 2) Обязательные поля в `results.jsonl` (registry-ready)

Каждая строка измерения (или агрегат) должна содержать **плоский набор ключей** (как в `run_bench.py`) для импорта в Postgres.

### 2.1 Identity (обязательно)
- `component_id` (канонично, стабильно)
- `component_part`:
  - `whole`
  - `substep:<name>` (строгое имя подшага)
- `owner`: `dataprocessor|fetcher|models`
- `stage`: `baseline|v1|v2` (или `null`, если не влияет)
- `unit`: `frame|frame_pair|audio_segment|prompt|video|...`
- `runtime`: `triton|inprocess|...`
- `model_signature` (если есть модель; иначе `null`)
- `model_branch` (если влияет на cost; иначе `null`)
- `device_profile` (JSON; см. §2.3)
- `input_bucket` (JSON; см. §2.4)
- `knobs` (JSON; см. §3)
- `producer_version`, `git_commit`, `git_dirty`
- `artifact_uri` (raw артефакт; в MVP допустим `file://...`, в проде `s3://...`)

### 2.2 Метрики (обязательно)
Минимум для планирования:
- `latency_ms` (сырые сэмплы) **или** агрегаты `latency_ms_mean_stable_per_unit`, `latency_ms_p95`
- `cpu_rss_mb` (текущий RSS) и/или `cpu_rss_peak_mb` (пик за прогон)
- `gpu_mem_mb` (если доступно) и/или `vram_peak_mb`/`vram_delta_mb` (см. ниже)

Примечание:
- Для Triton сейчас в harness обычно есть `gpu_mem_mb` через NVML (если `pynvml` установлен). Для scheduler важнее иметь **delta/peak**, поэтому новые harness’ы должны писать это явно.

### 2.3 `device_profile` (обязательно, JSON)
Минимальный набор ключей:
- `os`, `kernel`
- `cpu_name`, `cpu_cores_logical`
- `ram_total_mb`
- `gpu_name`, `vram_mb`, `driver`

Рекомендуется дополнительно (если доступно):
- `cuda_version`, `cudnn_version`, `torch_version`, `triton_server_version`
- `gpu_uuid` (если несколько GPU)

### 2.4 `input_bucket` (обязательно, JSON)
Должно однозначно описывать “масштаб работы”:
- Для visual/frame-level: `width`, `height`, `fps_bucket` (или `fps`), `frames_bucket`
- Для optical flow: `pair_stride`, `pairs_bucket`
- Для audio: `sr`, `segment_sec`, `segments_bucket`
- Для per-video стадий: `duration_bucket_sec`, `has_audio`, `has_subtitles`, ...

Правило: если `input_bucket` пустой — scheduler не сможет переносить cost между видео корректно.

---

## 3) Knobs / оси, которые обязаны быть покрыты бенчмарками

Мы явно разделяем уровни планирования:

### 3.1 L3: Component/model-level (внутри компонента)
Обязательные оси (где применимо):
- `knobs.model_batch_size` (или `batch_size`) — **управляет только scheduler**
- `knobs.num_workers` / `knobs.intra_component_parallelism` (например параллельные сегменты/тайлы)
- `knobs.prefetch` / `knobs.queue_depth` (если есть очереди)

### 3.2 L2: Processor-level (внутри Audio/Visual/Text Processor)
Обязательные оси (где применимо):
- `knobs.processor_inflight_units` (сколько unit одновременно “в полёте” внутри процессора)
- `knobs.processor_workers` (threads/processes)
- `knobs.segment_parallelism` (для аудио сегментов)

Важно:
- “2 видео параллельно” обычно **не равно** “1 видео ×2” — поэтому такие оси должны иметь бенч-покрытие.

### 3.3 L1: Inter-video concurrency (на уровне DynamicBatch)
Обязательные оси:
- `knobs.max_parallel_videos` (число параллельных видео)

Ожидание: scheduler будет перебирать эти варианты, пока не появится online replanning.

---

## 4) Какие именно бенчмарки нужны (матрица)

Комбинаторика может быть огромной. Поэтому вводим **обязательный минимум** + **расширение**.

### 4.1 Обязательный минимум (MVP) — для каждого GPU/CPU профиля
Для каждого компонента (и его важных substeps):
- **Whole** cost на `unit` для `model_batch_size ∈ {1, 2, 4, 8, 16}` (или релевантное множество)
- **Substeps**: минимум по одному прогону на `batch=1` и на “целевом” batch (например 8) чтобы поймать пики

Для каждого процессора:
- Processor-level “inflight” прогон: `processor_inflight_units ∈ {1, 2, 4, 8, 16}` (или до насыщения)
- CPU/RAM/VRAM фиксируются как метрики (пики!)

### 4.2 Расширение (рекомендуется) — чтобы scheduler видел “почти все варианты”
Добавляем оси:
- `max_parallel_videos ∈ {1..K}` (K зависит от железа)
- `segment_parallelism ∈ {1, 2, 4, 8, 16, 32, 40}` (пример из AudioProcessor)
- `processor_workers ∈ {1, 2, 4, 8}`

### 4.3 Interference / co-scheduling бенчмарки (обязательно для GPU-heavy параллелизма)
Если мы хотим разрешить параллельное выполнение нескольких GPU-heavy задач на одной GPU, **нужны отдельные бенчи**, которые измеряют:
- **память** при одновременном запуске A+B (VRAM peak)
- **время** (slowdown коэффициент относительно одиночного запуска)

Формат:
- `component_id = <processor_or_system_scope>`
- `component_part = substep:co_run::<A>+<B>` (или аналогично)
- `knobs = { "co_run": ["A","B"], "max_parallel_videos": X, ... }`

Без таких бенчей scheduler должен считать это режимом “unsafe” и не выбирать.

---

## 5) Стадии (stages) — что обязано быть документировано

Scheduler планирует на уровне:
- “выше процессоров” (Global orchestrator / DynamicBatch),
- стадий внутри DataProcessor,
- стадий внутри каждого процессора (Visual/Audio/Text) и ключевых компонентов.

Требование:
- Для каждого процессора должен существовать документ “Stage Map”, где перечислены:
  - stage id (стабильный),
  - входные артефакты,
  - выходные артефакты,
  - какие компоненты входят в стадию,
  - где возможен параллелизм и какие knobs его контролируют.

Формат Stage Map будет добавлен отдельным документом (следующий шаг).

---

## 6) Acceptance criteria (что считаем “успехом”)

На одном и том же `device_profile`:
- Scheduler может перечислить варианты и выбрать план.
- При разных симулированных budgets (VRAM/RAM/CPU) **не происходит деградации** (OOM), а scheduler выбирает другие варианты.
- Фактические метрики после прогона **почти не отличаются** от registry:
  - VRAM: запас порядка **~300MB**,
  - RAM: запас порядка **~500MB**,
  - latency: допускается небольшой drift, но без крупных расхождений.

На другом профиле (например Colab: 16GB VRAM + 16GB RAM):
- Scheduler выбирает новые варианты (больше batch_size/параллелизма) и ускоряет обработку.

---

## 8) Runtime report (plan vs fact) — обязательный выход для scheduler

Чтобы scheduler мог сравнивать “план” (registry costs) с “фактом” (реальный прогон) и потом улучшать планирование,
каждый run должен иметь **best-effort runtime report** в `run_rs_path`:

- `run_rs_path/_reports/scheduler_runtime_report.json`
- `schema_version = scheduler_runtime_report_v1`

Минимум полей:
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `created_at`
- `scheduler_knobs` (что было применено)
- `per_processor.*.duration_ms`
- best-effort memory peaks (если доступны в env): `rss_peak_mb`, `gpu_used_peak_mb`
- `per_processor.*.per_extractor/per_component` с:
  - `wall_ms` (замер orchestration)
  - `reported_ms` (если компонент возвращает)
  - `segments_count`/`units_count` (если применимо)
  - `effective_knobs` (фактически применённые knobs; иногда scheduler просит X, но компонент может clamp’ить)

Примечание:
- Этот report **не заменяет** Benchmark Registry (он “про факт”), но является источником для валидации бенчей и дрейфа.

---

## 7) Практическое правило: как добавлять новый benchmark

1) В spec/коде явно укажи:
- `component_id`, `component_part`, `unit`, `runtime`
- `input_bucket`, `knobs`
- `model_signature`, `model_branch` (если применимо)

2) Прогоняй:
- whole + substeps (если есть “тяжёлые куски”)
- минимум по batch сетке

3) Проверь, что `results.jsonl` содержит все поля из §2 (иначе импортер в DB будет либо отбрасывать, либо делать запись “непригодной” для scheduler).


