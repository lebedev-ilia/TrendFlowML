## Benchmark Registry — контракт хранения и использования бенчмарков (DynamicBatch)

Цель: DynamicBatch scheduler должен получать **актуальные, версионированные и воспроизводимые** cost‑оценки для:
- DataProcessor (Segmenter/Visual/Audio/Text),
- Fetcher,
- Models (inference/heads),
на разных устройствах и при разных версиях моделей/алгоритмов.

Ключевая проблема: “одна цифра на компонент” недостаточна. Компоненты должны быть покрыты:
- **whole-component** метрикой (end‑to‑end),
- **substeps** (части компонента), чтобы учитывать “скрытые” пики/циклы и объяснять расхождения.

---

## 1) Двухуровневое хранение (обязательно)

### 1.1 Object storage (S3/MinIO) — raw артефакты
Храним неизменяемые артефакты бенчмарка:
- исходный JSON с сериями измерений,
- логи stdout/stderr,
- (опционально) NPZ/CSV для профилировщиков,
- метаданные запуска (env, commit, config knobs).

Правило:
- raw артефакты **append-only** (не перезаписываем), идентифицируем по `artifact_id`.

### 1.2 DB (Postgres) — нормализованные поля + ссылки
Храним “быстрые для запросов” поля, которые нужны scheduler’у:
- unit-cost per unit,
- memory peaks,
- drift/restart flags,
- hard caps/constraints,
плюс ссылку на raw артефакт: `artifact_uri`.

Правило:
- в DB тоже **append-only**; “активная версия” выбирается по `valid_to is null` или `is_active=true` (см. ниже).

---

## 2) Identity keys (что отличает одну запись cost от другой)

Запись в registry описывает **не абстрактный компонент**, а конкретный runtime‑профиль:

- **component_id**: каноническое имя сущности (см. §3)
- **component_part**: `whole` или `substep:<name>` (см. §3)
- **owner**: `dataprocessor|fetcher|models`
- **stage**: `baseline|v1|v2` (если cost зависит от stage)
- **unit**: `frame|frame_pair|segment|prompt|video|url|…`
- **runtime**: `triton|inprocess|ultralytics|…`
- **model_signature**: строка/хэш из Model System Rules (если есть модель; иначе `null`)
- **model_branch**: ветка/вариант модели (например `224/336/448`, `midas_384`, etc.) — если влияет на cost
- **input_bucket**: bucket входа (resolution/fps/duration/etc.) — JSON
- **knobs**: параметры, влияющие на cost (preset/branch/thresholds) — JSON
- **device_profile**: описание железа + драйверов — JSON
- **producer_version**: версия producer/компонента (например `core_clip:2.0`)
- **git_commit** (+ dirty flag)

Важно:
- **device_profile обязателен**, иначе scheduler не сможет выбирать правильные цифры для другой GPU/CPU.

---

## 3) Что именно бенчмаркуем: component + parts (substeps)

### 3.1 Канонические сущности
Мы вводим понятие **benchmark entity**:

- **Whole component**:
  - пример: `dataprocessor.visual.core_clip.clip_image`
  - пример: `dataprocessor.visual.cut_detection`
  - пример: `fetcher.download`
  - пример: `models.infer_baseline`

- **Component part (substep)**:
  - пример: `dataprocessor.visual.cut_detection.feature_ssim_only`
  - пример: `dataprocessor.visual.cut_detection.feature_farneback_flowmag_only`
  - пример: `dataprocessor.visual.core_object_detections.postprocess_nms`

Правило:
- scheduler использует **whole-component** cost для планирования “сколько единиц”, но для safety‑проверок и объяснимости должен иметь доступ к **substeps** (особенно memory peaks).

### 3.2 Обязательность substeps
Substeps обязательны для компонентов, где:
- есть несколько тяжёлых стадий,
- есть циклы/итерации, способные породить пики памяти,
- есть разные code paths по knobs/preset.

Пример: `cut_detection` (уже реализовано в бенчмарках) — это эталон.

---

## 4) Метрики (минимум для scheduler’а)

### 4.1 Unit-cost метрики
- `latency_ms_mean_stable_per_unit`
- `latency_ms_p95` (желательно)

### 4.2 Memory метрики
- CPU: `cpu_rss_peak_mb`
- GPU (Triton):  
  - `vram_triton_peak_mb`
  - `vram_triton_delta_run_mb` (ключевая для batch sizing)
  - `vram_triton_drift_mb` + `restart_recommended`

### 4.3 Constraints
- `max_batch_size_component` (hard cap)
- `cross_video_batching` (yes/no) + constraints
- `spikes` (bool) + краткое описание

---

## 5) Версионирование и политика обновления

### 5.1 Append-only + active selection
Мы **не обновляем** старые записи. Добавляем новые:
- `valid_from` = now
- `valid_to` = null
- у предыдущей активной версии ставим `valid_to` = now (или `is_active=false`)

### 5.2 Почему так
- воспроизводимость (можно объяснить исторические решения scheduler’а),
- откат,
- аудит.

---

## 6) Минимальная схема таблицы (MVP, Postgres)

Таблица: `benchmark_costs_v1`
DDL (MVP): `DynamicBatch/docs/BENCHMARK_REGISTRY_DDL_POSTGRES.sql`

Минимум полей:
- `id uuid pk`
- `component_id text not null`
- `component_part text not null` (`whole` | `substep:<name>`)
- `owner text not null`
- `stage text null`
- `unit text not null`
- `runtime text not null`
- `model_signature text null`
- `model_branch text null`
- `input_bucket jsonb not null`
- `knobs jsonb not null`
- `device_profile jsonb not null`
- `producer_version text not null`
- `git_commit text not null`
- `git_dirty bool not null`
- `schema_version text not null`
- `metrics jsonb not null`  
  (должно включать ключи из §4)
- `artifact_uri text not null`
- `created_at timestamptz not null`
- `valid_from timestamptz not null`
- `valid_to timestamptz null`

Индексы (MVP):
- `(component_id, runtime, model_signature)`
- `(owner, stage)`
- GIN по `input_bucket`, `device_profile`

---

## 7) Контракт доступа для scheduler’а

Scheduler запрашивает:
- `GET /benchmark-costs?component_id=...&runtime=...&model_signature=...&device_profile=...&input_bucket=...`

MVP реализация (локально):
- `FileCostProvider` читает `DataProcessor/docs/models_docs/resource_costs/*.json` как seed.
Позже:
- `DbCostProvider` читает из Postgres, с кешированием 30–60 секунд.
---

## Навигация

[Module README](../README.md) · [Vault](../../docs/MAIN_INDEX.md)
