# VisualProcessor: Production Readiness & Scale Plan (100k видео)

## Контекст и цель

Цель этого документа — зафиксировать **production‑чеклист** и **план доводки** VisualProcessor до режима массовой обработки (порядка **100k видео**) с предсказуемым throughput/стоимостью, воспроизводимым окружением, наблюдаемостью и quality gates на артефактах.

Определение “готов к production/100k”:

- **Воспроизводимость**: пайплайн разворачивается без ручных правок путей/venv и без “магии локальной машины”.
- **Надёжность**: внешние зависимости (особенно Triton) не превращают прогон в лотерею; есть деградация/ретраи/таймауты.
- **Идемпотентность**: повторный запуск не удваивает стоимость и не портит результаты.
- **Качество вывода**: артефакты валидны не только по schema, но и по базовым sanity/диапазонам; expected‑skips формализованы.
- **Эксплуатация**: метрики/логи/алерты позволяют быстро понять “что сломалось и где”.

---

## Рекомендуемая целевая инфраструктура (решение принято здесь)

Ниже — **практичный** вариант для production (масштабируемость + контроль стоимости), который хорошо ложится на текущую архитектуру с Triton и результатами в `dp_results`.

### Оркестрация: Kubernetes (рекомендуется) + минимальный fallback на VM

- **K8s** как основной оркестратор:
  - авто‑масштабирование воркеров,
  - изоляция ресурсов,
  - удобные rollout/rollback,
  - стандартизованные healthchecks и лимиты.
- **VM fallback** (если K8s недоступен): тот же набор контейнеров через docker compose/systemd, но с худшей масштабируемостью.

### Triton inference

- **Отдельный Deployment/StatefulSet Triton**, доступный только внутри кластера.
- **Model repository**:
  - минимально: PVC (ReadOnlyMany если возможно) + подготовка моделей job’ом,
  - лучше: object storage (S3/MinIO) + sidecar/инициализатор, который синхронизирует repo на локальный NVMe.
- **Сеть**: gRPC/HTTP endpoint Triton доступен воркерам по внутреннему Service.
- **Policy**: no‑network для моделей — веса поставляются как артефакты сборки/деплоя.

### Очередь задач (выбор)

В репо уже есть Redis и Celery (backend/fetcher), поэтому рациональный компромисс:

- **MVP/первые 100k**: **Celery + Redis** как task queue для DataProcessor задач (простая эксплуатация, быстро внедрить).
- **Ingest/streaming**: если видеопоток большой/неравномерный — **Kafka** для событий/загрузки (у Fetcher уже есть Kafka‑код), а в обработку конвертировать в Celery‑таски (или прямо Kafka‑consumer воркеры, если захотите уйти от Celery).

Практический совет:

- Если цель “100k видео за неделю/несколько дней” — Celery+Redis достаточно при корректных лимитах concurrency и backpressure.
- Если цель “100k/день и выше” — лучше уходить в Kafka‑ориентированную обработку и более строгий контроль партиций/consumer groups.

### Хранилище результатов

- **Локальный FS** приемлем только для dev/малых прогонов.
- Для production/100k:
  - `dp_results` как **storage backend** в **S3/MinIO** (артефакты, логи, render),
  - **Postgres** (или другой DB) как индекс: `video_id`, `run_id`, статусы, версии, ссылки на артефакты.
- Retention: сырые тяжёлые промежуточные артефакты — коротко, агрегаты/model‑facing — дольше (см. `DataProcessor/docs/contracts/PRIVACY_AND_RETENTION.md`).

### GPU/CPU пулы и базовый sizing (пример)

Рекомендуемая схема пулов:

- **gpu-triton**: GPU‑ноды только под Triton.
- **cpu-workers**: CPU‑ноды под orchestration и CPU‑only модули (их много).
- **gpu-workers (опционально)**: для компонентов, которые реально исполняются “in‑process” на GPU и не вынесены в Triton (если такие остаются).

Sizing нельзя “угадать” точно без 2–3 бенчмарков, но для планирования можно использовать модель:

1. Оцените среднее/percentile GPU‑время на видео: \(t_{gpu}\) минут/видео (с учётом Triton).
2. Тогда GPU‑часы на 100k: \(100000 \times t_{gpu} / 60\).
3. Кол-во GPU для срока \(D\) дней: \(\text{GPU} \approx \frac{100000 \times t_{gpu}}{60 \times 24 \times D}\).

Стартовый “разумный” baseline для **100k/неделя** (без гарантии, а как отправная точка):

- **Triton**: 8× GPU (например, L4/A10), включить динамический batching.
- **CPU workers**: 200–400 vCPU суммарно (многие модули CPU‑only).
- **gpu-workers**: 0–4× GPU (только если остаются in‑process GPU стадии).

---

## План доводки (чеклист) — по приоритетам

Ниже чеклист упорядочен так, чтобы каждый шаг давал измеримый эффект и снижал риск массовых прогонов.

### Пункт 1. Воспроизводимый runtime и переносимость (P0)

**Задачи:**

- Убрать хардкод абсолютных путей и локального venv из скриптов тестов/прогонов.
- Везде поддержать env overrides (`BASE_DIR`, `PYTHON`, `RESULTS_DIR`, `VIDEOS_DIR`, …).
- Зафиксировать “как запускать в контейнере” (пример команд и переменных).

**Definition of Done:**

- Скрипты `modules/*/scripts/run_tests.sh`, `wait_and_analyze.sh`, `DataProcessor/scripts/run_missing_visual_tests.sh` запускаются из любого пути, на любой машине/в контейнере.

**Пример запуска (контейнер/CI):**

```bash
export BASE_DIR=/workspace
export PYTHON=python3
export RESULTS_DIR=/workspace/DataProcessor/dp_results
export VIDEOS_DIR=/workspace/example/example_videos

bash DataProcessor/VisualProcessor/modules/action_recognition/scripts/run_tests.sh
```

### Пункт 2. Управление Triton‑зависимостями (P0)

- Preflight healthcheck (Triton reachable + модели готовы).
- Таймауты на infer/видео/стадию.
- Ретраи только на infra‑ошибки (с backoff + лимитом).
- Деградация: “skip Triton‑dependent stages” с явным статусом или fail‑fast батча (выбирается политикой).

**Текущее внедрение (минимум):**

- `DataProcessor/scripts/preflight_triton.py` — утилита проверки `GET /v2/health/ready` (+ опционально модели через `/v2/models/<name>/ready`), с retry/backoff.
  - Параметры: `--attempts`, `--timeout-sec`, `--models-preset` (например `core_low`), `--models` (ручной список).
- `DataProcessor/scripts/run_missing_visual_tests.sh` запускает preflight по умолчанию (fail‑fast).
  - Отключение: `SKIP_TRITON_PREFLIGHT=1`
  - Настройка: `TRITON_PREFLIGHT_TIMEOUT_SEC`, `TRITON_PREFLIGHT_ATTEMPTS`, `TRITON_MODELS_PRESET` (default `core_low`), `TRITON_MODELS` (доп. модели через запятую).

### Пункт 3. Идемпотентность + atomic writes + resume (P0)

- Детерминированный `run_id` или строгая политика его задания.
- Писать в temp и атомарно commit.
- Возможность докрутить упавший модуль без пересчёта всего.

См. также: `DataProcessor/docs/IDEMPOTENCY_REQUIREMENTS.md`.

**Текущее внедрение (частично):**

- `DataProcessor/VisualProcessor/utils/results_store.py` и `DataProcessor/VisualProcessor/utils/manifest.py` используют атомарную запись через `os.replace(...)`.
- `DataProcessor/main.py`:
  - создаёт `manifest.json` рано с `run.status="running"` и `root_path`,
  - по успешному завершению прогона выставляет `run.status="success"` (best-effort).
- `DataProcessor/VisualProcessor/utils/manifest.py` нормализует пути артефактов к **run‑local relative** (например `core_clip/embeddings.npz`), чтобы:
  - результаты были переносимы (FS ↔ S3),
  - idempotency/exists‑checks работали предсказуемо.
- `DataProcessor/api/services/idempotency.py` принимает статусы компонентов как `success|ok|empty|skipped` и корректно префиксует пути артефактов.

**Рекомендация по run_id (production):**

- Для at‑least‑once очереди используйте детерминированный `run_id = hash(video_id + config_hash + processor_versions)` или внешний idempotency key.
- Для ручных прогонов/отладки допускайте произвольный `run_id`, но тогда cache hit rate будет ниже.

### Пункт 4. Quality gates для артефактов (P0/P1)

- Единый интерфейс валидаторов и единая команда “run → validate”.
- Expected‑skips формализованы (no_faces / insufficient_data_for_PCA и т.п.) как “зелёные” статусы, а не падения.
- Семантические sanity checks (NaN rate, диапазоны, монотонность времени, пустые массивы).

**Текущее внедрение (минимум):**

- `DataProcessor/scripts/validate_run_manifest.py` — quality gate по `manifest.json`:
  - проверяет, что все компоненты `status in {ok, empty}`,
  - для `ok` требует артефакты и валидирует `.npz` через `VisualProcessor/utils/artifact_validator.py` (schema + базовые sanity),
  - для `empty` (expected‑skip) допускает отсутствие артефактов и (в strict режиме) требует `empty_reason`.

**Пример запуска (рекомендуется через VP venv):**

```bash
DataProcessor/VisualProcessor/.vp_venv/bin/python DataProcessor/scripts/validate_run_manifest.py \
  --manifest DataProcessor/dp_results/youtube/<video_id>/<run_id>/manifest.json \
  --require-known-schema \
  --strict-empty-reason
```

### Пункт 5. Observability (P1)

- Метрики latency per stage/module, success/skip/infra_fail ratios, retry counts.
- Метрики Triton (очереди, latency, batch sizes), GPU util/mem.
- Логи структурированные по `video_id/run_id/component`.
- Алерты по росту infra_fail и по drift ключевых фич.

**Текущее внедрение (минимум):**

- API:
  - `DataProcessor/api/endpoints/health.py` проверяет Storage/Redis/Triton и выдаёт `healthy|degraded|unhealthy`.
  - `DataProcessor/api/endpoints/metrics.py` (Prometheus) + метрики в `DataProcessor/api/services/metrics.py`.
- Артефакты:
  - `manifest.json` в каждом run: статусы/тайминги/ошибки/артефакты по компонентам.
  - `DataProcessor/scripts/summarize_run_manifest.py` пишет компактный отчёт:
    - `<run_dir>/_reports/run_manifest_summary.json` (schema `run_manifest_summary_v1`).
  - `DataProcessor/main.py` запускает этот summary best‑effort в конце run’а (не влияет на статус).

### Пункт 6. Performance / Capacity model (P1)

- Нагрузочный прогон на representative наборе: короткие/средние/длинные, с лицами/без, много людей/мало.
- Реальный throughput (видео/час/GPU, CPU‑мин/видео), оценка стоимости.
- Оптимизация: кеширование общих core‑артефактов, batching, early exit.

**Текущее внедрение (минимум):**

- `DataProcessor/scripts/capacity_report.py` — сканирует `dp_results`, читает `manifest.json`, считает распределения таймингов:
  - total/cpu/cuda p50/p90/p95/p99,
  - per-component тайминги и статусы,
  - оценка `gpu_hours_total` по сумме `cuda_ms` (очень грубо, как baseline).

**Пример запуска:**

```bash
python3 DataProcessor/scripts/capacity_report.py \
  --results-root DataProcessor/dp_results \
  --platform-id youtube \
  --max-runs 500 \
  --gpus 8
```

**Как читать результат:**

- `total p50/p95` — общая “стоимость” видео в миллисекундах (по сумме `duration_ms` компонентов из manifest).
- `cuda p50/p95` — нижняя оценка GPU‑нагрузки (по компонентам с `device_used=cuda`).
- `cpu p50/p95` — CPU‑нагрузка (остальные компоненты).
- `gpu_hours_total` — сколько GPU‑часов ушло на набор run’ов в отчёте (полезно для экстраполяции на 100k).

### Пункт 7. Storage backend для `dp_results` (P1/P2)

- S3/MinIO backend, индексация run’ов в DB.
- Политика ретеншена и архивации.

**Текущее внедрение (минимум):**

- API уже поддерживает Storage backend:
  - `DataProcessor/storage/fs.py` (`FileSystemStorage`)
  - `DataProcessor/storage/s3.py` (`S3Storage`, совместим с MinIO)
  - layout ключей: `DataProcessor/storage/paths.py` (`KeyLayout`, всё под `result_store/<platform>/<video>/<run>/...`)
- Миграция/синк локальных результатов:
  - `DataProcessor/scripts/sync_dp_results_to_storage.py`
    - берёт локальный run_dir (папка с `manifest.json`) или сканирует `--results-root`,
    - заливает файлы в Storage под `result_store/<platform>/<video>/<run>/...`,
    - пишет per-run index object в Storage: `result_store/_indexes/runs/<platform>/<video>/<run>.json` (S3‑friendly),
    - опционально пишет локальный `--index-jsonl` для последующего импорта в DB.

**Как включить S3/MinIO (env):**

- `TREND_STORAGE_BACKEND=s3`
- `S3_ENDPOINT=http://minio:9000`
- `S3_BUCKET=trendflow`
- `S3_PREFIX=trendflowml`
- `AWS_ACCESS_KEY_ID=...`
- `AWS_SECRET_ACCESS_KEY=...`
- `AWS_DEFAULT_REGION=us-east-1`

**Пример миграции 1 run:**

```bash
TREND_STORAGE_BACKEND=s3 S3_ENDPOINT=http://minio:9000 S3_BUCKET=trendflow S3_PREFIX=trendflowml \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
python3 DataProcessor/scripts/sync_dp_results_to_storage.py \
  --run-dir DataProcessor/dp_results/youtube/test_color_light_2/test_color_light_2 \
  --skip-existing
```

**Пример миграции батча (скан):**

```bash
python3 DataProcessor/scripts/sync_dp_results_to_storage.py \
  --results-root DataProcessor/dp_results \
  --max-runs 1000 \
  --skip-existing \
  --index-jsonl DataProcessor/dp_results/_reports/runs_index.jsonl
```

**Индекс в production:**

- Минимум без DB: использовать `result_store/_indexes/runs/...` (листануть префикс и получить список run’ов).
- Рекомендуется: Postgres таблица `runs` (platform_id, video_id, run_id, config_hash, status, created_at, storage_prefix, metrics jsonb).

### Пункт 8. CI: smoke + nightly full (P1)

- Быстрый smoke на 1–2 видео и минимальном наборе компонент.
- Ночной full прогон на 20 видео (GPU runner + Triton).

**Текущее внедрение (MVP):**

- GitHub Actions:
  - `.github/workflows/visualprocessor-smoke.yml` — быстрые проверки без GPU:
    - `py_compile` для `DataProcessor/scripts/*`,
    - валидация JSON схем `DataProcessor/VisualProcessor/schemas/*.json`,
    - `bash -n` для ключевых shell‑скриптов.
  - `.github/workflows/visualprocessor-nightly.yml` — nightly workflow для **self‑hosted GPU runner** (требуется Triton).

**Требуемые переменные/секреты для nightly:**

- `TRITON_HTTP_URL` или `TRITON_ENDPOINT` (например `http://triton:8000`)
- (опционально) настройка runner так, чтобы `DataProcessor/VisualProcessor/.vp_venv/bin/python` существовал

---

## Текущее состояние (по тестам модулей)

Сводка по 17 модулям и прогонам на 20 видео фиксируется в:

- `DataProcessor/VisualProcessor/modules/action_recognition/docs/ALL_MODULES_TESTING_STATUS.md`

Ключевой риск production‑готовности на масштабе — **инфраструктурная надёжность** (Triton, GPU, IO) и формализация expected‑skips.
---

## Навигация

[Module README](../README.md) · [VisualProcessor](MAIN_INDEX.md) · [DataProcessor](../../docs/MAIN_INDEX.md) · [Vault](../../../docs/MAIN_INDEX.md)
