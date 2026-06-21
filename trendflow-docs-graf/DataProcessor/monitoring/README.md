# Мониторинг и Observability для DataProcessor API

Этот каталог содержит конфигурацию для мониторинга DataProcessor API:
- **Prometheus** - сбор метрик
- **Grafana** - визуализация метрик
- **Jaeger** - distributed tracing

## Структура

```
monitoring/
├── prometheus/
│   ├── prometheus.yml      # Конфигурация Prometheus
│   └── alerts.yml          # Правила алертов
├── grafana/
│   ├── dashboards/
│   │   └── dataprocessor-overview.json  # Дашборд Grafana
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml            # Автозагрузка дашбордов
│       └── datasources/
│           └── prometheus.yml           # Автоподключение Prometheus
└── README.md
```

## Запуск мониторинга

### Через docker-compose

Мониторинг уже добавлен в `docker-compose.yml`:

```bash
docker-compose up -d prometheus grafana jaeger
```

### Доступ к сервисам

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Jaeger UI**: http://localhost:16686

## Prometheus

### Конфигурация

Файл `prometheus/prometheus.yml` настраивает:
- Scrape интервал: 10 секунд для API, 15 секунд для других
- Targets: dataprocessor-api, dataprocessor-worker
- Alert rules: загружаются из `alerts.yml`

### Алерты

Файл `prometheus/alerts.yml` содержит правила:

1. **DataProcessorQueueLengthHigh**
   - Условие: `sum(dataprocessor_queue_length) > 100`
   - Длительность: 5 минут
   - Severity: warning

2. **DataProcessorCrashedRunsHigh**
   - Условие: `crashed_runs / total_runs > 0.1`
   - Длительность: 5 минут
   - Severity: critical

3. **DataProcessorProcessingTimeHigh**
   - Условие: `current_time > baseline * 2`
   - Длительность: 10 минут
   - Severity: warning

4. **DataProcessorMemoryUsageHigh**
   - Условие: `sum(memory_bytes) > 8GB`
   - Длительность: 5 минут
   - Severity: warning

5. **DataProcessorActiveRunsHigh**
   - Условие: `active_runs > 30`
   - Длительность: 5 минут
   - Severity: warning

6. **DataProcessorFailureRateHigh**
   - Условие: `rate(failures_total[5m]) > 0.1`
   - Длительность: 5 минут
   - Severity: critical

## Grafana

### Дашборды

**DataProcessor API - Overview** (`dataprocessor-overview.json`):
- Queue Length по приоритетам
- Active Runs (stat panel с цветовой индикацией)
- Crashed Runs (total)
- Processing Time (95th percentile)
- Failure Rate по процессорам/компонентам
- Memory Usage по run_id
- Queue Wait Time (95th percentile)

### Provisioning

Grafana автоматически:
- Подключает Prometheus как datasource
- Загружает дашборды из `dashboards/`

## Jaeger

### Использование

1. **Включить tracing** в DataProcessor API:
   ```bash
   export ENABLE_TRACING=true
   export TRACING_EXPORTER=jaeger
   export JAEGER_AGENT_HOST=jaeger
   export JAEGER_AGENT_PORT=6831
   ```

2. **Просмотр трейсов**:
   - Открыть Jaeger UI: http://localhost:16686
   - Выбрать сервис: `dataprocessor-api`
   - Найти трейсы по run_id или другим атрибутам

### Экспортеры

Поддерживаются два экспортера:
- **Jaeger** (UDP agent) - по умолчанию
- **OTLP** (gRPC/HTTP) - для OpenTelemetry Collector

## OpenTelemetry

### Инструментация

DataProcessor API автоматически инструментирует:
- FastAPI endpoints (через FastAPIInstrumentor)
- HTTP запросы (через httpx instrumentation)

### Spans

Каждый запрос создаёт span с атрибутами:
- `run_id` - UUID run'а
- `video_id` - ID видео
- `platform_id` - ID платформы
- `http.method` - HTTP метод
- `http.route` - HTTP route
- `http.status_code` - HTTP статус код

### Настройка

Переменные окружения:
- `ENABLE_TRACING=true` - включить tracing
- `TRACING_EXPORTER=jaeger` или `otlp` - выбрать экспортер
- `JAEGER_AGENT_HOST=jaeger` - хост Jaeger agent
- `JAEGER_AGENT_PORT=6831` - порт Jaeger agent
- `OTLP_ENDPOINT=http://localhost:4317` - OTLP endpoint
- `SERVICE_NAME=dataprocessor-api` - имя сервиса
- `SERVICE_VERSION=0.1.0` - версия сервиса

## Метрики

**Расшифровка полей (в т.ч. `python_gc_*`, `process_*` из библиотеки):** [METRICS_REFERENCE.md](METRICS_REFERENCE.md)

### Доступные метрики

**Queue метрики**:
- `dataprocessor_queue_length{priority}` - длина очереди по приоритетам
- `dataprocessor_queue_wait_seconds` - время ожидания в очереди

**Processing метрики**:
- `dataprocessor_processing_seconds{processor,component}` - время обработки
- `dataprocessor_failures_total{processor,component,error_type}` - количество ошибок

**Resource метрики**:
- `dataprocessor_memory_bytes{run_id}` - использование памяти
- `dataprocessor_active_runs` - количество активных run'ов
- `dataprocessor_crashed_runs_total` - количество упавших run'ов

### Endpoint

Метрики доступны по адресу:
```
GET http://dataprocessor-api:8000/api/v1/metrics
```

## Батч 60+ видео (Audit v4)

Подготовка к большому прогону описана в [CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md](../docs/audit_v4/CHECKLIST_LARGE_VIDEO_BATCH_60PLUS.md) (фазы наблюдаемости — раздел **§5** чеклиста).

**Локальный E2E, Prometheus+Grafana рабочие** (как поднять, порты, проверка): [OBSERVABILITY_STACK_LOCAL_E2E.md](../docs/audit_v4/OBSERVABILITY_STACK_LOCAL_E2E.md).

- URL выше (`localhost`) относятся к **локальному** `docker-compose`. Для батча в проде/на кластере владелец наблюдаемости должен зафиксировать **боевые** URL Prometheus и Grafana и доступы (см. [BATCH_60PLUS_OWNER_TASKS.md](../docs/audit_v4/BATCH_60PLUS_OWNER_TASKS.md)).
- **П. 4.6 чеклиста (CLI vs API):** если прогон идёт в обход HTTP API, заранее согласуйте, как метрики с воркера попадут в Prometheus (тот же scrape target, отдельный exporter и т.д.).
- Реестр видео (заготовка): [VIDEO_REGISTRY_60PLUS.yaml](../docs/audit_v4/VIDEO_REGISTRY_60PLUS.yaml).
- Разрезы `processor` / `component` для **полного** прогона через API→worker→`main.py`: в ответе `ProcessorService` задаются **`pipeline` / `main_py`** (сквозной subprocess, не отдельные экстракторы) — подробности и пилот — [METRICS_LABELS_INVENTORY_60PLUS.md](../docs/audit_v4/METRICS_LABELS_INVENTORY_60PLUS.md).

### NPZ / визуальные ряды: маски и ожидаемые NaN (чеклист **2.9**)

Текущий Grafana-дашборд в этом репозитории отражает **очередь, время, ошибки API** — не содержимое NPZ. Если на батче строите панели или offline-отчёты по **кадровым/сегментным** фичам, соблюдайте:

1. **Не считать NaN багом**, пока не сверились с **`valid_mask` / `present` / `segment_mask` / `empty_reason`** в артефакте и с `SCHEMA.md` / `README.md` модуля. Часть полей **по контракту** не определена вне ROI (нет лица, нет текста, выключенная ветка признаков).
2. **Сначала доля валидных**, потом агрегат по значениям: например, mean только по `finite & mask`; иначе «рост NaN» на датасете без лиц — артефакт контента, а не деградация пайплайна.
3. **Вероятности top‑k** в ряде модулей **не обязаны** суммироваться в 1 — см. [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](../docs/audit_v4/PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.3.
4. **Разная длина оси N** у модулей одного run — не смешивать в одном графике без выравнивания по `frame_indices` / документации Segmenter.
5. Для типовых картин по L2 см. заметки в [RUN_LOG.md](../docs/audit_v4/RUN_LOG.md) по модулям (например, `face_present`, `landmarks_present`, полностью NaN фичи в `shot_quality`).

При появлении отдельного дашборда «качество NPZ» — продублируйте в описании панелей ссылку на этот подраздел и на соответствующий **`components/.../*_audit_v4.md`**.

### Разные **N** кадров / окон (Segmenter) — чеклист **2.10**

Один и тот же `run_id` может иметь **разное число опорных кадров или сегментов** у разных компонентов (Segmenter: разные **families**, разные политики downsampling). Пример из аудита: **N=48** vs **N=120** (`loudness` family `primary` vs `text_scoring`) — см. [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](../docs/audit_v4/PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.3 и заметки в [RUN_LOG.md](../docs/audit_v4/RUN_LOG.md).

**Политика батча 70:**

- Не сравнивать и не усреднять по run **несовместимые ряды** без явного join по времени / общему индексу из артефакта.
- В [VIDEO_REGISTRY_60PLUS.yaml](../docs/audit_v4/VIDEO_REGISTRY_60PLUS.yaml) тег **`segmenter_N`** — про **стратификацию контента** (короткое/длинное видео, богатый Segmenter и т.д.), а не единый скаляр N; фактические размеры осей — из **`metadata.json` / NPZ** после прогона.
- Эталон «полный» профиль и Segmenter **required**: [BATCH_FULL_PROFILE_REFERENCE.md](../docs/audit_v4/BATCH_FULL_PROFILE_REFERENCE.md).

### **`meta.models_used` в NPZ** — чеклист **2.11**

Цель батча: нет «тихих» пропусков модели там, где шёл inference. Ниже — **согласованные исключения** (пустой список или иная семантика **не** считается дефектом без отдельного тикета):

| Зона | Поведение | Источник |
|------|-----------|----------|
| Audio: чисто сигнальные экстракторы | **`models_used: []`** ожидаем | отчёты `onset`, `chroma`, `band_energy`, `loudness` и др. |
| Text: агрегаторы без собственной модели | **`[]` на уровне NPZ**, модель у embedder в цепочке | [asr_text_proxy_audio_features_audit_v4.md](../docs/audit_v4/components/text_processor/asr_text_proxy_audio_features_audit_v4.md), [comments_aggregator_audit_v4.md](../docs/audit_v4/components/text_processor/comments_aggregator_audit_v4.md) |
| Visual | Запись модели по контракту модуля; **известный долг** — точечно закрывать при аудите (напр. исторически **`shot_quality`** / CLIP из `impl_meta`) | [AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md](../docs/audit_v4/AUDIT_V4_L1_CROSS_PROCESSORS_SUMMARY.md), [VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md](../docs/audit_v4/VISUAL_PROCESSOR_AUDIT_V4_SUMMARY.md) |
| Наблюдаемость | Расхождение **`device_used`** vs `models_used[].device` у отдельных run — документировать, не смешивать с «модель не записана» | [clap_extractor_audit_v4.md](../docs/audit_v4/components/audio_processor/clap_extractor_audit_v4.md) |

После батча при появлении новых **пустых `models_used`** при явном GPU-inference — занести в отчёт модуля и в [RUN_LOG.md](../docs/audit_v4/RUN_LOG.md).

### **OCR и `retain_raw_ocr_text`** — чеклист **2.12**

В боевом [`global_config.yaml`](../configs/global_config.yaml) для цепочек OCR / `text_scoring` задано **`retain_raw_ocr_text: false`** (приватность): в NPZ **нет** сырого текста, есть хэши/длины/метаданные — см. [ocr_extractor_audit_v4.md](../docs/audit_v4/components/visual_processor/core/ocr_extractor_audit_v4.md), [RUN_LOG.md](../docs/audit_v4/RUN_LOG.md) (блок `ocr_extractor`). **Метрики качества OCR** на батче интерпретировать по **наличию детекций, conf, счётчикам**, а не по полному тексту кадра. Для отладки OCR — **отдельный внутренний прогон** с `retain_raw_ocr_text: true` (dev-конфиги `audit_v3`, не смешивать с прод-метриками батча). Контекст: [PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md](../docs/audit_v4/PLAN_PREP_LARGE_VIDEO_BATCH_60PLUS.md) §8.3.

## Пилот 15 видео: Prometheus и Grafana

**Что даст дашборд в этом репозитории.** Grafana показывает **операционные** метрики DataProcessor: очередь, активные/упавшие run, время и ошибки в разрезе лейблов `processor` / `component`, память, ожидание в очереди. Этого **недостаточно**, чтобы увидеть «все фичи из всех компонентов» в смысле **содержимого NPZ** (кадры, поля, маски) — туда идут **manifest**, L2-скрипты и offline-отчёты; см. разделы [батча 60+](#батч-60-видео-audit-v4) выше.

**Два сценария scrape.**

| Сценарий | Как поднять | Что проверить |
|----------|-------------|----------------|
| **Всё в Docker** (как `docker compose up` для api + worker + prometheus) | `docker compose up -d prometheus grafana` в каталоге `DataProcessor/`. Используется `prometheus/prometheus.yml`: targets `dataprocessor-api:8000` и `dataprocessor-worker:8001`. | В UI Prometheus → **Status → Targets**: оба job **UP**. У worker в compose задано `DP_WORKER_METRICS_PORT=8001` — гистограммы `dataprocessor_processing_seconds` и `dataprocessor_failures_total` по **pipeline** приходят с **процесса worker** (`/metrics`), не только с API. |
| **API и worker на хосте, Prometheus в Docker** (типичный E2E) | `backend/scripts/setup_e2e_infra.sh` (шаг **[6/6]**) поднимает Prometheus+Grafana так же, как команда вручную из `DataProcessor/`:<br>`docker compose -f docker-compose.yml -f monitoring/docker-compose.prometheus-override-e2e.yml up -d prometheus grafana`<br>Полный стек: `./backend/scripts/start_e2e_stack.sh --with-infra` (после инфраструктуры — приложения). В `e2e_env.sh` задано **`DP_WORKER_METRICS_PORT=8003`** (порт **8001** в E2E занят Backend API). | Targets **UP** на `host.docker.internal:8002` (`/api/v1/metrics`) и `host.docker.internal:8003` (`/metrics`). |

**Переменные окружения (worker).** `DP_WORKER_METRICS_PORT` — порт `prometheus_client` в процессе воркера; без него в Prometheus попадут только метрики API, а **не** гистограммы/счётчики, которые обновляет worker.

| Источник | Job (пример) | Path | Назначение |
|----------|----------------|------|------------|
| Процесс API | `dataprocessor-api` / `...-e2e-host` | `/api/v1/metrics` | Очередь, активные run, память (как зарегистрировано в API) |
| Процесс worker | `dataprocessor-worker` / `...-e2e-host` | `/metrics` | `dataprocessor_processing_seconds`, `dataprocessor_failures_total` (основной путь субпроцесса `main.py`) |

**Grafana:** после запуска — http://localhost:3000 (часто `admin`/`admin`, см. compose). Дашборд **DataProcessor API - Overview** подключён к Prometheus автоматически (provisioning).

## Ссылки

- [Архитектура мониторинга](../../docs/DATAPROCESSOR_API_ARCHITECTURE.md#L2165)
- [Prometheus документация](https://prometheus.io/docs/)
- [Grafana документация](https://grafana.com/docs/)
- [Jaeger документация](https://www.jaegertracing.io/docs/)
- [OpenTelemetry документация](https://opentelemetry.io/docs/)
---

## Навигация

[DataProcessor](../docs/MAIN_INDEX.md) · [Vault](../../docs/MAIN_INDEX.md)
