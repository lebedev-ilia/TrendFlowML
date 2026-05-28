# Контракт Backend ↔ DataProcessor

Этот документ описывает контракт между Backend API и DataProcessor worker для запуска анализа видео.

> **Актуальный код (зафиксировано для портфолио, 2026-04):** в `backend/app/tasks/` зарегистрированы задачи **`process_analysis_job`** (v2 **AnalysisJob**, путь «видео в продукте») и **`process_ingestion_run`** (по **IngestionRun** после Fetcher). Отдельной задачи **`process_run`** в этом модуле нет. Разделы ниже с примерами `process_run`, `Run` и `runs.py` как создателем run отражают **legacy-контракт и эволюцию**; живые маршруты: `app/routers/analysis.py`, `app/routers/runs.py`, адаптер `app/services/dataprocessor_adapter.py`. См. также [RUNS_AND_WORKERS.md](../RUNS_AND_WORKERS.md), [DEMO_AND_PORTFOLIO.md](../DEMO_AND_PORTFOLIO.md).

---

## 1) Обзор

Backend создаёт задачу анализа (`AnalysisJob` в v2 или `Run` в legacy), ставит её в очередь Celery, и DataProcessor worker выполняет обработку, записывая результаты в файловое хранилище и обновляя статус в БД.

---

## 2) Текущий контракт (Legacy API)

### 2.1 Создание задачи

**Backend** (`backend/app/routers/runs.py`):
```python
run = Run(user_id=user.id, video_id=payload.video_id, profile_id=payload.profile_id, status="queued")
db.add(run)
process_run.delay(run.id)  # Celery task
```

**Celery Task** (`backend/app/tasks/`):
- Принимает `run_id` (UUID строка)
- Читает `Run`, `Video`, `AnalysisProfile` из БД
- Запускает `DataProcessor/main.py` через subprocess
- Передаёт параметры командной строки

### 2.2 Параметры командной строки DataProcessor

DataProcessor ожидает следующие аргументы:
```
--video-path <path>
--output <frames_dir_base>
--chunk-size 64
--visual-cfg-path <path>
--profile-path <path>
--dag-path <path>
--dag-stage baseline
--platform-id <platform_id>
--video-id <video_id>
--run-id <run_id>
--sampling-policy-version v1
--dataprocessor-version dev
--rs-base <result_store_base>
```

### 2.3 Структура данных

**Run** (legacy):
- `id`: UUID строка
- `user_id`: UUID строка
- `video_id`: UUID строка
- `profile_id`: UUID строка (nullable)
- `config_hash`: строка
- `status`: `queued|running|succeeded|failed|cancelled`
- `stage`: `segmenter|audio|text|visual|render`

**Video** (legacy):
- `id`: UUID строка
- `platform_id`: строка (`upload`, `youtube`)
- `video_id`: строка (каноничный ID)
- `source_type`: строка (`upload`, `link`)

**AnalysisProfile** (legacy):
- `id`: UUID строка
- `config_json`: JSON объект
- `config_hash`: строка (sha256 от нормализованного JSON)

### 2.4 Результаты

DataProcessor пишет:
- `manifest.json` в `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`
- NPZ артефакты в `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/*.npz`
- `state_events.jsonl` в `state/<platform_id>/<video_id>/<run_id>/state_events.jsonl`

Backend после завершения:
- Читает `manifest.json`
- Регистрирует артефакты в таблице `artifacts`
- Обновляет статус `Run.status` и `RunComponent.status`

---

## 3) Новый контракт (V2 API)

### 3.1 Создание задачи

**Backend** (`backend/app/routers/v2_analysis.py`):
```python
job = AnalysisJob(
    workspace_id=workspace.id,
    video_id=video.id,
    triggered_by_user_id=user.id,
    processing_config_id=config.id,
    model_version_id="v1",
    status=AnalysisStatus.queued,
)
db.add(job)
process_analysis_job.delay(job.id)  # Celery task (новый)
```

### 3.2 Структура данных V2

**AnalysisJob** (v2):
- `id`: UUID
- `workspace_id`: UUID
- `video_id`: UUID
- `triggered_by_user_id`: UUID
- `processing_config_id`: UUID
- `model_version_id`: строка
- `status`: `AnalysisStatus` enum (`queued|processing|completed|failed|canceled`)
- `retry_count`: int
- `error_message`: строка (nullable)
- `started_at`: datetime (nullable)
- `completed_at`: datetime (nullable)

**Video** (v2):
- `id`: UUID
- `channel_id`: UUID
- `external_video_id`: строка (nullable)
- `title`: строка
- `description`: строка (nullable)
- `duration_seconds`: int
- `video_type`: `VideoType` enum (`shorts|video`)
- `source_type`: `SourceType` enum (`upload|link`)
- `source_url`: строка (nullable)
- `storage_path`: строка (nullable)
- `file_size_mb`: float (nullable)
- `checksum`: строка (nullable)

**Channel** (v2):
- `id`: UUID
- `workspace_id`: UUID
- `platform`: строка (`youtube`, `tiktok`, ...)
- `external_channel_id`: строка (nullable)
- `channel_name`: строка

### 3.3 Адаптер для DataProcessor

Чтобы DataProcessor продолжал работать с legacy форматом, создаём адаптер:

**`backend/app/services/dataprocessor_adapter.py`**:
```python
def prepare_dataprocessor_payload(analysis_job: AnalysisJob) -> dict:
    """Преобразует AnalysisJob (v2) в формат, понятный DataProcessor (legacy)"""
    # Маппинг v2 → legacy
    # Возвращает параметры для subprocess
```

### 3.4 Результаты V2

DataProcessor пишет те же файлы, но backend:
- Обновляет `AnalysisJob.status` и `AnalysisJob.completed_at`
- Создаёт `Prediction` записи из `manifest.json`
- Регистрирует артефакты (можно использовать legacy `artifacts` или создать v2 таблицу)

---

## 4) Маппинг Legacy → V2

### 4.1 Run → AnalysisJob

| Legacy (Run) | V2 (AnalysisJob) |
|--------------|-------------------|
| `id` | `id` |
| `user_id` | `triggered_by_user_id` |
| `video_id` | `video_id` |
| `profile_id` | `processing_config_id` |
| `config_hash` | (из `processing_config`) |
| `status` | `status` (enum) |
| `stage` | (убрано, детализация через компоненты) |
| `created_at` | `created_at` |
| `started_at` | `started_at` |
| `finished_at` | `completed_at` |

### 4.2 Video → Video

| Legacy (Video) | V2 (Video) |
|----------------|------------|
| `id` | `id` |
| `platform_id` | (из `channel.platform`) |
| `video_id` | `external_video_id` |
| `source_type` | `source_type` (enum) |
| `title` | `title` |
| `description` | `description` |

### 4.3 Profile → ProcessingConfig

**Проблема**: В v2 нет прямой замены `AnalysisProfile`.

**Варианты решения**:
1. Использовать legacy `analysis_profiles` как есть
2. Создать `core.processing_configs` таблицу
3. Хранить `processing_config_id` как UUID, который ссылается на legacy `analysis_profiles.id`

**Рекомендация**: Вариант 3 (временный), затем мигрировать на вариант 2.

---

## 5) Celery Tasks

### 5.1 Legacy Task

**`backend/app/tasks/`**:
```python
@celery_app.task
def process_run(run_id: str):
    # Читает Run из БД
    # Запускает DataProcessor
    # Обновляет статус
```

### 5.2 V2 Task (новый)

**`backend/app/tasks/`**:
```python
@celery_app.task
def process_analysis_job(job_id: UUID):
    # Читает AnalysisJob из БД
    # Использует адаптер для преобразования в legacy формат
    # Запускает DataProcessor
    # Обновляет AnalysisJob и создаёт Prediction
```

---

## 6) Обработка результатов

### 6.1 Legacy

После завершения DataProcessor:
1. Backend читает `manifest.json`
2. Регистрирует артефакты в `artifacts`
3. Обновляет `Run.status` и `RunComponent.status`

### 6.2 V2

После завершения DataProcessor:
1. Backend читает `manifest.json`
2. Обновляет `AnalysisJob.status = completed`
3. Создаёт `Prediction` записи из predictions в manifest
4. Регистрирует артефакты (legacy `artifacts` или v2 таблица)

---

## 7) Обратная совместимость

### Стратегия
1. **Адаптер**: преобразует v2 → legacy формат для DataProcessor
2. **Двойная запись**: результаты пишутся и в legacy, и в v2 таблицы
3. **Постепенный переход**: клиенты мигрируют на v2 API постепенно

### Риски
- Дублирование данных (legacy + v2)
- Сложность поддержки двух форматов
- Потенциальные рассинхронизации

### Митигация
- Использовать транзакции для атомарности
- Периодически синхронизировать данные
- Чёткий план завершения миграции

---

## 8) Примеры использования

### 8.1 Создание AnalysisJob через V2 API

```python
# POST /api/v2/analysis
{
    "workspace_id": "uuid",
    "video_id": "uuid",
    "processing_config_id": "uuid"
}

# Backend создаёт AnalysisJob и ставит в очередь
process_analysis_job.delay(job.id)
```

### 8.2 Адаптер преобразует в legacy формат

```python
# dataprocessor_adapter.py
payload = prepare_dataprocessor_payload(analysis_job)
# payload содержит все параметры для DataProcessor
# в legacy формате (run_id, video_id, platform_id, config_hash)
```

### 8.3 DataProcessor работает как обычно

DataProcessor не знает о v2, получает legacy формат и работает как раньше.

---

## 9) Чеклист реализации

- [ ] Создать `backend/app/services/dataprocessor_adapter.py`
- [x] Создать `backend/app/tasks/` с `process_analysis_job`
- [ ] Обновить `backend/app/routers/v2_analysis.py` для использования нового task
- [ ] Добавить создание `Prediction` после завершения анализа
- [ ] Протестировать полный цикл: создание → обработка → результаты

---

**Последнее обновление**: 2025-01-XX  
**Ответственный**: Backend Team

