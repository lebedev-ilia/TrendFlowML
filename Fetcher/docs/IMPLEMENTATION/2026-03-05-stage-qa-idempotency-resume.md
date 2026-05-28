# Idempotency и Resume после сбоя

**Дата**: 2026-03-05  
**Стадия**: Quality Assurance Checklist  
**Статус**: ✅ Реализовано

## Обзор

Реализована поддержка идемпотентности pipeline stages и возможность resume после сбоя worker'а. Это критически важно для обеспечения надёжности и предотвращения дубликатов при повторных запусках.

## Требования

Из Quality Assurance Checklist:
- ✅ Idempotency of pipeline stages
- ✅ Resume after worker crash
- ✅ Checksum validation (несовпадение приводит к fail/retry)
- ✅ Cache reuse correctness (нет лишних скачиваний)

## Реализация

### 1. Модуль `fetcher/idempotency.py`

**Функции**:

1. **`check_video_exists(platform, platform_video_id)`**:
   - Проверяет наличие видео в кеше по platform и platform_video_id

2. **`check_artifact_exists(video_id, artifact_type, status)`**:
   - Проверяет наличие артефакта для видео в БД

3. **`check_artifact_in_storage(artifact, bucket)`**:
   - Проверяет существование артефакта в object storage

4. **`validate_artifact_checksum(artifact, bucket)`**:
   - Скачивает артефакт, вычисляет SHA256 checksum и сравнивает с сохранённым
   - Возвращает `(is_valid, error_message)`

5. **`is_stage_idempotent(platform, platform_video_id, stage, validate_checksum=False)`**:
   - Проверяет, можно ли пропустить stage (уже выполнен)
   - Проверяет наличие видео, артефакта в БД и storage
   - Опционально проверяет checksum для валидации целостности
   - Возвращает `(can_skip, reason)`

**Маппинг stages → artifact_types**:
- `metadata` → `metadata_file`
- `download` → `video_file`
- `comments` → `comments_file`

### 2. Модуль `fetcher/resume.py`

**Функции**:

1. **`get_incomplete_runs(statuses=None)`**:
   - Получает список незавершённых run'ов для resume
   - Фильтрует по статусам (по умолчанию все незавершённые, кроме FINAL и FAILED)
   - Возвращает только run'ы с `finished_at IS NULL`

2. **`get_missing_artifacts_for_run(run_id)`**:
   - Определяет, какие артефакты отсутствуют для run'а
   - Возвращает список типов отсутствующих артефактов

3. **`determine_next_stage(run_id)`**:
   - Определяет следующую stage для resume на основе отсутствующих артефактов
   - Приоритет: `metadata` → `download` → `comments` → `finalize`
   - Возвращает название следующей stage или `None` если всё готово

### 3. Интеграция в Workers

**Изменения в `fetcher/workers/metadata.py`, `video.py`, `comments.py`**:

- Добавлена проверка идемпотентности перед выполнением stage
- Если stage уже выполнен (артефакт существует в БД и storage), worker пропускает выполнение
- Логируется причина пропуска для observability

**Пример**:
```python
# Получаем platform_video_id из video_source
with session_scope() as db:
    video_source = db.query(VideoSource).filter(...).first()
    platform_video_id = video_source.normalized_video_id or source

can_skip, reason = is_stage_idempotent(platform, platform_video_id, "metadata")
if can_skip:
    logger.info(f"Metadata stage skipped (idempotent): {reason}")
    return

# Выполняем stage только если не идемпотентен
adapter.fetch_metadata(source, run_id=run_id)
```

## Checksum Validation

- Checksum вычисляется при upload всех артефактов (SHA256)
- Функция `validate_artifact_checksum()` позволяет проверить целостность артефакта
- Опциональная проверка checksum в `is_stage_idempotent()` для валидации при идемпотентности

## Cache Reuse

- Проверка кеша в `orchestrator.check_cache()` перед постановкой задач
- Проверка идемпотентности в workers предотвращает повторные скачивания
- Distributed lock для video download предотвращает дубликаты при параллельных запусках

## Использование

### Проверка идемпотентности

```python
from fetcher.idempotency import is_stage_idempotent

can_skip, reason = is_stage_idempotent("youtube", "dQw4w9WgXcQ", "metadata")
if can_skip:
    print(f"Stage can be skipped: {reason}")
```

### Resume после сбоя

```python
from fetcher.resume import get_incomplete_runs, determine_next_stage

# Найти незавершённые run'ы
incomplete_runs = get_incomplete_runs()

for run in incomplete_runs:
    next_stage = determine_next_stage(str(run.id))
    if next_stage:
        print(f"Run {run.id} needs {next_stage} stage")
        # Запустить соответствующую задачу
```

## Метрики

- Логирование пропущенных stages для observability
- Метрики cache hits/misses уже интегрированы в orchestrator

## Тестирование

**Рекомендуемые тесты**:

1. **Unit тесты**:
   - `test_is_stage_idempotent()`: Проверка идемпотентности для разных stages
   - `test_validate_artifact_checksum()`: Проверка checksum validation
   - `test_determine_next_stage()`: Определение следующей stage для resume

2. **Integration тесты**:
   - `test_idempotent_metadata_worker()`: Повторный запуск metadata worker не создаёт дубликаты
   - `test_idempotent_video_worker()`: Повторный запуск video worker не скачивает повторно
   - `test_resume_after_crash()`: Resume после сбоя продолжает с правильной stage

## Связанные файлы

- `fetcher/idempotency.py` - модуль проверки идемпотентности
- `fetcher/resume.py` - модуль resume после сбоя
- `fetcher/workers/metadata.py` - интеграция идемпотентности
- `fetcher/workers/video.py` - интеграция идемпотентности
- `fetcher/workers/comments.py` - интеграция идемпотентности
- `fetcher/checksums.py` - вычисление checksum
- `fetcher/orchestrator.py` - проверка кеша

## Статус

✅ **Реализовано**:
- Модуль `idempotency.py` с функциями проверки идемпотентности
- Модуль `resume.py` с функциями для resume после сбоя
- Интеграция в workers для пропуска уже выполненных stages
- Проверка checksum для валидации целостности артефактов
- Проверка существования артефактов в БД и storage

## Следующие шаги

- Добавить периодическую задачу для автоматического resume незавершённых run'ов
- Добавить метрики для отслеживания идемпотентных пропусков
- Интегрировать resume в orchestrator для автоматического продолжения после сбоя

