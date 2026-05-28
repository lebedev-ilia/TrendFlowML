# Требования к идемпотентности Processors

**Версия**: 1.0  
**Дата**: 2024-01-XX  
**Ссылка**: `DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md` (строка 2246-2269)

## Обзор

DataProcessor API использует **At-least-once execution** модель с идемпотентными processors для обеспечения надежности и устойчивости к сбоям.

## Принцип At-least-once

**Почему At-least-once, а не Exactly-once?**

- Exactly-once в distributed системе = дорого и сложно
- Проще: разрешить повторный запуск
- Сделать processors идемпотентными

## Требования к идемпотентности

### 1. Проверка существующих результатов

**Перед запуском обработки run'а:**

1. Проверить существование `manifest.json` в Storage
2. Если manifest существует и run имеет статус `success`:
   - Проверить наличие всех артефактов компонентов в Storage
   - Если все артефакты существуют → использовать кэш (идемпотентность)
   - Вернуть существующий результат без повторной обработки

**Реализация:**
- Функция `check_existing_result()` в `api/services/idempotency.py`
- Вызывается в worker перед запуском subprocess
- Проверяет manifest.json и артефакты в Storage

### 2. Использование кэша при повторном запуске

**Если run уже обработан:**

- Обновить статус run'а на `SUCCESS` (если еще не обновлен)
- Добавить событие `processing_completed_from_cache` в event stream
- ACK сообщение в очереди
- Завершить обработку без запуска subprocess

**Реализация:**
- Worker проверяет существующий результат перед запуском обработки
- Если результат найден → использует кэш, не запускает subprocess

### 3. Частичная обработка (Resume)

**Если run частично обработан:**

- Использовать checkpoint систему для определения последнего процессора
- Resume с последнего процессора через `--resume-from` флаг
- Processors должны поддерживать resume логику

**Реализация:**
- Checkpoint система (`api/services/checkpoint.py`)
- Определение последнего процессора через `determine_last_processor()`
- Передача `last_processor` в subprocess для resume

## Требования к Processors

### 1. Идемпотентность на уровне компонента

**Каждый processor должен:**

1. Проверять существование своих артефактов перед обработкой
2. Если артефакты уже существуют и валидны → использовать их
3. Если артефакты отсутствуют или невалидны → обработать заново

**Пример для core_clip:**
```python
# Проверить существование artifact.npz
artifact_path = f"{rs_path}/core_clip/features.npz"
if os.path.exists(artifact_path):
    # Проверить валидность артефакта
    if is_valid_artifact(artifact_path):
        logger.info("Using existing artifact (idempotency)")
        return load_artifact(artifact_path)
    else:
        logger.warning("Existing artifact is invalid, reprocessing")
        # Обработать заново
```

### 2. Resume поддержка

**Processors должны поддерживать resume:**

1. Принимать параметр `--resume-from` для указания последнего процессора
2. Пропускать уже обработанные процессоры
3. Продолжать с последнего процессора

**Пример:**
```bash
python main.py \
    --video-path /path/to/video.mp4 \
    --run-id abc123 \
    --resume-from audio  # Пропустить segmenter, начать с audio
```

### 3. Атомарная запись артефактов

**Артефакты должны записываться атомарно:**

1. Использовать временный файл для записи
2. Переместить временный файл в финальное место атомарно
3. Это предотвращает частично записанные артефакты при сбоях

**Пример:**
```python
# Атомарная запись через временный файл
temp_path = f"{artifact_path}.tmp"
np.savez_compressed(temp_path, **data)
os.rename(temp_path, artifact_path)  # Атомарная операция
```

## Проверка идемпотентности

### 1. Проверка существующих результатов

**Функция:** `check_existing_result()`

**Логика:**
1. Проверить существование `manifest.json`
2. Проверить статус run'а (должен быть `success`)
3. Проверить наличие всех артефактов компонентов в Storage
4. Вернуть существующий результат если все проверки пройдены

### 2. Проверка компонента

**Функция:** `check_component_result()`

**Логика:**
1. Проверить manifest.json
2. Найти компонент в manifest
3. Проверить статус компонента (должен быть `success`)
4. Проверить наличие артефактов компонента в Storage
5. Вернуть информацию о компоненте если все проверки пройдены

## Обработка ошибок

### 1. Ошибки при проверке кэша

**Если проверка кэша завершилась ошибкой:**

- Не использовать кэш (fail-safe)
- Обработать run заново
- Логировать ошибку для диагностики

### 2. Частично записанные артефакты

**Если артефакт существует, но поврежден:**

- Не использовать кэш
- Обработать компонент заново
- Перезаписать артефакт

### 3. Несоответствие manifest и артефактов

**Если manifest указывает на артефакты, которых нет:**

- Не использовать кэш
- Обработать run заново
- Обновить manifest после обработки

## Метрики и мониторинг

### Метрики для отслеживания:

1. **Cache hit rate**: Процент run'ов, использующих кэш
2. **Cache miss rate**: Процент run'ов, обрабатываемых заново
3. **Resume rate**: Процент run'ов, продолжающихся с checkpoint
4. **Idempotency errors**: Количество ошибок при проверке кэша

### События для логирования:

1. `processing_completed_from_cache`: Run использовал кэш
2. `resume_from_checkpoint`: Run продолжен с checkpoint
3. `cache_check_failed`: Ошибка при проверке кэша

## Тестирование

### Unit тесты:

1. `test_check_existing_result()`: Проверка существующего результата
2. `test_check_component_result()`: Проверка результата компонента
3. `test_idempotency_cache_hit()`: Использование кэша при повторном запуске
4. `test_idempotency_cache_miss()`: Обработка заново при отсутствии кэша

### Integration тесты:

1. `test_idempotent_run()`: Повторный запуск уже обработанного run'а
2. `test_partial_resume()`: Resume частично обработанного run'а
3. `test_cache_with_missing_artifacts()`: Обработка при отсутствии артефактов

## Ссылки

- Архитектура: `DataProcessor/docs/DATAPROCESSOR_API_ARCHITECTURE.md` (строка 2246-2269)
- Реализация: `api/services/idempotency.py`
- Checkpoint система: `api/services/checkpoint.py`
- Worker: `api/services/worker.py`

