# Strict State Machine для Event Ordering Correctness

**Дата**: 2026-03-05  
**Стадия**: Quality Assurance Checklist  
**Статус**: ✅ Реализовано

## Обзор

Реализована строгая state machine для управления переходами статусов run'ов в Fetcher. Это предотвращает недопустимые переходы между статусами (например, COMPLETED → PENDING) и обеспечивает корректный порядок событий.

## Требования

Из Quality Assurance Checklist:
- ✅ Event ordering correctness (state machine не "прыгает" назад)

## Реализация

### 1. Модуль `fetcher/state_machine.py`

**Константы статусов**:
- `RUN_STATUS_PENDING`
- `RUN_STATUS_NORMALIZING_SOURCE`
- `RUN_STATUS_CHECKING_CACHE`
- `RUN_STATUS_FETCHING_METADATA`
- `RUN_STATUS_FETCHING_CHANNEL`
- `RUN_STATUS_FETCHING_COMMENTS`
- `RUN_STATUS_DOWNLOADING_VIDEO`
- `RUN_STATUS_UPLOADING_ARTIFACTS`
- `RUN_STATUS_FINALIZING`
- `RUN_STATUS_COMPLETED`
- `RUN_STATUS_FAILED`

**Таблица разрешенных переходов** (`ALLOWED_TRANSITIONS`):

```python
ALLOWED_TRANSITIONS = {
    "PENDING": ["NORMALIZING_SOURCE", "FAILED"],
    "NORMALIZING_SOURCE": ["CHECKING_CACHE", "FAILED"],
    "CHECKING_CACHE": ["FETCHING_METADATA", "FINALIZING", "FAILED"],
    "FETCHING_METADATA": ["FETCHING_CHANNEL", "FETCHING_COMMENTS", "DOWNLOADING_VIDEO", "UPLOADING_ARTIFACTS", "FAILED"],
    "FETCHING_CHANNEL": ["FETCHING_COMMENTS", "DOWNLOADING_VIDEO", "UPLOADING_ARTIFACTS", "FAILED"],
    "FETCHING_COMMENTS": ["DOWNLOADING_VIDEO", "UPLOADING_ARTIFACTS", "FAILED"],
    "DOWNLOADING_VIDEO": ["UPLOADING_ARTIFACTS", "FAILED"],
    "UPLOADING_ARTIFACTS": ["FINALIZING", "FAILED"],
    "FINALIZING": ["COMPLETED", "FAILED"],
    "COMPLETED": [],  # Финальное состояние
    "FAILED": [],  # Финальное состояние
}
```

**Функции**:

1. **`can_transition(from_status, to_status)`**:
   - Проверяет, разрешен ли переход между статусами
   - Возвращает `True` если переход разрешен, `False` иначе

2. **`validate_transition(from_status, to_status, run_id=None)`**:
   - Валидирует переход между статусами
   - Вызывает `ValueError` если переход не разрешен
   - Логирует предупреждение при недопустимом переходе

3. **`get_allowed_transitions(from_status)`**:
   - Возвращает список разрешенных переходов из текущего статуса

4. **`is_final_status(status)`**:
   - Проверяет, является ли статус финальным (COMPLETED или FAILED)

### 2. Интеграция в Orchestrator

**Изменения в `fetcher/orchestrator.py`**:

- Все обновления статуса используют `validate_transition()` перед изменением
- Валидация переходов:
  - `PENDING` → `NORMALIZING_SOURCE`
  - `NORMALIZING_SOURCE` → `CHECKING_CACHE`
  - `CHECKING_CACHE` → `FETCHING_METADATA` или `FINALIZING` (cache hit)
  - `CHECKING_CACHE` → `FAILED` (при ошибке)

**Особенности**:
- При переходе в `FAILED` из любого промежуточного статуса валидация может быть пропущена с предупреждением (для обработки критических ошибок)

### 3. Интеграция в Tasks

**Изменения в `fetcher/tasks.py`**:

- `finalize_task`: Валидация перехода `FINALIZING` → `COMPLETED` или `FAILED`
- При ошибке: Валидация перехода в `FAILED` с предупреждением, если переход не разрешен

## Примеры использования

### Валидация перехода

```python
from fetcher.state_machine import validate_transition, RUN_STATUS_COMPLETED, RUN_STATUS_FINALIZING

# Валидация разрешенного перехода
validate_transition(RUN_STATUS_FINALIZING, RUN_STATUS_COMPLETED, run_id="...")
# OK

# Валидация недопустимого перехода
validate_transition(RUN_STATUS_COMPLETED, RUN_STATUS_PENDING, run_id="...")
# ValueError: Invalid status transition: COMPLETED → PENDING
```

### Проверка разрешенных переходов

```python
from fetcher.state_machine import can_transition, get_allowed_transitions

# Проверка разрешенности перехода
can_transition("FINALIZING", "COMPLETED")  # True
can_transition("COMPLETED", "PENDING")  # False

# Получение списка разрешенных переходов
get_allowed_transitions("FINALIZING")  # ["COMPLETED", "FAILED"]
```

## Защита от недопустимых переходов

State machine предотвращает:

1. **Переходы назад**: `COMPLETED` → `PENDING`, `FINALIZING` → `FETCHING_METADATA`
2. **Пропуск этапов**: `PENDING` → `FINALIZING` (минуя нормализацию и проверку кеша)
3. **Переходы из финальных состояний**: `COMPLETED` → любое другое состояние
4. **Недопустимые параллельные переходы**: `FETCHING_METADATA` → `COMPLETED` (минуя финализацию)

## Обработка ошибок

- При недопустимом переходе вызывается `ValueError` с описанием проблемы
- При критических ошибках переход в `FAILED` может быть разрешен с предупреждением
- Все недопустимые переходы логируются для мониторинга

## Метрики

- Логирование всех переходов статусов для observability
- Предупреждения при недопустимых переходах

## Тестирование

**Рекомендуемые тесты**:

1. **Unit тесты**:
   - `test_can_transition()`: Проверка разрешенных переходов
   - `test_validate_transition()`: Валидация переходов с исключениями
   - `test_final_statuses()`: Проверка финальных состояний

2. **Integration тесты**:
   - `test_orchestrator_state_transitions()`: Проверка переходов в orchestrator
   - `test_tasks_state_transitions()`: Проверка переходов в tasks
   - `test_invalid_transitions_rejected()`: Проверка отклонения недопустимых переходов

## Связанные файлы

- `fetcher/state_machine.py` - модуль state machine
- `fetcher/orchestrator.py` - интеграция валидации переходов
- `fetcher/tasks.py` - интеграция валидации переходов
- `docs/BACKEND_CONTRACTS.md` - список статусов Fetcher
- `docs/PIPELINE_ORCHESTRATION.md` - дизайн state machine

## Статус

✅ **Реализовано**:
- Модуль `state_machine.py` с таблицей разрешенных переходов
- Функции валидации переходов
- Интеграция в orchestrator и tasks
- Предотвращение недопустимых переходов

## Следующие шаги

- Добавить метрики для отслеживания недопустимых переходов (если они происходят)
- Добавить автоматическое восстановление при обнаружении недопустимого состояния
- Интегрировать state machine в workers для валидации переходов на уровне задач

