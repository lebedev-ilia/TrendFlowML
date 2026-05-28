## Stage QA.1 — Error Classification (Quality Assurance Checklist)

**Дата**: 2026-03-05  
**Статус**: ✅ завершено  
**Чеклист**: Fetcher `Quality Assurance Checklist → Retry safety`

---

## 1. Выполненные задачи

- ✅ Создан модуль классификации ошибок (`fetcher/errors.py`)
- ✅ Реализованы базовые классы `RetryableError` и `NonRetryableError`
- ✅ Реализованы конкретные типы ошибок для каждого класса
- ✅ Реализована функция `is_retryable_error()` для определения retryable ошибок
- ✅ Реализована функция `get_error_category()` для категоризации ошибок
- ✅ Интегрирована классификация ошибок во все Celery tasks

---

## 2. Технические детали

### 2.1. Error Classes

**Файл**: `fetcher/errors.py`

**Retryable Errors**:
- `NetworkError` — сетевые ошибки (connection timeout, DNS failure)
- `TimeoutError` — timeout ошибки
- `RateLimitError` — rate limit ошибки (429 Too Many Requests)
- `TransientStorageError` — временные ошибки storage (503 Service Unavailable)
- `CircuitBreakerOpenError` — circuit breaker открыт

**Non-Retryable Errors**:
- `VideoNotFoundError` — видео не найдено
- `PrivateVideoError` — видео приватное
- `AgeRestrictedError` — видео с возрастными ограничениями
- `InvalidInputError` — невалидный вход
- `LogicError` — логическая ошибка
- `AuthenticationError` — ошибка аутентификации
- `MissingDependencyError` — отсутствует зависимость

### 2.2. Error Detection

**Функция**: `is_retryable_error(error: Exception) -> bool`

**Логика**:
1. Проверка по типу исключения (isinstance)
2. Проверка по строковому представлению (паттерны в error message)
3. По умолчанию считает retryable (консервативный подход)

**Паттерны**:
- Retryable: "timeout", "connection", "429", "503", "rate limit", "circuit breaker"
- Non-retryable: "not found", "404", "private", "age restricted", "invalid", "401", "403", "missing", "removed"

### 2.3. Integration

**Изменённые файлы**: `fetcher/tasks.py`

**Изменения**:
- В `fetch_metadata_task`, `download_video_task`, `fetch_comments_task`, `finalize_task`:
  - Вызов `get_error_category(e)` для категоризации
  - Проверка `is_retryable_error(e)` перед retry
  - Retry только если `is_retryable_error(e) and retries < max_retries`
  - Non-retryable ошибки и превышение лимита retry приводят к fail-fast
  - Логирование включает `error_category` для мониторинга

---

## 3. Технические детали

### 3.1. Conservative Approach

- По умолчанию ошибки считаются retryable (лучше повторить, чем пропустить из-за временной ошибки)
- Неизвестные типы ошибок логируются как warning, но считаются retryable

### 3.2. Error Category Logging

- Каждая ошибка логируется с полем `error_category` ("retryable" или "non_retryable")
- Это позволяет мониторить распределение ошибок по категориям

---

## 4. Тестирование

### 4.1. Ручные проверки

- ✅ Retryable ошибки корректно определяются и ретраятся
- ✅ Non-retryable ошибки корректно определяются и не ретраятся
- ✅ Превышение лимита retry приводит к fail-fast
- ✅ Error category корректно логируется

### 4.2. Edge cases

- ✅ Обработка неизвестных типов ошибок (считаются retryable)
- ✅ Обработка ошибок без строкового представления
- ✅ Обработка ошибок с несколькими паттернами (приоритет non-retryable)

---

## 5. Известные ограничения

- Паттерн-based detection может давать false positives/negatives
- Нет поддержки кастомных правил классификации для разных платформ
- Нет метрики `fetcher_errors_by_category_total` для мониторинга распределения ошибок

---

## 6. Следующие шаги

- Добавить метрику `fetcher_errors_by_category_total` для мониторинга
- Рассмотреть использование более точных методов детектирования (например, по HTTP status codes)
- Добавить поддержку кастомных правил классификации через конфигурацию
- Интегрировать error classification в метрики (например, `fetcher_videos_failed_total` с label `error_category`)

---

## 7. Ссылки

- Чеклист: `Quality Assurance Checklist → Retry safety`
- Код: `fetcher/errors.py`
- Интеграция: `fetcher/tasks.py` (все Celery tasks)
- Документация: `Fetcher/docs/PLATFORM_ADAPTERS.md` (раздел 2.2. Ошибки и retry)

