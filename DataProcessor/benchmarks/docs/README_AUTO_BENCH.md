# Автоматический бенчмарк-раннер для core_optical_flow

Скрипт `auto_bench.py` автоматизирует процесс запуска бенчмарков компонента `core_optical_flow` с различными конфигурациями RAFT моделей, batch sizes и количеством кадров.

## Возможности

- **Автоматическая поддержка всех RAFT пресетов**: `raft_256`, `raft_384`, `raft_512`
- **Автоматическая защита от OOM**: автоматическое уменьшение batch_size при ошибках памяти
- **Cleanup и перезапуск Triton на каждой попытке**: чистое состояние системы для каждого измерения
- **Множественные попытки**: запуск каждого бенчмарка несколько раз для статистики
- **Автоматическая агрегация результатов**: объединение результатов всех попыток

## Конфигурация

Скрипт настраивается через переменные в начале файла:

```python
# Benchmark configuration
PREP = "raft_256"  # Будет переопределено для всех пресетов
COMPONENT = "core_optical_flow"
VIDEO_PATH = "example/example_videos/-F71yZij1Uc.mp4"
TRITON_URL = "http://localhost:8000"
FRAMES_LIST = [2, 4, 8, 32, 64, 100, 304]
BATCHES = [1, 8, 16]
ATTEMPTS = 4

# Docker configuration
DOCKER_IMAGE = "nvcr.io/nvidia/tritonserver:24.08-py3"
DOCKER_CONTAINER_NAME = "triton-bench"
```

## Использование

### Базовое использование

```bash
cd DataProcessor/benchmarks
python auto_bench.py
```

Скрипт автоматически:
1. Прогонит все конфигурации для всех RAFT пресетов (`raft_256`, `raft_384`, `raft_512`)
2. Для каждого пресета запустит все комбинации batch sizes и frame counts
3. Для каждой комбинации выполнит несколько попыток (по умолчанию 4)
4. Агрегирует результаты и сохранит в `summary/res_{prep}_{batch}_{frames}.json`

## Как это работает

### Последовательность выполнения

Для каждой комбинации `(prep, batch, frames)`:

1. **Для каждой попытки (attempt)**:
   - **Cleanup системы**: очистка GPU памяти, системного кеша
   - **Остановка предыдущего Triton**: если контейнер был запущен
   - **Запуск Triton**: запуск Docker контейнера с нужной моделью
   - **Ожидание готовности**: автоматическое ожидание готовности Triton (через `--wait-triton`)
   - **Запуск бенчмарка**: выполнение `run_component_bench.py`
   - **Остановка Triton**: остановка контейнера после завершения

2. **Агрегация результатов**: после всех попыток результаты агрегируются через `aggregate_benchmark_results.py`

3. **Очистка**: удаление временных файлов из `out_component/`

### Автоматическая защита от OOM

Скрипт автоматически обрабатывает ошибки нехватки памяти (OOM):

- **Обнаружение OOM**: анализ stderr/stdout на наличие признаков OOM:
  - `"cuda out of memory"`
  - `"cuda_error_out_of_memory"`
  - `"out of memory"`
  - `"ResourceExhaustedError"`
  - `"memory allocation failed"`

- **Автоматическое уменьшение batch_size**: если обнаружен OOM и `batch_size > 1`:
  - Batch size уменьшается вдвое: `new_batch = max(1, current_batch // 2)`
  - Попытка повторяется с уменьшенным batch size
  - Процесс повторяется до успеха или до `batch_size = 1`

**Пример:**
```
[auto_bench] Running benchmark: batch=16, frames=304, attempt=1/4
[auto_bench] ERROR: Benchmark failed (OOM detected)
[auto_bench] Retrying with reduced batch_size: 8
[auto_bench] Running benchmark: batch=8, frames=304, attempt=1/4
[auto_bench] ✓ Benchmark completed successfully
```

### Поддержка всех RAFT пресетов

Скрипт автоматически прогоняет все три пресета:

- **raft_256**: быстрая модель, меньше VRAM
- **raft_384**: средняя модель, средний VRAM
- **raft_512**: медленная модель, больше VRAM (чаще требует уменьшения batch_size)

Для каждого пресета:
- Используется соответствующая директория моделей: `triton/models_raft_{256,384,512}`
- Передается соответствующий `--triton-preprocess-preset`
- Результаты сохраняются с префиксом пресета: `res_raft_{256,384,512}_{batch}_{frames}.json`

## Структура результатов

Результаты сохраняются в `benchmarks/summary/`:

```
summary/
├── res_raft_256_1_2.json
├── res_raft_256_1_8.json
├── res_raft_256_8_64.json
├── res_raft_384_1_2.json
├── res_raft_384_8_64.json
├── res_raft_512_1_2.json
└── ...
```

Каждый JSON файл содержит агрегированные метрики:
- `Duration (s)`: среднее время выполнения
- `Peak CPU %`: пиковая загрузка CPU
- `Peak GPU %`: пиковая загрузка GPU
- `Component Delta VRAM (MB)`: изменение VRAM при выполнении компонента
- `Component Delta RAM (MB)`: изменение RAM при выполнении компонента

## Требования

- **Docker**: должен быть установлен и запущен
- **Triton модели**: должны быть в директориях:
  - `DataProcessor/triton/models_raft_256/`
  - `DataProcessor/triton/models_raft_384/`
  - `DataProcessor/triton/models_raft_512/`
- **Права sudo**: для cleanup системы (или используйте `--skip-cleanup` в cleanup скрипте)
- **GPU**: с достаточным количеством VRAM (особенно для `raft_512`)

## Важные особенности

### Cleanup и перезапуск Triton на каждой попытке

**Критически важно**: Cleanup системы и перезапуск Triton происходят перед **каждой** попыткой (attempt), а не только перед группой бенчмарков. Это обеспечивает:

- Чистое состояние системы для каждого измерения
- Отсутствие влияния предыдущих попыток на текущую
- Более точные и воспроизводимые результаты

Последовательность для каждой попытки:
1. Остановка предыдущего Triton (если был запущен)
2. Cleanup системы (очистка кэша, GPU памяти, буферов)
3. Запуск Triton с нужной моделью
4. Ожидание готовности Triton (автоматически)
5. Запуск бенчмарка
6. Остановка Triton

Это означает, что для бенчмарка с 4 попытками будет выполнено:
- 4 раза cleanup
- 4 раза запуск Triton
- 4 раза остановка Triton

### Автоматическое ожидание Triton

Скрипт использует флаг `--wait-triton` при запуске `run_component_bench.py`, что означает:
- Бенчмарк автоматически ждет готовности Triton (через health check)
- Не требуется ручное нажатие Enter
- Таймаут ожидания: 30 секунд (настраивается через `--triton-timeout`)

### Обработка ошибок

- **OOM ошибки**: автоматическое уменьшение batch_size и повтор
- **Ошибки запуска Triton**: пропуск попытки с предупреждением
- **Ошибки бенчмарка**: логирование и продолжение следующей попытки
- **Ошибки агрегации**: предупреждение, но продолжение работы

## Примеры вывода

### Успешный запуск

```
================================================================================
AUTOMATED BENCHMARK RUNNER
================================================================================
Component: core_optical_flow
Preprocess preset: raft_256 (will run for all: raft_256, raft_384, raft_512)
Video: example/example_videos/-F71yZij1Uc.mp4
Batches: [1, 8, 16]
Frames: [2, 4, 8, 32, 64, 100, 304]
Attempts per configuration: 4
================================================================================

================================================================================
CONFIGURATION: prep=raft_256, batch=1, frames=2
================================================================================

[auto_bench] Attempt 1/4
[auto_bench] Cleaning system...
[auto_bench] ✓ Cleanup successful
[auto_bench] Stopping existing Triton container (if any)...
[auto_bench] Starting Triton server in Docker...
[auto_bench] ✓ Triton container started: 1ee0da66fb35
[auto_bench] Running benchmark: batch=1, frames=2, attempt=1/4
[auto_bench] ✓ Benchmark completed successfully
[auto_bench] Stopping Triton container...
[auto_bench] ✓ Triton container stopped
...
```

### Обработка OOM

```
[auto_bench] Running benchmark: batch=16, frames=304, attempt=1/4
[auto_bench] ERROR: Benchmark failed with returncode 1
[auto_bench] OOM detected, retrying with reduced batch_size: 8
[auto_bench] Running benchmark: batch=8, frames=304, attempt=1/4
[auto_bench] ✓ Benchmark completed successfully
```

## Настройка

### Изменение количества попыток

```python
ATTEMPTS = 4  # Изменить на нужное значение
```

### Изменение списка кадров

```python
FRAMES_LIST = [2, 4, 8, 32, 64, 100, 304]  # Добавить/удалить значения
```

### Изменение batch sizes

```python
BATCHES = [1, 8, 16]  # Добавить/удалить значения
```

### Изменение таймаута ожидания Triton

В функции `run_benchmark()` измените:
```python
"--triton-timeout", "30.0",  # Изменить на нужное значение
```

## Отключение кэширования

Кэширование текстовых эмбеддингов автоматически отключено при запуске через `run_component_bench.py` (флаг `--disable-text-cache` для `core_clip`).

## Известные ограничения

1. **Время выполнения**: полный прогон всех конфигураций может занять несколько часов
2. **VRAM требования**: `raft_512` с большими batch sizes может требовать много VRAM
3. **Docker ресурсы**: частые перезапуски Triton могут нагружать систему
4. **Cleanup требует sudo**: для полной очистки системы нужны права root

## Будущие улучшения

- [ ] Параллельный запуск нескольких конфигураций (с ограничением по GPU)
- [ ] Автоматическое определение оптимального batch_size на основе VRAM
- [ ] Поддержка других компонентов (не только `core_optical_flow`)
- [ ] Интеграция с системой планирования задач
- [ ] Экспорт результатов в различные форматы (CSV, Excel)

---

**Авторы**: AI Assistant + Ilya  
**Дата**: 2026-01-17  
**Версия документа**: 2.0
