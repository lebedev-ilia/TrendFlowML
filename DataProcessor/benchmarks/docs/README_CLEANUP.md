# System Cleanup Script

Скрипт для очистки системы перед запуском бенчмарков. Обеспечивает "чистый старт" для точных измерений.

## Возможности

- **Очистка GPU памяти**: Освобождение GPU памяти и завершение GPU процессов
- **Очистка системного кэша**: Сброс page cache, dentries и inodes (требует root)
- **Синхронизация файловой системы**: Запись всех буферов на диск
- **Очистка Python кэша**: Удаление `__pycache__` директорий

## Использование

### Базовая очистка (без root)

```bash
python benchmarks/cleanup_system.py
```

Выполняет:
- Проверку и очистку GPU памяти (если возможно)
- Синхронизацию файловой системы
- Очистку Python кэша

**Примечание**: Для полной очистки системного кэша требуется root.

### Полная очистка (с root)

**Важно**: При использовании `sudo` нужно указать полный путь к Python или использовать `python3`:

```bash
# Вариант 1: Использовать python3 (рекомендуется)
sudo python3 benchmarks/cleanup_system.py

# Вариант 2: Использовать полный путь к Python из виртуального окружения
sudo /home/ilya/Рабочий\ стол/TrendFlowML/DataProcessor/.data_venv/bin/python benchmarks/cleanup_system.py

# Вариант 3: Активировать venv и использовать sudo -E (сохраняет переменные окружения)
source DataProcessor/.data_venv/bin/activate
sudo -E python benchmarks/cleanup_system.py
```

Выполняет все операции, включая:
- Сброс системного кэша (page cache, dentries, inodes)
- Полную очистку GPU памяти

### Принудительная очистка GPU процессов

Если есть запущенные процессы, использующие GPU:

```bash
# Без root (если есть права на завершение процессов)
python benchmarks/cleanup_system.py --force

# С root для полной очистки (используйте python3 или полный путь)
sudo python3 benchmarks/cleanup_system.py --force
```

Опция `--force` автоматически завершает процессы, использующие GPU.

### Только GPU очистка

Если нужно очистить только GPU:

```bash
python benchmarks/cleanup_system.py --only-gpu
```

Или с принудительным завершением процессов:

```bash
python benchmarks/cleanup_system.py --only-gpu --force

# С root (если нужны дополнительные права):
sudo python3 benchmarks/cleanup_system.py --only-gpu --force
```

### Дополнительные опции

```bash
python benchmarks/cleanup_system.py \
    --skip-python-cache \  # Пропустить очистку Python кэша
    --skip-fsync           # Пропустить синхронизацию файловой системы
```

## Примеры использования

### Перед запуском бенчмарка

```bash
# 1. Очистка системы (используйте python3 или полный путь)
sudo python3 benchmarks/cleanup_system.py --force

# 2. Запуск Triton (в другом терминале)
# ... запуск Triton ...

# 3. Запуск бенчмарка
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --frames-count 1 \
    --triton-http-url http://localhost:8000
```

### Быстрая очистка только GPU

```bash
# Если нужно быстро освободить GPU без полной очистки системы
python benchmarks/cleanup_system.py --only-gpu --force

# С root (если нужны дополнительные права):
sudo python3 benchmarks/cleanup_system.py --only-gpu --force
```

## Что делает скрипт

### 1. Очистка GPU памяти

- Проверяет запущенные процессы, использующие GPU
- Предупреждает о найденных процессах (если не используется `--force`)
- Завершает процессы при использовании `--force`
- Проверяет финальное состояние GPU памяти

### 2. Очистка системного кэша (требует root)

- Сбрасывает page cache (кэш страниц памяти)
- Сбрасывает dentries (кэш структуры каталогов)
- Сбрасывает inodes (кэш метаданных файлов)

**Примечание**: Эта операция безопасна, но может замедлить последующие операции до тех пор, пока кэш не восстановится.

### 3. Синхронизация файловой системы

- Принудительно записывает все буферы на диск
- Обеспечивает, что все данные сохранены

### 4. Очистка Python кэша

- Находит все директории `__pycache__` в проекте
- Удаляет их для освобождения места

## Безопасность

- Скрипт **не удаляет** пользовательские данные
- Скрипт **не удаляет** файлы проекта
- Сброс системного кэша безопасен (кэш восстановится автоматически)
- GPU процессы завершаются только при использовании `--force`

## Требования

- `nvidia-smi` (для очистки GPU) - обычно устанавливается с CUDA
- Python 3.8+ (используйте `python3` при запуске с `sudo`)
- Root права (для полной очистки системного кэша)

## Важные замечания при использовании sudo

При использовании `sudo` переменные окружения (включая виртуальные окружения) не передаются по умолчанию. Рекомендуемые способы:

1. **Использовать `python3`** (если установлен системно):
   ```bash
   sudo python3 benchmarks/cleanup_system.py
   ```

2. **Использовать полный путь к Python**:
   ```bash
   sudo /path/to/venv/bin/python benchmarks/cleanup_system.py
   ```

3. **Использовать `sudo -E`** (сохраняет переменные окружения):
   ```bash
   source DataProcessor/.data_venv/bin/activate
   sudo -E python benchmarks/cleanup_system.py
   ```

## Выходные данные

Скрипт выводит подробную информацию о каждом шаге:

```
======================================================================
System Cleanup for Benchmark Preparation
======================================================================

[cleanup] Clearing GPU memory...
[cleanup]   ✓ GPU memory: 0/8192 MB

[cleanup] Clearing system cache...
[cleanup]   ✓ Dropped page cache, dentries, and inodes

[cleanup] Syncing filesystem...
[cleanup]   ✓ Filesystem synced

[cleanup] Clearing Python cache...
[cleanup]   ✓ Removed 15 __pycache__ directory(ies)

======================================================================
Final Status
======================================================================
[cleanup] GPU Memory Status:
[cleanup]   GPU 0 (NVIDIA GeForce RTX 3090): 0 / 24564 MB

======================================================================
✓ Cleanup completed successfully (4/4 steps)
======================================================================
```

## Интеграция с бенчмарками

Рекомендуется использовать скрипт очистки перед каждым запуском бенчмарка:

```bash
#!/bin/bash
# Пример скрипта для запуска бенчмарка

# Очистка системы (используйте python3 для sudo)
echo "Cleaning system..."
sudo python3 benchmarks/cleanup_system.py --force

# Запуск Triton (в фоне или в другом терминале)
# ...

# Запуск бенчмарка
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path "$1" \
    --frames-count 1 \
    --triton-http-url http://localhost:8000
```

## Примечания

- Очистка системного кэша может занять несколько секунд
- После очистки кэша первые операции могут быть медленнее
- GPU процессы завершаются корректно (SIGTERM, затем SIGKILL при необходимости)
- Скрипт проверяет наличие всех необходимых инструментов перед использованием

