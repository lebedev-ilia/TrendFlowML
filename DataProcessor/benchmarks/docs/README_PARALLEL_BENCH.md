# Multi-Threaded Component Benchmark Harness

Этот скрипт позволяет проводить бенчмарки компонентов в нескольких потоках одновременно и измерять системные ресурсы (GPU VRAM, RAM, загрузка CPU и GPU) во время параллельного выполнения.

## Возможности

1. **Параллельный запуск компонента**: Запускает несколько экземпляров компонента одновременно в разных потоках
2. **Мониторинг ресурсов**: Отслеживает использование CPU, GPU, RAM и VRAM во время всех параллельных запусков
3. **Статистика выполнения**: Собирает метрики для каждого потока (длительность, успешность, коды возврата)
4. **Агрегированные отчеты**: Генерирует HTML и JSON отчеты с агрегированными данными
5. **Анализ производительности**: Вычисляет среднюю, минимальную и максимальную длительность, пропускную способность

## Использование

### Базовый пример

```bash
python benchmarks/run_component_parallel_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --threads 4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --batch-size 1
```

**Что произойдет:**
1. Скрипт замерит ресурсы до Triton
2. Попросит запустить Triton вручную (или используйте `--wait-triton` для автоматического ожидания)
3. Замерит ресурсы после Triton
4. Запустит 4 экземпляра компонента `core_clip` одновременно
5. Будет мониторить ресурсы системы во время всех запусков
6. Сгенерирует отчеты с результатами

### Автоматическое ожидание Triton

```bash
python benchmarks/run_component_parallel_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --threads 4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --wait-triton \
    --triton-timeout 300
```

### Обработка полного видео

```bash
python benchmarks/run_component_parallel_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --threads 8 \
    --full-video \
    --triton-http-url http://localhost:8000 \
    --batch-size 16
```

### Использование ModelManager spec'ов

```bash
python benchmarks/run_component_parallel_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --threads 4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --triton-image-model-spec clip_image_224_triton \
    --triton-text-model-spec clip_text_triton \
    --batch-size 1
```

### Пример для core_depth_midas

```bash
python benchmarks/run_component_parallel_bench.py \
    --component core_depth_midas \
    --video-path /path/to/video.mp4 \
    --threads 4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --triton-model-spec midas_384_triton \
    --triton-preprocess-preset midas_384 \
    --batch-size 4
```

Или с явным указанием модели:

```bash
python benchmarks/run_component_parallel_bench.py \
    --component core_depth_midas \
    --video-path /path/to/video.mp4 \
    --threads 4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --triton-model-name midas_384 \
    --triton-preprocess-preset midas_384 \
    --batch-size 4
```

## Параметры командной строки

### Обязательные параметры

- `--component`: Имя компонента (например, `core_clip`)
- `--video-path`: Путь к видео файлу
- `--threads`: Количество параллельных потоков
- `--triton-http-url`: URL Triton сервера (например, `http://localhost:8000`)

### Параметры конфигурации

- `--frames-count`: Количество кадров для обработки (по умолчанию: 1)
- `--full-video`: Обработать все кадры видео
- `--batch-size`: Размер батча для компонента (по умолчанию: 1)
- `--frames-dir`: Путь к директории с кадрами (создается автоматически через Segmenter, если не указан)
- `--out-dir`: Выходная директория (по умолчанию: `benchmarks/out_component_parallel/<timestamp>`)

### Параметры Triton

- `--wait-triton`: Ждать, пока Triton станет доступен (автоматическая проверка)
- `--triton-timeout`: Таймаут ожидания Triton в секундах (по умолчанию: 300)
- `--triton-image-model-spec`: Имя ModelManager spec для image модели
- `--triton-text-model-spec`: Имя ModelManager spec для text модели
- `--triton-image-model-name`: Имя Triton image модели (по умолчанию: `clip_image_224`)
- `--triton-text-model-name`: Имя Triton text модели (по умолчанию: `clip_text`)
- `--triton-preprocess-preset`: Preset для препроцессинга CLIP (`openai_clip_224`, `openai_clip_336`, `openai_clip_448`)
- `--triton-http-timeout-sec`: Таймаут HTTP клиента Triton в секундах (по умолчанию: 60.0)
- `--triton-image-datatype`: Тип данных для image входа (`UINT8` или `FP32`, по умолчанию: `UINT8`)
- `--triton-text-datatype`: Тип данных для text входа (`INT64`, по умолчанию: `INT64`)

### Другие параметры

- `--dp-models-root`: Переменная окружения `DP_MODELS_ROOT` (для загрузки Places365 prompts)

## Результаты

Скрипт генерирует следующие файлы в выходной директории:

1. **`results.json`**: Полные результаты в JSON формате
2. **`report.html`**: HTML отчет с таблицами, графиками и метриками

### Структура JSON результатов

```json
{
  "component": "core_clip",
  "video_path": "/path/to/video.mp4",
  "frames_dir": "/path/to/frames",
  "frames_count": 10,
  "frame_indices": [0, 1, 2, ...],
  "batch_size": 1,
  "num_threads": 4,
  "triton_http_url": "http://localhost:8000",
  "total_wall_time_sec": 12.345,
  "threads": [
    {
      "thread_id": 0,
      "run_id": "thread_0_abc12345",
      "success": true,
      "duration_sec": 3.123,
      "returncode": 0,
      "component_timing": {...}
    },
    ...
  ],
  "statistics": {
    "successful": 4,
    "failed": 0,
    "avg_duration_sec": 3.123,
    "min_duration_sec": 3.045,
    "max_duration_sec": 3.201,
    "throughput_runs_per_sec": 0.324
  },
  "resources": {
    "before_triton": {...},
    "after_triton": {...},
    "after_execution": {...},
    "peaks": {
      "cpu_util_peak_pct": 85.3,
      "ram_used_peak_mb": 10240,
      "gpu_util_peak_pct": 95.2,
      "vram_used_peak_mb": 4096
    },
    "time_series": [...],
    "peak_timestamps": {...}
  },
  "created_at": "2024-01-01T12:00:15"
}
```

### Метрики в отчете

HTML отчет содержит следующие метрики:

1. **Конфигурация теста**:
   - Компонент, путь к видео, количество кадров, размер батча
   - Количество потоков, успешные/неуспешные запуски

2. **Статистика выполнения**:
   - Средняя длительность выполнения
   - Минимальная и максимальная длительность
   - Общее время выполнения (wall time)
   - Пропускная способность (runs/second)

3. **Результаты по потокам**:
   - Таблица с результатами каждого потока
   - Статус (успех/неудача), длительность, код возврата

4. **Пиковое использование ресурсов**:
   - Пиковая загрузка CPU
   - Пиковое использование RAM
   - Пиковая загрузка GPU
   - Пиковое использование GPU VRAM

5. **Графики использования ресурсов**:
   - CPU utilization over time
   - GPU utilization over time
   - Memory usage over time (RAM и VRAM)

## Пример использования

1. **Подготовка**: Убедитесь, что Triton не запущен (или остановите его)

2. **Запуск скрипта**:
   ```bash
   python benchmarks/run_component_parallel_bench.py \
       --component core_clip \
       --video-path test_video.mp4 \
       --threads 4 \
       --frames-count 10 \
       --triton-http-url http://localhost:8000 \
       --batch-size 1
   ```

3. **Скрипт замерит ресурсы до Triton** и выведет:
   ```
   [bench] Step 1: Measuring resources BEFORE Triton startup...
   [bench] GPU VRAM before Triton: 0/8192 MB
   [bench] RAM before Triton: 8192/16384 MB
   ```

4. **Скрипт остановится и попросит запустить Triton**:
   ```
   [bench] ======================================================================
   [bench] Step 2: MANUAL STEP - Start Triton now
   [bench] ----------------------------------------------------------------------
   [bench] Please:
   [bench]   1. Start Triton Inference Server in another terminal
   [bench]   2. Ensure it's available at: http://localhost:8000
   [bench]   3. Press Enter here when Triton is ready
   [bench] ======================================================================
   ```

5. **В другом терминале запустите Triton** и вернитесь к скрипту, нажмите Enter

6. **Скрипт замерит ресурсы после Triton** и запустит компоненты параллельно:
   ```
   [bench] Step 5: Running 4 component instances in parallel with resource monitoring...
   [bench] Thread 0 (thread_0_abc12345): ✓ (3.123s)
   [bench] Thread 1 (thread_1_def67890): ✓ (3.045s)
   [bench] Thread 2 (thread_2_ghi13579): ✓ (3.201s)
   [bench] Thread 3 (thread_3_jkl24680): ✓ (3.156s)
   ```

7. **По завершении** скрипт выведет сводку и создаст отчеты:
   ```
   ================================================================================
   PARALLEL BENCHMARK SUMMARY
   ================================================================================
   Component: core_clip
   Threads: 4
   Frames per thread: 10
   Successful: 4/4
   Total wall time: 3.234 seconds
   Average duration per thread: 3.131 seconds
   Min duration: 3.045 seconds
   Max duration: 3.201 seconds
   Throughput: 1.237 runs/second
   
   Peak Resource Usage:
     CPU utilization: 85.3%
     RAM: 10240 MB
     GPU utilization: 95.2%
     GPU VRAM: 4096 MB
   ================================================================================
   ```

## Поддерживаемые компоненты

В настоящее время поддерживаются:
- `core_clip`: CLIP эмбеддинги для кадров
- `core_depth_midas`: MiDaS depth maps для кадров

Для добавления поддержки других компонентов см. функции `_run_core_clip_thread()` и `_run_core_depth_midas_thread()` в скрипте.

## Отличия от `run_component_bench.py`

- **`run_component_bench.py`**: Запускает компонент один раз и детально измеряет ресурсы на разных этапах (до Triton, после Triton, после компонента)
- **`run_component_parallel_bench.py`**: Запускает компонент в нескольких потоках одновременно и измеряет ресурсы системы во время параллельного выполнения

Оба скрипта дополняют друг друга:
- Используйте `run_component_bench.py` для детального анализа одного запуска
- Используйте `run_component_parallel_bench.py` для анализа масштабируемости и параллельной производительности

## Требования

- Python 3.8+
- Доступ к GPU с nvidia-smi
- Triton Inference Server (запущенный или запускаемый вручную)
- Установленные зависимости из DynamicBatch и VisualProcessor

## Примечания

- Каждый поток использует уникальный `run_id`, поэтому результаты сохраняются в разные директории
- Скрипт использует `EnhancedResourceMonitor` для мониторинга ресурсов во время выполнения
- Фреймы создаются автоматически через Segmenter, если `--frames-dir` не указан
- Результаты сохраняются в `benchmarks/out_component_parallel/<timestamp>` по умолчанию
- HTML отчет можно открыть в браузере для удобного просмотра графиков и метрик

## Анализ результатов

### Пропускная способность

Пропускная способность (throughput) показывает, сколько запусков компонента можно выполнить в секунду при параллельном выполнении. Высокая пропускная способность означает хорошую масштабируемость.

### Использование ресурсов

- **CPU utilization**: Показывает, насколько загружен CPU. Если близко к 100%, возможно, CPU является узким местом.
- **GPU utilization**: Показывает, насколько загружен GPU. Для GPU-интенсивных компонентов должна быть высокой.
- **RAM/VRAM**: Показывает использование памяти. Если пиковые значения близки к максимуму, может потребоваться больше памяти или меньше параллельных потоков.

### Масштабируемость

Сравните результаты с разным количеством потоков:
- Линейное масштабирование: время выполнения уменьшается пропорционально количеству потоков
- Сублинейное масштабирование: узкие места (GPU, память, сеть) ограничивают параллелизм
- Ухудшение производительности: слишком много потоков создают конкуренцию за ресурсы

