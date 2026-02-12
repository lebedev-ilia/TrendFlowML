# Component-Level Benchmark Harness

Этот скрипт позволяет проводить бенчмарки компонентов с детальным мониторингом ресурсов системы.

## Возможности

1. **Фиксация ресурсов до запуска тритона**: GPU VRAM, RAM, загрузка CPU и GPU
2. **Фиксация ресурсов после запуска тритона**: ресурсы системы после запуска Triton
3. **Мониторинг ресурсов во время выполнения компонента**: пиковые значения памяти и загрузки
4. **Фиксация ресурсов после выполнения компонента**: финальные значения ресурсов
5. **Запуск компонента с различными конфигурациями**:
   - Один кадр (`--frames-count 1`)
   - Несколько кадров (`--frames-count 10`)
   - Полное видео (`--full-video`)
6. **Генерация отчетов**: HTML, JSON и таблица в консоли

## Использование

### Процесс работы скрипта

1. **Запуск скрипта**: Скрипт сразу замеряет ресурсы системы (до запуска Triton)
2. **Ручной запуск Triton**: Скрипт останавливается и просит вас запустить Triton вручную
3. **Подтверждение**: После запуска Triton нажимаете Enter, скрипт проверяет доступность
4. **Замер после Triton**: Скрипт фиксирует ресурсы после запуска Triton
5. **Запуск компонента**: Скрипт запускает компонент с мониторингом ресурсов
6. **Результаты**: Генерируются HTML и JSON отчеты

### Базовый пример (ручной запуск Triton)

```bash
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --frames-count 1 \
    --triton-http-url http://localhost:8000 \
    --batch-size 1
```

**Что произойдет:**
1. Скрипт замерит ресурсы до Triton
2. Выведет сообщение: "Please start Triton now, then press Enter..."
3. **В это время вы запускаете Triton в другом терминале**
4. Нажимаете Enter, скрипт проверит доступность Triton
5. Скрипт продолжится: замерит ресурсы после Triton и запустит компонент

### Обработка нескольких кадров

```bash
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --batch-size 8
```

### Обработка полного видео

```bash
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --full-video \
    --triton-http-url http://localhost:8000 \
    --batch-size 8
```

### Автоматическое ожидание Triton

Если вы хотите, чтобы скрипт автоматически ждал запуска Triton:

```bash
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --frames-count 1 \
    --triton-http-url http://localhost:8000 \
    --wait-triton \
    --triton-timeout 300
```

### Использование ModelManager spec'ов

```bash
python benchmarks/run_component_bench.py \
    --component core_clip \
    --video-path /path/to/video.mp4 \
    --frames-count 1 \
    --triton-http-url http://localhost:8000 \
    --triton-image-model-spec clip_image_224_triton \
    --triton-text-model-spec clip_text_triton \
    --batch-size 1
```

### Пример для core_depth_midas

```bash
python benchmarks/run_component_bench.py \
    --component core_depth_midas \
    --video-path /path/to/video.mp4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --triton-model-spec midas_384_triton \
    --triton-preprocess-preset midas_384 \
    --batch-size 4
```

Или с явным указанием модели:

```bash
python benchmarks/run_component_bench.py \
    --component core_depth_midas \
    --video-path /path/to/video.mp4 \
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
- `--triton-http-url`: URL Triton сервера (например, `http://localhost:8000`)

### Параметры конфигурации

- `--frames-count`: Количество кадров для обработки (по умолчанию: 1)
- `--full-video`: Обработать все кадры видео
- `--batch-size`: Размер батча для компонента (по умолчанию: 1)
- `--frames-dir`: Путь к директории с кадрами (создается автоматически через Segmenter, если не указан)
- `--out-dir`: Выходная директория (по умолчанию: `benchmarks/out_component/<timestamp>`)

### Параметры Triton

- `--wait-triton`: Ждать, пока Triton станет доступен (автоматическая проверка)
- `--triton-timeout`: Таймаут ожидания Triton в секундах (по умолчанию: 300)
- `--triton-image-model-spec`: Имя ModelManager spec для image модели (для core_clip)
- `--triton-text-model-spec`: Имя ModelManager spec для text модели (для core_clip)

## Результаты

Скрипт генерирует следующие файлы в выходной директории:

1. **`results.json`**: Полные результаты в JSON формате
2. **`report.html`**: HTML отчет с таблицами и метриками

### Структура JSON результатов

```json
{
  "component": "core_clip",
  "video_path": "/path/to/video.mp4",
  "frames_dir": "/path/to/frames",
  "frames_count": 1,
  "frame_indices": [0],
  "batch_size": 1,
  "triton_http_url": "http://localhost:8000",
  "component_duration_sec": 0.123,
  "component_success": true,
  "component_returncode": 0,
  "resources": {
    "before_triton": {
      "timestamp_iso": "2024-01-01T12:00:00",
      "cpu_mem_total_mb": 16384,
      "cpu_mem_used_mb": 8192,
      "cpu_mem_free_mb": 8192,
      "gpu_mem_total_mb": 8192,
      "gpu_mem_used_mb": 0,
      "gpu_mem_free_mb": 8192,
      "cpu_util_pct": 10.5,
      "gpu_util_pct": 0.0
    },
    "after_triton": {
      "timestamp_iso": "2024-01-01T12:00:05",
      "gpu_mem_used_mb": 2048,
      ...
    },
    "after_component": {
      "timestamp_iso": "2024-01-01T12:00:10",
      ...
    },
    "peaks": {
      "cpu_util_peak_pct": 85.3,
      "ram_used_peak_mb": 10240,
      "gpu_util_peak_pct": 95.2,
      "vram_used_peak_mb": 4096
    }
  },
  "created_at": "2024-01-01T12:00:15"
}
```

### Метрики в отчете

HTML отчет содержит следующие метрики:

1. **Ресурсы до запуска Triton**:
   - GPU VRAM (использовано / всего)
   - GPU загрузка (%)
   - RAM (использовано / всего)
   - CPU загрузка (%)

2. **Ресурсы после запуска Triton**:
   - GPU VRAM (использовано / всего)
   - GPU загрузка (%)
   - RAM (использовано / всего)
   - CPU загрузка (%)

3. **Ресурсы после выполнения компонента**:
   - GPU VRAM (использовано / всего)
   - GPU загрузка (%)
   - RAM (использовано / всего)
   - CPU загрузка (%)

4. **Пиковые ресурсы во время выполнения компонента**:
   - Пиковая GPU VRAM
   - Пиковая GPU загрузка
   - Пиковая RAM
   - Пиковая CPU загрузка

5. **Время выполнения компонента**:
   - Длительность выполнения (секунды)
   - Успешность выполнения
   - Код возврата

6. **Таблица сравнения**:
   - Сравнение всех метрик на разных этапах
   - Дельты между этапами (Triton overhead, Component overhead)

## Пример использования

1. **Подготовка**: Убедитесь, что Triton не запущен (или остановите его)

2. **Запуск скрипта**:
   ```bash
   python benchmarks/run_component_bench.py \
       --component core_clip \
       --video-path test_video.mp4 \
       --frames-count 1 \
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

5. **В другом терминале запустите Triton**:
   ```bash
   # Например:
   docker run --gpus all -p 8000:8000 -v /models:/models nvcr.io/nvidia/tritonserver:latest tritonserver --model-repository=/models
   # или
   tritonserver --model-repository=/path/to/models
   ```

6. **Вернитесь к скрипту и нажмите Enter** - скрипт проверит доступность Triton:
   ```
   [bench] Verifying Triton is available...
   [bench] ✓ Triton is available!
   ```

7. **Скрипт замерит ресурсы после Triton** и запустит компонент

7. **По завершении** скрипт выведет сводку и создаст отчеты:
   ```
   ================================================================================
   BENCHMARK SUMMARY
   ================================================================================
   Component: core_clip
   Frames processed: 1
   Component duration: 0.123 seconds
   
   Resource Metrics:
     GPU VRAM before Triton: 0/8192 MB
     GPU VRAM after Triton: 2048/8192 MB
     GPU VRAM after component: 2048/8192 MB
     GPU VRAM peak: 4096 MB
     ...
   ================================================================================
   ```

## Поддерживаемые компоненты

В настоящее время поддерживаются:
- `core_clip`: CLIP эмбеддинги для кадров
- `core_depth_midas`: MiDaS depth maps для кадров
- `core_optical_flow`: Optical flow motion curve для кадров (требует минимум 2 кадра)

Для добавления поддержки других компонентов см. функцию `_run_component()` в скрипте.

### Пример для core_optical_flow

```bash
python benchmarks/run_component_bench.py \
    --component core_optical_flow \
    --video-path /path/to/video.mp4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --triton-model-spec raft_256_triton \
    --triton-preprocess-preset raft_256 \
    --batch-size 8
```

Или с явным указанием модели:

```bash
python benchmarks/run_component_bench.py \
    --component core_optical_flow \
    --video-path /path/to/video.mp4 \
    --frames-count 10 \
    --triton-http-url http://localhost:8000 \
    --triton-model-name raft_256 \
    --triton-preprocess-preset raft_256 \
    --batch-size 8
```

**Важно для core_optical_flow:**
- Требуется минимум 2 кадра (для вычисления flow между парами кадров)
- Если указано `--frames-count 1`, скрипт автоматически увеличит до 2
- Поддерживаемые пресеты: `raft_256`, `raft_384`, `raft_512`
- Batch size указывает количество пар кадров в одном запросе Triton

## Требования

- Python 3.8+
- Доступ к GPU с nvidia-smi
- Triton Inference Server (запущенный или запускаемый вручную)
- Установленные зависимости из DynamicBatch и VisualProcessor

## Примечания

- Скрипт использует `ResourceMonitor` из DynamicBatch для мониторинга ресурсов во время выполнения
- Фреймы создаются автоматически через Segmenter, если `--frames-dir` не указан
- Результаты сохраняются в `benchmarks/out_component/<timestamp>` по умолчанию
- HTML отчет можно открыть в браузере для удобного просмотра

