# Руководство по тестированию AudioProcessor

## Обзор

Это руководство описывает процесс тестирования AudioProcessor с поочередным добавлением компонентов и измерением производительности.

## Подготовка к тестам

### 1. Конфигурация через global_config.yaml

Все тесты проводятся через верхний оркестратор `DataProcessor/main.py` с использованием `global_config.yaml`.

**Важно**: 
- Для **single-file mode** (одиночная обработка): используйте `DataProcessor/main.py` с `--global-config`
  - Оптимизации (`batch_processing.enabled: true`) в single-file mode включают GPU batching сегментов и CPU parallelism внутри одного файла
- Для **batch mode** (батчевая обработка нескольких файлов): запускайте напрямую `AudioProcessor/run_cli.py` с `--audio-input-dir` или `--audio-input-list`

### 2. Структура тестирования

1. **Поочередное добавление компонентов**: начните с верха списка в `global_config.yaml` (extractors в порядке: clap, tempo, loudness, asr, ...)
2. **Два режима тестирования** (для single-file mode через DataProcessor/main.py):
   - **Без оптимизаций**: 
     - `batch_processing.enabled: false`
     - `scheduler.segment_parallelism: 1`
     - `scheduler.clap_batch_size: 1`
   - **С оптимизациями**: 
     - `batch_processing.enabled: true` (включает GPU batching сегментов и CPU parallelism внутри одного файла)
     - `scheduler.segment_parallelism: 16` (параллелизм для CPU extractors)
     - `scheduler.clap_batch_size: 16` (размер батча для GPU extractors)
     - `batch_processing.enable_gpu_batching: true` (GPU batching для ML моделей)
     - `batch_processing.enable_cpu_parallel: true` (CPU parallelism для CPU extractors)
     - `batch_processing.enable_video_parallel: true` (параллелизм на уровне видео)

**Примечание**: В single-file mode `batch_processing.enabled: true` не означает обработку нескольких файлов, а включает оптимизации внутри одного файла (GPU batching сегментов, CPU parallelism).

### 3. Фиксация времени

Время обработки сохраняется в:
- **NPZ meta**: `stage_timings_ms` и `timings_by_extractor` (для каждого extractor'а)
- **Batch results**: `processing_time` и `timings.wall_clock.elapsed_s` (для каждого файла в batch mode)
- **Manifest**: `duration_ms` для каждого компонента

### 4. HTML Render

HTML render генерируется автоматически для каждого extractor'а:
- **Single-file mode**: `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/_render/render_context.json`
- **Batch mode**: `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/_render/render_context.json` (для каждого файла)

## Анализ производительности

При тестировании может наблюдаться минимальное ускорение от оптимизаций (особенно на коротких видео). Это нормально и объясняется следующими факторами:

1. **Фиксированное время инициализации модели** (~19% от общего времени) не зависит от `batch_size`
2. **Последовательная загрузка/предобработка сегментов** (~21% от общего времени) не оптимизируется через `clap_batch_size`
3. **Небольшое количество сегментов** на коротких видео (например, 29 сегментов для 28.8 секунд видео)
4. **GPU inference уже быстрый** для одного сегмента

Подробный анализ см. в `docs/PERFORMANCE_ANALYSIS.md`.

## Процесс тестирования

### Шаг 1: Подготовка global_config.yaml

Начните с минимальной конфигурации (только первый extractor):

```yaml
processors:
  audio:
    enabled: true
    required: false
    device: "auto"
    
    scheduler:
      segment_parallelism: 1
      max_inflight: null
      clap_batch_size: 1
    
    # Batch processing: отключено для первого теста
    batch_processing:
      enabled: false
    
    extractors:
      clap:
        enabled: true
      tempo:
        enabled: false
      loudness:
        enabled: false
      # ... остальные extractors disabled
```

### Шаг 2: Тест без оптимизаций

Убедитесь, что в `global_config.yaml`:
```yaml
    batch_processing:
      enabled: false
    scheduler:
      segment_parallelism: 1
      clap_batch_size: 1
```

```bash
python3 DataProcessor/main.py \
  --video-path /path/to/video.mp4 \
  --global-config DataProcessor/configs/global_config.yaml \
  --run-audio \
  --platform-id youtube \
  --video-id test_video_1 \
  --run-id test_run_1_no_optimizations
```

**Что проверять**:
- Время обработки в NPZ meta (`stage_timings_ms`, `timings_by_extractor`)
- Корректность HTML render: `result_store/.../<component_name>/_render/render_context.json`
- Корректность NPZ файла
- Запишите время обработки для сравнения

### Шаг 3: Тест с оптимизациями

Обновите `global_config.yaml`:

```yaml
    scheduler:
      segment_parallelism: 16  # Параллелизм для CPU extractors
      max_inflight: null  # null = segment_parallelism
      clap_batch_size: 16  # Размер батча для GPU extractors
    
    batch_processing:
      enabled: true  # Enable batch processing optimizations (GPU batching + CPU parallelism)
      max_video_workers: null  # Number of parallel workers for video-level processing (null = auto)
      enable_video_parallel: true  # Enable parallel processing of multiple videos
      max_segment_workers: null  # Number of parallel workers for segment-level processing (null = auto)
      enable_segment_parallel: true  # Enable parallel processing of segments
      enable_gpu_batching: true  # Use extract_batch() for GPU extractors with supports_batch=true
      max_segments_per_gpu_batch: null  # Limit batch size for GPU extractors (null = no limit)
      enable_cpu_parallel: true  # Parallelize CPU extractors across documents using ThreadPoolExecutor
```

```bash
python3 DataProcessor/main.py \
  --video-path /path/to/video.mp4 \
  --global-config DataProcessor/configs/global_config.yaml \
  --run-audio \
  --platform-id youtube \
  --video-id test_video_1 \
  --run-id test_run_1_with_optimizations
```

**Что проверять**:
- Время обработки (должно быть меньше, чем без оптимизаций)
- Ускорение: сравните с результатом без оптимизаций
- Корректность HTML render
- Корректность NPZ файла
- Запишите время обработки и ускорение

### Шаг 4: Добавление следующего компонента

Обновите `global_config.yaml`, включив следующий extractor:

```yaml
    extractors:
      clap:
        enabled: true
      tempo:
        enabled: true  # Добавлен
      loudness:
        enabled: false
```

Повторите шаги 2 и 3 для нового набора extractors.

### Шаг 5: Batch mode тестирование (опционально)

Для batch mode запускайте напрямую `AudioProcessor/run_cli.py`:

```bash
# Подготовка: создайте директорию с несколькими frames_dirs
mkdir -p /path/to/batch_frames_dirs
# Скопируйте frames_dirs в эту директорию

# Тест без оптимизаций
python3 AudioProcessor/run_cli.py \
  --audio-input-dir /path/to/batch_frames_dirs \
  --rs-base /path/to/result_store \
  --platform-id youtube \
  --run-id batch_test_1_no_optimizations \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --extractors clap,tempo,loudness \
  --no-batch-gpu \
  --no-batch-cpu-parallel

# Тест с оптимизациями
python3 AudioProcessor/run_cli.py \
  --audio-input-dir /path/to/batch_frames_dirs \
  --rs-base /path/to/result_store \
  --platform-id youtube \
  --run-id batch_test_1_with_optimizations \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --extractors clap,tempo,loudness \
  --batch-max-workers 4
```

## Документирование результатов

### Формат документации

Создайте документ `docs/TEST_RESULTS.md` со следующей структурой:

```markdown
# Результаты тестирования AudioProcessor

## Тестовая конфигурация
- Дата: YYYY-MM-DD
- Видео: <video_id>
- Платформа: youtube
- Устройство: cuda/cpu

## Результаты по компонентам

### clap_extractor

#### Без оптимизаций
- Время обработки: X.XX секунд
- Stage timings:
  - load_input_ms: XX
  - run_extractors_ms: XX
  - save_npz_ms: XX
- Per-extractor timing: XX ms
- HTML render: ✅ / ❌
- NPZ валидация: ✅ / ❌

#### С оптимизациями
- Время обработки: X.XX секунд
- Ускорение: X.XXx
- Stage timings: ...
- HTML render: ✅ / ❌
- NPZ валидация: ✅ / ❌

### tempo_extractor
...
```

### Извлечение метрик времени

**Из NPZ файла**:
```python
import numpy as np

npz = np.load("result_store/.../clap_extractor/clap_extractor_features.npz", allow_pickle=True)
meta = npz["meta"].item() if hasattr(npz["meta"], "item") else npz["meta"]

stage_timings = meta.get("stage_timings_ms", {})
timings_by_extractor = meta.get("timings_by_extractor", {})

print(f"Total time: {stage_timings.get('run_extractors_ms', 0)} ms")
print(f"CLAP time: {timings_by_extractor.get('clap', {}).get('wall_ms', 0)} ms")
```

**Из batch results** (batch mode):
```python
# batch_results содержит список результатов для каждого файла
for result in batch_results:
    file_id = result["file_id"]
    processing_time = result["processing_time"]
    timings = result["timings"]
    print(f"{file_id}: {processing_time} seconds")
```

## Проверка HTML Render

### Расположение файлов

- **Render-context JSON**: `result_store/.../<component_name>/_render/render_context.json`
- **HTML файл** (если генерируется): `result_store/.../<component_name>/_render/render.html`

### Проверка корректности

1. **Проверка существования файла**:
   ```bash
   ls -la result_store/.../<component_name>/_render/render_context.json
   ```

2. **Проверка структуры JSON**:
   ```python
   import json
   
   with open("render_context.json", "r") as f:
       render = json.load(f)
   
   # Проверка обязательных полей
   assert "component" in render
   assert "summary" in render
   assert "timeline" in render
   assert "meta" in render
   ```

3. **Проверка HTML** (если генерируется):
   - Откройте HTML файл в браузере
   - Проверьте корректность отображения графиков и данных
   - Проверьте интерактивные элементы (если есть)

## Troubleshooting

### Проблема: Render не генерируется

**Причина**: Render генерируется только для успешно обработанных extractors.

**Решение**: 
- Проверьте статус extractor'а в NPZ meta (`status="ok"`)
- Проверьте логи на наличие ошибок render

### Проблема: Время не сохраняется

**Причина**: Время сохраняется только при успешной обработке.

**Решение**:
- Проверьте статус в NPZ meta
- Проверьте наличие `stage_timings_ms` и `timings_by_extractor` в meta

### Проблема: Batch mode не работает через DataProcessor/main.py

**Причина**: `DataProcessor/main.py` не поддерживает batch mode напрямую (он всегда передает `--frames-dir` для одного видео).

**Решение**: 
- Для batch mode запускайте напрямую `AudioProcessor/run_cli.py` с `--audio-input-dir` или `--audio-input-list`
- Или используйте single-file mode через `DataProcessor/main.py` для каждого видео отдельно

## Следующие шаги

После завершения тестирования всех компонентов:

1. Создайте сводную таблицу результатов
2. Проанализируйте ускорение от оптимизаций
3. Выявите узкие места производительности
4. Документируйте рекомендации по настройке для production

