# Конфигурация параллелизма для тестирования AudioProcessor

## Фиксированные настройки для тестов с оптимизациями

Для обеспечения воспроизводимости результатов тестирования используются фиксированные настройки параллелизма из `global_config.yaml`.

### Конфигурация без оптимизаций (baseline)

```yaml
processors:
  audio:
    scheduler:
      segment_parallelism: 1
      max_inflight: null  # null = segment_parallelism
      clap_batch_size: 1
    
    batch_processing:
      enabled: false
      enable_gpu_batching: false
      enable_cpu_parallel: false
      enable_video_parallel: false
```

**Использование**: Базовый тест для измерения времени обработки без оптимизаций.

### Конфигурация с оптимизациями (production-ready)

```yaml
processors:
  audio:
    scheduler:
      segment_parallelism: 16  # Параллелизм для CPU extractors
      max_inflight: null  # null = segment_parallelism
      clap_batch_size: 16  # Размер батча для GPU extractors
    
    batch_processing:
      enabled: true  # Enable batch processing optimizations (GPU batching + CPU parallelism)
      max_video_workers: null  # Number of parallel workers for video-level processing (null = auto, typically os.cpu_count())
      enable_video_parallel: true  # Enable parallel processing of multiple videos
      max_segment_workers: null  # Number of parallel workers for segment-level processing (null = auto, for CPU extractors)
      enable_segment_parallel: true  # Enable parallel processing of segments
      enable_gpu_batching: true  # Use extract_batch() for GPU extractors with supports_batch=true
      max_segments_per_gpu_batch: null  # Limit batch size for GPU extractors (null = no limit)
      enable_cpu_parallel: true  # Parallelize CPU extractors across documents using ThreadPoolExecutor
```

**Использование**: Тест для измерения ускорения от оптимизаций.

## Параметры и их влияние

### Scheduler knobs

- **`segment_parallelism`** (16):
  - Количество параллельных воркеров для CPU extractors на уровне сегментов
  - Влияет на: tempo, loudness, spectral, quality, mfcc, mel, onset, chroma, rhythmic, voice_quality, hpss, key, band_energy, spectral_entropy
  - Увеличение ускоряет обработку CPU-bound extractors

- **`clap_batch_size`** (16):
  - Размер батча для CLAP extractor (GPU batching)
  - Влияет на: clap_extractor
  - Увеличение ускоряет обработку, но требует больше VRAM

- **`max_inflight`** (null):
  - Максимальное количество одновременно выполняемых задач
  - null = равен `segment_parallelism`
  - Ограничивает пиковое использование ресурсов

### Batch processing flags

- **`enable_gpu_batching`** (true):
  - Включает GPU batching для ML моделей (ASR, diarization, emotion, CLAP, source_separation)
  - Группирует сегменты из всех видео в один батч для оптимизации GPU inference
  - Значительно ускоряет обработку GPU extractors

- **`enable_cpu_parallel`** (true):
  - Включает параллелизм CPU extractors через ThreadPoolExecutor
  - Параллелизует обработку сегментов внутри одного видео
  - Ускоряет CPU-bound extractors

- **`enable_video_parallel`** (true):
  - Включает параллелизм на уровне видео (для batch mode)
  - Обрабатывает несколько видео параллельно
  - В single-file mode не используется

- **`max_video_workers`** (null):
  - Количество параллельных воркеров для video-level parallelism
  - null = auto (обычно os.cpu_count())
  - Используется только в batch mode

- **`max_segments_per_gpu_batch`** (null):
  - Ограничение размера батча для GPU extractors
  - null = без ограничений
  - Полезно для ограничения VRAM usage

## Рекомендации по настройке

### Для тестирования (фиксированные значения)

Используйте значения из этого документа для воспроизводимости результатов.

### Для production

- **CPU-bound extractors**: Увеличьте `segment_parallelism` до количества CPU cores
- **GPU extractors**: Увеличьте `clap_batch_size` в зависимости от доступной VRAM
- **Batch mode**: Установите `max_video_workers` в зависимости от количества доступных CPU cores

## Измерение производительности

При тестировании сравнивайте:
1. **Baseline** (без оптимизаций): `segment_parallelism=1`, `clap_batch_size=1`, `batch_processing.enabled=false`
2. **Optimized** (с оптимизациями): `segment_parallelism=16`, `clap_batch_size=16`, `batch_processing.enabled=true`

**Метрики для сравнения**:
- Общее время обработки (`stage_timings_ms.run_extractors_ms`)
- Время каждого extractor'а (`timings_by_extractor.<extractor>.wall_ms`)
- Ускорение = время_baseline / время_optimized

## Примеры конфигураций

### Минимальная (для отладки)
```yaml
scheduler:
  segment_parallelism: 1
  clap_batch_size: 1
batch_processing:
  enabled: false
```

### Средняя (для тестирования)
```yaml
scheduler:
  segment_parallelism: 8
  clap_batch_size: 8
batch_processing:
  enabled: true
  enable_gpu_batching: true
  enable_cpu_parallel: true
```

### Максимальная (production, из global_config.yaml)
```yaml
scheduler:
  segment_parallelism: 16
  clap_batch_size: 16
batch_processing:
  enabled: true
  enable_gpu_batching: true
  enable_cpu_parallel: true
  enable_video_parallel: true
```

