## `voice_quality_extractor` (Voice quality metrics)

### Назначение

Извлекает **метрики качества голоса** для оценки стабильности и гармоничности голоса. Использует прокси-метрики jitter, shimmer и HNR-подобную метрику на основе оценки f0 и анализа амплитуды.

**Версия**: 3.0.1  
**Категория**: voice  
**GPU**: опционально (torchcrepe может использовать CUDA для ускорения f0 estimation)  
**schema_version**: `voice_quality_extractor_npz_v1` (см. `SCHEMA.md`, `schemas/voice_quality_extractor_npz_v1.json`)

### Входы

- **`audio/audio.wav`** (Segmenter contract) — полное аудио для `run()`
- **`audio/segments.json`** (Segmenter contract, family `voice_quality`) — сегменты для `run_segments()`

### Выходы

Схема NPZ: `voice_quality_extractor_npz_v1` (`schemas/…`, `docs/SCHEMA.md`).

#### Audit v4 — заметки по NPZ

- На reference **A**: tabular **F=30**, **1 NaN** по **`f0_method`** (строка в float) — **исправлено**: строка только в **`meta`**; после перезапуска **F=29**.
- **`device_used`**: в **meta**, не в tabular (савер не добавляет).
- Observability (audit v4.2): `meta.stage_timings_ms`, `meta.voice_quality_resource_profile` (env: `AP_VOICE_QUALITY_RESOURCE_PROFILE=1`)

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Jitter метрики (feature-gated: `--voice-quality-enable-jitter`)

- **`vq_jitter`**: вариативность основной частоты (f0)
  - Вычисляется как стандартное отклонение разностей f0, нормализованное на среднее f0
  - Низкие значения → стабильный голос
  - Высокие значения → нестабильный голос (дрожание)
  - Единицы: безразмерная величина (нормализованная)
  - Диапазон: [0.0, 1.0] (типично)

- **`vq_jitter_mean`**: среднее абсолютных разностей f0
- **`vq_jitter_std`**: стандартное отклонение разностей f0
- **`vq_jitter_min`**: минимальная абсолютная разность f0
- **`vq_jitter_max`**: максимальная абсолютная разность f0

#### Shimmer метрики (feature-gated: `--voice-quality-enable-shimmer`)

- **`vq_shimmer`**: вариативность амплитуды
  - Вычисляется как стандартное отклонение разностей амплитуд по окнам, нормализованное на среднюю амплитуду
  - Низкие значения → стабильная амплитуда
  - Высокие значения → нестабильная амплитуда (мерцание)
  - Единицы: безразмерная величина (нормализованная)
  - Диапазон: [0.0, 1.0] (типично)

- **`vq_shimmer_mean`**: среднее абсолютных разностей амплитуд
- **`vq_shimmer_std`**: стандартное отклонение разностей амплитуд
- **`vq_shimmer_min`**: минимальная абсолютная разность амплитуд
- **`vq_shimmer_max`**: максимальная абсолютная разность амплитуд

#### HNR метрики (feature-gated: `--voice-quality-enable-hnr`)

- **`vq_hnr_like_db`**: HNR-подобная метрика (Harmonic-to-Noise Ratio)
  - Вычисляется как отношение энергий автокорреляции (lag1) к нулевой лаг
  - Выражается в децибелах (dB)
  - Высокие значения → более гармоничный звук (меньше шума)
  - Низкие значения → более шумный звук
  - Единицы: децибелы (dB)
  - Диапазон: [-100.0, 100.0] (типично)

- **`vq_hnr_mean`**: среднее HNR по окнам
- **`vq_hnr_std`**: стандартное отклонение HNR по окнам
- **`vq_hnr_min`**: минимальное HNR по окнам
- **`vq_hnr_max`**: максимальное HNR по окнам

#### F0 статистики (feature-gated: `--voice-quality-enable-f0-stats`)

- **`vq_f0_mean`**: среднее значение f0 (Hz)
- **`vq_f0_std`**: стандартное отклонение f0 (Hz)
- **`vq_f0_min`**: минимальное значение f0 (Hz)
- **`vq_f0_max`**: максимальное значение f0 (Hz)
- **`vq_f0_median`**: медиана f0 (Hz)
- **`vq_f0_stability`**: стабильность f0 (коэффициент вариации, нормализованный: `1.0 / (1.0 + std/mean)`)
  - Диапазон: [0.0, 1.0]
  - Высокие значения → более стабильный голос
- **`vq_voice_presence_ratio`**: доля времени с присутствием голоса (f0 > 0)
  - Диапазон: [0.0, 1.0]

#### Quality scores (всегда включены, если включены jitter, shimmer и HNR)

- **`vq_voice_quality_score`**: композитная оценка качества голоса (0.0-1.0)
  - Комбинация jitter, shimmer и HNR
  - Формула: `(1.0 - jitter) * 0.33 + (1.0 - shimmer) * 0.33 + hnr_norm * 0.34`
  - Высокие значения → лучше качество голоса

- **`vq_breathiness_score`**: оценка "дыхательности" голоса (0.0-1.0)
  - На основе HNR: низкий HNR = более "дыхательный" голос
  - Формула: `max(0.0, min(1.0, (50.0 - hnr) / 100.0))`
  - Высокие значения → более "дыхательный" голос

#### Временные серии (feature-gated: `--voice-quality-enable-time-series`)

- **`f0`**: временная серия f0 значений (float32[]) или путь к `.npy` файлу (`f0_npy`)
- **`amps`**: временная серия амплитуд по окнам (float32[]) или путь к `.npy` файлу (`amps_npy`)
- **`hnr_vals`**: временная серия HNR значений по окнам (float32[]) или путь к `.npy` файлу (`hnr_vals_npy`)
- **`segment_centers_sec`**: центры сегментов в секундах (float32[], для `run_segments()`)
- **`segment_durations_sec`**: длительности сегментов в секундах (float32[], для `run_segments()`)

**Примечание**: Большие временные серии (>10000 элементов) сохраняются в `.npy` файлы в `_artifacts/` и регистрируются в `manifest.json`.

#### Метаданные

- `device_used`: устройство обработки (`"cpu"`, `"cuda"` или `"auto"`)
- `sample_rate`: частота дискретизации аудио (Hz)
- `duration`: длительность аудио (секунды)
- `segments_count`: количество сегментов (для `run_segments()`)
- `f0_method`: метод оценки f0 (`"yin"`, `"pyin"`, `"torchcrepe"`) — в **NPZ** в **`meta`**, не в `feature_values`
- `f0_fmin`: минимальная частота f0 для оценки (Hz)
- `f0_fmax`: максимальная частота f0 для оценки (Hz)
- `hnr_frame_ms`: размер окна для HNR вычисления (миллисекунды)
- `rms_mask_threshold`: порог RMS для маскирования тихих участков
- `average_channels`: использовалось ли усреднение каналов
- `voice_quality_contract_version`: версия контракта (`"voice_quality_contract_v1"`)
- `_features_enabled`: список включённых групп фичей (для отладки)

### Feature Dependencies

- **Jitter** зависит от оценки f0 (требует `f0_method` и `f0_fmin`/`f0_fmax`)
- **Shimmer** не зависит от других фичей
- **HNR** не зависит от других фичей
- **F0 stats** зависит от оценки f0 (требует `f0_method` и `f0_fmin`/`f0_fmax`)
- **Quality scores** зависят от jitter, shimmer и HNR (все три должны быть включены)
- **Time series** зависит от включённых фичей (f0, amps, hnr_vals)

**Опциональная интеграция с `pitch_extractor`**:
- Если `pitch_extractor` выполнен перед `voice_quality_extractor`, его результаты f0 могут быть использованы вместо собственной оценки f0
- Работает при совпадении families (voice_quality и pitch используют одни сегменты) или для `run()` (full audio)
- При несовпадении сегментов — своя оценка f0 для каждого сегмента

### Render (dev-only)

Offline HTML render: vanilla `<canvas>`, без CDN. График jitter vs segment_center_sec.

### Алгоритмы

#### 1. Оценка f0 (fail-fast, no-fallback)

1. **Опциональная интеграция с pitch_extractor**: если доступны результаты `pitch_extractor`, используются их f0 значения
2. **Самостоятельная оценка f0**: если интеграция недоступна, используется выбранный метод:
   - **YIN** (librosa): быстрый, подходит для речи
   - **PYIN** (librosa): более точный, учитывает voiced/unvoiced frames
   - **torchcrepe**: наиболее точный, требует GPU (опционально)
3. **Фильтрация**: удаление NaN и отрицательных значений
4. **Fail-fast**: если оценка f0 не удалась или нет валидных значений → `RuntimeError`

#### 2. Jitter (вариативность f0)

1. **Вычисление разностей**: `df0 = diff(f0)`
2. **Нормализация**: `jitter = std(df0) / (mean(f0) + epsilon)`
3. **Дополнительные метрики**: mean, std, min, max абсолютных разностей

#### 3. Shimmer (вариативность амплитуды)

1. **Разбиение на окна**: размер окна = 30 мс, шаг = 10 мс
2. **Вычисление амплитуд**: RMS для каждого окна
3. **Маскирование**: исключение окнов с RMS < `rms_mask_threshold`
4. **Вычисление shimmer**: `shimmer = std(diff(amps_masked)) / (mean(amps_masked) + epsilon)`
5. **Дополнительные метрики**: mean, std, min, max абсолютных разностей

#### 4. HNR-подобная метрика

1. **Разбиение на окна**: размер окна = `hnr_frame_ms` (по умолчанию 40 мс)
2. **Автокорреляция**: для каждого окна вычисляется автокорреляция
3. **Вычисление HNR**: `hnr = 20 * log10(|r1| / r0 + epsilon)`, где r0 = lag0, r1 = lag1
4. **Усреднение**: среднее значение по всем окнам
5. **Дополнительные метрики**: mean, std, min, max HNR по окнам

### Конфигурация

#### Через global_config.yaml

```yaml
processors:
  audio:
    extractors:
      voice_quality:
        enabled: false
        sample_rate: 16000
        hnr_frame_ms: 20.0
        rms_mask_threshold: 1e-4
        f0_fmin: 50.0
        f0_fmax: 500.0
        f0_method: "pyin"  # pyin|yin|torchcrepe
        average_channels: true
        feature_flags:
          enable_audio_normalization: false
          enable_jitter: false
          enable_shimmer: false
          enable_hnr: false
          enable_f0_stats: false
          enable_time_series: false
```

#### Через Python API

```python
{
    "device": "auto",                          # "auto" | "cuda" | "cpu"
    "sample_rate": 22050,                      # Частота дискретизации (Hz)
    "average_channels": True,                  # Усреднять ли каналы для многоканального аудио
    "hnr_frame_ms": 40.0,                     # Размер окна для HNR (миллисекунды)
    "rms_mask_threshold": 0.01,                # Порог RMS для маскирования тихих участков
    "f0_fmin": 50.0,                           # Минимальная частота f0 (Hz)
    "f0_fmax": 500.0,                          # Максимальная частота f0 (Hz)
    "f0_method": "yin",                        # "yin" | "pyin" | "torchcrepe"
    "enable_audio_normalization": False,        # Нормализация аудио перед обработкой
    "enable_jitter": False,                     # Включить jitter метрики
    "enable_shimmer": False,                   # Включить shimmer метрики
    "enable_hnr": False,                       # Включить HNR метрики
    "enable_f0_stats": False,                  # Включить f0 статистики
    "enable_time_series": False,               # Включить временные серии
}
```

### Feature Gating (Audit v3)

Базовый preset по умолчанию: jitter + shimmer + hnr включены.

- `--voice-quality-enable-jitter` (default: True): jitter метрики
- `--voice-quality-disable-jitter`: отключить jitter
- `--voice-quality-enable-shimmer` (default: True): shimmer метрики
- `--voice-quality-disable-shimmer`: отключить shimmer
- `--voice-quality-enable-hnr` (default: True): HNR метрики
- `--voice-quality-disable-hnr`: отключить HNR
- `--voice-quality-enable-f0-stats`: включить f0 статистики
- `--voice-quality-enable-time-series`: включить временные серии

### Параметры

#### Основные параметры

- **`sample_rate`** (int, default: 22050): Частота дискретизации для обработки (Hz)
  - Влияние на скорость: меньшие значения → быстрее обработка, но менее точные результаты
  - Влияние на стоимость: линейная зависимость от sample_rate
  - Рекомендуется: 16000-22050 Hz для речи, 44100-48000 Hz для музыки
- **`average_channels`** (bool, default: True): Усреднять каналы для многоканального аудио
  - Влияние на скорость: незначительное (только при загрузке аудио)
  - Влияние на стоимость: незначительное
- **`hnr_frame_ms`** (float, default: 40.0): Размер окна для HNR вычисления (миллисекунды)
  - Влияние на скорость: большие значения → меньше окон → быстрее (~-10% latency при увеличении в 2 раза)
  - Влияние на стоимость: обратно пропорционально количеству окон
  - Диапазон: [10.0, 100.0] мс
- **`rms_mask_threshold`** (float, default: 0.01): Порог RMS для маскирования тихих участков
  - Влияние на скорость: незначительное (только фильтрация)
  - Влияние на стоимость: незначительное
  - Диапазон: [0.0, 1.0]

#### Параметры f0 оценки

- **`f0_fmin`** (float, default: 50.0): Минимальная частота f0 для оценки (Hz)
  - Диапазон: [20.0, f0_fmax)
  - Типично для речи: 50-100 Hz
  - Влияние на скорость: незначительное (только на диапазон поиска)
  - Влияние на стоимость: незначительное
- **`f0_fmax`** (float, default: 500.0): Максимальная частота f0 для оценки (Hz)
  - Диапазон: (f0_fmin, 2000.0]
  - Типично для речи: 300-500 Hz
  - Влияние на скорость: незначительное (только на диапазон поиска)
  - Влияние на стоимость: незначительное
- **`f0_method`** (str, default: "yin"): Метод оценки f0
  - `"yin"`: быстрый, подходит для речи (librosa, CPU-only)
    - Δ latency: baseline (~5-10 ms/segment для коротких сегментов)
    - Δ cost: baseline
    - Рекомендуется для: быстрой обработки, когда точность не критична
  - `"pyin"`: более точный, учитывает voiced/unvoiced frames (librosa, CPU-only)
    - Δ latency: +50-100% относительно YIN (~10-20 ms/segment)
    - Δ cost: +50-100% относительно YIN
    - Рекомендуется для: баланса скорости и точности
  - `"torchcrepe"`: наиболее точный, требует torchcrepe (опционально GPU)
    - Δ latency: 
      - CPU: +200-300% относительно YIN (~20-40 ms/segment)
      - GPU (CUDA): +50-100% относительно YIN (~10-20 ms/segment, с ускорением на GPU)
    - Δ cost: +200-300% относительно YIN (может использовать GPU VRAM ~100-500MB)
    - Рекомендуется для: максимальной точности, особенно с GPU
    - **Важно**: требует `torchcrepe` пакет, автоматически использует CUDA если доступно

#### Опциональные параметры

- **`torchcrepe_model`** (str, default: "tiny"): Модель для torchcrepe (только если `f0_method="torchcrepe"`)
  - `"tiny"`: быстрая модель (~2x быстрее full, менее точная)
    - Δ latency: baseline для torchcrepe (~10-20 ms/segment на GPU)
    - Δ cost: baseline для torchcrepe (~100MB VRAM)
  - `"full"`: точная модель (медленнее, но точнее)
    - Δ latency: +100% относительно tiny (~20-40 ms/segment на GPU)
    - Δ cost: +200% относительно tiny (~300MB VRAM)
  - Рекомендуется: `"tiny"` для production, `"full"` для максимальной точности

- **`enable_audio_normalization`** (bool, default: False): Нормализация аудио перед обработкой
  - Влияние на скорость: незначительное (~+1-2% latency)
  - Влияние на стоимость: незначительное (~+1-2% cost)
- **`pitch_payload`** (Optional[Dict], default: None): Результаты от `pitch_extractor` для использования их f0
  - Влияние на скорость: значительное ускорение (~-50% latency, если pitch_extractor уже выполнен)
  - Влияние на стоимость: экономия на повторной оценке f0 (~-50% cost)
  - Рекомендуется: использовать интеграцию с pitch_extractor для оптимизации
- **`progress_callback`** (Optional[Callable], default: None): Callback для отображения прогресса обработки
  - Формат: `(metric_name: str, current: int, total: int, message: str) -> None`
  - Для `run()`: обновляется на этапах загрузки аудио, оценки f0, вычисления метрик, сохранения артефактов
  - Для `run_segments()`: обновляется каждые 10% сегментов с количеством обработанных сегментов
  - Влияние на скорость: незначительное (только вызов callback)
  - Влияние на стоимость: незначительное
- **`artifacts_dir`** (Optional[str], default: None): Директория для сохранения `.npy` файлов (per-run storage)
  - Используется для сохранения больших временных серий (>10000 элементов) в `.npy` файлы
  - Если не указан, большие массивы сохраняются напрямую в payload (может увеличить размер NPZ)
  - Рекомендуется: указывать для batch processing с изоляцией артефактов между файлами

#### Feature flags (влияние на скорость и стоимость)

- **`enable_jitter`** (bool, default: False): Включить jitter метрики
  - Δ latency: +5-10% (вычисление разностей f0)
  - Δ cost: +5-10%
- **`enable_shimmer`** (bool, default: False): Включить shimmer метрики
  - Δ latency: +10-15% (оконные операции для амплитуд)
  - Δ cost: +10-15%
- **`enable_hnr`** (bool, default: False): Включить HNR метрики
  - Δ latency: +15-20% (автокорреляция по окнам)
  - Δ cost: +15-20%
- **`enable_f0_stats`** (bool, default: False): Включить f0 статистики
  - Δ latency: +2-5% (статистические вычисления)
  - Δ cost: +2-5%
- **`enable_time_series`** (bool, default: False): Включить временные серии
  - Δ latency: +5-10% (сохранение временных серий)
  - Δ cost: +10-20% (дополнительные .npy файлы, если >10000 элементов)
  - Влияние на NPZ size: значительное увеличение (может быть >10x при больших сериях)

### Progress Reporting

`voice_quality_extractor` поддерживает `progress_callback` для отображения прогресса обработки:

- **Для `run()`**: прогресс обновляется на этапах:
  - Загрузка аудио (0/6)
  - Оценка f0 (1/6)
  - Вычисление метрик (2/6)
  - Сохранение артефактов (5/6)
- **Для `run_segments()`**: прогресс обновляется каждые 10% сегментов
  - Отображается количество обработанных сегментов и процент выполнения
  - Поддерживается как последовательная, так и параллельная обработка
- **Формат callback**: `(metric_name: str, current: int, total: int, message: str) -> None`

### Обработка ошибок

Экстрактор использует **no-fallback policy** (fail-fast):

- **Отсутствие аудио**: `RuntimeError` с error_code `voice_quality_audio_load_failed`
- **Ошибка оценки f0**: `RuntimeError` с error_code `voice_quality_f0_estimation_failed`
- **Ошибка librosa**: `RuntimeError` с error_code `voice_quality_librosa_failed`
- **Недостаточно данных**: `RuntimeError` с error_code `voice_quality_insufficient_data`
- **Ошибка валидации**: `RuntimeError` с error_code `voice_quality_validation_failed`
- **Неизвестная ошибка**: `RuntimeError` с error_code `voice_quality_unknown`

### Валидация

#### Валидация параметров (fail-fast)

- `sample_rate > 0`
- `hnr_frame_ms > 0`
- `rms_mask_threshold >= 0`
- `f0_fmin > 0` и `f0_fmin < f0_fmax`
- `f0_fmin >= 20.0` и `f0_fmax <= 2000.0`
- `f0_method` ∈ `["yin", "pyin", "torchcrepe"]`

#### Валидация выходных данных

- Проверка на NaN/inf во всех метриках
- Проверка диапазонов: jitter, shimmer ∈ [0.0, 1.0]
- Проверка HNR на экстремальные значения (warning, не error)

### Parallelization

#### Внутренний параллелизм (внутри компонента)

- **Segment-level parallelism**: поддерживается через `segment_parallelism` и `max_inflight` (для `run_segments()`)
  - Использует `ThreadPoolExecutor` для параллельной обработки сегментов внутри одного файла
  - Количество потоков контролируется параметром `segment_parallelism` (по умолчанию 1)
  - Thread-safety: экстрактор thread-safe для параллельной обработки сегментов
  - Ограничения: параллелизм ограничен I/O операциями (загрузка аудио) и CPU вычислениями (f0 оценка, метрики)

#### Внешний параллелизм (выше компонента)

- **Batch processing**: поддерживается через `extract_batch_segments()` с CPU parallelism для обработки сегментов из нескольких видео одновременно
  - Использует `ThreadPoolExecutor` для параллельной обработки файлов
  - Количество воркеров контролируется параметром `max_workers` (по умолчанию `os.cpu_count()`)
  - Каждый файл обрабатывается изолированно через `run_segments()`
  - Изоляция артефактов: каждый файл имеет свой `artifacts_dir` для сохранения `.npy` файлов
  - Требования к изоляции: разные `run_id`, разные `result_store` пути, разные `artifacts_dir`
  - Ограничения: нет shared mutable state между файлами (кроме read-only моделей/корпусов)

#### Комбинированный подход

- Внутренний параллелизм (сегменты внутри файла) + внешний параллелизм (несколько файлов)
- Рекомендуется: использовать внешний параллелизм для обработки нескольких файлов, внутренний параллелизм для обработки сегментов внутри одного файла
- GPU: не используется (CPU-only)

### Performance characteristics

**Resource costs**:
- **CPU**: низкие-умеренные (YIN/PYIN, оконные операции, автокорреляция)
- **GPU**: не используется
- **Estimated duration**: ~1.5 секунды для типичного аудио файла (полное аудио)
- **Per-segment**: ~0.1-0.2 секунды на сегмент (зависит от размера сегмента)

**Параметры производительности**:
- `hnr_frame_ms`: большие значения → меньше окон → быстрее, но менее детально
- `rms_mask_threshold`: влияет на количество обрабатываемых данных
- `f0_method`: YIN быстрее PYIN, torchcrepe медленнее (но точнее)
- Размер аудио: линейная зависимость от длительности

**Batch processing**:
- **CPU parallelism**: поддерживается через `extract_batch_segments()` с `ThreadPoolExecutor`
- **Масштабирование**: линейное ускорение при увеличении количества CPU ядер (до лимита I/O)
- **Изоляция**: каждый файл обрабатывается изолированно, нет shared mutable state между файлами
- **Артефакты**: каждый файл имеет свой `artifacts_dir` для сохранения `.npy` файлов (per-run storage)

**Критически важные оптимизации для производительности**:
- **Segment parallelism**: **ОБЯЗАТЕЛЬНО** для ускорения обработки
  - По умолчанию: `segment_parallelism=1` (последовательная обработка) → **очень медленно**
  - Рекомендуется: `segment_parallelism=4-8` для CPU, `segment_parallelism=2-4` для GPU (torchcrepe)
  - Ожидаемое ускорение: **~4-8x** при `segment_parallelism=8` на многоядерном CPU
  - Пример: 29 сегментов × 6.7 сек = 194 сек → с `segment_parallelism=8`: ~25-30 сек
- **GPU ускорение**: доступно через `torchcrepe` с `f0_method="torchcrepe"` и `device="cuda"`
  - Автоматическое использование CUDA если доступно
  - Ускорение: **~2-4x** относительно CPU для torchcrepe
  - VRAM: ~100MB (tiny модель) или ~300MB (full модель)
  - Рекомендация: использовать `torchcrepe_model="tiny"` для production (быстрее в 2x)
- **Интеграция с pitch_extractor**: рекомендуется для избежания повторной оценки f0
  - Ускорение: **~50%** если pitch_extractor уже выполнен
  - Экономия: избежание дублирования вычислений f0
  - Настройка: запустить `pitch_extractor` перед `voice_quality_extractor`

### Quality validation

#### Sanity checks

- Jitter, shimmer ∈ [0.0, 1.0]
- HNR ∈ [-100.0, 100.0] (типично)
- F0 stats: f0_min ≤ f0_mean ≤ f0_max
- Voice presence ratio ∈ [0.0, 1.0]
- Quality scores ∈ [0.0, 1.0]

#### Статистические инварианты

- Jitter и shimmer нормализованы (безразмерные)
- HNR выражается в dB
- F0 stability нормализован к [0.0, 1.0]

### Visualization

Рекомендуемые типы визуализации для UI/сайта:

1. **Timeline графики**:
   - F0 timeline: временная серия f0 значений (line chart)
   - Amps timeline: временная серия амплитуд (line chart)
   - HNR timeline: временная серия HNR значений (line chart)

2. **Распределения**:
   - F0 distribution: гистограмма распределения f0 значений
   - Jitter/Shimmer distribution: гистограмма для сегментов (для `run_segments()`)

3. **Метрики**:
   - Jitter, shimmer, HNR: bar charts или gauge charts
   - Quality scores: gauge charts или progress bars
   - F0 stats: box plots или violin plots

4. **Интерактивные элементы**:
   - Tooltips с детальной информацией о метриках
   - Zoom для timeline графиков
   - Фильтры по сегментам (для `run_segments()`)

5. **Сравнение**:
   - Side-by-side сравнение метрик между сегментами
   - Heatmap для сегментных метрик (для `run_segments()`)

**Примеры визуализаций**:
- F0 timeline: line chart с временной осью (секунды) и осью частоты (Hz)
- Quality scores: gauge chart с диапазоном [0.0, 1.0]
- Jitter/Shimmer: bar chart с метриками по сегментам

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **librosa**: библиотека для YIN/PYIN алгоритмов оценки f0
- **torchcrepe** (опционально): библиотека для точной оценки f0
- **pitch_extractor** (опционально): интеграция для использования более точных оценок f0

### Различия с другими extractors

- **`quality_extractor`**: фокусируется на техническом качестве аудио (DC offset, clipping, SNR), а не на качестве голоса
- **`pitch_extractor`**: фокусируется на оценке f0, а не на метриках качества голоса (jitter/shimmer/HNR)
- **`voice_quality_extractor`**: фокусируется на качестве голоса (jitter, shimmer, HNR) на основе f0 и амплитуды

### Примечания

1. **Прокси-метрики**: это не точные медицинские метрики jitter/shimmer/HNR, а упрощенные прокси для быстрой оценки
2. **Методы f0**: YIN быстрый, PYIN точнее, torchcrepe наиболее точный (но медленнее)
3. **Маскирование**: тихие участки исключаются для более точной оценки метрик качества голоса
4. **Нормализация**: все метрики нормализованы для сравнения между записями разной громкости
5. **Диапазон f0**: по умолчанию 50-500 Hz (типичный для речи), может быть настроен для других типов голоса/музыки
6. **Окна**: размеры окон оптимизированы для речи, могут потребовать настройки для других типов аудио
7. **HNR-подобная**: это упрощенная версия классической HNR метрики, использует только lag1 автокорреляции
8. **Интеграция с pitch_extractor**: позволяет использовать более точные оценки f0 из `pitch_extractor` вместо собственной оценки
9. **Batch processing**: компонент поддерживает батчевую обработку нескольких файлов через `extract_batch_segments()` с изоляцией данных между файлами

### Медицинские приложения

**Внимание**: Эти метрики являются прокси и не должны использоваться для медицинской диагностики без дополнительной валидации. Для клинических применений требуются более точные алгоритмы и калибровка.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
