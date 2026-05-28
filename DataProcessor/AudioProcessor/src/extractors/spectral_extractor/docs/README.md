## `spectral_extractor` (Spectral features)

### Назначение

Извлекает базовые спектральные признаки из аудио сигнала: спектральный центроид, ширина полосы, плоскостность, rolloff, скорость пересечения нуля (ZCR), контраст и дополнительные метрики (спектральный наклон и плоскостность в дБ).

**Версия**: 2.0.1  
**Категория**: spectral  
**GPU**: не требуется

### Входы

- **`audio/audio.wav`** (Segmenter contract) — полное аудио для `run()`
- **`audio/segments.json`** (Segmenter contract, family `spectral`) — сегменты для `run_segments()`

### Выходы

NPZ: `spectral_extractor_npz_v2` — см. `docs/SCHEMA.md`, `schemas/spectral_extractor_npz_v2.json`.

#### Audit v4 — заметки по NPZ

- **До фикса** на reference **A**: в tabular было **4 NaN** — **`device_used`** (строка → `as_float`), **`hop_length`**, **`n_fft`**, **`duration`** (в `run_segments` не попадали в payload). **Исправлено:** `npz_savers/spectral.py` (убран `device_used` из tabular) + `main.py` `run_segments` (проброс `hop_length`, `n_fft`, `duration` = охват оси сегментов).
- После перезапуска **F=46** (было **47** с псевдо-полем `device_used`).
- Observability (audit v4.2): `meta.stage_timings_ms`, `meta.spectral_resource_profile` (env: `AP_SPECTRAL_RESOURCE_PROFILE=1`)

Экстрактор возвращает `ExtractorResult` с payload, содержащим:

#### Основные спектральные признаки (feature-gated: `--spectral-enable-basic-features`)

Для каждого признака вычисляются статистики: `mean`, `std`, `min`, `max`, `median`.

- **`spectral_centroid_stats`**: центроид спектра (средневзвешенная частота, Hz)
  - Характеризует "яркость" звука
  - Высокие значения → более яркий/высокочастотный звук
  - Диапазон: > 0 Hz
  
- **`spectral_bandwidth_stats`**: ширина полосы (Hz)
  - Характеризует разброс частот вокруг центроида
  - Высокие значения → более широкий спектр
  - Диапазон: > 0 Hz
  
- **`spectral_flatness_stats`**: плоскостность спектра (0.0-1.0)
  - Характеризует "шумоподобность" звука
  - 1.0 → белый шум, 0.0 → чистый тон
  - Диапазон: [0.0, 1.0]
  
- **`spectral_rolloff_stats`**: частота rolloff (Hz)
  - Частота, ниже которой находится 85% спектральной энергии
  - Характеризует верхнюю границу частотного диапазона
  - Диапазон: > 0 Hz
  
- **`zcr_stats`**: скорость пересечения нуля (crossings/sec)
  - Количество переходов сигнала через ноль в секунду
  - Высокие значения → более шумный/высокочастотный сигнал
  - Диапазон: [0.0, 1.0]

#### Контраст (feature-gated: `--spectral-enable-contrast`)

- **`spectral_contrast_stats`**: спектральный контраст (усредненный по полосам)
  - Характеризует различие между пиками и впадинами в спектре
  - Высокие значения → более структурированный звук
  - Диапазон: ≥ 0

- **`spectral_contrast_bands`**: полные данные контраста по частотным полосам (список массивов)
  - Включается только если `keep_contrast_bands=True`
  - Представляет контраст для каждой частотной полосы отдельно

- **`spectral_contrast_variance`**: дисперсия контраста по полосам (дополнительная метрика)

#### Продвинутые признаки (feature-gated: `--spectral-enable-advanced-features`)

- **`spectral_slope_stats`**: спектральный наклон (dB/Hz)
  - Наклон спектральной огибающей (линейная регрессия)
  - Отрицательные значения → спад высоких частот
  - Положительные значения → усиление высоких частот
  - Диапазон: может быть отрицательным

- **`spectral_flatness_db_stats`**: плоскостность в децибелах (dB)
  - Логарифмическая версия плоскостности
  - Более удобна для анализа широкого диапазона значений
  - Диапазон: может быть отрицательным

- **`spectral_slope_stability`**: стабильность наклона (дополнительная метрика)
  - Вычисляется как `1.0 / (1.0 + std(slope))`
  - Высокие значения → более стабильный наклон

#### Дополнительные метрики для ML/аналитики (всегда включены, если включены basic_features)

- **`spectral_centroid_median`**: медиана центроида (Hz)
- **`spectral_bandwidth_ratio`**: относительная ширина полосы (`bandwidth / centroid`)
- **`spectral_rolloff_ratio`**: относительный rolloff (`rolloff / sample_rate`)
- **`spectral_flatness_entropy`**: энтропия плоскостности (энтропия распределения)
- **`spectral_features_correlation`**: корреляция между признаками (словарь парных корреляций)

#### Per-segment arrays (Audit v3)

- **`segment_start_sec`**, **`segment_end_sec`**, **`segment_center_sec`**: каноническая ось сегментов (float32[N])
- **`segment_mask`**: маска валидных сегментов (bool[N])
- **`centroid_mean_by_segment`**, **`bandwidth_mean_by_segment`**, **`flatness_mean_by_segment`**, **`rolloff_mean_by_segment`**, **`zcr_mean_by_segment`**: mean по сегменту (float32[N], basic_features)
- **`contrast_mean_by_segment`**: mean контраста по сегменту (float32[N], contrast)
- **`slope_mean_by_segment`**: mean slope по сегменту (float32[N], advanced_features)

Failed сегменты: `segment_mask=false`, значения в массивах = NaN.

#### Временные серии (feature-gated: `--spectral-enable-time-series`, legacy для `run()`)

- **`centroid_series`**, **`bandwidth_series`**, **`flatness_series`**, **`rolloff_series`**, **`zcr_series`**, **`contrast_series`**, **`slope_series`**: frame-level серии (только для `run()` на полном аудио)

#### Метаданные

- `device_used`: устройство обработки (`"cpu"`, `"cuda"` или `"auto"`)
- `sample_rate`: частота дискретизации аудио (Hz)
- `hop_length`: размер шага между кадрами (samples)
- `n_fft`: размер окна FFT (samples)
- `average_channels`: использовалось ли усреднение каналов
- `keep_contrast_bands`: сохраняются ли полные данные контраста
- `duration`: длительность аудио (секунды)
- `segments_count`: количество сегментов (для `run_segments()`)
- `spectral_contract_version`: версия контракта (`"spectral_contract_v1"`)
- `_features_enabled`: список включённых групп фичей (для отладки)

### Feature Dependencies

- **`spectral_flatness_db_stats`** зависит от **`spectral_flatness_stats`** (требует включения `--spectral-enable-basic-features`)
- **`spectral_features_correlation`** зависит от всех базовых признаков (требует включения `--spectral-enable-basic-features`)
- **`spectral_contrast_variance`** зависит от **`spectral_contrast_stats`** (требует включения `--spectral-enable-contrast`)
- **`spectral_slope_stability`** зависит от **`spectral_slope_stats`** (требует включения `--spectral-enable-advanced-features`)

### Конфигурация

#### CLI аргументы

```bash
# Параметры обработки
--spectral-sample-rate 22050          # Частота дискретизации (Hz)
--spectral-hop-length 512             # Размер шага между кадрами (samples)
--spectral-n-fft 2048                 # Размер окна FFT (samples)
--spectral-average-channels           # Усреднять каналы для многоканального аудио
--spectral-keep-contrast-bands        # Сохранять полные данные контраста по полосам
--spectral-enable-normalization       # Включить нормализацию аудио перед обработкой

# Feature gating (все opt-in, по умолчанию все выключены)
--spectral-enable-basic-features      # Включить базовые признаки (centroid, bandwidth, flatness, rolloff, ZCR)
--spectral-enable-contrast            # Включить контраст (contrast stats + contrast_bands)
--spectral-enable-advanced-features   # Включить продвинутые признаки (slope, flatness_db)
--spectral-enable-time-series         # Включить временные серии для всех признаков
```

#### Python API

```python
from src.extractors.spectral_extractor import SpectralExtractor

extractor = SpectralExtractor(
    device="auto",
    sample_rate=22050,
    hop_length=512,
    n_fft=2048,
    average_channels=True,
    keep_contrast_bands=True,
    enable_normalization=False,
    enable_basic_features=True,
    enable_contrast=False,
    enable_advanced_features=False,
    enable_time_series=False,
    progress_callback=None,
    artifacts_dir=None,
)
```

### Алгоритмы

Все признаки извлекаются с использованием библиотеки **librosa**:

1. **Спектральный центроид**: взвешенное среднее частот по амплитуде спектра (`librosa.feature.spectral_centroid`)
2. **Ширина полосы**: стандартное отклонение частот вокруг центроида (`librosa.feature.spectral_bandwidth`)
3. **Плоскостность**: геометрическое среднее / арифметическое среднее спектральных амплитуд (`librosa.feature.spectral_flatness`)
4. **Rolloff**: частота, ниже которой находится 85% энергии (`librosa.feature.spectral_rolloff`)
5. **ZCR**: количество переходов через ноль на кадр (`librosa.feature.zero_crossing_rate`)
6. **Контраст**: различие между пиками и впадинами в разных частотных полосах (`librosa.feature.spectral_contrast`)
7. **Спектральный наклон**: линейная регрессия спектральной огибающей в дБ (вычисляется через STFT и МНК)

### Обработка ошибок

Экстрактор использует **no-fallback policy** (fail-fast):

- **Ошибка загрузки аудио**: `spectral_audio_load_failed`
- **Ошибка вычисления centroid**: `spectral_centroid_failed`
- **Ошибка вычисления bandwidth**: `spectral_bandwidth_failed`
- **Ошибка вычисления flatness**: `spectral_flatness_failed`
- **Ошибка вычисления rolloff**: `spectral_rolloff_failed`
- **Ошибка вычисления ZCR**: `spectral_zcr_failed`
- **Ошибка вычисления contrast**: `spectral_contrast_failed`
- **Ошибка вычисления slope**: `spectral_slope_failed`
- **Ошибка валидации**: `spectral_validation_failed`
- **Неизвестная ошибка**: `spectral_unknown`

Все ошибки включают детальный `error_code` в сообщении об ошибке.

### Валидация

#### Валидация параметров (fail-fast)

- `sample_rate > 0`
- `hop_length > 0`
- `n_fft > 0` и `n_fft >= 512`
- `hop_length <= n_fft`

#### Валидация выходных данных

- Проверка диапазонов значений (например, centroid > 0, flatness ∈ [0, 1], rolloff > 0)
- Проверка NaN/inf в массивах
- Проверка консистентности (min ≤ mean ≤ max для всех stats)
- Проверка типов и размерностей

### Обработка многоканального аудио

Экстрактор автоматически преобразует многоканальное аудио в моно:

- Если `average_channels=True`: усредняет все каналы (по умолчанию)
- Если `average_channels=False`: использует первый канал

### Нормализация аудио

Опциональная нормализация аудио перед обработкой (включается через `--spectral-enable-normalization`):

- Использует `AudioUtils.normalize_audio()` для нормализации амплитуды
- Может улучшить стабильность и точность спектральных признаков

### Segmenter Contract

Экстрактор поддерживает работу на сегментах от Segmenter:

- **`run()`**: работает на полном аудио (`audio/audio.wav`)
- **`run_segments()`**: работает на сегментах из `audio/segments.json` (family `spectral`)

Для `run_segments()`:
- Читает `families.spectral.segments[]` из `audio/segments.json`
- Использует `start_sample/end_sample` для загрузки сегментов через `AudioUtils.load_audio_segment()`
- Агрегирует результаты по всем сегментам (статистики и временные серии)

### Progress Reporting

Экстрактор поддерживает progress reporting через callback:

- Для `run()`: обновление прогресса для каждого признака (centroid, bandwidth, flatness, rolloff, ZCR, contrast, slope)
- Для `run_segments()`: обновление прогресса каждые 10% сегментов

### Per-run Storage

Большие временные серии (>1000 элементов) сохраняются в `.npy` файлы:

- Путь: `result_store/<platform_id>/<video_id>/<run_id>/spectral_extractor/_artifacts/*.npy`
- Регистрация в `manifest.json.components[].artifacts[]` (type=`"npy"`)

### Visualization

**Цель**: Представить выход компонента в формате, удобном для ручного просмотра и валидации.

#### Рекомендации для UI/сайта

1. **Timeline визуализация** (рекомендуется):
   - Отображать временные серии признаков (centroid, bandwidth, flatness, rolloff, ZCR) на временной шкале
   - Использовать line charts с разными цветами для каждого признака
   - Добавить tooltips с точными значениями при наведении
   - Масштабирование и панорамирование для детального просмотра
   - Легенда для включения/выключения отдельных признаков

2. **Распределения** (рекомендуется):
   - Гистограммы для статистик (mean, std, min, max, median) каждого признака
   - Box plots для сравнения признаков между сегментами
   - Violin plots для более детального анализа распределений

3. **Корреляционная матрица** (опционально):
   - Heatmap для `spectral_features_correlation`
   - Интерактивные элементы для фильтрации и масштабирования
   - Цветовая кодировка: синий (отрицательная корреляция), красный (положительная корреляция)

4. **Дополнительные метрики** (рекомендуется):
   - Отображать `spectral_centroid_median`, `spectral_bandwidth_ratio`, `spectral_rolloff_ratio` как отдельные карточки
   - Использовать gauge charts для `spectral_flatness_entropy` и `spectral_slope_stability`
   - Прогресс-бары для нормализованных значений

5. **Контраст по полосам** (опционально):
   - Если включен `spectral_contrast_bands`, отображать как stacked area chart или heatmap по частотным полосам
   - Цветовая кодировка по интенсивности контраста

6. **Интерактивные элементы**:
   - Фильтры по диапазонам значений признаков
   - Выбор признаков для отображения
   - Экспорт данных в CSV/JSON
   - Сравнение между сегментами/файлами

**Типы графиков**:
- **Line charts**: для временных серий (centroid, bandwidth, flatness, rolloff, ZCR, contrast, slope)
- **Bar charts**: для статистик (mean, std, min, max, median)
- **Box plots**: для сравнения признаков между сегментами
- **Heatmaps**: для корреляционной матрицы и contrast bands
- **Gauge charts**: для нормализованных метрик (flatness_entropy, slope_stability)
- **Stacked area charts**: для contrast bands по частотным полосам

#### Render (dev-only, Audit v3)

Offline-only HTML render (vanilla canvas, без CDN): графики `centroid_mean_by_segment` и `flatness_mean_by_segment` vs `segment_center_sec`.

Используйте `render_spectral_extractor_html()` для генерации HTML страницы с результатами:

```python
from src.core.renderer import render_spectral_extractor_html

render_spectral_extractor_html(
    npz_path="result_store/.../spectral_extractor/spectral_extractor_features.npz",
    output_path="spectral_debug.html"
)
```

HTML страница включает:
- Summary (sample_rate, hop_length, n_fft, device, duration, segments_count)
- Таблицы статистик для всех включённых признаков
- Дополнительные метрики
- Временные серии (если включены)
- Raw JSON данные

### Sampling / units-of-processing requirements

**Стратегия выборки**: Компонент использует сегменты из Segmenter (family `spectral`).

**Требования к сегментам**:
- Сегменты должны быть предоставлены через `audio/segments.json` (family `spectral`)
- Компонент не генерирует сегменты сам (Segmenter — единственный владелец sampling)
- Для `run()`: обрабатывает полное аудио (`audio/audio.wav`)
- Для `run_segments()`: обрабатывает сегменты из `families.spectral.segments[]`

**Параметры выборки** (управляются Segmenter):
- Используется универсальная нелинейная кривая для family `spectral`
- Ссылка на `docs/contracts/SEGMENTER_CONTRACT.md` как источник истины

**Требования для разных длительностей**:
- Короткие аудио (< 10 сек): минимум 1 сегмент
- Средние аудио (10-60 сек): 5-20 сегментов
- Длинные аудио (> 60 сек): 20-100 сегментов (зависит от нелинейной кривой Segmenter)

### Models

**GPU Models**: Не используются

**CPU Models**: Не используются

**Библиотеки**:
- **librosa**: используется для всех спектральных операций (FFT, spectral features)
  - Runtime: `inprocess`
  - Engine: `numpy` (CPU-only)
  - Precision: `fp32`
  - Device: `cpu`

**Примечание**: Компонент не использует ML-модели, все вычисления выполняются через librosa (спектральный анализ).

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **Потоки**: не используется (librosa операции выполняются последовательно)
- **Батчинг**: не используется (обработка одного аудио/сегмента за раз)

**Внешний параллелизм** (выше компонента):
- **Batch processing**: компонент поддерживает `extract_batch_segments()` для параллельной обработки сегментов из нескольких видео
- **CPU parallelism**: используется `ThreadPoolExecutor` для параллельной обработки файлов в batch режиме
- **Количество воркеров**: контролируется через `max_workers` (по умолчанию: `min(len(audio_files), os.cpu_count())`)
- **Требования к изоляции**: разные `run_id`, разные `result_store` пути, разные `tmp_path` для каждого файла
- **Thread-safety**: компонент thread-safe для параллельной обработки разных файлов (нет shared mutable state)

**Комбинированный подход**:
- Внутренняя обработка: последовательная (librosa операции)
- Внешняя обработка: параллельная через ThreadPoolExecutor (batch processing)

**Ограничения**:
- Параллелизм ограничен количеством CPU ядер
- Память: каждый поток обрабатывает один файл, peak memory зависит от размера аудио файла
- GPU: не используется

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (FFT и спектральные операции)
- **GPU**: не используется
- **Estimated duration**: ~1.2 секунды для типичного аудио файла (полное аудио)
- **Estimated duration (segments)**: ~0.1-0.2 секунды на сегмент (зависит от длины сегмента)

**Параметры производительности**:
- `hop_length`: меньшие значения → больше кадров → выше точность, но медленнее
  - **Δ latency**: ~+10-20 ms/frame при уменьшении hop_length в 2 раза
  - **Δ cost**: незначительное (CPU-only)
- `n_fft`: большие значения → лучше частотное разрешение, но медленнее
  - **Δ latency**: ~+5-15 ms/frame при увеличении n_fft в 2 раза
  - **Δ cost**: незначительное (CPU-only)
- `keep_contrast_bands`: `False` → меньше данных в payload, быстрее передача
  - **Δ latency**: ~-5-10 ms (экономия на сериализации)
  - **Δ cost**: незначительное
- `enable_time_series`: `False` → меньше данных в payload, быстрее передача
  - **Δ latency**: ~-10-50 ms (экономия на сериализации и сохранении .npy)
  - **Δ cost**: незначительное
- `enable_normalization`: `True` → дополнительная обработка, немного медленнее
  - **Δ latency**: ~+5-10 ms/frame
  - **Δ cost**: незначительное

**Batch processing ускорение**:
- При обработке N файлов параллельно через `extract_batch_segments()`:
  - Ускорение: ~N× (на многоядерных CPU, до лимита памяти)
  - Peak RAM: увеличивается пропорционально количеству параллельных воркеров

### Batch processing

**Поддержка**: Компонент поддерживает batch processing через `extract_batch_segments()`.

**Реализация**:
- Использует `ThreadPoolExecutor` для параллельной обработки сегментов из нескольких видео
- Количество воркеров контролируется через `max_workers` (по умолчанию: `min(len(audio_files), os.cpu_count())`)
- Каждый файл обрабатывается изолированно (разные `tmp_path`, разные `artifacts_dir`)

**Использование**:
```python
from src.extractors.spectral_extractor import SpectralExtractor

extractor = SpectralExtractor(
    enable_basic_features=True,
    enable_time_series=False,
)

# Batch processing
audio_files_with_segments = [
    {
        "input_uri": "path/to/audio1.wav",
        "tmp_path": "path/to/tmp1",
        "segments": segments1,
        "file_id": "file1",
    },
    {
        "input_uri": "path/to/audio2.wav",
        "tmp_path": "path/to/tmp2",
        "segments": segments2,
        "file_id": "file2",
    },
]

results = extractor.extract_batch_segments(
    audio_files_with_segments,
    max_workers=4,  # Количество параллельных воркеров
)
```

**Ускорение**:
- При обработке N файлов параллельно: ускорение ~N× (на многоядерных CPU)
- Ограничения: количество CPU ядер, доступная RAM

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **librosa**: библиотека для спектрального анализа
- **numpy**: численные операции
- **Segmenter**: источник сегментов для `run_segments()`

### Примечания

1. **FFT параметры**: `n_fft` должен быть достаточно большим для хорошего частотного разрешения (обычно 2048 или 4096)
2. **Hop length**: меньшие значения дают более детальную временную информацию, но увеличивают объем данных
3. **Усреднение каналов**: рекомендуется для многоканального аудио для получения репрезентативных спектральных признаков
4. **Feature gating**: Audit v3 — `enable_basic_features=True` по умолчанию; contrast/advanced/time_series opt-in
5. **Временные серии**: большие серии (>1000 элементов) автоматически сохраняются в `.npy` файлы для экономии памяти
6. **Contract versioning**: используется `spectral_contract_version="spectral_contract_v1"` для валидации совместимости с downstream extractors
7. **Batch processing**: компонент batch-safe и поддерживает параллельную обработку нескольких файлов через `extract_batch_segments()`
8. **Audit v3**: schema `spectral_extractor_npz_v2`, canonical axis, per-segment arrays, no payload, NaN для missing
