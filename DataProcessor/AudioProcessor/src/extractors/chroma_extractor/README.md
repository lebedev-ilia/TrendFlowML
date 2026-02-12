## `chroma_extractor` (Audio signal processing extractor)

### Назначение

Извлекает **хрома-признаки** (12-полосный профиль классов высот, pitch class profile) с автоматической оценкой строя, нормализацией и статистическими агрегатами. Хрома-фичи отражают гармоническое содержание аудио и полезны для анализа музыки и речи.

**Версия**: 2.0.0  
**Категория**: spectral  
**GPU**: не требуется

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter) — полный аудио файл
- **`audio/segments.json`** (Segmenter contract `audio_segments_v1`) — для `run_segments()`:
  - `families.chroma.segments[]` — сегменты для обработки хрома-фич

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/chroma_extractor/chroma_extractor_features.npz` (**фиксированное имя**)

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

#### Полезные поля payload:

**Статистические агрегаты (feature-gated):**
- `chroma_mean`: средние значения по 12 хрома-классам (list[float], если `enable_basic_stats=True`)
- `chroma_std`: стандартные отклонения (list[float], если `enable_basic_stats=True`)
- `chroma_min`: минимальные значения (list[float], если `enable_basic_stats=True`)
- `chroma_max`: максимальные значения (list[float], если `enable_basic_stats=True`)
- `chroma_median`: медианные значения (list[float], если `enable_extended_stats=True`)
- `chroma_p25`: 25-й перцентиль (list[float], если `enable_extended_stats=True`)
- `chroma_p75`: 75-й перцентиль (list[float], если `enable_extended_stats=True`)
- `chroma_stats_vector`: конкатенированный вектор всех статистик (если `enable_stats_vector=True`)

**Временные ряды (feature-gated):**
- `chroma`: полная хрома-спектрограмма (12 x frames) как numpy array (если `enable_time_series=True` и размер <= threshold)
- `chroma_npy`: путь к сохраненному .npy файлу в `_artifacts/` (если размер > threshold)

**Дополнительные метрики для ML/аналитики:**
- `tuning_estimate`: оценка строя (semitones, float)
- `chroma_dominant_class`: индекс доминирующего хрома-класса (int, 0-11)
- `chroma_dominant_energy`: энергия доминирующего класса (float)
- `chroma_harmonic_stability`: стабильность гармонического содержания (float, 0-1)
- `chroma_entropy`: энтропия распределения хрома-классов (float)
- `chroma_contrast`: контраст между классами (max - min, float)
- `chroma_centroid`: центроид распределения хрома-классов (float)
- `chroma_rolloff`: rolloff частоты (95% энергии, int)

**Метаданные:**
- `sample_rate`: частота дискретизации (int)
- `hop_length`: размер hop для STFT/CQT (int)
- `n_fft`: размер FFT окна (int, для STFT mode)
- `duration`: длительность аудио в секундах (float)
- `device_used`: устройство обработки (str)
- `chroma_frames`: количество кадров в хрома-спектрограмме (int)
- `n_chroma`: количество хрома-классов (int, по умолчанию 12)
- `chroma_type`: тип хрома ("cqt" | "stft")
- `normalize`: тип нормализации (None | "l1" | "l2")
- `segments_count`: количество обработанных сегментов (int, для `run_segments()`)
- `segment_centers_sec`: центры сегментов в секундах (float32[N], для `run_segments()`)
- `segment_durations_sec`: длительности сегментов в секундах (float32[N], для `run_segments()`)

### Feature Dependencies

**Зависимости между фичами:**
- `chroma_stats_vector` зависит от `enable_basic_stats` и/или `enable_extended_stats` (конкатенирует все включенные статистики)
- Все статистики (`chroma_mean`, `chroma_std`, и т.д.) зависят от вычисления хрома-спектрограммы
- Дополнительные метрики (например, `chroma_dominant_class`) зависят от вычисления статистик

**Зависимости от других extractors:**
- Нет зависимостей от других extractors

### Models

ML модели **не используются** (signal processing only). `models_used[]` пустой.

**Библиотеки:**
- **librosa**: основная библиотека для хрома-фич (CQT или STFT-based)
  - **Triton**: ❌ Нет (in-process)
  - **Runtime**: `inprocess`
  - **Engine**: `librosa` (signal processing)
  - **Precision**: `fp32`
  - **Device**: `cpu`

### Sampling / units-of-processing requirements

**Стратегия выборки**: Segmenter-owned (универсальная нелинейная кривая)

Chroma extractor использует сегменты из Segmenter contract (family `chroma`). Segmenter является единственным владельцем sampling — компонент не генерирует семплинг сам.

**Требования к сегментам**:
- Сегменты должны быть предоставлены через `audio/segments.json` (family `chroma`)
- Для `run_segments()` требуется наличие family `chroma` в `segments.json`
- Если family `chroma` отсутствует, extractor завершается с ошибкой (no-fallback policy)

**Параметры нелинейной кривой** (Segmenter-owned):
- Ссылка на `docs/contracts/SEGMENTER_CONTRACT.md` как источник истины
- Параметры кривой определяются Segmenter и передаются через `segments.json`

**Ограничения**:
- Минимальная длительность сегмента: определяется Segmenter
- Максимальная длительность сегмента: определяется Segmenter
- Coverage requirements: определяется Segmenter

### Конфигурация

**Параметры конфигурации компонента**:

| Параметр | Тип | Значение по умолчанию | Допустимые значения | Описание | Δ latency | Δ cost |
|----------|-----|----------------------|---------------------|----------|-----------|--------|
| `device` | str | `"auto"` | `"auto"` \| `"cpu"` | Устройство для обработки | 0 ms | 0 ₽ |
| `sample_rate` | int | `22050` | `> 0` | Частота дискретизации | +0.1 ms/frame при увеличении на 1kHz | +0.01 ₽/frame |
| `hop_length` | int | `512` | `> 0` | Размер hop для STFT/CQT | -0.05 ms/frame при увеличении на 64 | -0.005 ₽/frame |
| `n_fft` | int | `4096` | `> 0` | Размер FFT окна (для STFT mode) | +0.02 ms/frame при увеличении на 512 | +0.002 ₽/frame |
| `mix_to_mono` | bool | `true` | `true` \| `false` | Сводить стерео в моно | 0 ms | 0 ₽ |
| `chroma_type` | str | `"cqt"` | `"cqt"` \| `"stft"` | Тип хрома (no-fallback policy) | -0.1 ms/frame для stft | -0.01 ₽/frame |
| `normalize` | str | `"l1"` | `null` \| `"l1"` \| `"l2"` | Нормализация по кадрам | +0.01 ms/frame | +0.001 ₽/frame |
| `n_chroma` | int | `12` | `> 0` | Количество хрома-классов | 0 ms | 0 ₽ |
| `fmin` | float | `null` | `> 0.0` \| `null` | Минимальная частота (Hz) | 0 ms | 0 ₽ |
| `fmax` | float | `null` | `> 0.0` \| `null` | Максимальная частота (Hz) | 0 ms | 0 ₽ |
| `n_bins` | int | `null` | `> 0` \| `null` | Количество бинов для CQT | 0 ms | 0 ₽ |
| `enable_audio_normalization` | bool | `false` | `true` \| `false` | Нормализация аудио перед обработкой | +0.05 ms/frame | +0.005 ₽/frame |
| `enable_basic_stats` | bool | `false` | `true` \| `false` | Включить базовые статистики | +0.05 ms/frame | +0.005 ₽/frame |
| `enable_extended_stats` | bool | `false` | `true` \| `false` | Включить расширенные статистики | +0.1 ms/frame | +0.01 ₽/frame |
| `enable_stats_vector` | bool | `false` | `true` \| `false` | Включить компактный вектор статистик | +0.01 ms/frame | +0.001 ₽/frame |
| `enable_time_series` | bool | `false` | `true` \| `false` | Включить временные серии | +0.1 ms/frame | +0.01 ₽/frame |

**Источник оценки**: бенчмарк/профилирование на типичных аудио файлах (30-120 секунд)

**Пример конфигурации (минимум)**:
```python
{
    "device": "auto",
    "sample_rate": 22050,
    "hop_length": 512,
    "chroma_type": "cqt",
    "enable_basic_stats": true,
}
```

**Пример конфигурации (расширенный)**:
```python
{
    "device": "auto",
    "sample_rate": 22050,
    "hop_length": 512,
    "n_fft": 4096,
    "mix_to_mono": true,
    "chroma_type": "cqt",
    "normalize": "l1",
    "n_chroma": 12,
    "fmin": null,
    "fmax": null,
    "n_bins": null,
    "enable_audio_normalization": false,
    "enable_basic_stats": true,
    "enable_extended_stats": true,
    "enable_stats_vector": true,
    "enable_time_series": false,
}
```

### Features

**Features contract**: компонент имеет явный механизм выбора выходных фич через аргументы/конфиг.

**Дефолтный набор фич**: все фичи выключены по умолчанию (opt-in подход). Для включения фич используйте соответствующие флаги.

**Перечень всех возможных фич**:

#### Basic Stats (`enable_basic_stats=True`)
- `chroma_mean`: средние значения по 12 хрома-классам (list[float], shape: (12,))
  - Формат: per-run
  - Единица: chroma energy (0-1 после нормализации)
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame
- `chroma_std`: стандартные отклонения по 12 хрома-классам (list[float], shape: (12,))
  - Формат: per-run
  - Единица: chroma energy std
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame
- `chroma_min`: минимальные значения по 12 хрома-классам (list[float], shape: (12,))
  - Формат: per-run
  - Единица: chroma energy
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame
- `chroma_max`: максимальные значения по 12 хрома-классам (list[float], shape: (12,))
  - Формат: per-run
  - Единица: chroma energy
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame

#### Extended Stats (`enable_extended_stats=True`)
- `chroma_median`: медианные значения по 12 хрома-классам (list[float], shape: (12,))
  - Формат: per-run
  - Единица: chroma energy
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `chroma_p25`: 25-й перцентиль по 12 хрома-классам (list[float], shape: (12,))
  - Формат: per-run
  - Единица: chroma energy
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `chroma_p75`: 75-й перцентиль по 12 хрома-классам (list[float], shape: (12,))
  - Формат: per-run
  - Единица: chroma energy
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame

#### Stats Vector (`enable_stats_vector=True`)
- `chroma_stats_vector`: конкатенированный вектор всех статистик (numpy array, shape: (N,))
  - Формат: per-run
  - Единица: chroma energy (конкатенация всех включенных статистик)
  - Влияние на стоимость: +0.01 ms/frame, +0.001 ₽/frame
  - Зависимости: требует хотя бы одну из `enable_basic_stats` или `enable_extended_stats`

#### Time Series (`enable_time_series=True`)
- `chroma`: полная хрома-спектрограмма (numpy array, shape: (12, frames))
  - Формат: per-run (весь аудио файл) или per-segment (для run_segments)
  - Единица: chroma energy (0-1 после нормализации)
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
  - Если `chroma.size > threshold` (12 * 500), сохраняется в `.npy` файл в `_artifacts/chroma.npy`

#### Additional Metrics (always computed)
- `tuning_estimate`: оценка строя (semitones, float)
  - Формат: per-run
  - Единица: semitones
  - Влияние на стоимость: включено в базовую стоимость
- `chroma_dominant_class`: индекс доминирующего хрома-класса (int, 0-11)
  - Формат: per-run
  - Единица: class index (0-11)
  - Влияние на стоимость: включено в базовую стоимость
- `chroma_dominant_energy`: энергия доминирующего класса (float)
  - Формат: per-run
  - Единица: chroma energy
  - Влияние на стоимость: включено в базовую стоимость
- `chroma_harmonic_stability`: стабильность гармонического содержания (float, 0-1)
  - Формат: per-run
  - Единица: stability score (0-1)
  - Влияние на стоимость: включено в базовую стоимость
- `chroma_entropy`: энтропия распределения хрома-классов (float)
  - Формат: per-run
  - Единица: entropy
  - Влияние на стоимость: включено в базовую стоимость
- `chroma_contrast`: контраст между классами (max - min, float)
  - Формат: per-run
  - Единица: chroma energy difference
  - Влияние на стоимость: включено в базовую стоимость
- `chroma_centroid`: центроид распределения хрома-классов (float)
  - Формат: per-run
  - Единица: class index (weighted)
  - Влияние на стоимость: включено в базовую стоимость
- `chroma_rolloff`: rolloff частоты (95% энергии, int)
  - Формат: per-run
  - Единица: frame index
  - Влияние на стоимость: включено в базовую стоимость

**В meta артефакта фиксируется**: `_features_enabled[]` — список включенных фич.

**Feature Gating**:

Все фичи управляются персональными флагами (opt-in):

- `--chroma-enable-basic-stats`: базовые статистики (`chroma_mean`, `chroma_std`, `chroma_min`, `chroma_max`)
- `--chroma-enable-extended-stats`: расширенные статистики (`chroma_median`, `chroma_p25`, `chroma_p75`)
- `--chroma-enable-stats-vector`: компактный вектор статистик (`chroma_stats_vector`)
- `--chroma-enable-time-series`: временные серии (`chroma` spectrogram)

**Зависимости фичей:**
- `chroma_stats_vector` требует хотя бы одну из `enable_basic_stats` или `enable_extended_stats`

### Алгоритм работы

1. **Загрузка аудио**: через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. **Опциональная нормализация аудио**: если `enable_audio_normalization=True`
3. **Сведение в моно** (опционально): если `mix_to_mono=True` и аудио стерео
4. **Оценка строя**: через `librosa.estimate_tuning()` для коррекции хрома-фич
5. **Вычисление хрома** (no-fallback policy):
   - **CQT mode**: `librosa.feature.chroma_cqt()` — более точный для музыки
   - **STFT mode**: `librosa.feature.chroma_stft()` — для STFT-based хрома
   - Если выбранный метод не работает → fail-fast с error_code
6. **Нормализация** (опционально):
   - **L1**: нормализация по сумме (каждый кадр суммируется в 1.0)
   - **L2**: нормализация по L2-норме
7. **Статистические агрегаты** (feature-gated): mean, std, min, max, median, p25, p75 по каждому из 12 хрома-классов
8. **Дополнительные метрики**: доминирующий класс, стабильность, энтропия, контраст, центроид, rolloff
9. **Временные ряды** (feature-gated):
   - Сохранение inline (если размер <= threshold) или в .npy файл в `_artifacts/`

### Особенности

- **12 хрома-классов**: C, C#, D, D#, E, F, F#, G, G#, A, A#, B (октавно-инвариантные)
- **Автоматический тюнинг**: оценка строя для коррекции хрома-фич
- **Два режима**: CQT (предпочтительно для музыки) или STFT (явный выбор, no-fallback)
- **Нормализация**: опциональная L1 или L2 нормализация по кадрам
- **Эффективное хранение**: временные ряды опциональны и могут сохраняться в .npy файл
- **Segment-based обработка**: поддержка `run_segments()` для обработки сегментов от Segmenter

### Error Handling

Детальные error codes:
- `chroma_audio_load_failed`: Ошибка загрузки аудио
- `chroma_tuning_failed`: Ошибка оценки строя
- `chroma_cqt_failed`: Ошибка CQT метода (no-fallback)
- `chroma_stft_failed`: Ошибка STFT метода (no-fallback)
- `chroma_normalization_failed`: Ошибка нормализации
- `chroma_statistics_failed`: Ошибка вычисления статистик
- `chroma_validation_failed`: Ошибка валидации выходных данных
- `chroma_unknown`: Неизвестная ошибка

**No-fallback policy**: Если выбранный метод (CQT или STFT) не работает, extractor завершается с ошибкой (fail-fast), без автоматического fallback.

### Performance characteristics

**Resource costs**:
- **CPU**: O(N * log(N)) для CQT/STFT, где N — длина аудио
- **Память**: O(12 * frames) для хрома-спектрограммы
- **Estimated duration**: ~1.2 секунд для типичного аудио

**Единица обработки**: 
- `run()`: весь аудио файл (не сегментированный)
- `run_segments()`: сегменты из `families.chroma.segments[]`

**Типичные значения (preset="default")**:

| Audio Duration | Latency per unit | CPU RAM peak | Notes |
|----------------|------------------|--------------|-------|
| 30 sec | ~1.2 sec | ~50 MB | typical |
| 60 sec | ~2.4 sec | ~100 MB | typical |
| 120 sec | ~4.8 sec | ~200 MB | typical |

**Для аудио с длительностью N секунд**: Total latency ≈ N × 0.04 sec (примерно)

**Оптимизации**:
- Эффективная обработка сегментов с прогресс-репортингом
- Сохранение больших массивов в `.npy` файлы для оптимизации размера NPZ
- CPU parallelism через `extract_batch_segments()` для обработки нескольких файлов одновременно

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **Потоки**: не используется (однопоточная обработка)
- **Батчинг**: не используется (обработка по одному сегменту)
- **Асинхронная обработка**: не используется

**Внешний параллелизм** (выше компонента):
- ✅ Можно запускать несколько экземпляров компонента параллельно на разных видео (разные run_id)
- ✅ Требования к изоляции: разные `artifacts_dir` для каждого файла (обеспечивается через `AudioFileContext`)
- ✅ Ограничения: нет shared resources, нет locks

**Batch processing**:
- ✅ Поддерживается через `extract_batch_segments()` с CPU parallelism
- ✅ Использует `ThreadPoolExecutor` для параллельной обработки сегментов из нескольких видео одновременно
- ✅ Каждый файл обрабатывается изолированно через `run_segments()`
- ✅ Количество параллельных воркеров контролируется через `max_workers` (по умолчанию `os.cpu_count()`)

**Thread-safety**: компонент thread-safe для параллельной обработки разных файлов (нет shared mutable state).

**Требования к памяти**: peak memory при параллельном выполнении ≈ N × single_file_memory, где N — количество параллельных воркеров.

**Комбинированный подход**: внутренний однопоточный + внешний параллелизм через `extract_batch_segments()`.

### Visualization

**Рекомендации по визуализации для UI/сайта:**

1. **Timeline визуализация**:
   - Отобразить хрома-спектрограмму как heatmap (12 классов по вертикали, время по горизонтали)
   - Использовать цветовую схему (например, Viridis) для отображения энергии
   - Добавить интерактивные tooltips с названиями хрома-классов (C, C#, D, и т.д.)

2. **Статистики**:
   - Bar chart для `chroma_mean` по 12 классам
   - Line chart для `chroma_std` (вариативность)
   - Box plot для `chroma_median`, `chroma_p25`, `chroma_p75`

3. **Дополнительные метрики**:
   - Отобразить доминирующий класс (highlight в bar chart)
   - Показать стабильность гармонического содержания (progress bar или gauge)
   - Отобразить энтропию и контраст (scalar values)

4. **Интерактивные элементы**:
   - Zoom для timeline
   - Фильтры по хрома-классам
   - Сравнение между сегментами (если используется `run_segments()`)

**Локальный HTML renderer для дебага:**
- Используйте `render_chroma_extractor_html()` для генерации HTML страницы с raw данными
- HTML включает интерактивные графики (Plotly) для визуализации хрома-спектрограммы и статистик
- Только для локального использования, не в production артефактах

### Примеры использования

#### Загрузка результатов

```python
import numpy as np

data = np.load("chroma_extractor_features.npz", allow_pickle=True)
payload = data["payload"].item()

# Статистики по 12 хрома-классам (если enable_basic_stats=True)
chroma_mean = payload.get("chroma_mean")  # [mean_C, mean_C#, ..., mean_B]
chroma_std = payload.get("chroma_std")
chroma_median = payload.get("chroma_median")  # если enable_extended_stats=True

# Полный вектор статистик (если enable_stats_vector=True)
stats_vector = payload.get("chroma_stats_vector")  # [mean_12, std_12, min_12, max_12, median_12, p25_12, p75_12]

# Временные ряды (если enable_time_series=True)
if "chroma" in payload:
    chroma_ts = np.array(payload["chroma"])  # shape: (12, frames)
elif "chroma_npy" in payload:
    chroma_ts = np.load(payload["chroma_npy"])  # shape: (12, frames)

# Дополнительные метрики
tuning_estimate = payload.get("tuning_estimate")
dominant_class = payload.get("chroma_dominant_class")
harmonic_stability = payload.get("chroma_harmonic_stability")
```

#### Анализ гармонического содержания

```python
# Найти доминирующий хрома-класс
chroma_mean = payload.get("chroma_mean")
if chroma_mean is not None:
    dominant_class = payload.get("chroma_dominant_class", np.argmax(chroma_mean))
    chroma_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"Доминирующий класс: {chroma_names[dominant_class]} ({chroma_mean[dominant_class]:.3f})")

# Проверка гармонической стабильности
harmonic_stability = payload.get("chroma_harmonic_stability")
print(f"Гармоническая стабильность: {harmonic_stability:.3f} (выше = стабильнее)")

# Энтропия распределения
entropy = payload.get("chroma_entropy")
print(f"Энтропия: {entropy:.3f} (выше = более разнообразное распределение)")
```

#### Визуализация хрома-спектрограммы

```python
import matplotlib.pyplot as plt

if "chroma" in payload or "chroma_npy" in payload:
    chroma_ts = np.array(payload["chroma"]) if "chroma" in payload else np.load(payload["chroma_npy"])
    
    plt.figure(figsize=(12, 4))
    plt.imshow(chroma_ts, aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel("Time (frames)")
    plt.ylabel("Chroma Class")
    plt.yticks(range(12), ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    plt.title("Chroma Spectrogram")
    plt.colorbar(label="Chroma Energy")
    plt.show()
```

### Связанные компоненты

- **Segmenter**: предоставляет аудио файл и сегменты (`families.chroma.segments[]`)
- **librosa**: основная библиотека для хрома-фич и оценки строя

### Quality validation & human-friendly inspection

#### Как проверить качество выхода компонента

**1. Автоматическая оценка**:
- Проверка валидности выходных данных через `_validate_output()`:
  - Отсутствие NaN/Inf значений в хрома-спектрограмме
  - Правильная размерность (12 хрома-классов)
  - Разумные диапазоны значений (0-1 после нормализации)
  - Консистентность статистик

**2. Human-friendly визуализация**:
- **Chroma spectrogram heatmap**: отображение хрома-спектрограммы как heatmap (12 классов по вертикали, время по горизонтали)
- **Статистики**: bar chart для `chroma_mean` по 12 классам, line chart для `chroma_std`
- **Дополнительные метрики**: отображение доминирующего класса, стабильности, энтропии
- **Интерактивные элементы**: zoom для timeline, фильтры по хрома-классам

**Локальный HTML renderer для дебага**:
- Используйте `render_chroma_extractor_html()` для генерации HTML страницы с интерактивными графиками
- Доступен через `src/core/renderer.py`

**3. Статистическая валидация**:
- Проверка разумности значений:
  - `chroma_mean` в диапазоне [0, 1] после нормализации
  - `chroma_std` >= 0 и разумные значения
  - `tuning_estimate` в диапазоне [-1, 1] semitones (разумные значения)
  - `chroma_harmonic_stability` в диапазоне [0, 1]
  - `chroma_entropy` >= 0 и разумные значения
- Отсутствие аномальных значений (NaN где не ожидается, бесконечности)

**4. Интеграция с downstream компонентами**:
- Проверка корректности чтения артефактов downstream компонентами
- Формат данных соответствует ожиданиям downstream
- Временные метки правильно выровнены

### Примечания

1. **CQT vs STFT**: CQT (Constant-Q Transform) предпочтительнее для музыки, STFT — для общего использования. Выбор метода явный через `--chroma-type`, без fallback.
2. **Тюнинг**: автоматическая оценка строя помогает корректно извлекать хрома-фичи для нестандартных строев
3. **Нормализация**: L1 нормализация делает каждый кадр суммой 1.0, что полезно для сравнения относительных энергий
4. **Временные ряды**: опциональны из-за размера, могут сохраняться в .npy файл для больших аудио
5. **Feature gating**: все фичи по умолчанию выключены, включаются через персональные флаги
6. **Contract versioning**: `chroma_contract_version="chroma_contract_v1"` для совместимости с downstream extractors
7. **Batch processing**: поддерживается через `extract_batch_segments()` с CPU parallelism для обработки нескольких файлов одновременно
8. **Stage timings**: измеряются и сохраняются в `stage_timings_ms` для анализа производительности
