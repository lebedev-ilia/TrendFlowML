## `chroma_extractor` (Audio signal processing extractor)

### Назначение

Извлекает **хрома-признаки** (12-полосный профиль классов высот, pitch class profile) с автоматической оценкой строя, нормализацией и статистическими агрегатами. Хрома-фичи отражают гармоническое содержание аудио и полезны для анализа музыки и речи.

**Версия**: 2.1.1 (Audit v4.2: stage timings + profiling)  
**Категория**: spectral  
**GPU**: не требуется

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter) — полный аудио файл
- **`audio/segments.json`** (Segmenter contract `audio_segments_v1`) — для `run_segments()`:
  - **required family**: `families.chroma.segments[]` — сегменты для обработки (no-fallback)

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/chroma_extractor/chroma_extractor_features.npz` (**фиксированное имя**)

Схема (Audit v3): `chroma_extractor_npz_v1` (см. `schemas/chroma_extractor_npz_v1.json` + `SCHEMA.md`).

#### Audit v4 — сводка артефакта NPZ (ключи верхнего уровня)

| Ключ / группа | `dtype` / форма | Tier | Заметка |
|---------------|-----------------|------|---------|
| `feature_names`, `feature_values` | `object[F]`, `float32[F]` | model_facing | 12× `chroma_mean_*` + entropy, stability, contrast, `chroma_dominant_energy`. **`chroma_dominant_class` только в массиве**, не в таблице. |
| `chroma_mean` | `float32[12]` | model_facing | Согласован с 12 именами в таблице; при агрегате сегментов сумма часто ≈ 1 (взвешенное среднее по L1-кадрам). |
| `chroma_entropy`, `chroma_harmonic_stability`, `chroma_contrast`, `chroma_dominant_energy` | скаляры `float32` | model_facing | Дублируются в tabular. |
| `chroma_dominant_class` | `int32` скаляр | analytics | 0…11 |
| `tuning_estimate` | `float32` скаляр | analytics | Semitones; при сбое оценки → `0.0`, `meta.tuning_failed=true`. |
| `segment_*`, `chroma_mean_by_segment` | см. ниже | analytics | Только при `run_segments()` **и** `enable_time_series=True`. |
| `chroma` | `float32[12,T]` | debug | Только режим **`run()`** (полный клип): если включён time series и матрица не слишком большая; иначе удаляется из payload и `meta.chroma_time_series_omitted=true`. В **`run_segments()` полная спектрограмма в NPZ не кладётся** — только per-segment матрица ниже. |
| `meta` | `object` (dict) | debug | В т.ч. `features_enabled`, `chroma_type`, `normalize`, `duration_sec`, … |

**Audit 4.2 (observability, meta):**
- `meta.stage_timings_ms`: тайминги стадий (мс) для `run()` и `run_segments()`.
- `meta.chroma_resource_profile` (optional): best-effort снимки RSS/VMS/GPU; включение `AP_CHROMA_RESOURCE_PROFILE=1`.

**Model-facing (Audit v3, всегда):**
- `chroma_mean`: `float32[12]`
- `chroma_entropy`: `float32` scalar
- `chroma_harmonic_stability`: `float32` scalar
- `chroma_contrast`: `float32` scalar
- `chroma_dominant_energy`: `float32` scalar

**Analytics (Audit v3, всегда):**
- `tuning_estimate`: `float32` scalar (если оценка не удалась → `0.0` + `meta.tuning_failed=true`)
- `chroma_dominant_class`: `int32` scalar (0..11)

**Debug time series (`enable_time_series`, режим `run()`):**
- `chroma`: `float32[12,T]` при лимите размера; иначе ключ отсутствует и `meta.chroma_time_series_omitted=true`.

**Segment-aligned sequence (`enable_time_series`, режим `run_segments()`):**
- `segment_centers_sec`: `float32[N]`
- `segment_durations_sec`: `float32[N]`
- `segment_mask`: `bool[N]` (strict alignment; “пропуски” не удаляются)
- `chroma_mean_by_segment`: `float32[N,12]` (для `segment_mask=false` = `NaN`)
- Полного ключа `chroma` в этом режиме **нет** (по контракту Audit v3); флаг `time_series` в `meta.features_enabled` означает включение **сегментной** выгрузки и/или спектрограммы в `run()`.

**Не входят в NPZ (Audit v3 minimal path):** внутренние `chroma_centroid` / `chroma_rolloff` из legacy-веток не сериализуются; ориентируйтесь на поля выше и `SCHEMA.md`.

**Метаданные (часть в `meta` артефакта, см. общий билдер meta):**
- `sample_rate`, `hop_length`, `n_fft`, `duration_sec`, `device_used`, `chroma_type`, `normalize`, `segments_count`, `features_enabled`, `chroma_contract_version`, …

### Feature Dependencies

**Зависимости между фичами (Audit v3):**
- minimal metrics зависят от вычисления chroma (CQT/STFT) и L1-normalization.
- segment sequence (`chroma_mean_by_segment`) доступна только при `enable_time_series=True`.

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
| `normalize` | str | `"l1"` | `"l1"` | Нормализация по кадрам (Audit v3 фиксировано) | +0.01 ms/frame | +0.001 ₽/frame |
| `n_chroma` | int | `12` | **только `12`** (Audit v3) | Количество хрома-классов (иное значение — fail-fast) | 0 ms | 0 ₽ |
| `fmin` | float | `null` | `> 0.0` \| `null` | Минимальная частота (Hz) | 0 ms | 0 ₽ |
| `fmax` | float | `null` | `> 0.0` \| `null` | Максимальная частота (Hz) | 0 ms | 0 ₽ |
| `n_bins` | int | `null` | `> 0` \| `null` | Количество бинов для CQT | 0 ms | 0 ₽ |
| `enable_audio_normalization` | bool | `false` | `true` \| `false` | Нормализация аудио перед обработкой | +0.05 ms/frame | +0.005 ₽/frame |
| `enable_time_series` | bool | `false` | `true` \| `false` | Debug time series / segment sequence | +0.1 ms/frame | +0.01 ₽/frame |

**Источник оценки**: бенчмарк/профилирование на типичных аудио файлах (30-120 секунд)

**Пример конфигурации (минимум, Audit v3)**:
```python
{
    "device": "auto",
    "sample_rate": 22050,
    "hop_length": 512,
    "chroma_type": "cqt",
}
```

**Пример конфигурации (legacy, не Audit v3)**:
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
    "enable_time_series": false,
}
```

### Features

**Features contract**: компонент имеет явный механизм выбора выходных фич через аргументы/конфиг.

**Дефолтный набор фич**: все фичи выключены по умолчанию (opt-in подход). Для включения фич используйте соответствующие флаги.

**Перечень всех возможных фич**:

#### Audit v3 feature scope

- статистики (basic/extended/stats_vector) **не поддерживаются** в audited контракте (fail-fast если включить).

#### Basic Stats (`enable_basic_stats=True`) — legacy
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

#### Extended Stats (`enable_extended_stats=True`) — legacy
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

#### Stats Vector (`enable_stats_vector=True`) — legacy
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
  - Audit v3: внешние `.npy` не используются; если массив слишком большой, time series опускается и фиксируется `meta.chroma_time_series_omitted=true`.

#### Additional Metrics (Audit v3 minimal — всегда в NPZ)

См. блок «Полезные поля» выше: `tuning_estimate`, `chroma_dominant_class`, `chroma_dominant_energy`, `chroma_harmonic_stability`, `chroma_entropy`, `chroma_contrast`. Дополнительные производные метрики без отдельных ключей в minimal path не экспортируются.

**В `meta` артефакта фиксируется**: `features_enabled` — список включённых фич (например `["time_series"]`).

**Feature Gating**:

Все фичи управляются персональными флагами (opt-in):

- `--chroma-enable-basic-stats`: базовые статистики (`chroma_mean`, `chroma_std`, `chroma_min`, `chroma_max`)
- `--chroma-enable-extended-stats`: расширенные статистики (`chroma_median`, `chroma_p25`, `chroma_p75`)
- `--chroma-enable-stats-vector`: компактный вектор статистик (`chroma_stats_vector`)
- `--chroma-enable-time-series`: временные серии (`chroma` spectrogram)

**Зависимости фичей (legacy):**
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
9. **Временные ряды** (feature-gated, debug):
   - Сохранение inline в NPZ (если размер <= threshold); иначе опускается (см. выше).

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

**Локальный HTML renderer для дебага (Audit v3):**
- `render_chroma_extractor_html()` генерирует offline HTML страницу (без CDN/Plotly)
- только для локального использования, не в production артефактах

### Примеры использования

#### Загрузка результатов

```python
import numpy as np

data = np.load("chroma_extractor_features.npz", allow_pickle=True)
meta = data["meta"].item()

# Minimal metrics (Audit v3)
chroma_mean = np.asarray(data["chroma_mean"])  # float32[12]
entropy = float(np.asarray(data["chroma_entropy"]))

# Tabular (опционально)
names = data["feature_names"].tolist()
values = data["feature_values"].astype(np.float64)

# Debug time series — только для run(), не для run_segments()
if "chroma" in data:
    chroma_ts = np.asarray(data["chroma"])  # shape: (12, frames)

tuning_estimate = float(np.asarray(data["tuning_estimate"]))
dominant_class = int(np.asarray(data["chroma_dominant_class"]))
harmonic_stability = float(np.asarray(data["chroma_harmonic_stability"]))
```

#### Анализ гармонического содержания

```python
# Найти доминирующий хрома-класс
chroma_mean = np.asarray(data["chroma_mean"])
if chroma_mean.size:
    dominant_class = int(np.asarray(data["chroma_dominant_class"]))
    chroma_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"Доминирующий класс: {chroma_names[dominant_class]} ({chroma_mean[dominant_class]:.3f})")

# Проверка гармонической стабильности
harmonic_stability = float(np.asarray(data["chroma_harmonic_stability"]))
print(f"Гармоническая стабильность: {harmonic_stability:.3f} (выше = стабильнее)")

# Энтропия распределения
entropy = float(np.asarray(data["chroma_entropy"]))
print(f"Энтропия: {entropy:.3f} (выше = более разнообразное распределение)")
```

#### Визуализация хрома-спектрограммы

```python
import matplotlib.pyplot as plt

if "chroma" in data:
    chroma_ts = np.array(data["chroma"])
    
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
7. **Key интеграция (in-memory)**: в рамках одного прогона AudioProcessor `chroma_extractor` предоставляет `_shared_chroma` (не сохраняется в NPZ) для ускорения `key_extractor` (опционально).
7. **Batch processing**: поддерживается через `extract_batch_segments()` с CPU parallelism для обработки нескольких файлов одновременно
8. **Stage timings**: измеряются и сохраняются в `stage_timings_ms` для анализа производительности
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [FEATURE_DESCRIPTION (root)](../FEATURE_DESCRIPTION.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
