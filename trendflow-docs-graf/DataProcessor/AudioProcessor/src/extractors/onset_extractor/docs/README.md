## `onset_extractor` (Audio Tier‑1, optional)

### Назначение

Определяет **онсеты (атаки звука)** — моменты начала новых звуковых событий в аудио сигнале. Онсеты важны для анализа ритма, сегментации музыки, обнаружения ударных инструментов и других задач музыкального анализа.

**Версия**: 2.0.1 (Audit v4.2 observability)  
**Категория**: rhythm  
**GPU**: не требуется (CPU-only обработка)

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** (Segmenter contract, family `onset`) — сегменты для `run_segments()`

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/onset_extractor/onset_extractor_features.npz` (**фиксированное имя**)

Схема: **`onset_extractor_npz_v2`** (`schemas/onset_extractor_npz_v2.json`, `docs/SCHEMA.md`).

#### Audit v4 — заметки по NPZ

- **Tabular:** только числа; строковый **`backend`** ранее попадал в tabular и давал **NaN** через `as_float` — **исправлено в `npz_savers/onset.py`**, значение в **`meta.backend`**.

#### Audit v4.2 observability:
- `meta.stage_timings_ms` (dict): тайминги этапов (ms), пишутся в NPZ meta
- `meta.onset_resource_profile` (dict|None): best-effort snapshot RSS/VMS (если включено)
  - включение: `AP_ONSET_RESOURCE_PROFILE=1`

#### Полезные поля payload (feature-gated):

**Basic features** (`--onset-enable-basic-features`):
- `onset_times`: массив времен онсетов в секундах (numpy array, dtype: float32)
- `onset_count`: количество обнаруженных онсетов (int)
- `onset_density_per_sec`: плотность онсетов (количество онсетов в секунду) (float)
- `insufficient_onsets`: флаг, указывающий что обнаружено <= 1 онсет (bool)

**Interval stats** (`--onset-enable-interval-stats`):
- `avg_interval_sec`: средний интервал между онсетами в секундах (float, None если онсетов <= 1)
- `interval_std`: стандартное отклонение интервалов между онсетами (float, None если интервалов нет)
- `interval_min`: минимальный интервал между онсетами (float, None если интервалов нет)
- `interval_max`: максимальный интервал между онсетами (float, None если интервалов нет)
- `interval_median`: медианный интервал между онсетами (float, None если интервалов нет)

**Rhythmic metrics** (`--onset-enable-rhythmic-metrics`):
- `onset_regularity_score`: регулярность ритма (0-1, где 1 = идеально регулярный)
- `onset_clustering_score`: мера кластеризации онсетов по времени (0-1)
- `onset_tempo_estimate`: оценка BPM из интервалов между онсетами (float)
- `onset_syncopation_score`: мера синкопированности (0-1, где 1 = максимальная синкопированность)
- `onset_strength_mean`: средняя сила онсетов (float)
- `onset_strength_std`: стандартное отклонение силы онсетов (float)
- `onset_density_variance`: вариация плотности онсетов по времени (float)
- `onset_tempo_consistency`: согласованность с tempo_extractor (0-1, если доступен) (float)

**Time series** (`--onset-enable-time-series`):
- `onset_times`: массив времен онсетов в секундах (numpy array, dtype: float32)
- Если `onset_times.size > 10000`, сохраняется в `.npy` файл в `_artifacts/onset_times.npy`

**Per-segment data** (для `run_segments()`):
- `segment_centers_sec`: центры сегментов в секундах (float32[L])
- `segment_durations_sec`: длительности сегментов в секундах (float32[L])
- `segments_count`: количество сегментов (int)

**Метаданные:**
- `sample_rate`: частота дискретизации (int)
- `hop_length`: размер hop для анализа (int, по умолчанию 512)
- `duration`: длительность аудио в секундах (float)
- `device_used`: устройство обработки (str, обычно "cpu")
- `backend`: используемый backend (`"librosa"` \| `"essentia"`) — в **NPZ** в **`meta`**, не в `feature_values`
- `onset_contract_version`: версия контракта ("onset_contract_v1")

### Models

ML модели **не используются** (signal processing only). `models_used[]` пустой.

**Библиотеки:**
- **librosa**: основная библиотека для обнаружения онсетов (default backend)
  - **Triton**: ❌ Нет (in-process)
  - **Runtime**: `inprocess`
  - **Engine**: `librosa` (signal processing)
  - **Precision**: `fp32`
  - **Device**: `cpu`
- **essentia** (опционально): более точный алгоритм обнаружения онсетов (если доступен)
  - **Triton**: ❌ Нет (in-process)
  - **Runtime**: `inprocess`
  - **Engine**: `essentia` (signal processing)
  - **Precision**: `fp32`
  - **Device**: `cpu`

### Feature Dependencies

- **Опциональная интеграция с `tempo_extractor`**: если доступны результаты `tempo_extractor`, используется для валидации/улучшения результатов (метрика `onset_tempo_consistency`).

### Конфигурация

**Параметры конфигурации компонента**:

| Параметр | Тип | Значение по умолчанию | Допустимые значения | Описание | Δ latency | Δ cost |
|----------|-----|----------------------|---------------------|----------|-----------|--------|
| `device` | str | `"auto"` | `"auto"` \| `"cpu"` | Устройство для обработки | 0 ms | 0 ₽ |
| `sample_rate` | int | `22050` | `> 0` | Частота дискретизации | +0.1 ms/frame при увеличении на 1kHz | +0.01 ₽/frame |
| `hop_length` | int | `512` | `> 0` | Размер hop для анализа онсетов | -0.05 ms/frame при увеличении на 64 | -0.005 ₽/frame |
| `pre_max` | int | `3` | `>= 0` | Количество кадров до максимума для пикового детектора | +0.01 ms/frame при увеличении на 1 | +0.001 ₽/frame |
| `post_max` | int | `3` | `>= 0` | Количество кадров после максимума для пикового детектора | +0.01 ms/frame при увеличении на 1 | +0.001 ₽/frame |
| `pre_avg` | int | `3` | `>= 0` | Количество кадров до для усреднения | +0.01 ms/frame при увеличении на 1 | +0.001 ₽/frame |
| `post_avg` | int | `5` | `>= 0` | Количество кадров после для усреднения | +0.01 ms/frame при увеличении на 1 | +0.001 ₽/frame |
| `delta` | float | `0.2` | `>= 0.0` | Минимальная разница для обнаружения онсета | 0 ms | 0 ₽ |
| `wait` | int | `10` | `>= 0` | Минимальное количество кадров между онсетами | 0 ms | 0 ₽ |
| `backend` | str | `"librosa"` | `"librosa"` \| `"essentia"` | Backend для обнаружения онсетов (no-fallback policy) | -0.1 ms/frame для essentia | -0.01 ₽/frame |
| `units` | str | `"time"` | `"time"` \| `"frames"` | Единицы измерения для онсетов | 0 ms | 0 ₽ |
| `backtrack` | bool | `false` | `true` \| `false` | Включить backtrack для обнаружения онсетов | +0.05 ms/frame | +0.005 ₽/frame |
| `energy` | bool | `false` | `true` \| `false` | Использовать энергетический детектор | +0.1 ms/frame | +0.01 ₽/frame |
| `normalize` | bool | `false` | `true` \| `false` | Нормализовать onset envelope | +0.02 ms/frame | +0.002 ₽/frame |
| `enable_basic_features` | bool | `false` | `true` \| `false` | Включить базовые фичи | +0.01 ms/frame | +0.001 ₽/frame |
| `enable_interval_stats` | bool | `false` | `true` \| `false` | Включить статистики интервалов | +0.05 ms/frame | +0.005 ₽/frame |
| `enable_rhythmic_metrics` | bool | `false` | `true` \| `false` | Включить ритмические метрики | +0.1 ms/frame | +0.01 ₽/frame |
| `enable_time_series` | bool | `false` | `true` \| `false` | Включить временные серии | +0.02 ms/frame | +0.002 ₽/frame |
| `enable_audio_normalization` | bool | `false` | `true` \| `false` | Включить нормализацию аудио перед обработкой | +0.05 ms/frame | +0.005 ₽/frame |

**Источник оценки**: бенчмарк/профилирование на типичных аудио файлах (30-120 секунд)

**Пример конфигурации (минимум)**:
```python
{
    "device": "auto",
    "sample_rate": 22050,
    "hop_length": 512,
    "backend": "librosa",
    "enable_basic_features": true,
}
```

**Пример конфигурации (расширенный)**:
```python
{
    "device": "auto",
    "sample_rate": 22050,
    "hop_length": 512,
    "pre_max": 3,
    "post_max": 3,
    "pre_avg": 3,
    "post_avg": 5,
    "delta": 0.2,
    "wait": 10,
    "backend": "librosa",
    "units": "time",
    "backtrack": false,
    "energy": false,
    "normalize": false,
    "enable_basic_features": true,
    "enable_interval_stats": true,
    "enable_rhythmic_metrics": true,
    "enable_time_series": false,
    "enable_audio_normalization": false,
}
```

#### Python API

```python
from src.extractors.onset_extractor import OnsetExtractor

extractor = OnsetExtractor(
    device="auto",
    sample_rate=22050,
    hop_length=512,
    pre_max=3,
    post_max=3,
    pre_avg=3,
    post_avg=5,
    delta=0.2,
    wait=10,
    backend="librosa",
    units="time",
    backtrack=False,
    energy=False,
    normalize=False,
    enable_basic_features=True,
    enable_interval_stats=False,
    enable_rhythmic_metrics=False,
    enable_time_series=False,
    enable_audio_normalization=False,
    tempo_payload=None,  # Optional: results from tempo_extractor for validation
    progress_callback=None,
    artifacts_dir=None,
)
```

### Features

**Features contract**: компонент имеет явный механизм выбора выходных фич через аргументы/конфиг.

**Дефолтный набор фич**: все фичи выключены по умолчанию (opt-in подход). Для включения фич используйте соответствующие флаги.

**Перечень всех возможных фич**:

#### Basic Features (`enable_basic_features=True`)
- `onset_times`: массив времен онсетов в секундах (numpy array, dtype: float32)
  - Формат: per-run (весь аудио файл) или per-segment (для run_segments)
  - Единица: секунды
  - Влияние на стоимость: +0.01 ms/frame, +0.001 ₽/frame
- `onset_count`: количество обнаруженных онсетов (int)
  - Формат: per-run
  - Единица: количество
  - Влияние на стоимость: +0.01 ms/frame, +0.001 ₽/frame
- `onset_density_per_sec`: плотность онсетов (количество онсетов в секунду) (float)
  - Формат: per-run
  - Единица: онсетов/сек
  - Влияние на стоимость: +0.01 ms/frame, +0.001 ₽/frame
- `insufficient_onsets`: флаг, указывающий что обнаружено <= 1 онсет (bool)
  - Формат: per-run
  - Единица: boolean
  - Влияние на стоимость: +0.01 ms/frame, +0.001 ₽/frame

#### Interval Stats (`enable_interval_stats=True`)
- `avg_interval_sec`: средний интервал между онсетами в секундах (float, None если онсетов <= 1)
  - Формат: per-run
  - Единица: секунды
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame
- `interval_std`: стандартное отклонение интервалов между онсетами (float, None если интервалов нет)
  - Формат: per-run
  - Единица: секунды
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame
- `interval_min`: минимальный интервал между онсетами (float, None если интервалов нет)
  - Формат: per-run
  - Единица: секунды
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame
- `interval_max`: максимальный интервал между онсетами (float, None если интервалов нет)
  - Формат: per-run
  - Единица: секунды
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame
- `interval_median`: медианный интервал между онсетами (float, None если интервалов нет)
  - Формат: per-run
  - Единица: секунды
  - Влияние на стоимость: +0.05 ms/frame, +0.005 ₽/frame

#### Rhythmic Metrics (`enable_rhythmic_metrics=True`)
- `onset_regularity_score`: регулярность ритма (0-1, где 1 = идеально регулярный) (float)
  - Формат: per-run
  - Единица: score (0-1)
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `onset_clustering_score`: мера кластеризации онсетов по времени (0-1) (float)
  - Формат: per-run
  - Единица: score (0-1)
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `onset_tempo_estimate`: оценка BPM из интервалов между онсетами (float)
  - Формат: per-run
  - Единица: BPM
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `onset_syncopation_score`: мера синкопированности (0-1, где 1 = максимальная синкопированность) (float)
  - Формат: per-run
  - Единица: score (0-1)
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `onset_strength_mean`: средняя сила онсетов (float)
  - Формат: per-run
  - Единица: strength
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `onset_strength_std`: стандартное отклонение силы онсетов (float)
  - Формат: per-run
  - Единица: strength
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `onset_density_variance`: вариация плотности онсетов по времени (float)
  - Формат: per-run
  - Единица: variance
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame
- `onset_tempo_consistency`: согласованность с tempo_extractor (0-1, если доступен) (float)
  - Формат: per-run
  - Единица: score (0-1)
  - Влияние на стоимость: +0.1 ms/frame, +0.01 ₽/frame

#### Time Series (`enable_time_series=True`)
- `onset_times`: массив времен онсетов в секундах (numpy array, dtype: float32)
  - Формат: per-run (весь аудио файл) или per-segment (для run_segments)
  - Единица: секунды
  - Влияние на стоимость: +0.02 ms/frame, +0.002 ₽/frame
  - Если `onset_times.size > 10000`, сохраняется в `.npy` файл в `_artifacts/onset_times.npy`

**В meta артефакта фиксируется**: `_features_enabled[]` — список включенных фич.

**Feature Gating**:

Все фичи управляются персональными флагами (opt-in):

- `--onset-enable-basic-features`: базовые фичи (onset_times, onset_count, onset_density_per_sec, insufficient_onsets)
- `--onset-enable-interval-stats`: статистики интервалов (avg_interval_sec, interval_std, interval_min, interval_max, interval_median)
- `--onset-enable-rhythmic-metrics`: ритмические метрики (regularity, clustering, tempo_estimate, syncopation, strength, density_variance, tempo_consistency)
- `--onset-enable-time-series`: временные серии (onset_times как time series)

### Алгоритм работы

1. **Загрузка аудио**: через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. **Опциональная нормализация**: если `enable_audio_normalization=True`, аудио нормализуется перед обработкой
3. **Выбор канала**: если аудио многоканальное, выбирается канал с максимальной RMS энергией
4. **Обнаружение онсетов** (no-fallback policy):
   - **Backend "essentia"**: использование `essentia.standard.OnsetRate()` (fail-fast если недоступен)
   - **Backend "librosa"**: использование `librosa.onset.onset_detect()` (fail-fast при ошибке)
5. **Вычисление интервалов**: разности между последовательными онсетами
6. **Вычисление статистик**:
   - Статистики интервалов (если `enable_interval_stats=True`)
   - Ритмические метрики (если `enable_rhythmic_metrics=True`)
7. **Валидация выходных данных**: полная валидация (NaN/inf, диапазоны, консистентность)
8. **Опциональная валидация с tempo_extractor**: если доступны результаты `tempo_extractor`, вычисляется `onset_tempo_consistency`
9. **Сохранение больших массивов**: если `onset_times.size > 10000`, сохраняется в `.npy` файл в `_artifacts/`

### Error Handling

Детальные error codes:
- `onset_audio_load_failed`: ошибка загрузки аудио
- `onset_essentia_failed`: ошибка backend "essentia" (недоступен или произошла ошибка)
- `onset_librosa_failed`: ошибка backend "librosa"
- `onset_validation_failed`: ошибка валидации выходных данных
- `onset_insufficient_data`: недостаточно данных для обработки
- `onset_unknown`: неизвестная ошибка

**No-fallback policy**: если выбранный backend недоступен или произошла ошибка, extractor завершается с ошибкой (fail-fast), без автоматического fallback.

### Sampling / units-of-processing requirements

**Стратегия выборки**: Segmenter-owned (универсальная нелинейная кривая)

Onset extractor использует сегменты из Segmenter contract (family `onset`). Segmenter является единственным владельцем sampling — компонент не генерирует семплинг сам.

**Требования к сегментам**:
- Сегменты должны быть предоставлены через `audio/segments.json` (family `onset`)
- Для `run_segments()` требуется наличие family `onset` в `segments.json`
- Если family `onset` отсутствует, extractor завершается с ошибкой (no-fallback policy)

**Параметры нелинейной кривой** (Segmenter-owned):
- Ссылка на `docs/contracts/SEGMENTER_CONTRACT.md` как источник истины
- Параметры кривой определяются Segmenter и передаются через `segments.json`

**Ограничения**:
- Минимальная длительность сегмента: определяется Segmenter
- Максимальная длительность сегмента: определяется Segmenter
- Coverage requirements: определяется Segmenter

### Performance characteristics

**Resource costs**:
- **CPU**: O(N * log(N)) для анализа онсетов, где N — длина аудио
- **Память**: O(N) для временных массивов
- **Estimated duration**: ~0.8 секунд для типичного аудио

**Единица обработки**: 
- `run()`: весь аудио файл (не сегментированный)
- `run_segments()`: сегменты от Segmenter (family `onset`)

**Типичные значения (preset="default")**:

| Audio Duration | Latency per unit | CPU RAM peak | Notes |
|----------------|------------------|--------------|-------|
| 30 sec | ~0.8 sec | ~50 MB | typical |
| 60 sec | ~1.5 sec | ~100 MB | typical |
| 120 sec | ~3.0 sec | ~200 MB | typical |

**Для аудио с длительностью N секунд**: Total latency ≈ N × 0.026 sec (примерно)

**Оптимизации**:
- Использование Essentia если доступен (более эффективный алгоритм)
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

**Рекомендуемые визуализации для UI/сайта**:

1. **Timeline визуализация**:
   - Линейный график с временной осью (X) и отметками онсетов (вертикальные линии или маркеры)
   - Интерактивные tooltips с временем онсета
   - Zoom и pan для детального просмотра

2. **Распределение интервалов**:
   - Гистограмма интервалов между онсетами
   - Показ статистик (mean, median, std, min, max)

3. **Ритмические метрики**:
   - Gauge/radial charts для regularity_score, clustering_score, syncopation_score
   - Числовые значения для tempo_estimate, strength_mean/std, density_variance

4. **Сравнение с tempo**:
   - Если доступен `onset_tempo_consistency`, показать согласованность с `tempo_extractor`
   - Сравнение `onset_tempo_estimate` с `tempo_bpm` из `tempo_extractor`

5. **Плотность онсетов**:
   - График плотности онсетов по времени (скользящее окно)
   - Показ вариации плотности (`onset_density_variance`)

**Локальный HTML renderer для дебага**:
- Используйте `render_onset_extractor_html()` для генерации HTML страницы с интерактивными графиками
- Доступен через `src/core/renderer.py`

### Примеры использования

#### Загрузка результатов

```python
import numpy as np

data = np.load("onset_extractor_features.npz", allow_pickle=True)
payload = data["payload"].item()

# Времена онсетов (если включено time_series)
onset_times = payload.get("onset_times")  # numpy array, shape: (onset_count,)
print(f"Найдено онсетов: {payload.get('onset_count')}")
print(f"Времена онсетов: {onset_times}")

# Ритмические метрики (если включено rhythmic_metrics)
regularity = payload.get("onset_regularity_score")
tempo_estimate = payload.get("onset_tempo_estimate")
print(f"Регулярность ритма: {regularity:.3f}")
print(f"Оценка BPM: {tempo_estimate:.1f}")
```

#### Визуализация онсетов

```python
import matplotlib.pyplot as plt
import numpy as np

onset_times = payload.get("onset_times")
duration = payload.get("duration")

# Визуализация временной линии
plt.figure(figsize=(12, 4))
plt.vlines(onset_times, 0, 1, colors='r', linestyles='solid', linewidth=2, label='Onsets')
plt.xlabel("Time (seconds)")
plt.ylabel("Onset Events")
plt.title("Onset Detection")
plt.xlim(0, duration)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

#### Анализ ритмических паттернов

```python
# Статистики интервалов (если включено interval_stats)
interval_std = payload.get("interval_std")
interval_min = payload.get("interval_min")
interval_max = payload.get("interval_max")
interval_median = payload.get("interval_median")

print(f"Стандартное отклонение интервалов: {interval_std:.3f} сек")
print(f"Минимальный интервал: {interval_min:.3f} сек")
print(f"Максимальный интервал: {interval_max:.3f} сек")
print(f"Медианный интервал: {interval_median:.3f} сек")

# Анализ регулярности ритма
regularity = payload.get("onset_regularity_score")
if regularity > 0.8:
    print("Ритм очень регулярный")
elif regularity < 0.5:
    print("Ритм нерегулярный")
```

#### Вычисление BPM из онсетов

```python
onset_times = payload.get("onset_times")
tempo_estimate = payload.get("onset_tempo_estimate")

if tempo_estimate and tempo_estimate > 0:
    print(f"Оценка BPM: {tempo_estimate:.1f}")
else:
    print("Недостаточно онсетов для оценки BPM")
```

#### Проверка качества обнаружения

```python
# Проверка достаточности онсетов
if payload.get("insufficient_onsets"):
    print("ВНИМАНИЕ: Обнаружено недостаточно онсетов (<= 1)")
    print("Результаты могут быть ненадежными")
else:
    print("Обнаружено достаточное количество онсетов")
    print(f"Плотность: {payload.get('onset_density_per_sec'):.3f} онсетов/сек")
```

#### Сравнение с другими компонентами

```python
# Онсеты можно использовать вместе с tempo_extractor
# для более точного анализа ритма

onset_times = payload.get("onset_times")
onset_count = payload.get("onset_count")
duration = payload.get("duration")

# Плотность онсетов
density = onset_count / duration if duration > 0 else 0

# Сравнение с ожидаемым темпом (если есть результаты от tempo_extractor)
tempo_consistency = payload.get("onset_tempo_consistency")
if tempo_consistency is not None:
    print(f"Согласованность с tempo_extractor: {tempo_consistency:.3f}")
```

### Связанные компоненты

- **Segmenter**: предоставляет аудио файл и сегменты (family `onset`)
- **librosa**: основная библиотека для обнаружения онсетов (default backend)
- **essentia** (опционально): более точный алгоритм обнаружения онсетов
- **tempo_extractor**: может использоваться вместе для комплексного анализа ритма (опциональная интеграция)
- **AudioUtils**: утилиты для загрузки аудио

### Quality validation & human-friendly inspection

#### Как проверить качество выхода компонента

**1. Автоматическая оценка**:
- Проверка валидности выходных данных через `_validate_output()`:
  - Отсутствие NaN/Inf значений в `onset_times`
  - Монотонное возрастание `onset_times`
  - Консистентность `onset_count == len(onset_times)`
  - Разумные диапазоны значений для всех метрик

**2. Human-friendly визуализация**:
- **Timeline визуализация**: линейный график с временной осью (X) и отметками онсетов (вертикальные линии или маркеры)
- **Распределение интервалов**: гистограмма интервалов между онсетами с показом статистик (mean, median, std, min, max)
- **Ритмические метрики**: gauge/radial charts для regularity_score, clustering_score, syncopation_score
- **Сравнение с tempo**: если доступен `onset_tempo_consistency`, показать согласованность с `tempo_extractor`
- **Плотность онсетов**: график плотности онсетов по времени (скользящее окно)

**Локальный HTML renderer для дебага**:
- Используйте `render_onset_extractor_html()` для генерации HTML страницы с интерактивными графиками
- Доступен через `src/core/renderer.py`

**3. Статистическая валидация**:
- Проверка разумности значений:
  - `onset_count >= 0`
  - `onset_density_per_sec >= 0` и разумные значения (обычно 0.1-10 онсетов/сек)
  - `avg_interval_sec >= 0` и разумные значения (обычно 0.1-10 секунд)
  - `onset_regularity_score` в диапазоне [0, 1]
  - `onset_tempo_estimate` в диапазоне [0, 300] BPM (разумные значения)
- Отсутствие аномальных значений (NaN где не ожидается, бесконечности)

**4. Интеграция с downstream компонентами**:
- Проверка корректности чтения артефактов downstream компонентами
- Формат данных соответствует ожиданиям downstream
- Временные метки правильно выровнены

### Примечания

1. **Essentia vs librosa**: Essentia предоставляет более точный алгоритм обнаружения онсетов, но требует установки. Librosa используется как default backend. **No-fallback policy**: если выбранный backend недоступен, extractor завершается с ошибкой.
2. **Параметры детектора**: параметры `pre_max`, `post_max`, `pre_avg`, `post_avg`, `delta`, `wait` настраивают чувствительность детектора онсетов.
3. **Недостаточные онсеты**: если обнаружено <= 1 онсет, статистики интервалов будут None, а флаг `insufficient_onsets` будет True.
4. **Многоканальное аудио**: автоматически выбирается канал с максимальной RMS энергией для анализа.
5. **Применение**: онсеты используются для сегментации музыки, обнаружения ударных, анализа ритма, синхронизации.
6. **Ритмические метрики**: интервалы между онсетами и их статистики помогают анализировать регулярность и структуру ритма.
7. **Feature gating**: все фичи управляются персональными флагами (opt-in), по умолчанию все фичи выключены.
8. **Per-run storage**: большие массивы (`onset_times.size > 10000`) сохраняются в `.npy` файлы в `_artifacts/` для оптимизации размера NPZ.
9. **Contract versioning**: используется `onset_contract_version="onset_contract_v1"` для совместимости с downstream extractors.
10. **Batch processing**: поддерживается через `extract_batch_segments()` с CPU parallelism для обработки нескольких файлов одновременно.
11. **Stage timings**: измеряются и сохраняются в `stage_timings_ms` для анализа производительности.
---

## Навигация

[FEATURE_DESCRIPTION](FEATURE_DESCRIPTION.md) · [SCHEMA](SCHEMA.md) · [TESTING_REPORT](TESTING_REPORT.md) · [AudioProcessor](../../../../docs/MAIN_INDEX.md) · [DataProcessor](../../../../../docs/MAIN_INDEX.md) · [Vault](../../../../../../docs/MAIN_INDEX.md)
