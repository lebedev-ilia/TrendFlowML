## `band_energy_extractor` (Audio signal processing extractor)

### Назначение

Извлекает **энергии по частотным полосам** (низ/середина/высокие) и их доли. Поддерживает фиксированные полосы или мел-шкалу, опционально возвращает временные ряды (per-frame энергии). Поддерживает segment-based обработку для отслеживания изменений частотного баланса.

**Версия**: 2.0.0  
**Категория**: spectral  
**GPU**: не требуется

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter) — полный аудио файл
- **`audio/segments.json`** (Segmenter contract `audio_segments_v1`) — для `run_segments()`:
  - `families.band_energy.segments[]` — сегменты для обработки энергий по полосам

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/band_energy_extractor/band_energy_extractor_features.npz` (**фиксированное имя**)

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

#### Полезные поля payload:

**Основной результат (всегда включен):**
- `band_edges`: список границ полос `[(lo, hi), ...]` в Hz
- `band_energies`: суммарные энергии по полосам (list[float])
- `band_energy_shares`: доли энергии по полосам (list[float], нормализованные, сумма = 1.0)
- `total_energy`: общая энергия сигнала (float)
- `method`: использованный метод ("essentia" | "librosa")

**Базовые статистики (feature-gated):**
- `band_energy_mean`: средние энергии по полосам (list[float], если `enable_basic_stats=True`)
- `band_energy_std`: стандартные отклонения энергий (list[float], если `enable_basic_stats=True`)
- `band_energy_median`: медианные энергии по полосам (list[float], если `enable_basic_stats=True`)

**Расширенные статистики (feature-gated):**
- `band_energy_min`: минимальные энергии по полосам (list[float], если `enable_extended_stats=True`)
- `band_energy_max`: максимальные энергии по полосам (list[float], если `enable_extended_stats=True`)
- `band_energy_p25`: 25-й перцентиль энергий (list[float], если `enable_extended_stats=True`)
- `band_energy_p75`: 75-й перцентиль энергий (list[float], если `enable_extended_stats=True`)

**Временные серии (feature-gated):**
- `band_energy_ts`: временные ряды энергий по полосам (list[list[float]], shape: [num_bands, frames], если `enable_time_series=True`)
- `segment_centers_sec`: центры сегментов в секундах (float32[N], если `enable_time_series=True` и `run_segments()`)
- `segment_durations`: длительности сегментов в секундах (float32[N], если `enable_time_series=True` и `run_segments()`)

**Метрики баланса (feature-gated):**
- `band_balance_score`: оценка баланса между полосами (float, 0.0-1.0, энтропия распределения, если `enable_balance_metrics=True`)
- `band_dominance`: индекс доминирующей полосы (int, если `enable_balance_metrics=True`)
- `band_contrast`: контраст между полосами (float, max - min, если `enable_balance_metrics=True`)

**Метрики динамики (feature-gated, для `run_segments()`):**
- `band_energy_stability`: стабильность распределения энергий (float, 0.0-1.0, если `enable_dynamics=True`)
- `band_transitions`: список переходов между доминирующими полосами (list[dict], если `enable_dynamics=True`)
  - Каждый элемент: `{"transition_index": int, "from_band": int, "to_band": int, "transition_time_sec": float}`
- `band_transitions_count`: количество переходов (int, если `enable_dynamics=True`)
- `band_transitions_rate`: частота переходов (transitions/sec, float, если `enable_dynamics=True`)
- `band_distribution`: распределение времени по доминирующим полосам (dict[int, float], если `enable_dynamics=True`)
- `band_diversity`: разнообразие доминирующих полос (int, количество уникальных полос, если `enable_dynamics=True`)

**Метаданные:**
- `sample_rate`: частота дискретизации (int)
- `n_fft`: размер FFT окна (int)
- `hop_length`: размер hop для STFT (int)
- `duration`: длительность аудио в секундах (float)
- `device_used`: устройство обработки (str)
- `band_energy_contract_version`: версия контракта ("band_energy_contract_v1")

### Feature Dependencies

**Зависимости между фичами:**
- `band_energy_stability` зависит от `enable_dynamics` и `enable_time_series` (требует временных серий)
- `band_transitions` и `band_transitions_count` зависят от `enable_dynamics` и `enable_time_series` (требуют временных серий)
- Все метрики динамики зависят от `enable_dynamics` и `enable_time_series` (требуют временных серий)
- `band_balance_score` зависит от `enable_balance_metrics` и вычисления `band_energy_shares`

**Зависимости от других extractors:**
- **spectral_extractor** (опционально): может использовать предвычисленный `stft_magnitude` и `frequencies` из `shared_features` для оптимизации (избегает повторного вычисления STFT)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

**Библиотеки:**
- **librosa**: основная библиотека для STFT и мел-шкалы
- **Essentia** (опционально): если доступен, используется как альтернативный метод

### Конфигурация

```python
{
    "sample_rate": 22050,           # Частота дискретизации
    "bands": None,                   # Список полос [(lo, hi), ...] в Hz, по умолчанию: [(0, 200), (200, 2000), (2000, nyq)]
    "n_fft": 2048,                   # Размер FFT окна
    "hop_length": 512,              # Размер hop для STFT
    "use_mel_bands": True,          # Использовать мел-шкалу вместо фиксированных полос
    "n_mels": 3,                    # Количество мел-полос (если use_mel_bands=True)
    "band_method": "auto",          # "essentia" | "librosa" | "auto" (явный выбор метода)
    "enable_audio_normalization": False,  # Нормализация аудио перед обработкой
    # Feature gating flags (все по умолчанию False)
    "enable_basic_stats": False,     # Включить базовые статистики (mean, std, median)
    "enable_extended_stats": False,  # Включить расширенные статистики (min, max, p25, p75)
    "enable_time_series": False,     # Включить временные серии (band_energy_ts)
    "enable_dynamics": False,        # Включить метрики динамики (для run_segments)
    "enable_balance_metrics": False,  # Включить метрики баланса
    "device": "auto"                 # "auto" | "cuda" | "cpu"
}
```

### Feature Gating

Все фичи контролируются через персональные флаги (по умолчанию все выключены, кроме базовых полей):

- `--band-energy-enable-basic-stats`: Включить базовые статистики (`band_energy_mean`, `band_energy_std`, `band_energy_median`)
- `--band-energy-enable-extended-stats`: Включить расширенные статистики (`band_energy_min`, `band_energy_max`, `band_energy_p25`, `band_energy_p75`)
- `--band-energy-enable-time-series`: Включить временные серии (`band_energy_ts`, `segment_centers_sec`, `segment_durations`)
- `--band-energy-enable-dynamics`: Включить метрики динамики (`band_energy_stability`, `band_transitions`, `band_transitions_count`, `band_transitions_rate`, `band_distribution`, `band_diversity`)
- `--band-energy-enable-balance-metrics`: Включить метрики баланса (`band_balance_score`, `band_dominance`, `band_contrast`)

**Зависимости фичей:**
- `enable_dynamics` требует `enable_time_series` (для отслеживания изменений по сегментам)

### Алгоритм работы

#### Метод выбора (явный, no-fallback):

1. **Явный выбор метода** через `band_method`:
   - **"essentia"**: только Essentia, fail-fast если недоступна
   - **"librosa"**: только librosa
   - **"auto"**: Essentia с fallback на librosa (если Essentia недоступна или ошибка)

#### Essentia путь (если выбран "essentia" или "auto"):

1. **Проверка Essentia**: если библиотека доступна, используется Essentia для frame-by-frame обработки
2. **Frame cutting**: разбиение аудио на кадры через `FrameCutter`
3. **Windowing**: применение окна Ханна
4. **Spectrum**: вычисление спектра мощности
5. **Биннинг**: вычисление энергий по полосам через маски частот
6. **Агрегация**: суммирование энергий по кадрам

#### Librosa путь:

1. **Загрузка аудио**: через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. **Опциональная нормализация аудио**: если `enable_audio_normalization=True`
3. **Сведение в моно**: усреднение каналов для стерео аудио
4. **Проверка минимальной длительности**: fail-fast если аудио < 1 секунды
5. **Вычисление STFT**:
   - **Оптимизация**: если `shared_features` содержит `stft_magnitude`, используется он
   - **STFT**: вычисление спектрограммы мощности через `librosa.stft()`
6. **Частотные полосы**:
   - **Фиксированные**: по умолчанию `[(0, 200), (200, 2000), (2000, nyq)]` Hz
   - **Мел-шкала**: если `use_mel_bands=True`, полосы строятся через `librosa.mel_frequencies()`
7. **Векторизованный биннинг**: эффективное вычисление энергий по полосам через матричное умножение масок
8. **Статистики** (feature-gated): mean, std, median, min, max, p25, p75 по временным рядам
9. **Метрики баланса** (feature-gated): balance score, dominance, contrast
10. **Временные серии** (feature-gated): сохранение per-frame энергий

#### Segment-based обработка (`run_segments()`):

1. **Загрузка сегментов**: из `families.band_energy.segments[]` от Segmenter
2. **Обработка каждого сегмента**: вычисление энергий по полосам для каждого сегмента
3. **Агрегация результатов**: усреднение энергий и shares по сегментам
4. **Временные серии** (если `enable_time_series=True`): сохранение последовательностей по сегментам
5. **Метрики динамики** (если `enable_dynamics=True`): вычисление стабильности, переходов, распределения

### Особенности

- **Явный выбор метода**: контроль через `band_method` (essentia/librosa/auto), no-fallback policy
- **Векторизованный биннинг**: эффективное вычисление через матричные операции (librosa)
- **Гибкие полосы**: поддержка фиксированных полос или мел-шкалы
- **Shared STFT**: может использовать предвычисленный STFT из `shared_features` для оптимизации
- **Segment-based processing**: поддержка `run_segments()` для отслеживания изменений частотного баланса
- **Dynamics metrics**: метрики динамики для анализа изменений по времени
- **Balance metrics**: метрики баланса для анализа частотного распределения

### Error Handling

Детальные error codes:
- `audio_load_failed`: Ошибка загрузки аудио
- `audio_too_short`: Аудио слишком короткое (< 1 секунды)
- `stft_computation_failed`: Ошибка вычисления STFT
- `band_computation_failed`: Ошибка вычисления энергий по полосам
- `essentia_unavailable`: Essentia недоступна (если выбран метод "essentia")
- `invalid_parameters`: Невалидные параметры (bands, n_fft, и т.д.)

**No-fallback policy:**
- Отсутствие обязательного входа → fail-fast с `RuntimeError`
- Пустой список segments → `ValueError("segments is empty (no-fallback)")`
- Невалидные параметры → `ValueError` с описанием
- Ошибки вычисления → `status="error"` с error_code

### Performance characteristics

**Resource costs:**
- **CPU**: O(N * log(N)) для STFT, где N — длина аудио
- **Память**: O(freq_bins * frames) для спектрограммы
- **Estimated duration**: ~0.9 секунд для типичного аудио (для `run()`)
- **Segment-based overhead**: для `run_segments()`, время пропорционально количеству сегментов

**Единица обработки:**
- `run()`: весь аудио файл
- `run_segments()`: сегменты от Segmenter (`families.band_energy.segments[]`)

### Parallelization

**Внутренний параллелизм** (внутри компонента):
- **CPU parallelism**: компонент поддерживает параллельную обработку сегментов из нескольких видео через `extract_batch_segments()`
- **ThreadPoolExecutor**: используется для параллельной обработки файлов в batch режиме
- **Количество потоков**: автоматически определяется как `min(len(audio_files), os.cpu_count())` или задаётся через `max_workers`
- **Thread-safety**: компонент thread-safe для параллельной обработки разных файлов (каждый файл имеет свой `artifacts_dir`)

**Внешний параллелизм** (выше компонента):
- Можно запускать несколько экземпляров компонента параллельно на разных видео (разные `run_id`, разные `result_store` пути)
- Требования к изоляции:
  - Каждый файл должен иметь свой `artifacts_dir` для изоляции артефактов
  - Каждый файл должен иметь свой `tmp_path` для временных файлов
  - Сегменты из разных файлов не должны пересекаться

**Batch processing:**
- Компонент поддерживает batch processing через `extract_batch_segments()`:
  - Сбор сегментов из всех видео
  - Параллельная обработка файлов через ThreadPoolExecutor
  - Изоляция результатов для каждого файла
- Для включения batch processing установите `supports_batch = True` (уже установлено)
- Batch processing контролируется через параметры:
  - `max_workers`: количество параллельных воркеров (None = auto)
  - `enable_cpu_parallel`: включение CPU параллелизма (через MainProcessor)

**Ограничения:**
- Память: peak memory при параллельном выполнении пропорционален количеству параллельных воркеров
- CPU: рекомендуется использовать не более `os.cpu_count()` воркеров для оптимальной производительности
- GPU: не используется (CPU-only компонент)

### Примеры использования

#### Загрузка результатов

```python
import numpy as np

data = np.load("band_energy_extractor_features.npz", allow_pickle=True)
payload = data["payload"].item()

# Основной результат
band_edges = payload["band_edges"]  # [(0.0, 200.0), (200.0, 2000.0), (2000.0, 11025.0)]
band_energies = payload["band_energies"]  # [e_low, e_mid, e_high]
band_shares = payload["band_energy_shares"]  # [share_low, share_mid, share_high], сумма = 1.0

print(f"Band energies: {band_energies}")
print(f"Band shares: {band_shares}")
```

#### Анализ частотного баланса

```python
# Проверка доминирующей полосы
shares = payload["band_energy_shares"]
dominant_band = np.argmax(shares)
band_names = ["Low", "Mid", "High"]
print(f"Доминирующая полоса: {band_names[dominant_band]} ({shares[dominant_band]:.2%})")

# Если включены метрики баланса
if "band_balance_score" in payload:
    print(f"Balance score: {payload['band_balance_score']:.3f}")
    print(f"Contrast: {payload['band_contrast']:.3f}")
```

#### Анализ статистик

```python
# Если включены базовые статистики
if "band_energy_mean" in payload:
    mean = payload["band_energy_mean"]
    std = payload["band_energy_std"]
    median = payload["band_energy_median"]
    print(f"Mean energies: {mean}")
    print(f"Std energies: {std}")
    print(f"Median energies: {median}")
```

#### Анализ динамики (для run_segments)

```python
# Если включены метрики динамики
if "band_energy_stability" in payload:
    print(f"Stability: {payload['band_energy_stability']:.3f}")
    print(f"Transitions: {payload['band_transitions_count']} transitions")
    print(f"Transition rate: {payload['band_transitions_rate']:.4f} transitions/sec")
    print(f"Band diversity: {payload['band_diversity']} unique bands")
    
    # Распределение времени по полосам
    distribution = payload["band_distribution"]
    print("Band distribution:")
    for band_idx, proportion in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  Band {band_idx}: {proportion:.2%}")
```

#### Использование с shared_features

```python
# Если spectral_extractor был запущен ранее
shared_features = {
    "stft_magnitude": stft_magnitude,  # shape: (freq_bins, frames)
    "frequencies": frequencies,  # shape: (freq_bins,)
}

result = band_energy_extractor.run(input_uri, tmp_path, shared_features=shared_features)
# Band energy extractor использует предвычисленный STFT, избегая повторного вычисления
```

### Связанные компоненты

- **Segmenter**: предоставляет аудио файл и сегменты (`families.band_energy.segments[]`)
- **librosa**: основная библиотека для STFT и мел-шкалы
- **essentia** (опционально): альтернативный метод обработки
- **spectral_extractor**: может предоставлять `stft_magnitude` и `frequencies` в `shared_features` для оптимизации

### Visualization

**Рекомендуемые типы визуализации:**

1. **Bar chart энергий**:
   - Столбчатая диаграмма энергий по полосам
   - Логарифмическая шкала для энергий

2. **Pie chart долей**:
   - Круговая диаграмма долей энергий по полосам
   - Процентное соотношение каждой полосы

3. **Timeline визуализация** (для `run_segments()`):
   - График изменений энергий по времени
   - Отдельные линии для каждой полосы
   - Логарифмическая шкала для энергий

4. **Statistics comparison**:
   - Сравнение статистик (mean, std, median) по полосам
   - Grouped bar chart

5. **Balance metrics**:
   - Индикатор баланса (balance score)
   - Индикатор доминирующей полосы
   - Контраст между полосами

**Интерактивные элементы:**
- Tooltips с детальной информацией о полосах
- Zoom для детального просмотра timeline
- Фильтры по полосам
- Переключение между различными метриками

### Примечания

1. **По умолчанию 3 полосы**: low [0-200 Hz), mid [200-2000 Hz), high [2000-nyq Hz)
2. **Мел-шкала**: если `use_mel_bands=True`, полосы строятся через мел-шкалу (более музыкально релевантно)
3. **Essentia vs librosa**: Essentia может быть быстрее для frame-by-frame обработки, librosa — более гибкий
4. **Векторизованный биннинг**: librosa использует матричное умножение для эффективного вычисления
5. **Shared STFT**: оптимизация для случаев, когда STFT уже вычислен другим экстрактором
6. **Segment-based processing**: полезно для длинных видео и отслеживания изменений частотного баланса
7. **Dynamics metrics**: метрики динамики полезны для ML моделей и аналитики
8. **Balance metrics**: метрики баланса помогают анализировать частотное распределение
