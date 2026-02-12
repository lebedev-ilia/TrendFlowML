## `spectral_entropy_extractor` (Audio signal processing extractor)

### Назначение

Извлекает **спектральную энтропию** и связанные метрики из аудио сигнала. Спектральная энтропия измеряет распределенность энергии по частотному спектру - высокие значения указывают на равномерное распределение (белый шум), низкие - на концентрированную энергию (тональные звуки). Дополнительно вычисляются spectral flatness (спектральная плоскость) и spectral spread (разброс частот). Поддерживает segment-based обработку для отслеживания изменений спектральных характеристик.

**Версия**: 2.0.0  
**Категория**: spectral  
**GPU**: не требуется

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter) — полный аудио файл
- **`audio/segments.json`** (Segmenter contract `audio_segments_v1`) — для `run_segments()`:
  - `families.spectral_entropy.segments[]` — сегменты для обработки спектральной энтропии

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/spectral_entropy_extractor/spectral_entropy_extractor_features.npz` (**фиксированное имя**)

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

#### Полезные поля payload:

**Базовые статистики (feature-gated):**
- `spectral_entropy_stats`: статистики энтропии (dict, если `enable_basic_stats=True`)
  - `mean`: среднее значение энтропии (float, биты)
  - `std`: стандартное отклонение энтропии (float, биты)
  - `min`, `max`, `p25`, `p75`: расширенные статистики (если `enable_extended_stats=True`)

**Flatness метрики (feature-gated):**
- `spectral_flatness_stats`: статистики flatness (dict, если `enable_flatness=True`)
  - `mean`: среднее значение flatness (float, 0.0-1.0)
  - `std`: стандартное отклонение flatness (float)
  - `min`, `max`, `p25`, `p75`: расширенные статистики (если `enable_extended_stats=True`)

**Spread метрики (feature-gated):**
- `spectral_spread_stats`: статистики spread (dict, если `enable_spread=True`)
  - `mean`: среднее значение spread (float, ≥ 0)
  - `std`: стандартное отклонение spread (float)
  - `min`, `max`, `p25`, `p75`: расширенные статистики (если `enable_extended_stats=True`)

**Временные серии (feature-gated):**
- `spectral_entropy_series`: временная серия энтропии (list[float], если `enable_time_series=True`)
- `spectral_flatness_series`: временная серия flatness (list[float], если `enable_time_series=True` и `enable_flatness=True`)
- `spectral_spread_series`: временная серия spread (list[float], если `enable_time_series=True` и `enable_spread=True`)

**Метрики динамики (feature-gated, для `run_segments()`):**
- `spectral_entropy_stability`: стабильность энтропии (float, variance, если `enable_dynamics=True`)
- `spectral_entropy_transitions_count`: количество значимых переходов (int, если `enable_dynamics=True`)
- `spectral_entropy_transitions_rate`: частота переходов (transitions/frame, float, если `enable_dynamics=True`)
- `spectral_entropy_distribution`: распределение времени по уровням энтропии (dict, если `enable_dynamics=True`)
  - `low`: доля времени с низкой энтропией (< 33-й перцентиль)
  - `medium`: доля времени со средней энтропией (33-67-й перцентиль)
  - `high`: доля времени с высокой энтропией (> 67-й перцентиль)
- `spectral_entropy_diversity`: разнообразие значений энтропии (float, 0.0-1.0, если `enable_dynamics=True`)

**Дополнительные метрики:**
- `spectral_entropy_variance`: дисперсия энтропии (float)
- `spectral_entropy_min/max`: минимальное/максимальное значение энтропии (float)
- `spectral_flatness_variance`: дисперсия flatness (float, если `enable_flatness=True`)
- `spectral_flatness_min/max`: минимальное/максимальное значение flatness (float, если `enable_flatness=True`)
- `spectral_spread_variance`: дисперсия spread (float, если `enable_spread=True`)
- `spectral_spread_min/max`: минимальное/максимальное значение spread (float, если `enable_spread=True`)

**Метаданные:**
- `sample_rate`: частота дискретизации (int)
- `n_fft`: размер FFT окна (int)
- `hop_length`: размер hop для STFT (int)
- `use_mel`: используется ли mel-шкала (bool)
- `n_mels`: количество mel-фильтров (int, если `use_mel=True`)
- `smoothing_window`: размер окна сглаживания (int)
- `duration`: длительность аудио в секундах (float)
- `segments_count`: количество сегментов (int, если `run_segments()`)
- `device_used`: устройство обработки (str)
- `spectral_entropy_contract_version`: версия контракта ("spectral_entropy_contract_v1")
- `_features_enabled`: список включенных фичей (list[str])

### Feature Dependencies

**Зависимости между фичами:**
- `spectral_flatness_stats` зависит от `enable_flatness`
- `spectral_spread_stats` зависит от `enable_spread`
- Все метрики динамики зависят от `enable_dynamics` и `enable_time_series` (требуют временных серий)
- Расширенные статистики (`min`, `max`, `p25`, `p75`) зависят от `enable_extended_stats` и соответствующих базовых фичей

**Зависимости от других extractors:**
- **spectral_extractor** (опционально): может использовать предвычисленный `stft_magnitude` или `mel_spectrogram` из `shared_features` для оптимизации (избегает повторного вычисления спектрограммы)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.

**Библиотеки:**
- **librosa**: основная библиотека для STFT, Mel-спектрограмм и обработки аудио

**Примечание**: Используется только librosa (явно документировано). Essentia не поддерживается для spectral entropy.

### Конфигурация

```python
{
    "sample_rate": 22050,           # Частота дискретизации
    "n_fft": 2048,                  # Размер FFT окна
    "hop_length": 512,              # Размер hop для STFT
    "average_channels": True,       # Усреднять ли каналы для многоканального аудио
    "smoothing_window": 0,           # Размер окна сглаживания (0 = без сглаживания)
    "use_mel": False,               # Использовать ли mel-шкалу вместо линейной
    "n_mels": 128,                  # Количество mel-фильтров (если use_mel=True)
    "enable_audio_normalization": False,  # Нормализация аудио перед обработкой
    # Feature gating flags (все по умолчанию False)
    "enable_basic_stats": False,     # Включить базовые статистики (mean, std) для entropy
    "enable_flatness": False,        # Включить метрики flatness
    "enable_spread": False,          # Включить метрики spread
    "enable_time_series": False,     # Включить временные серии для всех метрик
    "enable_extended_stats": False,  # Включить расширенные статистики (min, max, p25, p75)
    "enable_dynamics": False,        # Включить метрики динамики (для run_segments)
    "device": "auto"                 # "auto" | "cuda" | "cpu"
}
```

### Feature Gating

Все фичи контролируются через персональные флаги (по умолчанию все выключены):

- `--spectral-entropy-enable-basic-stats`: Включить базовые статистики (`spectral_entropy_stats` с mean, std)
- `--spectral-entropy-enable-flatness`: Включить метрики flatness (`spectral_flatness_stats`, `spectral_flatness_series`)
- `--spectral-entropy-enable-spread`: Включить метрики spread (`spectral_spread_stats`, `spectral_spread_series`)
- `--spectral-entropy-enable-time-series`: Включить временные серии (`spectral_entropy_series`, `spectral_flatness_series`, `spectral_spread_series`)
- `--spectral-entropy-enable-extended-stats`: Включить расширенные статистики (`min`, `max`, `p25`, `p75` для всех метрик)
- `--spectral-entropy-enable-dynamics`: Включить метрики динамики (`spectral_entropy_stability`, `spectral_entropy_transitions_count/rate`, `spectral_entropy_distribution`, `spectral_entropy_diversity`)

**Зависимости фичей:**
- `enable_dynamics` требует `enable_time_series` (для отслеживания изменений по сегментам)
- `enable_extended_stats` требует соответствующие базовые фичи (например, `enable_basic_stats` для entropy)

### Алгоритм работы

#### 1. Загрузка и предобработка аудио

1. Загрузка аудио через `AudioUtils.load_audio()` с ресемплированием до `sample_rate`
2. Проверка минимальной длительности (< 1 секунды → error)
3. Обработка многоканального аудио:
   - Если `average_channels=True`: усреднение всех каналов
   - Иначе: использование первого канала
4. Нормализация аудио (опционально, если `enable_audio_normalization=True`): peak normalization

#### 2. Вычисление спектрограммы мощности

**Интеграция с shared_features:**
- Если `spectral_extractor` был запущен и предоставил `stft_magnitude` или `mel_spectrogram` в `shared_features`, используется предвычисленная спектрограмма (оптимизация)

**Вариант A: STFT (по умолчанию, если `use_mel=False`):**
- `librosa.stft()` с параметрами:
  - `n_fft`: 2048 (по умолчанию)
  - `hop_length`: 512 (по умолчанию)
- Результат: комплексный спектр → `S = |STFT|^2` (спектр мощности)
- Shape: `[n_freq_bins, n_time_frames]`

**Вариант B: Mel-спектрограмма** (если `use_mel=True`):
- `librosa.feature.melspectrogram()` с параметрами:
  - `n_fft`: 2048
  - `hop_length`: 512
  - `n_mels`: 128 (по умолчанию)
  - `power`: 2.0
- Результат: mel-спектрограмма мощности
- Shape: `[n_mels, n_time_frames]`

#### 3. Вычисление спектральной энтропии

Для каждого временного кадра:
1. Нормировка спектра: `P = S / (sum(S) + eps)`
   - `P` - вероятностное распределение энергии по частотам
2. Вычисление энтропии Шеннона:
   ```
   entropy = -sum(P * log2(P + eps))
   ```
   - Высокие значения → равномерное распределение (белый шум)
   - Низкие значения → концентрированная энергия (тональные звуки)
   - Диапазон: [0, log2(n_freq_bins)]

#### 4. Вычисление Spectral Flatness (если `enable_flatness=True`)

Для каждого временного кадра:
```
flatness = exp(mean(log(P + eps))) / (mean(P) + eps)
```
- Геометрическое среднее / арифметическое среднее
- Значения в диапазоне [0, 1]
- 1.0 → белый шум
- 0.0 → тональный сигнал

#### 5. Вычисление Spectral Spread (если `enable_spread=True`)

Для каждого временного кадра:
1. Создание частотного индекса: `freq_idx = linspace(0, 1, n_freq_bins)`
2. Вычисление среднего: `mu = sum(freq_idx * P)`
3. Вычисление стандартного отклонения:
   ```
   spread = sqrt(sum((freq_idx - mu)^2 * P))
   ```
- Измеряет "ширину" распределения энергии по частотам
- Значения ≥ 0

#### 6. Опциональное сглаживание

Если `smoothing_window > 1`:
- Применяется скользящее среднее с окном размера `smoothing_window`
- Сглаживание применяется ко всем метрикам (entropy, flatness, spread)
- Используется `np.convolve()` с равномерным ядром

#### 7. Вычисление статистик и метрик

- Базовые статистики: `mean`, `std` для всех метрик (если `enable_basic_stats=True`)
- Расширенные статистики: `min`, `max`, `p25`, `p75` (если `enable_extended_stats=True`)
- Метрики динамики (для `run_segments()`, если `enable_dynamics=True`):
  - Стабильность: variance энтропии
  - Переходы: количество и частота значимых изменений
  - Распределение: доля времени по уровням энтропии
  - Разнообразие: количество уникальных значений

### Валидация

**Параметры:**
- `sample_rate > 0`, `n_fft > 0`, `hop_length > 0`
- `n_fft >= 512` (минимум для STFT)
- `hop_length <= n_fft`
- `n_mels >= 3` (если `use_mel=True`)
- `smoothing_window >= 0`

**Выходные данные:**
- Энтропия в диапазоне [0, log2(n_freq_bins)]
- Flatness в диапазоне [0, 1]
- Spread ≥ 0
- Нет NaN или Inf значений
- Размеры массивов согласованы

### Error Codes

Детальные error codes для observability:
- `audio_load_failed`: не удалось загрузить аудио
- `audio_too_short`: аудио слишком короткое (< 1 секунды)
- `stft_computation_failed`: ошибка вычисления STFT/Mel-спектрограммы
- `entropy_computation_failed`: ошибка вычисления энтропии
- `flatness_computation_failed`: ошибка вычисления flatness
- `spread_computation_failed`: ошибка вычисления spread
- `invalid_parameters`: невалидные параметры
- `validation_failed`: ошибка валидации выходных данных
- `spectral_entropy_unknown`: неизвестная ошибка

### Обработка ошибок

- **No-fallback policy**: отсутствие обязательных входов → fail-fast (`RuntimeError`)
- **Валидация параметров**: невалидные параметры → `ValueError` при инициализации
- **Валидация выходных данных**: невалидные выходы → `status="error"` с детальным error_code
- **Обработка сегментов**: ошибки в отдельных сегментах логируются, но не останавливают обработку (если хотя бы один сегмент успешен)

### Особенности

- **Три метрики**: энтропия, flatness и spread для комплексного анализа спектра
- **Гибкость спектрограммы**: поддержка как STFT, так и Mel-спектрограммы
- **Сглаживание**: опциональное временное сглаживание для стабильности
- **Многоканальное аудио**: автоматическая обработка стерео/многоканального аудио
- **Эффективность**: работает на CPU, не требует GPU
- **Интеграция с spectral_extractor**: переиспользование спектрограммы через `shared_features`
- **Segment-based обработка**: поддержка `run_segments()` для отслеживания изменений по времени
- **Feature gating**: гибкое управление выходными фичами через флаги
- **Нормализация аудио**: опциональная нормализация перед обработкой

### Parallelization

`spectral_entropy_extractor` поддерживает параллельную обработку для повышения производительности:

#### Внутренний параллелизм (внутри компонента)

- **CPU Parallelism**: При использовании `extract_batch_segments()` (в режиме батчевой обработки) компонент использует `ThreadPoolExecutor` для параллельной обработки сегментов из разных видео. Это позволяет эффективно утилизировать многоядерные CPU.
  - **Количество потоков**: Контролируется параметром `max_workers` в `extract_batch_segments()`, который по умолчанию определяется автоматически (количество CPU ядер или количество файлов).
  - **Операции**: Параллелизуется загрузка аудио сегментов и вычисление спектральной энтропии для каждого сегмента.

#### Внешний параллелизм (выше компонента)

- **Батчевая обработка**: `spectral_entropy_extractor` разработан для эффективной работы в режиме батчевой обработки нескольких аудио файлов (`MainProcessor.run_batch()`).
  - **Изоляция**: Каждый файл обрабатывается в своем изолированном контексте (`AudioFileContext`), что предотвращает конфликты данных и артефактов.
  - **Shared Features**: Компонент может принимать `shared_features` (например, `stft_magnitude` от `spectral_extractor`) для оптимизации, избегая повторных вычислений.

#### Ограничения и требования

- **Thread-safety**: Компонент является thread-safe, так как не использует общие изменяемые состояния между потоками, кроме read-only моделей/конфигураций.
- **Память CPU**: Пиковая память при параллельном выполнении пропорциональна количеству одновременно обрабатываемых сегментов и их длительности.

### Performance characteristics

**Resource costs**:
- **CPU**: умеренные (STFT/Mel-спектрограмма, вычисления энтропии)
- **GPU**: не используется
- **Estimated duration**: ~0.9 секунд для типичного аудио файла

**Параметры производительности**:
- `n_fft`: большие значения → лучше частотное разрешение, но медленнее
- `hop_length`: меньшие значения → больше временных кадров, но медленнее
- `use_mel`: Mel-спектрограмма может быть быстрее для больших файлов
- `smoothing_window`: сглаживание добавляет небольшие накладные расходы
- `enable_time_series`: временные серии увеличивают размер NPZ файла

### Связанные компоненты

- **AudioUtils**: загрузка и предобработка аудио
- **BaseExtractor**: базовый интерфейс экстрактора
- **librosa**: библиотека для STFT, Mel-спектрограмм и обработки аудио
- **spectral_extractor**: опциональная интеграция через `shared_features` для переиспользования спектрограммы

### Visualization

**Рекомендуемые типы визуализации:**

1. **Timeline графики** (для временных серий):
   - Line chart для `spectral_entropy_series`, `spectral_flatness_series`, `spectral_spread_series`
   - Показывает изменения метрик по времени
   - Полезно для анализа динамики спектральных характеристик

2. **Распределения**:
   - Histogram для значений энтропии, flatness, spread
   - Показывает распределение метрик по всему аудио
   - Полезно для понимания типичных значений

3. **Статистики**:
   - Bar chart для mean, std, min, max метрик
   - Сравнение статистик между разными метриками
   - Полезно для быстрого обзора характеристик

4. **Метрики динамики** (для `run_segments()`):
   - Pie chart для `spectral_entropy_distribution` (low/medium/high)
   - Line chart для transitions по времени
   - Показывает стабильность и изменения энтропии

**Интерактивные элементы:**
- Tooltips с точными значениями
- Zoom для детального просмотра timeline
- Фильтры по диапазонам значений
- Переключение между метриками (entropy/flatness/spread)

### Примечания

1. **Интерпретация энтропии**:
   - Высокая энтропия → равномерное распределение энергии (белый шум, шумная среда)
   - Низкая энтропия → концентрированная энергия (тональные звуки, музыкальные инструменты, речь)

2. **Spectral Flatness**:
   - Полезен для различения тональных и шумовых компонентов
   - Часто используется в аудио кодеках для определения типа сигнала

3. **Spectral Spread**:
   - Измеряет "ширину" спектра
   - Полезен для анализа тембра и характеристик звука

4. **Выбор между STFT и Mel**:
   - STFT: линейная частотная шкала, лучше для точного частотного анализа
   - Mel: логарифмическая шкала, лучше соответствует восприятию человека, часто используется в ML

5. **Сглаживание**:
   - Рекомендуется для визуализации и стабильности метрик
   - Может скрыть быстрые изменения в сигнале

6. **Многоканальное аудио**:
   - По умолчанию усредняются все каналы для репрезентативности
   - Можно использовать первый канал, установив `average_channels=False`

7. **Численная устойчивость**:
   - Используется `eps=1e-12` для избежания деления на ноль и log(0)

8. **Contract versioning**:
   - Версия контракта фиксируется в `spectral_entropy_contract_version` для совместимости с downstream extractors
